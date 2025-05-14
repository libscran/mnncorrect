#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "scran_tests/scran_tests.hpp"

#include "mnncorrect/AutomaticOrder.hpp"
#include <random>
#include <algorithm>
#include <cstddef>

TEST(AutomaticOrder, RedistributeCorrectedObservations) {
    constexpr std::size_t num_dim = 5;
    int num_total = 50;

    mnncorrect::internal::FindBatchNeighborsResults<int, double> batch_nns;
    batch_nns.target_ids = std::vector<int>{ 1, 3, 7, 11, 23, 31 };

    mnncorrect::internal::CorrectTargetResults correct_info;
    correct_info.batch.resize(num_total, 123456789);
    correct_info.batch[1] = 2;
    correct_info.batch[3] = 1;
    correct_info.batch[7] = 0;
    correct_info.batch[11] = 1;
    correct_info.batch[23] = 2;
    correct_info.batch[31] = 1;

    auto data = scran_tests::simulate_vector(num_total * num_dim, [&]{
        scran_tests::SimulationParameters params;
        params.seed = 69;
        return params;
    }());

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    mnncorrect::internal::RedistributeCorrectedObservationsWorkspace<int, double> work;
    work.offsets.resize(10); // filling it with some nonsense to check that the function wipes it.
    work.buffer.resize(1);

    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        std::vector<mnncorrect::internal::BatchInfo<int, double> > batches(3);
        mnncorrect::internal::redistribute_corrected_observations(
            num_dim,
            batch_nns,
            correct_info,
            data.data(),
            builder,
            num_threads,
            work,
            batches
        );

        for (int b = 0; b < 3; ++b) {
            const auto& curbatch = batches[b];
            EXPECT_EQ(curbatch.extras.size(), 1);

            std::vector<int> ids;
            if (b == 0) {
                ids = std::vector<int>{ 7 };
            } else if (b == 1) {
                ids = std::vector<int>{ 3, 11, 31 };
            } else {
                ids = std::vector<int>{ 1, 23 };
            }

            EXPECT_EQ(curbatch.extras[0].ids, ids);
            auto searcher = curbatch.extras[0].index->initialize();

            std::vector<int> idx;
            std::vector<double> dist;
            for (decltype(ids.size()) i = 0, end = ids.size(); i < end; ++i) {
                auto id = ids[i];
                searcher->search(data.data() + id * num_dim, 1, &idx, &dist);
                EXPECT_EQ(idx[0], i);
                EXPECT_EQ(dist[0], 0);
            }
        }
    }
}

/**********************************************************/

class AutomaticOrder2 : public mnncorrect::internal::AutomaticOrder<int, double, knncolle::Matrix<int, double> > {
public:
    template<typename ... Args_>
    AutomaticOrder2(Args_&&... args) : AutomaticOrder<int, double, knncolle::Matrix<int, double> >(std::forward<Args_>(args)...) {}

    const auto& get_batches() const { 
        return my_batches;
    }

    auto advance() {
        return next(true);
    }
};

class AutomaticOrderInitTest : public ::testing::Test {};

TEST_F(AutomaticOrderInitTest, Empty) {
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    AutomaticOrder2 overlord(
        5,
        std::vector<int>{}, 
        std::vector<const double*>{},
        static_cast<double*>(NULL),
        builder,
        /* num_neighbors = */ 10,
        /* tolerance = */ 3,
        mnncorrect::MergePolicy::RSS,
        /* num_threads = */ 1
    );
    EXPECT_TRUE(overlord.get_batches().empty());

    std::string msg;
    try {
        AutomaticOrder2(
            5,
            std::vector<int>{ 0 }, 
            std::vector<const double*>{},
            static_cast<double*>(NULL),
            builder,
            /* num_neighbors = */ 10,
            /* tolerance = */ 3,
            mnncorrect::MergePolicy::RSS,
            /* num_threads = */ 1
        );
    } catch (std::exception& e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("length of") != std::string::npos);
}

template<typename Index_, typename Float_>
static void check_initialization(
    std::size_t num_dim,
    const std::vector<mnncorrect::internal::BatchInfo<Index_, Float_> >& batches,
    const std::vector<const double*>& sources,
    const double* corrected)
{
    auto nbatches = batches.size();

    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        const auto& curbatch = batches[b];
        auto num_obs = curbatch.num_obs;
        EXPECT_EQ(curbatch.index->num_observations(), num_obs);

        auto searcher = curbatch.index->initialize();
        std::vector<int> indices;
        std::vector<double> distances;

        for (int c = 0; c < num_obs; ++c) {
            auto srcptr = sources[b] + c * num_dim;
            std::vector<double> srcvec(srcptr, srcptr + num_dim);
            auto corptr = corrected + (curbatch.offset + c) * num_dim;
            std::vector<double> corvec(corptr, corptr + num_dim);
            EXPECT_EQ(corvec, srcvec);

            searcher->search(corptr, 1, &indices, &distances);
            EXPECT_EQ(indices.size(), 1);
            EXPECT_EQ(indices[0], c);
            EXPECT_EQ(distances[0], 0);
        }
    }
}

template<typename Float_>
static std::vector<const double*> pointerize(const std::vector<std::vector<Float_> >& data) {
    std::vector<const double*> output;
    output.reserve(data.size());
    for (const auto& batch : data) {
        output.push_back(batch.data());
    }
    return output;
}

TEST_F(AutomaticOrderInitTest, Input) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 42 + ndim * 10 + sizes[b];
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Testing multiple threads to check for correct parallelization of index construction.
    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        std::vector<double> corrected(ndim * ntotal);
        AutomaticOrder2 overlord(
            ndim,
            sizes,
            ptrs,
            corrected.data(),
            builder,
            /* num_neighbors = */ 10,
            /* tolerance = */ 3,
            mnncorrect::MergePolicy::INPUT,
            num_threads
        );

        // Batches should be sorted by input order.
        const auto& batches = overlord.get_batches(); 
        EXPECT_EQ(batches[0].num_obs, 100);
        EXPECT_EQ(batches[0].offset, 0);
        EXPECT_EQ(batches[1].num_obs, 200);
        EXPECT_EQ(batches[1].offset, 100);
        EXPECT_EQ(batches[2].num_obs, 150);
        EXPECT_EQ(batches[2].offset, 300);

        check_initialization(ndim, batches, ptrs, corrected.data());
    }
}

TEST_F(AutomaticOrderInitTest, MaxSize) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 69 + ndim * 10 + sizes[b];
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> corrected(ndim * ntotal);
    AutomaticOrder2 overlord(
        ndim,
        sizes,
        ptrs,
        corrected.data(),
        builder,
        /* num_neighbors = */ 10,
        /* tolerance = */ 3,
        mnncorrect::MergePolicy::SIZE,
        /* num_threads = */ 1
    );

    // Batches should be sorted by size.
    const auto& batches = overlord.get_batches(); 
    EXPECT_EQ(batches[0].num_obs, 200);
    EXPECT_EQ(batches[0].offset, 100);
    EXPECT_EQ(batches[1].num_obs, 150);
    EXPECT_EQ(batches[1].offset, 300);
    EXPECT_EQ(batches[2].num_obs, 100);
    EXPECT_EQ(batches[2].offset, 0);

    std::vector<const double*> resorted_ptrs { ptrs[1], ptrs[2], ptrs[0] };
    check_initialization(ndim, batches, resorted_ptrs, corrected.data());
}

TEST_F(AutomaticOrderInitTest, MaxVariance) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 4269 + ndim * 10 + sizes[b];
            sparams.lower = -3.0 * (b + 1); // later batches are more variable.
            sparams.upper = 3.0 * (b + 1);
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Testing multiple threads to check for threading in the variance calculation.
    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        std::vector<double> corrected(ndim * ntotal);
        AutomaticOrder2 overlord(
            ndim,
            sizes,
            ptrs,
            corrected.data(),
            builder,
            /* num_neighbors = */ 10,
            /* tolerance = */ 3,
            mnncorrect::MergePolicy::VARIANCE,
            num_threads
        );

        // Batches should be sorted by variance.
        const auto& batches = overlord.get_batches(); 
        EXPECT_EQ(batches[0].num_obs, 150);
        EXPECT_EQ(batches[0].offset, 300);
        EXPECT_EQ(batches[1].num_obs, 200);
        EXPECT_EQ(batches[1].offset, 100);
        EXPECT_EQ(batches[2].num_obs, 100);
        EXPECT_EQ(batches[2].offset, 0);

        std::vector<const double*> resorted_ptrs { ptrs[2], ptrs[1], ptrs[0] };
        check_initialization(ndim, batches, resorted_ptrs, corrected.data());
    }
}

TEST_F(AutomaticOrderInitTest, MaxRss) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 50, 500 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 4269 + ndim * 10 + sizes[b];
            sparams.lower = -3.0 / (b + 1); // later batches are less variable.
            sparams.upper = 3.0 / (b + 1);
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> corrected(ndim * ntotal);
    AutomaticOrder2 overlord(
        ndim,
        sizes,
        ptrs,
        corrected.data(),
        builder,
        /* num_neighbors = */ 10,
        /* tolerance = */ 3,
        mnncorrect::MergePolicy::RSS,
        /* num_threads = */ 1
    );

    // Sorting by RSS; so even though the first batch is most variable,
    // it has fewer observations and so the second batch has a higher RSS.
    const auto& batches = overlord.get_batches(); 
    EXPECT_EQ(batches[0].num_obs, 500);
    EXPECT_EQ(batches[0].offset, 50);
    EXPECT_EQ(batches[1].num_obs, 50);
    EXPECT_EQ(batches[1].offset, 0);

    std::vector<const double*> resorted_ptrs { ptrs[1], ptrs[0] };
    check_initialization(ndim, batches, resorted_ptrs, corrected.data());
}

/**********************************************************/

class AutomaticOrderNextTest : public ::testing::TestWithParam<std::tuple<int, std::vector<int> > > {};

TEST_P(AutomaticOrderNextTest, Basic) {
    auto params = GetParam();
    auto k = std::get<0>(params);
    auto sizes = std::get<1>(params);
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    constexpr std::size_t ndim = 7;
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 4269 + ndim * 10 + sizes[b];
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> corrected(ndim * ntotal);
    AutomaticOrder2 overlord(
        ndim,
        sizes,
        ptrs,
        corrected.data(),
        builder,
        /* num_neighbors = */ k,
        /* tolerance = */ 3,
        mnncorrect::MergePolicy::INPUT,
        /* num_threads = */ 1
    );

    bool has_next = true;
    while (has_next) {
        has_next = overlord.advance();

        // Check that all observations are represented here.
        std::vector<unsigned char> present(ntotal);
        int accumulated = 0; 
        for (const auto& batch : overlord.get_batches()) {
            accumulated += batch.num_obs;
            std::fill_n(present.begin() + batch.offset, batch.num_obs, 1);
            for (const auto& extra : batch.extras) {
                accumulated += extra.ids.size();
                for (auto e : extra.ids) {
                    present[e] = 1;
                }
            }
        }

        EXPECT_EQ(accumulated, ntotal);
        EXPECT_EQ(std::accumulate(present.begin(), present.end(), 0), ntotal);
    }

    // Same results in parallel.
    std::vector<double> pcorrected(ndim * ntotal);
    AutomaticOrder2 poverlord(
        ndim,
        sizes,
        ptrs,
        pcorrected.data(),
        builder,
        /* num_neighbors = */ k,
        /* tolerance = */ 3,
        mnncorrect::MergePolicy::INPUT,
        /* num_threads = */ 3
    );
    poverlord.merge();

    EXPECT_EQ(corrected, pcorrected);
}

INSTANTIATE_TEST_SUITE_P(
    AutomaticOrder,
    AutomaticOrderNextTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // Number of neighbors
        ::testing::Values(
            std::vector<int>{10, 20},        
            std::vector<int>{10, 20, 30}, 
            std::vector<int>{100, 50, 80}, 
            std::vector<int>{50, 30, 100, 90},
            std::vector<int>{50, 40, 30, 20, 10}
        )
    )
);
