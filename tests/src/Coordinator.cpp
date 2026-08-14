#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "scran_tests/scran_tests.hpp"

#include "mnncorrect/Coordinator.hpp"
#include <random>
#include <algorithm>
#include <cstddef>

class Coordinator2 : public mnncorrect::Coordinator<int, double, knncolle::Matrix<int, double> > {
public:
    template<typename ... Args_>
    Coordinator2(Args_&&... args) : Coordinator<int, double, knncolle::Matrix<int, double> >(std::forward<Args_>(args)...) {}

    const auto& get_meta_batches() const { 
        return my_meta_batches;
    }

    const auto& get_meta_batch_assignments() const { 
        return my_meta_batch_assignments;
    }

    auto advance() {
        return next(true);
    }
};

/**********************************************************/

class CoordinatorInitTest : public ::testing::Test {
protected:
    template<typename Float_>
    static void check_initialization(std::size_t num_dim, const Coordinator2& overlord, const Float_* const data) {
        const auto& meta_batches = overlord.get_meta_batches();
        const auto& meta_batch_assignments = overlord.get_meta_batch_assignments();
        const int nbatches = meta_batches.size();

        for (int b = 0; b < nbatches; ++b) {
            const auto& cur_meta_batch = meta_batches[b];
            auto num_obs = cur_meta_batch.original_ids.size;
            EXPECT_EQ(cur_meta_batch.original_index->num_observations(), num_obs);

            auto searcher = cur_meta_batch.original_index->initialize();
            std::vector<int> indices;
            std::vector<double> distances;

            for (int c = 0; c < num_obs; ++c) {
                EXPECT_EQ(meta_batch_assignments[cur_meta_batch.original_ids.start + c], b);
                auto srcptr = data + (cur_meta_batch.original_ids.start + c) * num_dim;
                searcher->search(srcptr, 1, &indices, &distances);
                EXPECT_EQ(indices.size(), 1);
                EXPECT_EQ(indices[0], c);
                EXPECT_EQ(distances[0], 0);
            }
        }
    }
};

TEST_F(CoordinatorInitTest, Empty) {
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // No batches at all.
    {
        Coordinator2 overlord(
            5,
            0,
            std::vector<mnncorrect::Batch<int> >{}, 
            static_cast<double*>(NULL),
            builder,
            /* num_neighbors = */ 10,
            /* num_steps = */ 3,
            mnncorrect::MergePolicy::RSS,
            /* num_threads = */ 1
        );
        EXPECT_TRUE(overlord.get_meta_batches().empty());
    }

    // Empty batches only.
    {
        mnncorrect::Batch<int> batch;
        Coordinator2 overlord(
            5,
            0,
            std::vector<mnncorrect::Batch<int> >(5), 
            static_cast<double*>(NULL),
            builder,
            /* num_neighbors = */ 10,
            /* num_steps = */ 3,
            mnncorrect::MergePolicy::RSS,
            /* num_threads = */ 1
        );
        EXPECT_TRUE(overlord.get_meta_batches().empty());
    }
}

TEST_F(CoordinatorInitTest, Input) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };

    std::vector<mnncorrect::Batch<int> > batches;
    int ntotal = 0; 
    for (auto s : sizes) {
        mnncorrect::Batch<int> current;
        current.start = ntotal;
        current.size = s;
        batches.push_back(current);
        ntotal += s;
    }

    auto data = scran_tests::simulate_vector(ntotal * ndim, {});
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Testing multiple threads to check for correct parallelization of index construction.
    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        auto copy = data;
        Coordinator2 overlord(
            ndim,
            ntotal,
            batches,
            copy.data(),
            builder,
            /* num_neighbors = */ 10,
            /* num_steps = */ 3,
            mnncorrect::MergePolicy::INPUT,
            num_threads
        );

        // Batches should be sorted by input order.
        const auto& mbatches = overlord.get_meta_batches(); 
        EXPECT_EQ(mbatches[0].original_ids.size, 100);
        EXPECT_EQ(mbatches[0].original_ids.start, 0);
        EXPECT_EQ(mbatches[1].original_ids.size, 200);
        EXPECT_EQ(mbatches[1].original_ids.start, 100);
        EXPECT_EQ(mbatches[2].original_ids.size, 150);
        EXPECT_EQ(mbatches[2].original_ids.start, 300);

        check_initialization(ndim, overlord, data.data());
        EXPECT_EQ(data, copy); // not yet changed by initialization.
    }
}

TEST_F(CoordinatorInitTest, MaxSize) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };

    std::vector<mnncorrect::Batch<int> > batches;
    int ntotal = 0; 
    for (auto s : sizes) {
        mnncorrect::Batch<int> current;
        current.start = ntotal;
        current.size = s;
        batches.push_back(current);
        ntotal += s;
    }

    auto data = scran_tests::simulate_vector(ntotal * ndim, {});
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> copy = data;
    Coordinator2 overlord(
        ndim,
        ntotal,
        batches,
        copy.data(),
        builder,
        /* num_neighbors = */ 10,
        /* num_steps = */ 3,
        mnncorrect::MergePolicy::SIZE,
        /* num_threads = */ 1
    );

    // Batches should be sorted by size.
    const auto& mbatches = overlord.get_meta_batches(); 
    EXPECT_EQ(mbatches[0].original_ids.size, 200);
    EXPECT_EQ(mbatches[0].original_ids.start, 100);
    EXPECT_EQ(mbatches[1].original_ids.size, 150);
    EXPECT_EQ(mbatches[1].original_ids.start, 300);
    EXPECT_EQ(mbatches[2].original_ids.size, 100);
    EXPECT_EQ(mbatches[2].original_ids.start, 0);

    check_initialization(ndim, overlord, data.data());
    EXPECT_EQ(copy, data);
}

TEST_F(CoordinatorInitTest, MaxVariance) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };

    std::vector<mnncorrect::Batch<int> > batches;
    int ntotal = 0; 
    for (auto s : sizes) {
        mnncorrect::Batch<int> current;
        current.start = ntotal;
        current.size = s;
        batches.push_back(current);
        ntotal += s;
    }

    const int nbatches = batches.size();
    auto data = scran_tests::simulate_vector(ntotal * ndim, {});
    for (int b = 0; b < nbatches; ++b) {
        const std::size_t offset = batches[b].start * ndim;
        const std::size_t len = batches[b].size * ndim;
        for (std::size_t i = 0; i < len; ++i) {
            data[offset + i] *= (b + 1); // later batches are more variable
        }
    }
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Testing multiple threads to check for threading in the variance calculation.
    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        auto copy = data;
        Coordinator2 overlord(
            ndim,
            ntotal,
            batches,
            copy.data(),
            builder,
            /* num_neighbors = */ 10,
            /* num_steps = */ 3,
            mnncorrect::MergePolicy::VARIANCE,
            num_threads
        );

        // Batches should be sorted by variance.
        const auto& mbatches = overlord.get_meta_batches(); 
        EXPECT_EQ(mbatches[0].original_ids.size, 150);
        EXPECT_EQ(mbatches[0].original_ids.start, 300);
        EXPECT_EQ(mbatches[1].original_ids.size, 200);
        EXPECT_EQ(mbatches[1].original_ids.start, 100);
        EXPECT_EQ(mbatches[2].original_ids.size, 100);
        EXPECT_EQ(mbatches[2].original_ids.start, 0);

        check_initialization(ndim, overlord, data.data());
        EXPECT_EQ(copy, data);
    }
}

TEST_F(CoordinatorInitTest, MaxRss) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 50, 500 };

    std::vector<mnncorrect::Batch<int> > batches;
    int ntotal = 0; 
    for (auto s : sizes) {
        mnncorrect::Batch<int> current;
        current.start = ntotal;
        current.size = s;
        batches.push_back(current);
        ntotal += s;
    }

    const int nbatches = batches.size();
    auto data = scran_tests::simulate_vector(ntotal * ndim, {});
    for (int b = 0; b < nbatches; ++b) {
        const std::size_t offset = batches[b].start * ndim;
        const std::size_t len = batches[b].size * ndim;
        for (std::size_t i = 0; i < len; ++i) {
            data[offset + i] /= (b + 1); // later batches are less variable.
        }
    }
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    auto copy = data;
    Coordinator2 overlord(
        ndim,
        ntotal,
        batches,
        copy.data(),
        builder,
        /* num_neighbors = */ 10,
        /* num_steps = */ 3,
        mnncorrect::MergePolicy::RSS,
        /* num_threads = */ 1
    );

    // Sorting by RSS; so even though the first batch is most variable,
    // it has fewer observations and so the second batch has a higher RSS.
    const auto& mbatches = overlord.get_meta_batches(); 
    EXPECT_EQ(mbatches[0].original_ids.size, 500);
    EXPECT_EQ(mbatches[0].original_ids.start, 50);
    EXPECT_EQ(mbatches[1].original_ids.size, 50);
    EXPECT_EQ(mbatches[1].original_ids.start, 0);

    check_initialization(ndim, overlord, data.data());
}

/**********************************************************/

class CoordinatorNextTest : public ::testing::TestWithParam<std::tuple<int, std::vector<int> > > {};

TEST_P(CoordinatorNextTest, Basic) {
    auto params = GetParam();
    auto k = std::get<0>(params);
    auto sizes = std::get<1>(params);

    std::vector<mnncorrect::Batch<int> > batches;
    int ntotal = 0; 
    for (auto s : sizes) {
        mnncorrect::Batch<int> current;
        current.start = ntotal;
        current.size = s;
        batches.push_back(current);
        ntotal += s;
    }

    const std::size_t ndim = 6;
    auto data = scran_tests::simulate_vector(ntotal * ndim, {});
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    auto copy = data;
    Coordinator2 overlord(
        ndim,
        ntotal,
        batches,
        copy.data(),
        builder,
        /* num_neighbors = */ k,
        /* num_steps = */ 3,
        mnncorrect::MergePolicy::INPUT,
        /* num_threads = */ 1
    );

    bool has_next = true;
    while (has_next) {
        has_next = overlord.advance();

        // Check that all observations are represented here.
        std::vector<unsigned char> present(ntotal);
        for (const auto& mbatch : overlord.get_meta_batches()) {
            std::fill_n(present.begin() + mbatch.original_ids.start, mbatch.original_ids.size, 1);
            for (const auto& extra : mbatch.corrected) {
                ASSERT_GT(extra.ids.size(), 0);
                for (auto e : extra.ids) {
                    present[e] = 1;
                }
            }
        }
        for (auto p : present) {
            EXPECT_TRUE(p);
        }

        // Check that none of the batch assignments contain the just-corrected batch.
        auto nbatches = overlord.get_meta_batches().size();
        for (auto x : overlord.get_meta_batch_assignments()) {
            EXPECT_LT(x, nbatches);
        }

        // Check that the corrected index actually includes the redistributed observations.
        std::vector<int> indices;
        std::vector<double> distances;
        for (const auto& mbatch : overlord.get_meta_batches()) {
            for (const auto& extra : mbatch.corrected) {
                auto eindex = extra.index->initialize();
                for (std::size_t e = 0, esize = extra.ids.size(); e < esize; ++e) {
                    eindex->search(copy.data() + extra.ids[e] * ndim, 1, &indices, &distances);
                    EXPECT_EQ(distances[0], 0);
                    if (k > 1) {
                        // Only check this if the center of mass is not just a single point,
                        // as multiple MNN-involved cells in the target metabatch could be corrected to the same position.
                        EXPECT_EQ(indices[0], e);
                    }
                }
            }
        }
    }

    EXPECT_NE(copy, data); // indeed some correction was performed.

    // Same results in parallel.
    std::vector<double> pcopy = data;
    Coordinator2 poverlord(
        ndim,
        ntotal,
        batches,
        pcopy.data(),
        builder,
        /* num_neighbors = */ k,
        /* num_steps = */ 3,
        mnncorrect::MergePolicy::INPUT,
        /* num_threads = */ 3
    );
    poverlord.merge();
    EXPECT_EQ(copy, pcopy);
}

INSTANTIATE_TEST_SUITE_P(
    Coordinator,
    CoordinatorNextTest,
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
