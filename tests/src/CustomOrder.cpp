#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "scran_tests/scran_tests.hpp"
#include "knncolle/knncolle.hpp"

#include "mnncorrect/CustomOrder.hpp"
#include "mnncorrect/find_mutual_nns.hpp"
#include <random>
#include <algorithm>

struct CustomOrder2 : public mnncorrect::internal::CustomOrder<int, int, double> {
    template<typename ... Args_>
    CustomOrder2(Args_&& ... args) : mnncorrect::internal::CustomOrder<int, int, double>(std::forward<Args_>(args)...) {}

    const auto& get_neighbors_ref () const { 
        return my_neighbors_ref;
    }
    const auto& get_neighbors_target () const { 
        return my_neighbors_target;
    }

    size_t get_ncorrected() const { 
        return my_ncorrected;
    }

    void test_update(size_t latest) {
        update(latest);
        return;
    }
};

class CustomOrderTest : public ::testing::TestWithParam<std::tuple<int, std::vector<size_t>, bool> > {
protected:
    void SetUp() {
        auto param = GetParam();
        k = std::get<0>(param);
        sizes = std::get<1>(param);
        reversed = std::get<2>(param);

        data.resize(sizes.size());
        ptrs.resize(sizes.size());
        for (size_t b = 0; b < sizes.size(); ++b) {
            data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
                scran_tests::SimulationParameters sparams;
                sparams.lower = -2;
                sparams.upper = 2;
                sparams.seed = 42 + ndim * 10 + k * 100 + sizes[b];
                return sparams;
            }());
            ptrs[b] = data[b].data();
        }

        total_size = std::accumulate(sizes.begin(), sizes.end(), 0) * ndim;
    }

protected:
    // Constants.
    int ndim = 5;
    knncolle::VptreeBuilder<> builder;

    // Parameters.
    int k;
    std::vector<size_t> sizes;
    bool reversed;

    // Simulated.
    std::vector<std::vector<double> > data;
    std::vector<const double*> ptrs;
    int total_size;

public:
    static void compare_to_naive(const std::vector<int>& indices, const std::vector<double>& distances, const std::vector<std::pair<int, double> >& updated) {
        size_t n = indices.size();
        ASSERT_EQ(n, updated.size());
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(indices[i], updated[i].first);
            EXPECT_EQ(distances[i], updated[i].second);
        }
    }
};

TEST_P(CustomOrderTest, CheckInitialization) {
    std::vector<size_t> ordering(sizes.size());
    std::iota(ordering.begin(), ordering.end(), 0);
    if (reversed) {
        std::reverse(ordering.begin(), ordering.end());
    }

    std::vector<double> output(total_size);
    CustomOrder2 coords(
        ndim,
        sizes,
        ptrs,
        output.data(),
        builder,
        /* num_neighbors = */ k,
        /* order = */ ordering,
        /* mass_cap = */ -1,
        /* nthreads = */ 1
    );

    size_t ncorrected = coords.get_ncorrected();
    EXPECT_EQ(ncorrected, sizes[ordering[0]]);
    EXPECT_EQ(std::vector<double>(output.begin(), output.begin() + ncorrected * ndim), data[ordering[0]]);

    const auto& rneighbors = coords.get_neighbors_ref();
    EXPECT_EQ(rneighbors.size(), ncorrected);
    EXPECT_EQ(rneighbors[0].size(), k);

    const auto& lneighbors = coords.get_neighbors_target();
    EXPECT_EQ(lneighbors.size(), sizes[ordering[1]]);
    EXPECT_EQ(lneighbors[0].size(), k);
}

TEST_P(CustomOrderTest, CheckUpdate) {
    std::vector<size_t> ordering(sizes.size());
    std::iota(ordering.begin(), ordering.end(), 0);
    if (reversed) {
        std::reverse(ordering.begin(), ordering.end());
    }

    std::vector<CustomOrder2> all_coords;
    all_coords.reserve(2);
    std::vector<std::vector<double> > all_output;
    all_output.reserve(2);

    for (size_t t = 1; t <= 3; t += 2) {
        all_output.emplace_back(total_size);
        all_coords.emplace_back(
            ndim,
            sizes,
            ptrs,
            all_output.back().data(),
            builder,
            /* num_neighbors = */ k,
            /* order = */ ordering,
            /* mass_cap = */ -1,
            /* nthreads = */ t
        );
    }

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto& coords0 = all_coords[0];
        size_t sofar = coords0.get_ncorrected();
        auto current = ordering[b];
        size_t cursize = sizes[current];

        // Applying an update. We mock up some corrected data so that the builders work correctly.
        auto corrected = scran_tests::simulate_vector(cursize * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 6942 + b + ndim * k;
            return sparams;
        }());

        size_t output_offset = ndim * sofar;
        for (size_t i = 0; i < all_coords.size(); ++i) {
            std::copy(corrected.begin(), corrected.end(), all_output[i].data() + output_offset);
            all_coords[i].test_update(b);
        }

        // Check that the update to the neighbors works as expected.
        size_t new_sofar = coords0.get_ncorrected();
        EXPECT_EQ(sofar + cursize, new_sofar);

        if (b + 1 != sizes.size()) {
            auto next = ordering[b + 1];
            std::vector<int> indices;
            std::vector<double> distances;

            const auto& tdata = data[next];
            size_t tnum = sizes[next];

            const auto& rneighbors = coords0.get_neighbors_ref();
            EXPECT_EQ(rneighbors.size(), new_sofar);

            auto target_index = builder.build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, tnum, tdata.data()));
            auto target_searcher = target_index->initialize();
            for (size_t x = 0; x < new_sofar; ++x) {
                target_searcher->search(all_output[0].data() + x * ndim, k, &indices, &distances);
                compare_to_naive(indices, distances, rneighbors[x]);
            }

            const auto& tneighbors = coords0.get_neighbors_target();
            EXPECT_EQ(tneighbors.size(), tnum);

            auto ref_index = builder.build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, new_sofar, all_output[0].data()));
            auto ref_searcher = ref_index->initialize();
            for (size_t x = 0; x < tnum; ++x) {
                ref_searcher->search(tdata.data() + x * ndim, k, &indices, &distances);
                compare_to_naive(indices, distances, tneighbors[x]);
            }
        }
    }

    // Same results when run in parallel.
    EXPECT_EQ(all_output[0], all_output[1]);
}

INSTANTIATE_TEST_SUITE_P(
    CustomOrder,
    CustomOrderTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // Number of neighbors
        ::testing::Values(
            std::vector<size_t>{10, 20},        
            std::vector<size_t>{10, 20, 30}, 
            std::vector<size_t>{100, 50, 80}, 
            std::vector<size_t>{50, 30, 100, 90} 
        ),
        ::testing::Values(false, true) // whether to use the reverse order
    )
);

TEST(CustomOrder, InitializationError) {
    int ndim = 5;
    std::vector<size_t> sizes { 10, 20, 30 };
    std::vector<const double*> ptrs{ NULL, NULL, NULL };
    int k = 10;
    knncolle::VptreeBuilder<> builder;
    double* output = NULL;

    scran_tests::expect_error([&]() {
        std::vector<size_t> ordering;
        CustomOrder2 coords(
            ndim,
            sizes,
            ptrs,
            output,
            builder,
            /* num_neighbors = */ k,
            /* order = */ ordering,
            /* mass_cap = */ -1,
            /* nthreads = */ 1
        );
    }, "number of batches");

    scran_tests::expect_error([&]() {
        std::vector<size_t> ordering{ 0, 1, 2 };
        CustomOrder2 coords(
            ndim,
            sizes,
            std::vector<const double*>(),
            output,
            builder,
            /* num_neighbors = */ k,
            /* order = */ ordering,
            /* mass_cap = */ -1,
            /* nthreads = */ 1
        );
    }, "length of");

    scran_tests::expect_error([&]() {
        std::vector<size_t> ordering(sizes.size(), 1);
        CustomOrder2 coords(
            ndim,
            sizes,
            ptrs,
            output,
            builder,
            /* num_neighbors = */ k,
            /* order = */ ordering,
            /* mass_cap = */ -1,
            /* nthreads = */ 1
        );
    }, "duplicate");

    scran_tests::expect_error([&]() {
        std::vector<size_t> ordering(sizes.size(), 3);
        CustomOrder2 coords(
            ndim,
            sizes,
            ptrs,
            output,
            builder,
            /* num_neighbors = */ k,
            /* order = */ ordering,
            /* mass_cap = */ -1,
            /* nthreads = */ 1
        );
    }, "out-of-range");
}

TEST(CustomOrder, NoOp) {
    int ndim = 5;
    std::vector<size_t> sizes;
    std::vector<const double*> ptrs;
    int k = 10;
    knncolle::VptreeBuilder<> builder;
    double* output = NULL;
    std::vector<size_t> ordering;

    CustomOrder2 coords(
        ndim,
        sizes,
        ptrs,
        output,
        builder,
        /* num_neighbors = */ k,
        /* order = */ ordering,
        /* mass_cap = */ -1,
        /* nthreads = */ 1
    );

    EXPECT_TRUE(coords.get_num_pairs().empty());
}
