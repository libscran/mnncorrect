#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.
#include "utils.h"

#include "mnncorrect/compute_center_of_mass.hpp"
#include "knncolle/knncolle.hpp"

#include <cstddef>
#include <utility>
#include <vector>
#include <unordered_set>

static std::vector<int> mock_mnn_cells(int nobs, double density, unsigned long long seed) {
    std::vector<int> to_check;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<> runif;
    for (int o = 0; o < nobs; ++o) {
        if (runif(rng) < density) {
            to_check.push_back(o);
        }
    }
    return to_check;
}

class WalkAroundNeighborhoodTest : public ::testing::TestWithParam<std::tuple<int, int, double, bool> > {
protected:
    static void reference(
        std::size_t ndim,
        int position,
        const double* data,
        knncolle::Searcher<int, double, double>& searcher,
        int num_neighbors,
        int remaining_steps,
        std::vector<int>& indices,
        std::vector<double>& distances,
        mnncorrect::NeighborSet<int, double>& neighbors
    ) {
        if (neighbors[position].empty()) {
            searcher.search(data + position * ndim, num_neighbors, &indices, &distances);
            for (decltype(indices.size()) i = 0, end = indices.size(); i < end; ++i) {
                neighbors[position].emplace_back(indices[i], distances[i]);
            }
        }
        if (!remaining_steps) {
            return;
        }
        for (const auto& pair : neighbors[position]) {
            reference(
                ndim,
                pair.first,
                data,
                searcher,
                num_neighbors,
                remaining_steps - 1, 
                indices,
                distances,
                neighbors
            );
        }
    }
};

TEST_P(WalkAroundNeighborhoodTest, Basic) {
    auto params = GetParam();
    auto k = std::get<0>(params);
    auto steps = std::get<1>(params);
    auto density = std::get<2>(params);
    auto with_extras = std::get<2>(params);
    auto seed = k * 10 + steps + with_extras + density * 100;

    std::size_t ndim = 5;
    int nobs = 100;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = seed;
        return sparams;
    }());

    mnncorrect::MetaBatch<int, double> batch;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    if (with_extras) {
        batch.original_ids.start = 20; // [20, 70) is the main batch.
        batch.original_ids.size = 50;
        batch.corrected.emplace_back( // [70, 100) is the first extra.
            builder.build_unique(knncolle::SimpleMatrix(ndim, 30, vec.data() + 70 * ndim)),
            [&]{
                std::vector<int> extra(30);
                std::iota(extra.begin(), extra.end(), 70);
                return extra;
            }()
        );
        batch.corrected.emplace_back( // [0, 20) is the second extra.
            builder.build_unique(knncolle::SimpleMatrix(ndim, 20, vec.data())), // sprinkle in some extras.
            [&]{
                std::vector<int> extra(20);
                std::iota(extra.begin(), extra.end(), 0);
                return extra;
            }()
        );
    } else {
        batch.original_ids.start = 0;
        batch.original_ids.size = nobs;
    }
    batch.original_index = builder.build_unique(
        knncolle::SimpleMatrix(
            ndim,
            batch.original_ids.size,
            vec.data() + static_cast<std::size_t>(batch.original_ids.start) * ndim
        )
    );

    auto to_check = mock_mnn_cells(nobs, density, seed + 200);

    mnncorrect::NeighborhoodWalkWorkspace<int> workspace(nobs);
    mnncorrect::NeighborSet<int, double> neighbors(nobs);
    mnncorrect::walk_around_neighborhood(
        ndim,
        to_check,
        batch,
        vec.data(),
        k,
        steps,
        /* num_threads = */ 1,
        workspace,
        neighbors
    );

    // Checking against a reference.
    std::vector<int> indices;
    std::vector<double> distances;
    mnncorrect::NeighborSet<int, double> copy(nobs);
    auto full_index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));
    auto searcher = full_index->initialize();
    for (auto i : to_check) {
        reference(ndim, i, vec.data(), *searcher, k, steps, indices, distances, copy);
    }
    for (int o = 0; o < nobs; ++o) {
        EXPECT_EQ(neighbors[o], copy[o]);
    }

    // Check that it gives the same results for multiple threads. Also giving it some
    // dirty output containers to check that it sanitizes them. 
    workspace.ids.resize(1000, -1); 
    workspace.next_ids.resize(1000, -1); 
    workspace.all_ids.resize(1000, -1); 
    for (auto& nn : neighbors) {
        std::reverse(nn.begin(), nn.end());
    }
    mnncorrect::walk_around_neighborhood(
        ndim,
        to_check,
        batch,
        vec.data(),
        k,
        steps,
        /* num_threads = */ 3,
        workspace,
        neighbors
    );
    for (int o = 0; o < nobs; ++o) {
        EXPECT_EQ(neighbors[o], copy[o]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    WalkAroundNeighborhood,
    WalkAroundNeighborhoodTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // number of neighbors.
        ::testing::Values(0, 1, 2, 3), // number of steps.
        ::testing::Values(0.05, 0.1, 0.2), // density of the MNN-involved cells.
        ::testing::Values(false, true) // whether to check extras in the batch.
    )
);

TEST(WalkAroundNeighborhood, QuitEarly) {
    std::size_t ndim = 5;
    int nobs = 10;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, {});

    mnncorrect::MetaBatch<int, double> target;
    target.original_ids.start = 0;
    target.original_ids.size = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.original_index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    // Get some coverage on our loop break if we don't need to use all of the
    // steps, in this case because we've already covered all the observations
    // in the dataset after the first step.
    std::vector<int> to_check{ 5 };
    mnncorrect::NeighborhoodWalkWorkspace<int> workspace(nobs);
    mnncorrect::NeighborSet<int, double> neighbors(nobs);
    mnncorrect::walk_around_neighborhood(
        ndim,
        to_check,
        target,
        vec.data(),
        /* num_neighbors = */ nobs,
        /* num_steps = */ 3,
        /* num_threads = */ 3,
        workspace,
        neighbors
    );

    for (const auto& found : neighbors) {
        EXPECT_EQ(found.size(), nobs);
    }
}

/***************************************************/

class ComputeCenterOfMassTest : public ::testing::TestWithParam<std::tuple<int, int, double> > {
protected:
    static void reference(
        std::size_t ndim,
        int position,
        const mnncorrect::NeighborSet<int, double>& neighbors,
        const double* data,
        int remaining,
        double* output,
        std::unordered_set<int>& used
    ) {
        const auto& curneighbors = neighbors[position];
        for (auto pp : curneighbors) {
            if (used.find(pp.first) == used.end()) {
                auto ptr = data + ndim * pp.first;
                for (decltype(ndim) d = 0; d < ndim; ++d) {
                    output[d] += ptr[d];
                }
                used.insert(pp.first);
            }
            if (remaining > 0) {
                reference(ndim, pp.first, neighbors, data, remaining - 1, output, used);
            }
        }
    }
};

TEST_P(ComputeCenterOfMassTest, Basic) {
    auto params = GetParam();
    auto k = std::get<0>(params);
    auto steps = std::get<1>(params);
    auto density = std::get<2>(params);
    auto seed = k * 10 + steps + density * 100;

    std::size_t ndim = 5;
    int nobs = 212;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = k * 10 + steps;
        return sparams;
    }());

    mnncorrect::MetaBatch<int, double> target;
    target.original_ids.start = 0;
    target.original_ids.size = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.original_index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    auto to_check = mock_mnn_cells(nobs, density, seed + 200);

    std::size_t full_size = ndim * nobs;
    std::vector<double> centers(full_size);

    // Computing a center of mass.
    mnncorrect::NeighborhoodWalkWorkspace<int> workspace(nobs);
    mnncorrect::NeighborSet<int, double> neighbors(nobs);
    mnncorrect::compute_center_of_mass(
        ndim,
        to_check,
        target,
        vec.data(),
        k,
        steps,
        /* num_threads = */ 1,
        workspace,
        neighbors,
        centers.data()
    );

    // Computing the reference.
    std::vector<double> ref(ndim);
    std::unordered_set<int> used;
    mnncorrect::NeighborSet<int, double> ncopy(nobs);
    mnncorrect::walk_around_neighborhood(
        ndim,
        to_check,
        target,
        vec.data(),
        k,
        steps,
        /* num_threads = */ 1,
        workspace,
        ncopy
    );

    for (std::size_t i = 0, end = to_check.size(); i < end; ++i) {
        std::fill(ref.begin(), ref.end(), 0);
        used.clear();
        reference(
            ndim,
            to_check[i],
            ncopy,
            vec.data(),
            steps,
            ref.data(),
            used
        );
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            EXPECT_FLOAT_EQ(ref[d] / used.size(), centers[d + to_check[i] * ndim]);
        }
    }

    // Same results with more threads. 
    std::vector<double> pcenters(full_size);
    mnncorrect::compute_center_of_mass(
        ndim,
        to_check,
        target,
        vec.data(),
        k,
        steps,
        /* num_threads = */ 3,
        workspace,
        neighbors,
        pcenters.data()
    );
    EXPECT_EQ(centers, pcenters);
}

INSTANTIATE_TEST_SUITE_P(
    ComputeCenterOfMass,
    ComputeCenterOfMassTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // number of neighbors.
        ::testing::Values(0, 1, 2, 3), // number of steps.
        ::testing::Values(0.05, 0.1, 0.2) // density of MNN cells 
    )
);

TEST(ComputeCenterOfMassTest, QuitEarly) {
    std::size_t ndim = 5;
    int nobs = 10;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, {});

    mnncorrect::MetaBatch<int, double> target;
    target.original_ids.start = 0;
    target.original_ids.size = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.original_index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    std::vector<int> to_check{ 5 };
    mnncorrect::NeighborhoodWalkWorkspace<int> workspace(nobs);
    mnncorrect::NeighborSet<int, double> neighbors(nobs);

    // Gets some coverage on our loop break if we don't need to use all of the
    // steps, in this case because we've already covered all the observations
    // in the dataset after the first step.
    std::vector<double> center(ndim * nobs);
    mnncorrect::compute_center_of_mass(
        ndim,
        to_check,
        target,
        vec.data(),
        /* num_neighbors = */ nobs,
        /* num_steps = */ 3,
        /* num_threads = */ 1,
        workspace,
        neighbors,
        center.data()
    );

    // Checking that it is equal to the average of all points.
    std::vector<double> ref(ndim * nobs);
    for (decltype(nobs) o = 0; o < nobs; ++o) {
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            ref[d + to_check.front() * ndim] += vec[o * ndim + d];
        }
    }
    for (decltype(ndim) d = 0; d < ndim; ++d) {
        ref[d + to_check.front() * ndim] /= nobs;
    }
    scran_tests::compare_almost_equal_containers(center, ref, {});
}
