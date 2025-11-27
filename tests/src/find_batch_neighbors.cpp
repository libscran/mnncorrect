#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.
#include "utils.h"

#include "mnncorrect/find_batch_neighbors.hpp"

#include <vector>
#include <cstddef>
#include <algorithm>
#include <random>

class FindBatchNeighborsTest : public ::testing::TestWithParam<std::tuple<std::vector<int>, int, bool> > {
protected:
    const std::size_t num_dim = 5;
    int num_total;
    std::vector<double> simulated;
    std::vector<mnncorrect::internal::BatchInfo<int, double> > all_batches;
    std::unique_ptr<knncolle::Builder<int, double, double> > nn_builder;

    void assemble(const std::vector<int>& batch_sizes, bool extras) {
        auto num_batches = batch_sizes.size();
        num_total = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
        simulated = scran_tests::simulate_vector(num_dim * num_total, [&]{
            scran_tests::SimulateVectorParameters opt;
            opt.seed = num_total * num_batches;
            return opt;
        }());

        nn_builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >()));
        std::mt19937_64 rng(/* seed = */ num_total);
        all_batches = mock_batches(num_dim, batch_sizes, simulated.data(), extras, rng, *nn_builder);
    }
};

TEST_P(FindBatchNeighborsTest, Basic) {
    auto params = GetParam();
    assemble(std::get<0>(params), std::get<1>(params));
    auto num_neighbors = std::get<2>(params);

    // Creating the reference results first.
    std::vector<std::vector<int> > assignments = create_assignments(all_batches);

    std::vector<int> target_assignment(std::move(assignments.back())); 
    std::sort(target_assignment.begin(), target_assignment.end());
    assignments.pop_back();

    std::vector<int> reference_assignment;
    std::vector<mnncorrect::BatchIndex> batch_of_origin(num_total, -1);
    for (decltype(assignments.size()) br = 0, brend = assignments.size(); br < brend; ++br) {
        const auto& ref = assignments[br];
        reference_assignment.insert(reference_assignment.end(), ref.begin(), ref.end());
        for (auto r : ref) {
            batch_of_origin[r] = br;
        }
    }
    std::sort(reference_assignment.begin(), reference_assignment.end());

    std::vector<double> buffer;
    auto target_index = subset_and_index(num_dim, target_assignment, simulated.data(), *nn_builder, buffer);
    auto reference_index = subset_and_index(num_dim, reference_assignment, simulated.data(), *nn_builder, buffer);

    mnncorrect::internal::NeighborSet<int, double> expected(num_total);
    find_neighbors(num_dim, reference_assignment, simulated.data(), *target_index, target_assignment, num_neighbors, expected);
    find_neighbors(num_dim, target_assignment, simulated.data(), *reference_index, reference_assignment, num_neighbors, expected);

    // Now computing our friend.
    mnncorrect::internal::BatchInfo<int, double> target_batch(std::move(all_batches.back()));
    all_batches.pop_back();

    mnncorrect::internal::FindBatchNeighborsResults<int, double> computed;
    mnncorrect::internal::find_batch_neighbors(num_dim, num_total, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 1, computed);
    ASSERT_EQ(computed.neighbors.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(computed.neighbors[i], expected[i]);
    }

    // Making sure we get the same results on parallelization.
    mnncorrect::internal::find_batch_neighbors(num_dim, num_total, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 3, computed);
    ASSERT_EQ(computed.neighbors.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(computed.neighbors[i], expected[i]);
    }

    // Mutating the input object and checking that the shuffled residue is ignored.
    for (auto& compnn : computed.neighbors) {
        std::reverse(compnn.begin(), compnn.end());
    }
    mnncorrect::internal::find_batch_neighbors(num_dim, num_total, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 1, computed);
    ASSERT_EQ(computed.neighbors.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(computed.neighbors[i], expected[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FindBatchNeighbors,
    FindBatchNeighborsTest,
    ::testing::Combine(
        ::testing::Values( // batch sizes
            std::vector<int>{ 100, 200 },
            std::vector<int>{ 199, 201, 255 },
            std::vector<int>{ 54, 123, 78, 69 }
        ),
        ::testing::Values( // whether to include extras
            false, true
        ),
        ::testing::Values( // number of neighbors
            5, 10, 20
        )
    )
);
