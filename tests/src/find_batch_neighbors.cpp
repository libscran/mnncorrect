#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

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
    std::vector<std::vector<int> > assignments;
    std::unique_ptr<knncolle::Builder<int, double, double> > nn_builder;

    void assemble(const std::vector<int>& batch_sizes, bool extras) {
        auto num_batches = batch_sizes.size();
        num_total = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
        simulated = scran_tests::simulate_vector(num_dim * num_total, [&]{
            scran_tests::SimulationParameters opt;
            opt.seed = num_total * num_batches;
            return opt;
        }());

        assignments.resize(num_batches);
        all_batches.resize(num_batches);
        if (extras) {
            for (auto& batch : all_batches) {
                batch.extras.resize(num_batches);
            }
        }

        std::mt19937_64 rng(/* seed = */ num_total);
        int sofar = 0;
        for (decltype(num_batches) b = 0; b < num_batches; ++b) {
            auto bsize = batch_sizes[b];

            // Firstly adding the core stretch.
            int quarter = bsize / 4, half = bsize / 2;
            int start = rng() % quarter;
            int number = rng() % half + quarter;
            all_batches[b].offset = sofar + start;
            all_batches[b].num_obs = number;
            for (int i = 0; i < number; ++i) {
                assignments[b].push_back(i + start + sofar);
            }

            // Now adding anything before it.
            if (extras) {
                for (int i = 0; i < start; ++i) {
                    auto chosen = rng() % num_batches;
                    int index = i + sofar;
                    assignments[chosen].push_back(index);
                    all_batches[chosen].extras[b].ids.push_back(index);
                }

                int remaining = bsize - number - start;
                for (int i = 0; i < remaining; ++i) {
                    auto chosen = rng() % num_batches;
                    int index = i + sofar + start + number;
                    assignments[chosen].push_back(index);
                    all_batches[chosen].extras[b].ids.push_back(index);
                }
            }

            sofar += bsize;
        }

        // Creating the indices.
        std::vector<double> buffer;
        nn_builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >()));

        for (decltype(num_batches) b = 0; b < num_batches; ++b) {
            auto& batch = all_batches[b];
            buffer.resize(static_cast<std::size_t>(batch.num_obs) * num_dim);
            std::copy_n(simulated.begin() + static_cast<std::size_t>(batch.offset) * num_dim, buffer.size(), buffer.begin());
            batch.index = nn_builder->build_unique(knncolle::SimpleMatrix<int, double>(num_dim, batch.num_obs, buffer.data()));
            for (auto& extra : batch.extras) {
                extra.index = subset_and_index(extra.ids, buffer);
            }
        }
    }

    std::unique_ptr<knncolle::Prebuilt<int, double, double> > subset_and_index(const std::vector<int>& ids, std::vector<double>& buffer) const { 
        auto num_obs = ids.size();
        buffer.resize(static_cast<std::size_t>(num_obs) * num_dim);
        for (decltype(num_obs) e = 0; e < num_obs; ++e) {
            std::copy_n(
                simulated.begin() + static_cast<std::size_t>(ids[e]) * num_dim,
                num_dim,
                buffer.begin() + static_cast<std::size_t>(e) * num_dim
            );
        }
        return nn_builder->build_unique(knncolle::SimpleMatrix<int, double>(num_dim, num_obs, buffer.data()));
    }
};

TEST_P(FindBatchNeighborsTest, Basic) {
    auto params = GetParam();
    assemble(std::get<0>(params), std::get<1>(params));
    auto num_neighbors = std::get<2>(params);

    // Creating the reference results first.
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
    auto target_index = subset_and_index(target_assignment, buffer);
    auto reference_index = subset_and_index(reference_assignment, buffer);

    mnncorrect::internal::NeighborSet<int, double> expected(num_total);
    std::vector<int> indices;
    std::vector<double> distances;

    auto target_searcher = target_index->initialize();
    for (auto r : reference_assignment) {
        target_searcher->search(simulated.data() + static_cast<std::size_t>(r) * num_dim, num_neighbors, &indices, &distances);
        auto found = indices.size();
        for (decltype(found) i = 0; i < found; ++i) {
            expected[r].emplace_back(target_assignment[indices[i]], distances[i]);
        }
    }

    auto reference_searcher = reference_index->initialize();
    for (auto t : target_assignment) {
        reference_searcher->search(simulated.data() + static_cast<std::size_t>(t) * num_dim, num_neighbors, &indices, &distances);
        auto found = indices.size();
        for (decltype(found) i = 0; i < found; ++i) {
            expected[t].emplace_back(reference_assignment[indices[i]], distances[i]);
        }
    }

    // Now computing our friend.
    mnncorrect::internal::BatchInfo<int, double> target_batch(std::move(all_batches.back()));
    all_batches.pop_back();

    mnncorrect::internal::FindBatchNeighborsResults<int, double> computed;
    mnncorrect::internal::find_batch_neighbors(num_dim, num_total, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 1, computed);
    ASSERT_EQ(computed.neighbors.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(computed.neighbors[i], expected[i]);
    }
    EXPECT_EQ(computed.ref_ids, reference_assignment);
    EXPECT_EQ(computed.target_ids, target_assignment);
    EXPECT_EQ(computed.batch, batch_of_origin);

    // Making sure we get the same results on parallelization.
    mnncorrect::internal::find_batch_neighbors(num_dim, num_total, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 3, computed);
    ASSERT_EQ(computed.neighbors.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(computed.neighbors[i], expected[i]);
    }
    EXPECT_EQ(computed.ref_ids, reference_assignment);
    EXPECT_EQ(computed.target_ids, target_assignment);
    EXPECT_EQ(computed.batch, batch_of_origin);

    // Mutating the input object and checking that the shuffled residue is ignored.
    for (auto& compnn : computed.neighbors) {
        std::reverse(compnn.begin(), compnn.end());
    }
    std::reverse(computed.ref_ids.begin(), computed.ref_ids.end());
    std::reverse(computed.target_ids.begin(), computed.target_ids.end());
    std::reverse(computed.batch.begin(), computed.batch.end());
    mnncorrect::internal::find_batch_neighbors(num_dim, num_total, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 1, computed);
    ASSERT_EQ(computed.neighbors.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(computed.neighbors[i], expected[i]);
    }
    EXPECT_EQ(computed.ref_ids, reference_assignment);
    EXPECT_EQ(computed.target_ids, target_assignment);
    EXPECT_EQ(computed.batch, batch_of_origin);
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
