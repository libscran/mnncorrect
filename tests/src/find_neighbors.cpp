#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.
#include "utils.h"

#include "mnncorrect/find_neighbors.hpp"

#include <vector>
#include <cstddef>
#include <algorithm>
#include <random>

TEST(SubsetAndIndex, Basic) {
    const int num_dim = 6;
    const int num_total = 314;
    auto simulated = scran_tests::simulate_vector(num_dim * num_total, {});

    std::vector<int> subset, other;
    std::mt19937_64 rng(32897);
    std::uniform_real_distribution<> udist;
    for (int o = 0; o < num_total; ++o) {
        if (udist(rng) < 0.2) {
            subset.push_back(o);
        } else {
            other.push_back(o);
        }
    }
    ASSERT_LT(subset.size(), num_total);

    std::vector<double> buffer(num_dim * num_total, 123454); // putting in some initial gunk to check it's ignored.
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    auto subdex = mnncorrect::subset_and_index(num_dim, subset, simulated.data(), builder, buffer.data());
    
    EXPECT_EQ(subdex->num_dimensions(), num_dim);
    EXPECT_EQ(subdex->num_observations(), subset.size());

    auto searcher = subdex->initialize();
    std::vector<int> indices;
    std::vector<double> distances;
    const int num_sub = subset.size();
    for (int i = 0; i < num_sub; ++i) {
        searcher->search(simulated.data() + subset[i] * num_dim, 2, &indices, &distances);
        EXPECT_EQ(indices[0], i);
        EXPECT_NE(indices[1], i);
        EXPECT_EQ(distances[0], 0);
        EXPECT_GT(distances[1], 0);
    }

    for (auto o : other) {
        searcher->search(simulated.data() + o * num_dim, 1, &indices, &distances);
        EXPECT_GT(distances[0], 0);
    }
}

/**********************************/

TEST(FuseNnResults, Basic) {
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 }, { 7, 1.7 }, { 5, 2.5 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 7, 1.7 }, { 2, 2.2 } };
        EXPECT_EQ(output, expected);
    }

    // Not enough of 'base'.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 }, { 7, 1.7 }, { 5, 2.5 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 7, 1.7 }, { 5, 2.5 } };
        EXPECT_EQ(output, expected);
    }

    // Not enough of 'alt'.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        EXPECT_EQ(output, expected);
    }

    // Ties.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 4, 2.2 }, { 5, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 2, 1.1 }, { 3, 2.2 }, { 6, 3.3 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 1, 1.1 }, { 2, 1.1 }, { 3, 2.2 }, { 4, 2.2 } };
        EXPECT_EQ(output, expected);
    }

    // Empty.
    {
        std::vector<std::pair<int, double> > base, alt, output(5);
        mnncorrect::fuse_nn_results(base, alt, 0, output);
        EXPECT_TRUE(output.empty());
    }
}

class FuseNnResultsTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(FuseNnResultsTest, Randomized) {
    auto param = GetParam();
    auto nleft = std::get<0>(param);
    auto nright = std::get<1>(param);
    auto nkeep = std::get<2>(param);

    std::mt19937_64 rng(nleft * nright + nkeep);
    std::normal_distribution ndist;
    std::uniform_int_distribution udist(0, 10000000);
    auto comp = [](const auto& l, const auto& r) -> bool { return l.second < r.second; };

    std::vector<std::pair<int, double> > base;
    for (int l = 0; l < nleft; ++l) {
        base.emplace_back(udist(rng), ndist(rng));
    }
    std::sort(base.begin(), base.end(), comp);

    std::vector<std::pair<int, double> > alt;
    for (int r = 0; r < nright; ++r) {
        alt.emplace_back(udist(rng), ndist(rng));
    }
    std::sort(alt.begin(), alt.end(), comp);
    
    auto ref = base;
    ref.insert(ref.end(), alt.begin(), alt.end());
    std::sort(ref.begin(), ref.end(), comp);
    if (static_cast<std::size_t>(nkeep) < ref.size()) {
        ref.resize(nkeep);
    }

    std::vector<std::pair<int, double> > output;
    mnncorrect::fuse_nn_results(base, alt, nkeep, output);
    EXPECT_EQ(ref, output);
}

INSTANTIATE_TEST_SUITE_P(
    FuseNnResults,
    FuseNnResultsTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // left
        ::testing::Values(1, 5, 10), // right
        ::testing::Values(1, 5, 10) // number to keep
    )
);

TEST(FuseNnResults, Recovery) {
    // Recover the same NN results as just a direct search.
    std::size_t NR = 10;
    int NC = 100;
    auto contents = scran_tests::simulate_vector(NR * NC, []{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 69;
        return sparams;
    }());

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    auto prebuilt_full = builder.build_unique(knncolle::SimpleMatrix<int, double>(NR, NC, contents.data()));
    auto prebuilt_first = builder.build_unique(knncolle::SimpleMatrix<int, double>(NR, 50, contents.data()));
    auto prebuilt_second = builder.build_unique(knncolle::SimpleMatrix<int, double>(NR, NC - 50, contents.data() + 50 * NR));

    auto searcher_full = prebuilt_full->initialize();
    auto searcher_first = prebuilt_first->initialize();
    auto searcher_second = prebuilt_second->initialize();

    int k = 7;
    std::vector<int> indices;
    std::vector<double> distances;
    std::vector<std::pair<int, double> > paired_first, paired_second, paired_full, fused;

    for (int c = 0; c < NC; ++c) {
        auto ptr = contents.data() + c * NR;
        searcher_first->search(ptr, k, &indices, &distances);
        mnncorrect::fill_pair_vector(indices, distances, paired_first);

        searcher_second->search(ptr, k, &indices, &distances);
        for (auto& i : indices) {
            i += 50;
        }
        mnncorrect::fill_pair_vector(indices, distances, paired_second);

        searcher_full->search(ptr, k, &indices, &distances);
        mnncorrect::fill_pair_vector(indices, distances, paired_full);
        mnncorrect::fuse_nn_results(paired_first, paired_second, k, fused);
        EXPECT_EQ(paired_full, fused);
    }
}

/**********************************/

class FindNeighborsTest : public ::testing::TestWithParam<std::tuple<std::vector<int>, bool, int> > {
protected:
    const std::size_t num_dim = 5;
    int num_total;
    std::vector<double> simulated;
    std::vector<mnncorrect::MetaBatch<int, double> > all_batches;
    std::unique_ptr<knncolle::Builder<int, double, double> > nn_builder;

    void assemble(const std::vector<int>& batch_sizes, bool extras) {
        const std::size_t num_batches = batch_sizes.size();
        num_total = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
        simulated = scran_tests::simulate_vector(num_dim * num_total, [&]{
            scran_tests::SimulateVectorParameters opt;
            opt.seed = num_total * num_batches;
            return opt;
        }());

        nn_builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >()));
        std::mt19937_64 rng(/* seed = */ num_total + extras);

        // Mocking up the metabatches.
        all_batches.resize(num_batches);
        if (extras) {
            for (auto& batch : all_batches) {
                batch.corrected.resize(num_batches);
            }
        }

        int sofar = 0;
        for (std::size_t b = 0; b < num_batches; ++b) {
            const auto bsize = batch_sizes[b];

            if (extras) {
                // Firstly adding the core stretch.
                int quarter = bsize / 4, half = bsize / 2;
                int start = rng() % quarter;
                int number = rng() % half + quarter;
                all_batches[b].original_ids.start = sofar + start;
                all_batches[b].original_ids.size = number;

                // Now randomly distributing observations before and after this stretch into the other corrected batches.
                for (int i = 0; i < start; ++i) {
                    auto chosen = rng() % num_batches;
                    all_batches[chosen].corrected[b].ids.push_back(i + sofar);
                }

                int remaining = bsize - number - start;
                for (int i = 0; i < remaining; ++i) {
                    auto chosen = rng() % num_batches;
                    all_batches[chosen].corrected[b].ids.push_back(i + sofar + start + number);
                }
            } else {
                all_batches[b].original_ids.start = sofar; 
                all_batches[b].original_ids.size = bsize;
            }

            sofar += bsize;
        }

        // Shuffling the batches to check that the code doesn't assume them to be sorted by original_ids.start.
        std::shuffle(all_batches.begin(), all_batches.end(), rng);

        // Creating the indices.
        std::vector<double> buffer(num_total * num_dim);
        for (std::size_t b = 0; b < num_batches; ++b) {
            auto& batch = all_batches[b];
            auto ptr = simulated.data() + static_cast<std::size_t>(batch.original_ids.start) * num_dim;
            batch.original_index = nn_builder->build_unique(knncolle::SimpleMatrix<int, double>(num_dim, batch.original_ids.size, ptr));
            for (auto& extra : batch.corrected) {
                extra.index = mnncorrect::subset_and_index(num_dim, extra.ids, simulated.data(), *nn_builder, buffer.data());
            }
        }
    }
};

TEST_P(FindNeighborsTest, Basic) {
    auto params = GetParam();
    assemble(std::get<0>(params), std::get<1>(params));
    auto num_neighbors = std::get<2>(params);

    // Creating the reference results first.
    mnncorrect::NeighborSet<int, double> expected(num_total);
    {
        std::vector<std::vector<int> > assignments;
        assignments.reserve(all_batches.size());
        for (const auto& batch : all_batches) {
            std::vector<int> current(batch.original_ids.size);
            std::iota(current.begin(), current.end(), batch.original_ids.start);
            for (const auto& extra : batch.corrected) {
                current.insert(current.end(), extra.ids.begin(), extra.ids.end());
            }
            std::sort(current.begin(), current.end());
            assignments.emplace_back(std::move(current));
        }

        std::vector<int> target_assignment(std::move(assignments.back())); 
        assignments.pop_back();

        std::vector<int> reference_assignment;
        for (std::size_t br = 0, brend = assignments.size(); br < brend; ++br) {
            const auto& ref = assignments[br];
            reference_assignment.insert(reference_assignment.end(), ref.begin(), ref.end());
        }
        std::sort(reference_assignment.begin(), reference_assignment.end());

        std::vector<double> buffer(num_total * num_dim);
        auto target_index = mnncorrect::subset_and_index(num_dim, target_assignment, simulated.data(), *nn_builder, buffer.data());
        auto reference_index = mnncorrect::subset_and_index(num_dim, reference_assignment, simulated.data(), *nn_builder, buffer.data());

        find_neighbors(num_dim, reference_assignment, simulated.data(), *target_index, target_assignment, num_neighbors, expected);
        find_neighbors(num_dim, target_assignment, simulated.data(), *reference_index, reference_assignment, num_neighbors, expected);
    }

    // Now computing the neighbors from the meta-batches themselves.
    mnncorrect::MetaBatch<int, double> target_batch(std::move(all_batches.back()));
    all_batches.pop_back();

    {
        mnncorrect::NeighborSet<int, double> computed(num_total);
        mnncorrect::find_neighbors(num_dim, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 1, computed);
        for (int i = 0; i < num_total; ++i) {
            EXPECT_EQ(computed[i], expected[i]);
        }
    }

    // Making a dirty input object and checking that the existing input is ignored.
    {
        mnncorrect::NeighborSet<int, double> computed(num_total);
        for (auto& compnn : computed) {
            compnn.emplace_back(12323, 4334);
        }
        mnncorrect::find_neighbors(num_dim, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 1, computed);
        for (int i = 0; i < num_total; ++i) {
            EXPECT_EQ(computed[i], expected[i]);
        }
    }

    // Making sure we get the same results on parallelization.
    {
        mnncorrect::NeighborSet<int, double> computed(num_total);
        mnncorrect::find_neighbors(num_dim, all_batches, target_batch, simulated.data(), num_neighbors, /* num_threads = */ 3, computed);
        for (int i = 0; i < num_total; ++i) {
            EXPECT_EQ(computed[i], expected[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    FindNeighbors,
    FindNeighborsTest,
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
