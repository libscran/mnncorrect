#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.
#include "utils.h"

#include "mnncorrect/correct_target.hpp"
#include "mnncorrect/fuse_nn_results.hpp"
#include "knncolle/knncolle.hpp"

#include <cstddef>
#include <utility>
#include <vector>

TEST(CorrectTarget, BuildMnnOnlyIndex) {
    std::size_t ndim = 5;
    int nobs = 100;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 999;
        return sparams;
    }());

    std::vector<int> mnns; 
    for (int o = 1; o < nobs; o += 7) {
        mnns.push_back(o);
    }

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    std::vector<double> buffer;
    std::vector<int> indices;
    std::vector<double> distances;

    {
        auto mnn_index = mnncorrect::internal::build_mnn_only_index(ndim, vec.data(), mnns, builder, buffer);
        auto mnn_searcher = mnn_index->initialize();
        for (decltype(mnns.size()) i = 0, end = mnns.size(); i < end; ++i) {
            mnn_searcher->search(vec.data() + static_cast<std::size_t>(mnns[i]) * ndim, 1, &indices, &distances);
            EXPECT_EQ(indices[0], i);
            EXPECT_EQ(distances[0], 0);
        }
    }

    // Ignores gunk in the buffer.
    {
        std::fill(buffer.begin(), buffer.end(), -1);
        auto mnn_index = mnncorrect::internal::build_mnn_only_index(ndim, vec.data(), mnns, builder, buffer);
        auto mnn_searcher = mnn_index->initialize();
        for (decltype(mnns.size()) i = 0, end = mnns.size(); i < end; ++i) {
            mnn_searcher->search(vec.data() + static_cast<std::size_t>(mnns[i]) * ndim, 1, &indices, &distances);
            EXPECT_EQ(indices[0], i);
            EXPECT_EQ(distances[0], 0);
        }
    }
}

TEST(CorrectTarget, ForceSelf) {
    {
        std::vector<std::pair<int, double> > neighbors{ { 3, 1.11 }, { 104, 2.2 }, { 17, 3.0 } };
        mnncorrect::internal::force_self(neighbors, 13, 3);
        std::vector<std::pair<int, double> > expected{ { 13, 0 }, { 3, 1.11 }, { 104, 2.2 } };
        EXPECT_EQ(neighbors, expected);
    }

    {
        std::vector<std::pair<int, double> > neighbors{ { 3, 1.11 }, { 104, 2.2 }, { 17, 3.0 } };
        mnncorrect::internal::force_self(neighbors, 13, 4);
        std::vector<std::pair<int, double> > expected{ { 13, 0 }, { 3, 1.11 }, { 104, 2.2 }, { 17, 3.0 } };
        EXPECT_EQ(neighbors, expected);
    }

    {
        std::vector<std::pair<int, double> > neighbors{ { 13, 0.0 }, { 3, 1.11 }, { 104, 2.2 }, { 17, 3.0 } };
        auto expected = neighbors;
        mnncorrect::internal::force_self(neighbors, 13, 4);
        EXPECT_EQ(neighbors, expected);
    }
}

TEST(CorrectTarget, EnsureSort) {
    // Already sorted.
    {
        std::vector<std::pair<int, double> > neighbors{ { 3, 1.11 }, { 104, 2.2 }, { 17, 3.0 } };
        auto expected = neighbors;
        mnncorrect::internal::ensure_sort(neighbors);
        EXPECT_EQ(expected, neighbors);
    }

    // Resorts.
    {
        std::vector<std::pair<int, double> > neighbors{ { 3, 1.11 }, { 104, 0.22 }, { 17, 0.03 } };
        mnncorrect::internal::ensure_sort(neighbors);
        std::vector<std::pair<int, double> > expected{ { 17, 0.03 }, { 104, 0.22 }, { 3, 1.11 } };
        EXPECT_EQ(expected, neighbors);
    }

    // Respects indices during tie breaking.
    {
        std::vector<std::pair<int, double> > neighbors{ { 3, 1.11 }, { 0, 1.11 }, { 1, 1.11 } };
        mnncorrect::internal::ensure_sort(neighbors);
        std::vector<std::pair<int, double> > expected{ { 0, 1.11 }, { 1, 1.11 }, { 3, 1.11 } };
        EXPECT_EQ(expected, neighbors);
    }
}

class SearchForNeighborsFromMnnsTest : public ::testing::TestWithParam<std::tuple<std::vector<int>, int, bool> > {
protected:
    const std::size_t num_dim = 8;
    int num_total;
    std::vector<double> simulated;
    std::vector<mnncorrect::internal::BatchInfo<int, double> > all_batches;
    std::unique_ptr<knncolle::Builder<int, double, double> > nn_builder;

    void assemble(const std::vector<int>& batch_sizes, bool extras) {
        auto num_batches = batch_sizes.size();
        num_total = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
        simulated = scran_tests::simulate_vector(num_dim * num_total, [&]{
            scran_tests::SimulationParameters opt;
            opt.seed = num_total * num_batches;
            return opt;
        }());

        nn_builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >()));
        std::mt19937_64 rng(/* seed = */ num_total);
        all_batches = mock_batches(num_dim, batch_sizes, simulated.data(), extras, rng, *nn_builder);
    }
};

TEST_P(SearchForNeighborsFromMnnsTest, Basic) {
    auto params = GetParam();
    assemble(std::get<0>(params), std::get<1>(params));
    auto num_neighbors = std::get<2>(params);

    // Defining the reference results.
    std::vector<int> target_assignment, reference_assignment;
    {
        std::vector<std::vector<int> > assignments = create_assignments(all_batches);
        target_assignment.swap(assignments.back()); 
        assignments.pop_back();
        std::sort(target_assignment.begin(), target_assignment.end());

        std::vector<int> reference_assignment;
        std::vector<mnncorrect::BatchIndex> batch_of_origin(num_total, -1);
        for (decltype(assignments.size()) br = 0, brend = assignments.size(); br < brend; ++br) {
            const auto& ref = assignments[br];
            reference_assignment.insert(reference_assignment.end(), ref.begin(), ref.end());
        }
        std::sort(reference_assignment.begin(), reference_assignment.end());
    }

    mnncorrect::internal::NeighborSet<int, double> expected(num_total);
    {
        std::vector<double> buffer;
        auto target_index = subset_and_index(num_dim, target_assignment, simulated.data(), *nn_builder, buffer);
        auto reference_index = subset_and_index(num_dim, reference_assignment, simulated.data(), *nn_builder, buffer);
        find_neighbors(num_dim, reference_assignment, simulated.data(), *reference_index, reference_assignment, num_neighbors, expected);
        find_neighbors(num_dim, target_assignment, simulated.data(), *target_index, target_assignment, num_neighbors, expected);
    }

    // Checking it works with a designated subset of MNNs in each batch.
    mnncorrect::internal::BatchInfo<int, double> target_batch(std::move(all_batches.back()));
    all_batches.pop_back();

    std::vector<int> reference_mnns, target_mnns;
    {
        auto ref_size = reference_assignment.size();
        for (decltype(ref_size) i = 0; i < ref_size; i += 7) { // arbitrary subset.
            reference_mnns.push_back(reference_assignment[i]);
        }
        auto target_size = target_assignment.size();
        for (decltype(target_size) i = 2; i < target_size; i += 3) { // arbitrary subset.
            target_mnns.push_back(target_assignment[i]);
        }
    }

    mnncorrect::internal::NeighborSet<int, double> mnn_computed;
    {
        mnncorrect::internal::search_for_neighbors_from_mnns(
            num_dim, 
            num_total,
            reference_mnns,
            target_mnns,
            simulated.data(),
            all_batches,
            target_batch,
            num_neighbors,
            /* nthreads = */ 1,
            mnn_computed 
        );

        std::vector<unsigned char> is_mnn(num_total);
        for (auto r : reference_mnns) {
            EXPECT_EQ(mnn_computed[r], expected[r]);
            is_mnn[r] = true;
        }
        for (auto t : target_mnns) {
            EXPECT_EQ(mnn_computed[t], expected[t]);
            is_mnn[t] = true;
        }
        for (int i = 0; i < num_total; ++i) {
            EXPECT_NE(is_mnn[i], mnn_computed[i].empty());
        }
    }

    // Same results on parallelization.
    {
        mnncorrect::internal::NeighborSet<int, double> p_mnn_computed;
        mnncorrect::internal::search_for_neighbors_from_mnns(
            num_dim, 
            num_total,
            reference_mnns,
            target_mnns,
            simulated.data(),
            all_batches,
            target_batch,
            num_neighbors,
            /* nthreads = */ 3,
            p_mnn_computed 
        );
        for (int i = 0; i < num_total; ++i) {
            EXPECT_EQ(mnn_computed[i], p_mnn_computed[i]);
        }
    }

    // Unaffected by dirty inputs.
    {
        auto copy = mnn_computed;
        for (auto& comp : copy) {
            std::reverse(comp.begin(), comp.end());
        }
        mnncorrect::internal::search_for_neighbors_from_mnns(
            num_dim, 
            num_total,
            reference_mnns,
            target_mnns,
            simulated.data(),
            all_batches,
            target_batch,
            num_neighbors,
            /* nthreads = */ 1,
            copy 
        );
        for (int i = 0; i < num_total; ++i) {
            EXPECT_EQ(mnn_computed[i], copy[i]);
        }
    }

    // Check that it works with the full results.
    {
        mnncorrect::internal::NeighborSet<int, double> full_computed;
        mnncorrect::internal::search_for_neighbors_from_mnns(
            num_dim, 
            num_total,
            reference_assignment,
            target_assignment,
            simulated.data(),
            all_batches,
            target_batch,
            num_neighbors,
            /* nthreads = */ 1,
            full_computed 
        );
        for (int i = 0; i < num_total; ++i) {
            EXPECT_EQ(full_computed[i], expected[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    CorrectTarget,
    SearchForNeighborsFromMnnsTest, 
    ::testing::Combine(
        ::testing::Values( // batch sizes
            std::vector<int>{ 300, 100 },
            std::vector<int>{ 181, 231, 125 },
            std::vector<int>{ 155, 87, 133, 99 }
        ),
        ::testing::Values( // whether to include extras
            false, true
        ),
        ::testing::Values( // number of neighbors
            5, 10, 20
        )
    )
);

TEST(CorrectTarget, SearchForNeighborsFromMnnsZeroed) {
    // Testing that the enforced sorting and self-insertion works correctly.

    constexpr std::size_t num_dim = 10;
    std::vector<int> batch_sizes { 100, 200 };
    auto num_total = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);
    std::vector<double> simulated(num_total * num_dim, /* some arbitrary value */ 42);
    std::mt19937_64 rng(/* seed = */ 69);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    auto all_batches = mock_batches(num_dim, batch_sizes, simulated.data(), true, rng, builder);

    std::vector<int> reference_mnns, target_mnns;
    {
        std::vector<std::vector<int> > assignments = create_assignments(all_batches);
        const auto& target_assignments = assignments.back();
        auto num_target_ids = target_assignments.size();
        for (decltype(num_target_ids) tx = 0; tx < num_target_ids; ++tx) {
            if (tx % 2) {
                target_mnns.push_back(target_assignments[tx]);
            }
        }

        assignments.pop_back();
        for (decltype(assignments.size()) br = 0, brend = assignments.size(); br < brend; ++br) {
            const auto& ref_assignments = assignments[br];
            auto num_ref_ids = ref_assignments.size();
            for (decltype(num_ref_ids) rx = 0; rx < num_ref_ids; ++rx) {
                if (rx % 3 == 1) {
                    reference_mnns.push_back(ref_assignments[rx]);
                }
            }
        }
        std::sort(reference_mnns.begin(), reference_mnns.end());
    }

    mnncorrect::internal::BatchInfo<int, double> target_batch(std::move(all_batches.back()));
    all_batches.pop_back();
    mnncorrect::internal::NeighborSet<int, double> mnn_computed;
    mnncorrect::internal::search_for_neighbors_from_mnns(
        num_dim, 
        num_total,
        reference_mnns,
        target_mnns,
        simulated.data(),
        all_batches,
        target_batch,
        /* num_neighbors = */ 10,
        /* num_threads = */ 1,
        mnn_computed 
    );

    for (auto r : reference_mnns) {
        const auto& current = mnn_computed[r];
        EXPECT_TRUE(std::is_sorted(current.begin(), current.end()));
        bool found_self = false;
        for (auto pair : current) {
            found_self = found_self || pair.first == r;
            EXPECT_EQ(pair.second, 0);
        }
        EXPECT_TRUE(found_self);
    }

    for (auto t : target_mnns) {
        const auto& current = mnn_computed[t];
        EXPECT_TRUE(std::is_sorted(current.begin(), current.end()));
        bool found_self = false;
        for (auto pair : current) {
            found_self = found_self || pair.first == t;
            EXPECT_EQ(pair.second, 0);
        }
        EXPECT_TRUE(found_self);
    }
}

class SearchForNeighborsToMnnsTest : public ::testing::TestWithParam<std::tuple<std::vector<int>, int> > {};

TEST_P(SearchForNeighborsToMnnsTest, Basic) {
    auto params = GetParam();
    auto batch_sizes = std::get<0>(params);
    auto num_neighbors = std::get<1>(params);

    auto num_batches = batch_sizes.size();
    int num_total = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 0);

    const std::size_t num_dim = 8;
    std::vector<double> simulated = scran_tests::simulate_vector(num_dim * num_total, [&]{
        scran_tests::SimulationParameters opt;
        opt.seed = num_total * num_batches;
        return opt;
    }());

    std::vector<int> in_ref, in_target, ref_mnn, target_mnn;
    std::mt19937_64 rng(/* seed = */ num_total * 10 + num_neighbors);
    std::vector<int> batch_of_origin(num_total);
    std::vector<unsigned char> mnn(num_total);
    for (int i = 0; i < num_total; ++i) {
        bool is_mnn = rng() % 1000 < 100;
        bool is_target = rng() % 1000 < 300;
        batch_of_origin[i] = is_target;
        mnn[i] = is_mnn;
        if (is_target) {
            in_target.push_back(i);
            if (is_mnn) {
                target_mnn.push_back(i);
            }
        } else {
            in_ref.push_back(i);
            batch_of_origin[i] = 0;
            if (is_mnn) {
                ref_mnn.push_back(i);
            }
        }
    }

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    std::vector<double> buffer;
    auto ref_index = mnncorrect::internal::build_mnn_only_index(num_dim, simulated.data(), ref_mnn, builder, buffer);
    auto target_index = mnncorrect::internal::build_mnn_only_index(num_dim, simulated.data(), target_mnn, builder, buffer);

    mnncorrect::internal::NeighborSet<int, double> output;
    mnncorrect::internal::search_for_neighbors_to_mnns(
        num_dim,
        num_total,
        in_ref,
        in_target,
        simulated.data(),
        *ref_index,
        *target_index,
        num_neighbors,
        /* num_threads = */ 1,
        output
    ); 

    ASSERT_EQ(output.size(), num_total);
    for (int i = 0; i < num_total; ++i) {
        const auto& current = output[i];
        EXPECT_FALSE(current.empty());
        auto batch = batch_of_origin[i];
        const auto& mnn_ids = (batch ? target_mnn : ref_mnn);
        if (mnn[i]) {
            EXPECT_EQ(i, mnn_ids[current.front().first]);
        }
        for (auto pair : current) {
            EXPECT_LT(static_cast<std::size_t>(pair.first), mnn_ids.size());
        }
    }

    // Same results in parallel, and with a dirty input.
    auto copy = output;
    for (auto& current : copy) {
        std::reverse(current.begin(), current.end());
    }
    mnncorrect::internal::search_for_neighbors_to_mnns(
        num_dim,
        num_total,
        in_ref,
        in_target,
        simulated.data(),
        *ref_index,
        *target_index,
        num_neighbors,
        /* num_threads = */ 3,
        copy
    ); 
    for (int i = 0; i < num_total; ++i) {
        EXPECT_EQ(copy[i], output[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CorrectTarget,
    SearchForNeighborsToMnnsTest, 
    ::testing::Combine(
        ::testing::Values( // batch sizes
            std::vector<int>{ 250, 150 },
            std::vector<int>{ 136, 91, 215 },
            std::vector<int>{ 78, 93, 64, 51 }
        ),
        ::testing::Values( // number of neighbors
            5, 10, 20
        )
    )
);

TEST(CorrectTarget, InvertNeighbors) {
    mnncorrect::internal::NeighborSet<int, double> nns(20);
    std::vector<int> in_batch{ 3, 7, 11, 13 };

    nns[in_batch[0]].emplace_back(4, 0.12); // distances are more-or-less random. 
    nns[in_batch[0]].emplace_back(1, 0.32);
    nns[in_batch[0]].emplace_back(2, 1);

    nns[in_batch[1]].emplace_back(2, 0.3);
    nns[in_batch[1]].emplace_back(3, 0.5);
    nns[in_batch[1]].emplace_back(4, 1);

    nns[in_batch[2]].emplace_back(0, 0.1);
    nns[in_batch[3]].emplace_back(4, 0.01);

    auto inv = mnncorrect::internal::invert_neighbors(5, in_batch, nns, /* num_threads = */ 1);
    EXPECT_EQ(inv.size(), 5);
    {
        std::vector<std::pair<int, double> > exp0 { { 11, 0.1 } };
        EXPECT_EQ(inv[0], exp0);
        std::vector<std::pair<int, double> > exp1 { { 3, 0.32 } };
        EXPECT_EQ(inv[1], exp1);
        std::vector<std::pair<int, double> > exp2 { { 7, 0.3 }, { 3, 1.0 } };
        EXPECT_EQ(inv[2], exp2);
        std::vector<std::pair<int, double> > exp3 { { 7, 0.5 } };
        EXPECT_EQ(inv[3], exp3);
        std::vector<std::pair<int, double> > exp4 { { 13, 0.01 }, { 3, 0.12 }, { 7, 1.0 } };
        EXPECT_EQ(inv[4], exp4);
    }

    auto pinv = mnncorrect::internal::invert_neighbors(5, in_batch, nns, /* num_threads = */ 3);
    EXPECT_EQ(inv, pinv);
}

template<typename ... Args_>
double mean(Args_... args) {
    std::vector<double> values{ args... };
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

TEST(CorrectTarget, ComputeCenterOfMass) {
    std::size_t ndim = 3;
    int ntotal = 100;
    std::vector<double> data(ndim * ntotal);
    for (int i = 0; i < ntotal; ++i) {
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            data[d + ndim * i] = i + 0.1 * d; // filling with some simple numbers for easy testing.
        }
    }

    std::vector<int> in_mnns{ 5, 10, 20 };
    mnncorrect::internal::NeighborSet<int, double> neighbors_from(ntotal);
    neighbors_from[5] = std::vector<std::pair<int, double> >{ { 5, 0 }, { 4, 1 }, { 6, 1 } };
    neighbors_from[10] = std::vector<std::pair<int, double> >{ { 10, 0 }, { 9, 1 }, { 11, 1 }, { 12, 2 } };
    neighbors_from[20] = std::vector<std::pair<int, double> >{ { 20, 0 }, { 19, 1 }, { 21, 1 }, { 18, 2 }, { 22, 2 }, { 23, 3 } };

    // Checking that the redundant observations aren't added during the refining stage.
    {
        mnncorrect::internal::NeighborSet<int, double> neighbors_to(in_mnns.size());
        neighbors_to[0] = std::vector<std::pair<int, double> >{ { 5, 0 }, { 4, 1 } };
        neighbors_to[1] = std::vector<std::pair<int, double> >{ { 9, 1 }, { 12, 2 } };
        neighbors_to[2] = std::vector<std::pair<int, double> >{ { 20, 0 }, { 18, 2 }, { 22, 2 } };

        std::vector<double> running_means;
        mnncorrect::internal::compute_center_of_mass(ndim, in_mnns, neighbors_from, neighbors_to, data.data(), /* num_threads = */ 1, /* tolerance = */ 10, running_means);

        ASSERT_EQ(running_means.size(), in_mnns.size() * ndim);
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            auto shift = d * 0.1;
            EXPECT_FLOAT_EQ(running_means[d], mean(5.0, 4.0, 6.0) + shift);
            EXPECT_FLOAT_EQ(running_means[d + ndim], mean(10.0, 9.0, 11.0, 12.0) + shift);
            EXPECT_FLOAT_EQ(running_means[d + 2 * ndim], mean(20.0, 19.0, 21.0, 18.0, 22.0, 23.0) + shift);
        }

        // Same result with multiple threads.
        std::vector<double> pres;
        mnncorrect::internal::compute_center_of_mass(ndim, in_mnns, neighbors_from, neighbors_to, data.data(), /* num_threads = */ 2, /* tolerance = */ 10, pres);
        EXPECT_EQ(pres, running_means);
    }

    // Checking that new observations are added during the refining stage.
    {
        mnncorrect::internal::NeighborSet<int, double> neighbors_to(in_mnns.size());
        neighbors_to[0] = std::vector<std::pair<int, double> >{ { 5, 0 }, { 7, 2 } }; // at the end
        neighbors_to[1] = std::vector<std::pair<int, double> >{ { 10, 0 }, { 13, 1 }, { 12, 2 } }; // in the middle (distance is wrong but is only used for sorting).
        neighbors_to[2] = std::vector<std::pair<int, double> >{ { 17, 2 }, { 16, 3 }, { 23, 3 } }; // at the front (distance is wrong but is only used for sorting).

        std::vector<double> running_means;
        mnncorrect::internal::compute_center_of_mass(ndim, in_mnns, neighbors_from, neighbors_to, data.data(), /* num_threads = */ 1, /* tolerance = */ 10, running_means);

        ASSERT_EQ(running_means.size(), in_mnns.size() * ndim);
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            auto shift = d * 0.1;
            EXPECT_FLOAT_EQ(running_means[d], mean(5.0, 4.0, 6.0, 7.0) + shift);
            EXPECT_FLOAT_EQ(running_means[d + ndim], mean(10.0, 9.0, 11.0, 12.0, 13.0) + shift);
            EXPECT_FLOAT_EQ(running_means[d + 2 * ndim], mean(20.0, 19.0, 21.0, 18.0, 22.0, 23.0, 17.0, 16.0) + shift);
        }

        // Same result with multiple threads.
        std::vector<double> pres;
        mnncorrect::internal::compute_center_of_mass(ndim, in_mnns, neighbors_from, neighbors_to, data.data(), /* num_threads = */ 2, /* tolerance = */ 10, pres);
        EXPECT_EQ(pres, running_means);
    }

    // Checking that observations are correctly filtered out based on the tolerance.
    {
        mnncorrect::internal::NeighborSet<int, double> neighbors_to(in_mnns.size());

        // Distances are all wrong here, but are just used for sorting, so it's fine.
        neighbors_to[0] = std::vector<std::pair<int, double> >{ 
            { 9, 2 }, // Seed SD is 1, mean is 5, so adding 9 is ignored.
            { 7, 2 }, // Adding 7 is fine, bringing the mean to 5.6 and the SD to 1.29.
            { 1, 3 }, // Adding 1 is just out of range and is ignored.
            { 2, 3 } // 2 is fine.
        };
        neighbors_to[1] = std::vector<std::pair<int, double> >{ 
            { 14, 2 }, // mean is 10.5, SD is 1.29, so this is fine.
            { 17, 3 }, // mean is 11.2, SD is 1.92, so this is ignored (just).
            { 16, 3 }  // 16 is fine. 
        };
        neighbors_to[2] = std::vector<std::pair<int, double> >{
            { 50, 2 }, // mean is 20.5, SD is 1.87, but this would obviously be ignored anyway.
            { 26, 4 }, // adding 26 is fine (just), bringing the mean to 21.18 and the SD to 2.69.
            { 30, 5 }  // ignored again.
        };

        std::vector<double> running_means;
        mnncorrect::internal::compute_center_of_mass(ndim, in_mnns, neighbors_from, neighbors_to, data.data(), /* num_threads = */ 1, /* tolerance = */ 3, running_means);

        ASSERT_EQ(running_means.size(), in_mnns.size() * ndim);
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            auto shift = d * 0.1;
            EXPECT_FLOAT_EQ(running_means[d], mean(5.0, 4.0, 6.0, 7.0, 2.0) + shift);
            EXPECT_FLOAT_EQ(running_means[d + ndim], mean(10.0, 9.0, 11.0, 12.0, 14.0, 16.0) + shift);
            EXPECT_FLOAT_EQ(running_means[d + 2 * ndim], mean(20.0, 19.0, 21.0, 18.0, 22.0, 23.0, 26.0) + shift);
        }

        // Same result with multiple threads.
        std::vector<double> pres;
        mnncorrect::internal::compute_center_of_mass(ndim, in_mnns, neighbors_from, neighbors_to, data.data(), /* num_threads = */ 2, /* tolerance = */ 3, pres);
        EXPECT_EQ(pres, running_means);
    }
}

class CorrectTargetTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(CorrectTargetTest, Sanity) {
    std::size_t ndim = 5;
    auto param = GetParam();
    int nleft = std::get<0>(param);
    int nright = std::get<1>(param);
    int k = std::get<2>(param);

    int ntotal = nleft + nright;
    auto simulated = scran_tests::simulate_vector(ntotal * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.lower = -2;
        sparams.upper = 2;
        sparams.seed = 42 + nleft * 10 + nright + k;
        return sparams;
    }());

    for (int r = nleft; r < ntotal; ++r) {
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            simulated[r * ndim + d] += 10;
        }
    }

    // First, building the batch objects. We add an empty reference just so
    // that we can check that it corrects to the first batch.
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    std::vector<mnncorrect::internal::BatchInfo<int, double> > references(2);
    references[0].num_obs = 0;
    references[0].offset = 0;
    references[0].index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, 0, NULL));
    references[1].num_obs = nleft;
    references[1].offset = 0;
    references[1].index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nleft, simulated.data()));

    mnncorrect::internal::BatchInfo<int, double> target;
    target.num_obs = nright;
    target.offset = nleft;
    target.index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nright, simulated.data() + nleft * ndim));

    // Finding neighbors and MNNs.
    mnncorrect::internal::FindBatchNeighborsResults<int, double> batch_nns;
    mnncorrect::internal::find_batch_neighbors(ndim, ntotal, references, target, simulated.data(), k, /* num_threads = */ 1, batch_nns);

    mnncorrect::internal::FindClosestMnnWorkspace<int> mnn_work;
    mnncorrect::internal::FindClosestMnnResults<int> mnn_res;
    mnncorrect::internal::find_closest_mnn(batch_nns, mnn_work, mnn_res);

    // Actually running the correction now.
    mnncorrect::internal::CorrectTargetWorkspace<int, double> correct_work;
    mnncorrect::internal::CorrectTargetResults correct_res;

    auto copy = simulated;
    mnncorrect::internal::correct_target(
        ndim,
        ntotal,
        references,
        target,
        batch_nns,
        mnn_res,
        builder,
        k,
        /* num_threads = */ 1,
        /* tolerance = */ 3,
        copy.data(),
        correct_work,
        correct_res
    );

    for (int r = nleft; r < ntotal; ++r) {
        EXPECT_EQ(correct_res.batch[r], 1);
    }

    // Not entirely sure how to check for correctness here; 
    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
    {
        std::vector<double> left_means(ndim), right_means(ndim);
        for (int l = 0; l < nleft; ++l) {
            for (decltype(ndim) d = 0; d < ndim; ++d) {
                std::size_t offset = l * ndim + d;
                left_means[d] += copy[offset];
                EXPECT_EQ(copy[offset], simulated[offset]); // reference values are unchanged.
            }
        }
        for (int r = nright; r < ntotal; ++r) {
            for (decltype(ndim) d = 0; d < ndim; ++d) {
                right_means[d] += copy[r * ndim + d];
            }
        }
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            left_means[d] /= nleft;
            right_means[d] /= nright;
            double delta = std::abs(left_means[d] - right_means[d]);
            EXPECT_TRUE(delta < 1);
        }
    }

    // Same result with multiple threads, and after scrambling some of the inputs.
    {
        std::reverse(correct_work.ref_buffer.begin(), correct_work.ref_buffer.end());
        std::reverse(correct_work.target_buffer.begin(), correct_work.target_buffer.end());
        std::reverse(correct_work.neighbors_to.begin(), correct_work.neighbors_to.end());
        std::reverse(correct_work.neighbors_from.begin(), correct_work.neighbors_from.end());
        std::reverse(correct_work.mapping.begin(), correct_work.mapping.end());
        std::reverse(correct_res.batch.begin(), correct_res.batch.end());

        auto pcopy = simulated;
        mnncorrect::internal::correct_target(
            ndim,
            ntotal,
            references,
            target,
            batch_nns,
            mnn_res,
            builder,
            k,
            /* num_threads = */ 1,
            /* tolerance = */ 3,
            pcopy.data(),
            correct_work,
            correct_res
        );
        EXPECT_EQ(copy, pcopy);
        for (int r = nleft; r < ntotal; ++r) {
            EXPECT_EQ(correct_res.batch[r], 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    CorrectTarget,
    CorrectTargetTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);
