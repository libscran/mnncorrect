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

//class CorrectTargetTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
//protected:
//    void SetUp() {
//        auto param = GetParam();
//        nleft = std::get<0>(param);
//        nright = std::get<1>(param);
//        k = std::get<2>(param);
//
//        left = scran_tests::simulate_vector(nleft * ndim, [&]{
//            scran_tests::SimulationParameters sparams;
//            sparams.lower = -2;
//            sparams.upper = 2;
//            sparams.seed = 42 + nleft * 10 + nright + k;
//            return sparams;
//        }());
//
//        right = scran_tests::simulate_vector(nright * ndim, [&]{
//            scran_tests::SimulationParameters sparams;
//            sparams.lower = -2 + 5; // throw in a batch effect.
//            sparams.upper = 2 + 5;
//            sparams.seed = 69 + nleft * 10 + nright + k;
//            return sparams;
//        }());
//
//        knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
//        auto left_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nleft, left.data()));
//        auto right_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nright, right.data()));
//
//        auto neighbors_of_left = mnncorrect::internal::quick_find_nns(nleft, left.data(), *right_index, k, /* nthreads = */ 1);
//        auto neighbors_of_right = mnncorrect::internal::quick_find_nns(nright, right.data(), *left_index, k, /* nthreads = */ 1);
//        pairings = mnncorrect::internal::find_mutual_nns(neighbors_of_left, neighbors_of_right);
//    }
//
//    int ndim = 5;
//    int nleft, nright;
//    int k;
//    std::vector<double> left, right;
//    mnncorrect::internal::MnnPairs<int> pairings;
//};
//
//TEST_P(CorrectTargetTest, CappedFindNns) {
//    auto right_mnn = mnncorrect::internal::unique_right(pairings);
//    std::vector<double> subbuffer(right_mnn.size() * ndim);
//    mnncorrect::internal::subset_to_mnns(ndim, right.data(), right_mnn, subbuffer.data());
//
//    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
//    auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, right_mnn.size(), subbuffer.data()));
//    auto full = mnncorrect::internal::quick_find_nns(nright, right.data(), *index, k, /* nthreads = */ 1);
//
//    auto cap_out = mnncorrect::internal::capped_find_nns(nright, right.data(), *index, k, 23, /* nthreads = */ 1);
//    auto gap = cap_out.first;
//    const auto& capped = cap_out.second;
//
//    EXPECT_EQ(capped.size(), 23);
//    EXPECT_GT(gap, 1);
//    for (std::size_t c = 0; c < capped.size(); ++c) {
//        EXPECT_EQ(full[static_cast<std::size_t>(c * gap)], capped[c]);
//    }
//
//    // Same results in parallel.
//    auto pcap_out = mnncorrect::internal::capped_find_nns(nright, right.data(), *index, k, 23, /* nthreads = */ 3);
//    EXPECT_EQ(pcap_out.first, cap_out.first);
//    EXPECT_EQ(pcap_out.second, cap_out.second);
//}
//
//TEST_P(CorrectTargetTest, CenterOfMass) {
//    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
//
//    // Setting up the values for a reasonable comparison.
//    auto left_mnn = mnncorrect::internal::unique_left(pairings);
//    std::vector<double> buffer_left(left_mnn.size() * ndim);
//    {
//        mnncorrect::internal::subset_to_mnns(ndim, left.data(), left_mnn, buffer_left.data());
//        auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, left_mnn.size(), buffer_left.data()));
//        auto closest_mnn = mnncorrect::internal::quick_find_nns(nleft, left.data(), *index, k, /* nthreads = */ 1);
//        auto inverted = mnncorrect::internal::invert_neighbors(left_mnn.size(), closest_mnn, /* nthreads = */ 1);
//        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, inverted, left.data(), buffer_left.data(), /* minimum_required = */ k, /*  num_mads = */ 3, /* nthreads = */ 1);
//
//        // Same results in parallel.
//        std::vector<double> par_buffer_left(left_mnn.size() * ndim);
//        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, inverted, left.data(), par_buffer_left.data(), /* minimum_required = */ k, /* num_mads = */ 3, /* nthreads = */ 3);
//        EXPECT_EQ(par_buffer_left, buffer_left);
//    }
//
//    auto right_mnn = mnncorrect::internal::unique_right(pairings);
//    std::vector<double> buffer_right(right_mnn.size() * ndim);
//    {
//        mnncorrect::internal::subset_to_mnns(ndim, right.data(), right_mnn, buffer_right.data());
//        auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, right_mnn.size(), buffer_right.data()));
//        auto closest_mnn = mnncorrect::internal::quick_find_nns(nright, right.data(), *index, k, /* nthreads = */ 1);
//        auto inverted = mnncorrect::internal::invert_neighbors(right_mnn.size(), closest_mnn, /* nthreads = */ 1);
//        mnncorrect::internal::compute_center_of_mass(ndim, right_mnn, inverted, right.data(), buffer_right.data(), /* minimum_required = */ k, /* num_mads = */ 3, /* nthreads = */ 1);
//    }
//
//    // Checking that the centroids are all close to the expected values.
//    std::vector<double> left_means(ndim);
//    for (std::size_t s = 0; s < left_mnn.size(); ++s) {
//        for (int d = 0; d < ndim; ++d) {
//            left_means[d] += buffer_left[s * ndim + d];
//        }
//    }
//    for (auto m : left_means) {
//        EXPECT_LT(std::abs(m / left_mnn.size()), 0.5);
//    }
//
//    std::vector<double> right_means(ndim);
//    for (std::size_t s = 0; s < right_mnn.size(); ++s) {
//        for (int d = 0; d < ndim; ++d) {
//            right_means[d] += buffer_right[s * ndim + d];
//        }
//    }
//    for (auto m : right_means) {
//        EXPECT_LT(std::abs(m / right_mnn.size() - 5), 0.5);
//    }
//
//    // Center of mass calculations work correctly if it's all empty.
//    {
//        mnncorrect::internal::NeighborSet<int, double> empty_inverted(left_mnn.size());
//        std::vector<double> empty_buffer_left(left_mnn.size() * ndim);
//        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, empty_inverted, left.data(), empty_buffer_left.data(), /* minimum_required = */ k, /* num_mads = */ 3, /* nthreads = */ 1);
//
//        std::vector<double> expected(left_mnn.size() * ndim);
//        mnncorrect::internal::subset_to_mnns(ndim, left.data(), left_mnn, expected.data());
//        EXPECT_EQ(empty_buffer_left, expected);
//    }
//}
//
//TEST_P(CorrectTargetTest, Correction) {
//    double nmads = 3;
//    int iterations = 2;
//    double trim = 0.2;
//    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
//
//    std::vector<double> buffer(nright * ndim);
//    mnncorrect::internal::correct_target(
//        ndim,
//        nleft,
//        left.data(),
//        nright,
//        right.data(),
//        pairings,
//        builder,
//        k,
//        nmads,
//        iterations,
//        trim,
//        buffer.data(),
//        /* mass_cap = */ 0, 
//        /* nthreads = */ 1
//    );
//
//    // Not entirely sure how to check for correctness here; 
//    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
//    {
//        std::vector<double> left_means(ndim), right_means(ndim);
//        for (int l = 0; l < nleft; ++l) {
//            for (int d = 0; d < ndim; ++d) {
//                left_means[d] += left[l * ndim + d];
//            }
//        }
//        for (int r = 0; r < nright; ++r) {
//            for (int d = 0; d < ndim; ++d) {
//                right_means[d] += buffer[r * ndim + d];
//            }
//        }
//        for (int d = 0; d < ndim; ++d) {
//            left_means[d] /= nleft;
//            right_means[d] /= nright;
//            double delta = std::abs(left_means[d] - right_means[d]);
//            EXPECT_TRUE(delta < 1);
//        }
//    }
//
//    // Same result with multiple threads.
//    {
//        std::vector<double> par_buffer(nright * ndim);
//        mnncorrect::internal::correct_target(
//            ndim,
//            nleft,
//            left.data(),
//            nright,
//            right.data(),
//            pairings,
//            builder,
//            k,
//            nmads,
//            iterations,
//            trim,
//            par_buffer.data(),
//            /* mass_cap = */ 0, 
//            /* nthreads = */ 3 
//        );
//        EXPECT_EQ(par_buffer, buffer);
//    }
//
//    // Different results with a cap.
//    {
//        std::vector<double> cap_buffer(nright * ndim);
//        mnncorrect::internal::correct_target(
//            ndim,
//            nleft,
//            left.data(),
//            nright,
//            right.data(),
//            pairings,
//            builder,
//            k,
//            nmads,
//            iterations,
//            trim,
//            cap_buffer.data(),
//            /* mass_cap = */ 50,
//            /* nthreads = */ 1 
//        );
//        EXPECT_NE(cap_buffer, buffer);
//    }
//
//    // Unless the cap is larger than the number of observations.
//    {
//        std::vector<double> cap_buffer(nright * ndim);
//        mnncorrect::internal::correct_target(
//            ndim,
//            nleft,
//            left.data(),
//            nright,
//            right.data(),
//            pairings,
//            builder,
//            k,
//            nmads,
//            iterations,
//            trim,
//            cap_buffer.data(),
//            /* mass_cap = */ 5000,
//            /* nthreads = */ 1 
//        );
//        EXPECT_EQ(cap_buffer, buffer);
//    }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    CorrectTarget,
//    CorrectTargetTest,
//    ::testing::Combine(
//        ::testing::Values(100, 1000), // left
//        ::testing::Values(100, 1000), // right
//        ::testing::Values(10, 50)  // choice of k
//    )
//);
