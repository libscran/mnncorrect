#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.
#include "utils.h"

#include "mnncorrect/correct_target.hpp"
#include "mnncorrect/fuse_nn_results.hpp"
#include "knncolle/knncolle.hpp"

#include <cstddef>
#include <utility>
#include <vector>

class WalkAroundNeighborhoodTest : public ::testing::TestWithParam<std::tuple<int, int> > {
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
        mnncorrect::internal::NeighborSet<int, double>& neighbors)
    {
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

    std::size_t ndim = 5;
    int nobs = 100;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = k * 10 + steps;
        return sparams;
    }());

    mnncorrect::internal::BatchInfo<int, double> target;
    target.offset = 0;
    target.num_obs = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    std::vector<int> to_check { 1, 5, 17, 31, 53, 97 };
    mnncorrect::internal::CorrectTargetWorkspace<int, double> workspace;
    mnncorrect::internal::walk_around_neighborhood(
        ndim,
        nobs, 
        to_check,
        vec.data(),
        target,
        k,
        steps,
        /* num_threads = */ 1,
        workspace
    );

    // Checking against a reference.
    std::vector<int> indices;
    std::vector<double> distances;
    mnncorrect::internal::NeighborSet<int, double> copy(nobs);
    auto searcher = target.index->initialize();
    for (auto i : to_check) {
        reference(ndim, i, vec.data(), *searcher, k, steps, indices, distances, copy);
    }
    for (int o = 0; o < nobs; ++o) {
        EXPECT_EQ(workspace.neighbors[o], copy[o]);
    }

    // Check that it gives the same results for multiple threads. Also giving it some
    // dirty output containers to check that it sanitizes them. 
    auto pcopy = workspace.neighbors;
    workspace.ids.resize(1000, -1); 
    workspace.next_visit.resize(1000, -1); 
    for (auto& nn : workspace.neighbors) {
        std::reverse(nn.begin(), nn.end());
    }
    mnncorrect::internal::walk_around_neighborhood(
        ndim,
        nobs, 
        to_check,
        vec.data(),
        target,
        k,
        steps,
        /* num_threads = */ 3,
        workspace
    );
    for (int o = 0; o < nobs; ++o) {
        EXPECT_EQ(workspace.neighbors[o], copy[o]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CorrectTarget,
    WalkAroundNeighborhoodTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // number of neighbors.
        ::testing::Values(0, 1, 2, 3) // number of steps.
    )
);

TEST(CorrectTarget, WalkAroundNeighborhoodQuitEarly) {
    std::size_t ndim = 5;
    int nobs = 10;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, {});

    mnncorrect::internal::BatchInfo<int, double> target;
    target.offset = 0;
    target.num_obs = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    // Get some coverage on our loop break if we don't need to use all of the
    // steps, in this case because we've already covered all the observations
    // in the dataset after the first step.
    std::vector<int> to_check{ 5 };
    mnncorrect::internal::CorrectTargetWorkspace<int, double> workspace;
    mnncorrect::internal::walk_around_neighborhood(
        ndim,
        nobs, 
        to_check,
        vec.data(),
        target,
        /* num_neighbors = */ nobs,
        /* num_steps = */ 3,
        /* num_threads = */ 3,
        workspace
    );

    for (const auto& found : workspace.neighbors) {
        EXPECT_EQ(found.size(), nobs);
    }
}

/***************************************************/

class ComputeCenterOfMassTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    static void reference(
        std::size_t ndim,
        int position,
        const mnncorrect::internal::NeighborSet<int, double>& neighbors,
        const double* data,
        int remaining,
        double* output,
        std::unordered_set<int>& used) 
    {
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

    std::size_t ndim = 5;
    int nobs = 100;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = k * 10 + steps;
        return sparams;
    }());

    mnncorrect::internal::BatchInfo<int, double> target;
    target.offset = 0;
    target.num_obs = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    // Finding neighbors.
    std::vector<int> to_check { 1, 5, 17, 31, 53, 97 };
    mnncorrect::internal::CorrectTargetWorkspace<int, double> workspace;
    mnncorrect::internal::walk_around_neighborhood(
        ndim,
        nobs, 
        to_check,
        vec.data(),
        target,
        k,
        steps,
        /* num_threads = */ 1,
        workspace
    );

    // Now computing a center of mass.
    std::size_t full_size = ndim * to_check.size(); 
    workspace.ref_center_buffer.resize(full_size);
    mnncorrect::internal::compute_center_of_mass(
        ndim,
        to_check,
        vec.data(),
        steps,
        /* num_threads = */ 1,
        workspace.neighbors,
        workspace.ref_center_buffer.data()
    );

    // Computing the reference.
    std::vector<double> ref(ndim);
    std::unordered_set<int> used;
    for (std::size_t i = 0, end = to_check.size(); i < end; ++i) {
        std::fill(ref.begin(), ref.end(), 0);
        used.clear();
        reference(
            ndim,
            to_check[i],
            workspace.neighbors,
            vec.data(),
            steps,
            ref.data(),
            used
        );
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            EXPECT_FLOAT_EQ(ref[d] / used.size(), workspace.ref_center_buffer[d + i * ndim]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    CorrectTarget,
    ComputeCenterOfMassTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // number of neighbors.
        ::testing::Values(0, 1, 2, 3) // number of steps.
    )
);

TEST(CorrectTarget, ComputeCenterOfMassTestQuitEarly) {
    std::size_t ndim = 5;
    int nobs = 10;
    auto vec = scran_tests::simulate_vector(static_cast<std::size_t>(nobs) * ndim, {});

    mnncorrect::internal::BatchInfo<int, double> target;
    target.offset = 0;
    target.num_obs = nobs;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    target.index = builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, vec.data()));

    std::vector<int> to_check{ 5 };
    mnncorrect::internal::CorrectTargetWorkspace<int, double> workspace;
    mnncorrect::internal::walk_around_neighborhood(
        ndim,
        nobs, 
        to_check,
        vec.data(),
        target,
        /* num_neighbors = */ nobs,
        /* num_steps = */ 3,
        /* num_threads = */ 3,
        workspace
    );

    // Gets some coverage on our loop break if we don't need to use all of the
    // steps, in this case because we've already covered all the observations
    // in the dataset after the first step.
    std::vector<double> center(ndim);
    mnncorrect::internal::compute_center_of_mass(
        ndim,
        to_check,
        vec.data(),
        /* num_steps = */ 3,
        /* num_threads = */ 1,
        workspace.neighbors,
        center.data()
    );

    // Checking that it is equal to the average of all points.
    std::vector<double> ref(ndim);
    for (decltype(nobs) o = 0; o < nobs; ++o) {
        for (decltype(ndim) d = 0; d < ndim; ++d) {
            ref[d] += vec[o * ndim + d];
        }
    }
    for (decltype(ndim) d = 0; d < ndim; ++d) {
        EXPECT_FLOAT_EQ(center[d], ref[d] / nobs);
    }
}

/***************************************************/

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

/**********************************************************/

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

    std::vector<int> target_ids(nright);
    std::iota(target_ids.begin(), target_ids.end(), nleft);
    std::vector<mnncorrect::BatchIndex> batch_of_origin(ntotal);
    std::fill_n(batch_of_origin.begin(), nleft, 1);
    std::fill_n(batch_of_origin.begin() + nleft, nright, 2);

    // Finding neighbors and MNNs.
    mnncorrect::internal::FindBatchNeighborsResults<int, double> batch_nns;
    mnncorrect::internal::find_batch_neighbors(ndim, ntotal, references, target, simulated.data(), k, /* num_threads = */ 1, batch_nns);

    mnncorrect::internal::FindClosestMnnWorkspace<int> mnn_work;
    mnncorrect::internal::FindClosestMnnResults<int> mnn_res;
    mnncorrect::internal::find_closest_mnn(target_ids, batch_nns.neighbors, mnn_work, mnn_res);

    // Actually running the correction now. Note that this needs more steps
    // to search for the center of mass, to make sure the corrected points
    // are well-mixed enough that the means fall under the tolerance. 
    mnncorrect::internal::CorrectTargetWorkspace<int, double> correct_work;
    mnncorrect::internal::CorrectTargetResults<int> correct_res;

    auto copy = simulated;
    mnncorrect::internal::correct_target(
        ndim,
        ntotal,
        references,
        target,
        target_ids,
        batch_of_origin,
        mnn_res,
        builder,
        k,
        /* num_steps = */ 4,
        /* num_threads = */ 1,
        copy.data(),
        correct_work,
        correct_res
    );

    EXPECT_EQ(correct_res.reassignments.size(), 2);
    EXPECT_TRUE(correct_res.reassignments[0].empty());
    EXPECT_EQ(correct_res.reassignments[1].size(), nright);
    EXPECT_EQ(correct_res.reassignments[1].front(), nleft);
    EXPECT_EQ(correct_res.reassignments[1].back(), ntotal - 1);

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
            EXPECT_LT(delta, 1);
        }
    }

    // Same result with multiple threads, and after scrambling some of the
    // inputs to check that dirty containers are cleared before use.
    {
        std::reverse(correct_work.ref_center_buffer.begin(), correct_work.ref_center_buffer.end());
        std::reverse(correct_work.correction_buffer.begin(), correct_work.correction_buffer.end());
        std::reverse(correct_work.neighbors.begin(), correct_work.neighbors.end());
        std::reverse(correct_work.ids.begin(), correct_work.ids.end());
        std::reverse(correct_work.new_target_batch.begin(), correct_work.new_target_batch.end());
        std::reverse(correct_res.reassignments.begin(), correct_res.reassignments.end());

        auto pcopy = simulated;
        mnncorrect::internal::correct_target(
            ndim,
            ntotal,
            references,
            target,
            target_ids,
            batch_of_origin,
            mnn_res,
            builder,
            k,
            /* num_steps = */ 4,
            /* num_threads = */ 3,
            pcopy.data(),
            correct_work,
            correct_res
        );
        EXPECT_EQ(copy, pcopy);
        EXPECT_EQ(correct_res.reassignments.size(), 2);
        EXPECT_TRUE(correct_res.reassignments[0].empty());
        EXPECT_EQ(correct_res.reassignments[1].size(), nright);
        EXPECT_EQ(correct_res.reassignments[1].front(), nleft);
        EXPECT_EQ(correct_res.reassignments[1].back(), ntotal - 1);
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
