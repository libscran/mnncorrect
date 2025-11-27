#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.
#include "utils.h"

#include "scran_tests/scran_tests.hpp"
#include "knncolle/knncolle.hpp"

#include <random>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <utility>
#include <map>
#include <limits>

#include "mnncorrect/find_closest_mnn.hpp"

class FindClosestMnnTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    static std::pair<std::vector<int>, std::vector<int> > compute_reference(
        const mnncorrect::internal::NeighborSet<int, double>& all_neighbors, 
        const std::vector<int>& ref_ids,
        const std::vector<int>& target_ids) 
    {
        std::map<std::pair<int, int>, double> found;
        for (auto r : ref_ids) {
            const auto& current = all_neighbors[r];
            for (const auto& x : current) {
                found[std::pair<int, int>(r, x.first)] = x.second;
            }
        }

        std::pair<std::vector<int>, std::vector<int> > output;
        for (auto t : target_ids) {
            const auto& current = all_neighbors[t];

            int best_ref = -1;
            double best_dist = std::numeric_limits<double>::max();
            for (const auto& x : current) {
                auto it = found.find(std::pair<std::size_t, std::size_t>(x.first, t));
                if (it != found.end()) {
                    if (x.second <= best_dist) {
                        best_ref = x.first;
                        best_dist = x.second;
                    }
                }
            }

            if (best_ref >= 0) {
                output.first.push_back(t);
                output.second.push_back(best_ref);
            }
        }

        return output;
    }
};

TEST_P(FindClosestMnnTest, Check) {
    auto param = GetParam();
    int nref = std::get<0>(param);
    int ntarget = std::get<1>(param);
    int k = std::get<2>(param);

    int ndim = 7;
    int ntotal = nref + ntarget;
    auto simulated = scran_tests::simulate_vector(ntotal * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 42 + nref + ntarget * 10 + k;
        return sparams;
    }());

    std::vector<mnncorrect::BatchIndex> assignments(ntotal);
    std::fill(assignments.begin() + nref, assignments.end(), 1);
    std::mt19937_64 rng(/* seed = */ ntotal + k * nref);
    std::shuffle(assignments.begin(), assignments.end(), rng);

    std::vector<int> ref_ids, target_ids;
    for (int i = 0; i < ntotal; ++i) {
        if (assignments[i]) {
            target_ids.push_back(i);
        } else {
            ref_ids.push_back(i);
        }
    }

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Computing the reference.
    std::vector<double> buffer;
    auto ref_index = subset_and_index(ndim, ref_ids, simulated.data(), builder, buffer);
    auto target_index = subset_and_index(ndim, target_ids, simulated.data(), builder, buffer);

    mnncorrect::internal::NeighborSet<int, double> all_neighbors(ntotal);
    find_neighbors(ndim, ref_ids, simulated.data(), *target_index, target_ids, k, all_neighbors);
    find_neighbors(ndim, target_ids, simulated.data(), *ref_index, ref_ids, k, all_neighbors);

    auto expected_mnns = compute_reference(all_neighbors, ref_ids, target_ids);

    // Computing our values.
    mnncorrect::internal::FindClosestMnnWorkspace<int> workspace;
    mnncorrect::internal::FindClosestMnnResults<int> mnns;
    mnncorrect::internal::find_closest_mnn(target_ids, all_neighbors, workspace, mnns);
    EXPECT_EQ(expected_mnns.first, mnns.target_mnns);
    EXPECT_EQ(expected_mnns.second, mnns.ref_mnns);

    // Checking that we are unaffected by existing values in the workspace or results.
    std::reverse(mnns.target_mnns.begin(), mnns.target_mnns.end());
    std::reverse(mnns.ref_mnns.begin(), mnns.ref_mnns.end());
    for (auto& rb : workspace.reverse_neighbor_buffer) {
        std::reverse(rb.begin(), rb.end());
    }
    std::reverse(workspace.last_checked.begin(), workspace.last_checked.end());

    mnncorrect::internal::find_closest_mnn(target_ids, all_neighbors, workspace, mnns);
    EXPECT_EQ(expected_mnns.first, mnns.target_mnns);
    EXPECT_EQ(expected_mnns.second, mnns.ref_mnns);
}

INSTANTIATE_TEST_SUITE_P(
    FindClosestMnn,
    FindClosestMnnTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // ref
        ::testing::Values(100, 1000), // target
        ::testing::Values(10, 50) // num neighbors
    )
);
