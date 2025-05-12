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
        const std::vector<int>& left_ids,
        const std::vector<int>& right_ids) 
    {
        std::map<std::pair<int, int>, double> found;
        for (auto l : left_ids) {
            const auto& current = all_neighbors[l];
            for (const auto& x : current) {
                found[std::pair<int, int>(l, x.first)] = x.second;
            }
        }

        std::pair<std::vector<int>, std::vector<int> > output;
        for (auto r : right_ids) {
            const auto& current = all_neighbors[r];

            int best_ref = -1;
            double best_dist = std::numeric_limits<double>::max();
            for (const auto& x : current) {
                auto it = found.find(std::pair<std::size_t, std::size_t>(x.first, r));
                if (it != found.end()) {
                    if (x.second <= best_dist) {
                        best_ref = x.first;
                        best_dist = x.second;
                    }
                }
            }

            if (best_ref >= 0) {
                output.first.push_back(r);
                output.second.push_back(best_ref);
            }
        }

        return output;
    }
};

TEST_P(FindClosestMnnTest, Check) {
    auto param = GetParam();
    int nleft = std::get<0>(param);
    int nright = std::get<1>(param);
    int k = std::get<2>(param);

    int ndim = 7;
    int ntotal = nleft + nright;
    auto simulated = scran_tests::simulate_vector(ntotal * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 42 + nleft + nright * 10 + k;
        return sparams;
    }());

    std::vector<mnncorrect::BatchIndex> assignments(ntotal);
    std::fill(assignments.begin() + nleft, assignments.end(), 1);
    std::mt19937_64 rng(/* seed = */ ntotal + k * nleft);
    std::shuffle(assignments.begin(), assignments.end(), rng);

    std::vector<int> left_ids, right_ids;
    for (int i = 0; i < ntotal; ++i) {
        if (assignments[i]) {
            right_ids.push_back(i);
        } else {
            left_ids.push_back(i);
        }
    }

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Computing the reference.
    std::vector<double> buffer;
    auto left_index = subset_and_index(ndim, left_ids, simulated.data(), builder, buffer);
    auto right_index = subset_and_index(ndim, right_ids, simulated.data(), builder, buffer);

    mnncorrect::internal::NeighborSet<int, double> all_neighbors(ntotal);
    find_neighbors(ndim, left_ids, simulated.data(), *right_index, right_ids, k, all_neighbors);
    find_neighbors(ndim, right_ids, simulated.data(), *left_index, left_ids, k, all_neighbors);

    auto expected_mnns = compute_reference(all_neighbors, left_ids, right_ids);

    // Computing our values.
    mnncorrect::internal::FindBatchNeighborsResults<int, double> batch_nns;
    batch_nns.neighbors.swap(all_neighbors);
    batch_nns.ref_ids.swap(left_ids);
    batch_nns.target_ids.swap(right_ids);
    batch_nns.batch.swap(assignments);

    mnncorrect::internal::FindClosestMnnWorkspace<int> workspace;
    mnncorrect::internal::FindClosestMnnResults<int> mnns;
    mnncorrect::internal::find_closest_mnn(batch_nns, workspace, mnns);
    EXPECT_EQ(expected_mnns.first, mnns.target_mnns);
    EXPECT_EQ(expected_mnns.second, mnns.ref_mnns_partner);

    // Checking the uniques.
    std::set<int> sorted_set(mnns.ref_mnns_partner.begin(), mnns.ref_mnns_partner.end());
    std::vector<int> sorted(sorted_set.begin(), sorted_set.end());
    EXPECT_EQ(sorted, mnns.ref_mnns_unique);

    // Checking that we are unaffected by existing values in the workspace or results.
    std::reverse(mnns.target_mnns.begin(), mnns.target_mnns.end());
    std::reverse(mnns.ref_mnns_partner.begin(), mnns.ref_mnns_partner.end());
    std::reverse(mnns.ref_mnns_unique.begin(), mnns.ref_mnns_unique.end());

    for (auto& rb : workspace.reverse_neighbor_buffer) {
        std::reverse(rb.begin(), rb.end());
    }
    std::reverse(workspace.ref_mnn_buffer.begin(), workspace.ref_mnn_buffer.end());
    std::reverse(workspace.last_checked.begin(), workspace.last_checked.end());

    mnncorrect::internal::find_closest_mnn(batch_nns, workspace, mnns);
    EXPECT_EQ(expected_mnns.first, mnns.target_mnns);
    EXPECT_EQ(expected_mnns.second, mnns.ref_mnns_partner);
    EXPECT_EQ(sorted, mnns.ref_mnns_unique);
}

INSTANTIATE_TEST_SUITE_P(
    FindClosestMnn,
    FindClosestMnnTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50) // num neighbors
    )
);
