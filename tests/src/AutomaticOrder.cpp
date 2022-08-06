#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/AutomaticOrder.hpp"
#include <random>
#include <algorithm>
#include "order_utils.h"

TEST(AutomaticOrder, RunningVariances) {
    std::mt19937_64 rng(42);
    std::normal_distribution<> dist;

    int ndim = 12;
    size_t nobs = 34;

    std::vector<double> data;
    for (size_t s = 0; s < nobs; ++s) {
        for (int d = 0; d < ndim; ++d) {
            data.push_back(dist(rng));
        }
    }

    double ref = 0;
    for (int d = 0; d < ndim; ++d) {
        // First pass for the mean.
        double* pos = data.data() + d;
        double mean = 0;
        for (size_t s = 0; s < nobs; ++s, pos += ndim) {
            mean += *pos;
        }
        mean /= nobs;

        // Second pass for the variance.
        pos = data.data() + d;
        double variance = 0;
        for (size_t s = 0; s < nobs; ++s, pos += ndim) {
            variance += (*pos - mean) * (*pos - mean);
        }
        variance /= nobs - 1;
        ref += variance;
    }

    double running = mnncorrect::compute_total_variance(ndim, nobs, data.data(), false);
    EXPECT_FLOAT_EQ(running, ref);
    double rss = mnncorrect::compute_total_variance(ndim, nobs, data.data(), true);
    EXPECT_FLOAT_EQ(rss, ref * (nobs - 1));

    return;
}

struct AutomaticOrder2 : public mnncorrect::AutomaticOrder<int, double, Builder> {
    static constexpr mnncorrect::ReferencePolicy default_policy = mnncorrect::MaxSize;

    AutomaticOrder2(int nd, std::vector<size_t> no, std::vector<const double*> b, double* c, int k, mnncorrect::ReferencePolicy first = default_policy, int nthreads = 1) :
        AutomaticOrder<int, double, Builder>(nd, std::move(no), std::move(b), c, Builder(), k, first, nthreads) {}

    const std::vector<mnncorrect::NeighborSet<int, double> >& get_neighbors_ref () const { 
        return neighbors_ref;
    }
    const std::vector<mnncorrect::NeighborSet<int, double> >& get_neighbors_target () const { 
        return neighbors_target;
    }

    size_t get_ncorrected() const { 
        return ncorrected;
    }

    const std::set<size_t>& get_remaining () const { 
        return remaining; 
    }

    auto test_choose() const {
        return choose();
    }

    // The parallelized chooser is very complicated, so we test it against
    // the naive serial chooser, just in case.
    auto simple_choose() const {
        mnncorrect::MnnPairs<int> output;
        size_t chosen = 0;
        for (auto b : remaining) {
            auto tmp = mnncorrect::find_mutual_nns(neighbors_ref[b], neighbors_target[b]);
            if (tmp.num_pairs > output.num_pairs) {
                output = std::move(tmp);
                chosen = b;
            }
        }
       return std::pair<size_t, mnncorrect::MnnPairs<int> >(chosen, std::move(output));
    }

    void test_update(size_t latest) {
        update<false>(latest);
        return;
    }
};

class AutomaticOrderTest : public ::testing::TestWithParam<std::tuple<int, int, std::vector<size_t> > > {
protected:
    template<class Param>
    void assemble(Param param) {
        // Simulating values.
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist;

        ndim = std::get<0>(param);
        k = std::get<1>(param);
        sizes = std::get<2>(param);

        data.resize(sizes.size());
        ptrs.resize(sizes.size());
        for (size_t b = 0; b < sizes.size(); ++b) {
            for (size_t s = 0; s < sizes[b]; ++s) {
                for (int d = 0; d < ndim; ++d) {
                    data[b].push_back(dist(rng));
                }
            }
            ptrs[b] = data[b].data();
        }

        size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
        output.resize(total_size * ndim);
        return;
    }

    int ndim, k;
    std::vector<size_t> sizes;
    std::vector<std::vector<double> > data;
    std::vector<const double*> ptrs;
    std::vector<double> output;
};

TEST_P(AutomaticOrderTest, CheckInitialization) {
    assemble(GetParam());
    AutomaticOrder2 coords(ndim, sizes, ptrs, output.data(), k);

    size_t maxed = std::max_element(sizes.begin(), sizes.end()) - sizes.begin();
    const auto& ord = coords.get_order();
    EXPECT_EQ(ord.size(), 1);
    EXPECT_EQ(ord[0], maxed);

    size_t ncorrected = coords.get_ncorrected();
    EXPECT_EQ(ncorrected, sizes[maxed]);
    EXPECT_EQ(std::vector<double>(output.begin(), output.begin() + ncorrected * ndim), data[maxed]);
    EXPECT_EQ(coords.get_remaining().size(), sizes.size() - 1);

    const auto& rneighbors = coords.get_neighbors_ref(); 
    const auto& lneighbors = coords.get_neighbors_target();

    for (size_t b = 0; b < sizes.size(); ++b) {
        if (b == maxed) { 
            continue; 
        }

        EXPECT_EQ(rneighbors[b].size(), ncorrected);
        EXPECT_EQ(rneighbors[b][0].size(), k);
        EXPECT_EQ(lneighbors[b].size(), sizes[b]);
        EXPECT_EQ(lneighbors[b][0].size(), k);
    }
}

TEST_P(AutomaticOrderTest, CheckUpdate) {
    assemble(GetParam());
    AutomaticOrder2 coords(ndim, sizes, ptrs, output.data(), k);
    std::vector<char> used(sizes.size());
    used[coords.get_order()[0]] = true;

    // Compare to multi-threaded versions, using 2 or 3 threads.
    // We need to check this a bit more carefully because the
    // multithreading inside the AutomaticOrder class is wild.
    std::vector<double> par_output2(output.size());
    AutomaticOrder2 par_coords2(ndim, sizes, ptrs, par_output2.data(), k, AutomaticOrder2::default_policy, /* nthreads = */ 2);

    std::vector<double> par_output3(output.size());
    AutomaticOrder2 par_coords3(ndim, sizes, ptrs, par_output3.data(), k, AutomaticOrder2::default_policy, /* nthreads = */ 3);

    std::mt19937_64 rng(123456);
    std::normal_distribution<> dist;

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto chosen = coords.test_choose();
        EXPECT_FALSE(used[chosen.first]);
        used[chosen.first] = true;

        // Check that the MNN pair indices are correct.
        const auto& m = chosen.second.matches;
        EXPECT_TRUE(m.size() > 0);
        for (const auto& x : m) {
            EXPECT_TRUE(x.first < sizes[chosen.first]);
            for (const auto& y : x.second) {
                EXPECT_TRUE(y < coords.get_ncorrected());
            }
        }

        auto simpler = coords.simple_choose();
        EXPECT_EQ(chosen.first, simpler.first);
        EXPECT_EQ(chosen.second.num_pairs, simpler.second.num_pairs);
        EXPECT_EQ(chosen.second.matches, simpler.second.matches);

        // Applying an update. We mock up some corrected data so that the builders work correctly.
        size_t sofar = coords.get_ncorrected();
        double* fixed = output.data() + sofar * ndim;
        for (size_t s = 0; s < sizes[chosen.first]; ++s) {
            for (int d = 0; d < ndim; ++d) {
                fixed[s * ndim + d] = dist(rng);
            }
        }
        coords.test_update(chosen.first);

        // Check that the update works as expected.
        const auto& remaining = coords.get_remaining();
        EXPECT_EQ(remaining.size(), sizes.size() - b - 1);
        EXPECT_EQ(sofar + sizes[chosen.first], coords.get_ncorrected());

        const auto& ord = coords.get_order();
        EXPECT_EQ(ord.size(), b + 1);
        EXPECT_EQ(ord.back(), chosen.first);

        const auto& rneighbors = coords.get_neighbors_ref();
        for (auto r : remaining) {
            const auto& rcurrent = rneighbors[r];
            knncolle::VpTreeEuclidean<int, double> target_index(ndim, sizes[r], data[r].data());
            EXPECT_EQ(rcurrent.size(), coords.get_ncorrected());

            for (size_t x = 0; x < coords.get_ncorrected(); ++x) {
                auto naive = target_index.find_nearest_neighbors(output.data() + x * ndim, k);
                const auto& updated = rcurrent[x];
                compare_to_naive(naive, updated);
            }
        }

        knncolle::VpTreeEuclidean<> ref_index(ndim, coords.get_ncorrected(), output.data());
        const auto& tneighbors = coords.get_neighbors_target();
        for (auto r : remaining) {
            const auto& current = data[r];
            const auto& tcurrent = tneighbors[r];
            EXPECT_EQ(tcurrent.size(), sizes[r]);

            for (size_t x = 0; x < sizes[r]; ++x) {
                auto naive = ref_index.find_nearest_neighbors(current.data() + x * ndim, k);
                const auto& updated = tcurrent[x];
                compare_to_naive(naive, updated);
            }
        }

        // Doing the same for the parallelized runners.
        {
            auto par_chosen2 = par_coords2.test_choose();
            EXPECT_EQ(par_chosen2.first, chosen.first);
            EXPECT_EQ(chosen.second.num_pairs, par_chosen2.second.num_pairs);
            EXPECT_EQ(chosen.second.matches, par_chosen2.second.matches);

            double* par_fixed2 = par_output2.data() + sofar * ndim;
            std::copy(fixed, fixed + sizes[chosen.first] * ndim, par_fixed2);
            par_coords2.test_update(chosen.first);
        }

        {
            auto par_chosen3 = par_coords3.test_choose();
            EXPECT_EQ(par_chosen3.first, chosen.first);
            EXPECT_EQ(chosen.second.num_pairs, par_chosen3.second.num_pairs);
            EXPECT_EQ(chosen.second.matches, par_chosen3.second.matches);

            double* par_fixed3 = par_output3.data() + sofar * ndim;
            std::copy(fixed, fixed + sizes[chosen.first] * ndim, par_fixed3);
            par_coords3.test_update(chosen.first);
        }
    }

    // Same results when run in parallel.
    EXPECT_EQ(output, par_output2);
    EXPECT_EQ(output, par_output3);
}

TEST_P(AutomaticOrderTest, DifferentPolicies) {
    assemble(GetParam());

    // Choosing the smallest batch to amplify the variance.
    size_t chosen = std::min_element(sizes.begin(), sizes.end()) - sizes.begin();
    for (auto& d : data[chosen]) {
        d *= 10;
    }

    for (size_t iter = 0; iter < 4; ++iter) {
        mnncorrect::ReferencePolicy choice = mnncorrect::Input;
        if (iter == 1) {
            choice = mnncorrect::MaxSize;
        } else if (iter == 2) {
            choice = mnncorrect::MaxVariance;
        } else if (iter == 3) {
            choice = mnncorrect::MaxRss;
        }

        AutomaticOrder2 coords(ndim, sizes, ptrs, output.data(), k, choice);

        if (iter == 0) {
            EXPECT_EQ(coords.get_order()[0], 0);
        } else if (iter == 1) {
            auto first = coords.get_order()[0];
            for (auto s : sizes) {
                EXPECT_TRUE(sizes[first] >= s);
            }
        } else if (iter == 2) {
            EXPECT_EQ(coords.get_order()[0], chosen);
        } else if (iter == 3) {
            EXPECT_EQ(coords.get_order()[0], chosen);
        }

        std::mt19937_64 rng(123456);
        std::normal_distribution<> dist;
        std::vector<char> used(sizes.size());
        used[coords.get_order()[0]] = true;

        // Just checking that everything runs to completion under the non-default policies.
        if (choice != AutomaticOrder2::default_policy) {
            for (size_t b = 1; b < sizes.size(); ++b) {
                auto chosen = coords.test_choose();
                EXPECT_FALSE(used[chosen.first]);
                used[chosen.first] = true;

                // Applying an update. We mock up some corrected data so that the builders work correctly.
                size_t sofar = coords.get_ncorrected();
                double* fixed = output.data() + sofar * ndim;
                for (size_t s = 0; s < sizes[chosen.first]; ++s) {
                    for (int d = 0; d < ndim; ++d) {
                        fixed[s * ndim + d] = dist(rng);
                    }
                }

                coords.test_update(chosen.first);
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    AutomaticOrder,
    AutomaticOrderTest,
    ::testing::Combine(
        ::testing::Values(5), // Number of dimensions
        ::testing::Values(1, 5, 10), // Number of neighbors
        ::testing::Values(
            std::vector<size_t>{10, 20},        
            std::vector<size_t>{10, 20, 30}, 
            std::vector<size_t>{100, 50, 80}, 
            std::vector<size_t>{50, 30, 100, 90},
            std::vector<size_t>{50, 40, 30, 20, 10}
        )
    )
);
