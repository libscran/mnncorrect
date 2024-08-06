#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "scran_tests/scran_tests.hpp"

#include "mnncorrect/AutomaticOrder.hpp"
#include <random>
#include <algorithm>

TEST(AutomaticOrder, RunningVariances) {
    int ndim = 12;
    size_t nobs = 34;
    auto data = scran_tests::simulate_vector(ndim * nobs, scran_tests::SimulationParameters());

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

    std::vector<double> buffer(ndim);
    double running = mnncorrect::internal::compute_total_variance(ndim, nobs, data.data(), buffer, false);
    EXPECT_FLOAT_EQ(running, ref);

    double rss = mnncorrect::internal::compute_total_variance(ndim, nobs, data.data(), buffer, true);
    EXPECT_FLOAT_EQ(rss, ref * (nobs - 1));

    // Overlord function works, even with multiple threads.
    size_t nobs2 = 100;
    auto data2 = scran_tests::simulate_vector(ndim * nobs2, scran_tests::SimulationParameters());

    auto vars = mnncorrect::internal::compute_total_variances<double>(ndim, { nobs, nobs2 }, { data.data(), data2.data() }, false, 1);
    EXPECT_FLOAT_EQ(vars[0], running);
    EXPECT_FLOAT_EQ(vars[1], mnncorrect::internal::compute_total_variance(ndim, nobs2, data2.data(), buffer, false));

    auto pvars = mnncorrect::internal::compute_total_variances<double>(ndim, { nobs, nobs2 }, { data.data(), data2.data() }, false, 2);
    EXPECT_EQ(vars, pvars);
}

/****************************************************/

struct AutomaticOrder2 : public mnncorrect::internal::AutomaticOrder<int, int, double> {
    static constexpr mnncorrect::ReferencePolicy default_policy = mnncorrect::ReferencePolicy::MAX_SIZE;

    template<typename ... Args_>
    AutomaticOrder2(Args_&&... args) : AutomaticOrder<int, int, double>(std::forward<Args_>(args)...) {}

    const auto& get_neighbors_ref () const { 
        return my_neighbors_ref;
    }
    const auto& get_neighbors_target () const { 
        return my_neighbors_target;
    }

    size_t get_ncorrected() const { 
        return my_ncorrected;
    }

    const auto& get_remaining () const { 
        return my_remaining; 
    }

    auto test_choose() {
        return choose();
    }

    void test_update(size_t latest) {
        update<false>(latest);
        return;
    }
};

class AutomaticOrderTest : public ::testing::TestWithParam<std::tuple<int, std::vector<size_t> > > {
protected:
    void SetUp() {
        auto param = GetParam();
        k = std::get<0>(param);
        sizes = std::get<1>(param);

        data.resize(sizes.size());
        ptrs.resize(sizes.size());
        for (size_t b = 0; b < sizes.size(); ++b) {
            data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
                scran_tests::SimulationParameters sparams;
                sparams.lower = -2;
                sparams.upper = 2;
                sparams.seed = 42 + ndim * 10 + k * 100 + sizes[b];
                return sparams;
            }());
            ptrs[b] = data[b].data();
        }

        total_size = std::accumulate(sizes.begin(), sizes.end(), 0) * ndim;
    }

    int ndim = 5, k;
    std::vector<size_t> sizes;
    std::vector<std::vector<double> > data;
    std::vector<const double*> ptrs;
    int total_size;

public:
    static void compare_to_naive(const std::vector<int>& indices, const std::vector<double>& distances, const std::vector<std::pair<int, double> >& updated) {
        size_t n = indices.size();
        ASSERT_EQ(n, updated.size());
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(indices[i], updated[i].first);
            EXPECT_EQ(distances[i], updated[i].second);
        }
    }
};

TEST_P(AutomaticOrderTest, CheckInitialization) {
    std::vector<double> output(total_size);
    AutomaticOrder2 coords(
        ndim,
        sizes,
        ptrs,
        output.data(),
        std::make_unique<knncolle::VptreeBuilder<> >(),
        /* num_neighbors = */ k,
        /* ref_policy = */ AutomaticOrder2::default_policy,
        /* nobs_cap = */ -1,
        /* nthreads = */ 1
    );

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
    std::vector<AutomaticOrder2> all_coords;
    all_coords.reserve(3);
    std::vector<std::vector<double> > all_output(3);

    for (int t = 0; t < 3; ++t) {
        all_output[t].resize(total_size);
        all_coords.emplace_back(
            ndim,
            sizes,
            ptrs,
            all_output[t].data(),
            std::make_unique<knncolle::VptreeBuilder<> >(),
            /* num_neighbors = */ k,
            /* ref_policy = */ AutomaticOrder2::default_policy,
            /* nobs_cap = */ -1,
            /* nthreads = */ t + 1
        );
    }

    std::vector<char> used(sizes.size());
    used[all_coords.front().get_order()[0]] = true;

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto& coords0 = all_coords[0];
        size_t sofar = coords0.get_ncorrected();

        // The parallelized chooser with neighbor re-use is very complicated,
        // so we test it against the naive serial chooser, just in case.
        auto simpler = [&]{ 
            auto corrected = all_output[0].data();
            auto ref_index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, sofar, corrected));

            mnncorrect::internal::MnnPairs<int> output;
            size_t chosen = 0;
            for (auto r : coords0.get_remaining()) {
                auto target_to_ref = mnncorrect::internal::quick_find_nns(sizes[r], data[r].data(), *ref_index, /* k = */ k, /* num_threads = */ 1);
                auto target_index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, sizes[r], data[r].data()));
                auto ref_to_target = mnncorrect::internal::quick_find_nns(sofar, corrected, *target_index, /* k = */ k, /* num_threads = */ 1);

                auto tmp = mnncorrect::internal::find_mutual_nns(ref_to_target, target_to_ref);
                if (tmp.num_pairs > output.num_pairs) {
                    output = std::move(tmp);
                    chosen = r;
                }
            }

            return std::make_pair(chosen, std::move(output));
        }();

        EXPECT_FALSE(used[simpler.first]);
        used[simpler.first] = true;

        // Double-check that the MNN pair indices are sensible.
        const auto& m = simpler.second.matches;
        EXPECT_TRUE(m.size() > 0);
        for (const auto& x : m) {
            EXPECT_LT(x.first, sizes[simpler.first]);
            for (const auto& y : x.second) {
                EXPECT_LT(y, sofar);
            }
        }

        for (size_t i = 0; i < all_coords.size(); ++i) {
            auto chosen = all_coords[i].test_choose();
            EXPECT_EQ(chosen.first, simpler.first);
            EXPECT_EQ(chosen.second.num_pairs, simpler.second.num_pairs);
            EXPECT_EQ(chosen.second.matches, simpler.second.matches);
        }

        // Applying an update. We mock up some corrected data so that the builders work correctly.
        size_t cursize = sizes[simpler.first];
        auto corrected = scran_tests::simulate_vector(ndim * cursize, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = ndim * 1000 + k + b + 69;
            return sparams;
        }());

        size_t output_offset = ndim * sofar;
        for (size_t i = 0; i < all_coords.size(); ++i) {
            std::copy(corrected.begin(), corrected.end(), all_output[i].data() + output_offset);
            all_coords[i].test_update(simpler.first);
        }

        // Check that the update works as expected.
        const auto& remaining = coords0.get_remaining();
        EXPECT_EQ(remaining.size(), sizes.size() - b - 1);
        size_t new_sofar = coords0.get_ncorrected();
        EXPECT_EQ(sofar + sizes[simpler.first], new_sofar);

        const auto& ord = coords0.get_order();
        EXPECT_EQ(ord.size(), b + 1);
        EXPECT_EQ(ord.back(), simpler.first);

        const auto& rneighbors = coords0.get_neighbors_ref();
        for (auto r : remaining) {
            const auto& rcurrent = rneighbors[r];
            auto target_index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, sizes[r], data[r].data()));
            auto target_search = target_index->initialize();
            EXPECT_EQ(rcurrent.size(), new_sofar);

            std::vector<int> indices;
            std::vector<double> distances;
            for (size_t x = sofar; x < new_sofar; ++x) {
                target_search->search(all_output[0].data() + x * ndim, k, &indices, &distances);
                compare_to_naive(indices, distances, rcurrent[x]);
            }
        }

        auto ref_index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, new_sofar, all_output[0].data()));
        auto ref_search = ref_index->initialize();
        const auto& tneighbors = coords0.get_neighbors_target();
        for (auto r : remaining) {
            const auto& current = data[r];
            const auto& tcurrent = tneighbors[r];
            EXPECT_EQ(tcurrent.size(), sizes[r]);

            std::vector<int> indices;
            std::vector<double> distances;
            for (size_t x = 0; x < sizes[r]; ++x) {
                ref_search->search(current.data() + x * ndim, k, &indices, &distances);
                compare_to_naive(indices, distances, tcurrent[x]);
            }
        }
    }

    // Same results when run in parallel.
    EXPECT_EQ(all_output[0], all_output[1]);
    EXPECT_EQ(all_output[0], all_output[2]);
}

TEST_P(AutomaticOrderTest, DifferentPolicies) {
    // Choosing the smallest batch to amplify the variance,
    // so that it's clear that we're using a different policy.
    size_t chosen = std::min_element(sizes.begin(), sizes.end()) - sizes.begin();
    for (auto& d : data[chosen]) {
        d *= 10;
    }

    for (size_t iter = 0; iter < 4; ++iter) {
        mnncorrect::ReferencePolicy choice = mnncorrect::ReferencePolicy::INPUT;
        if (iter == 1) {
            choice = mnncorrect::ReferencePolicy::MAX_SIZE;
        } else if (iter == 2) {
            choice = mnncorrect::ReferencePolicy::MAX_VARIANCE;
        } else if (iter == 3) {
            choice = mnncorrect::ReferencePolicy::MAX_RSS;
        }

        std::vector<double> output(total_size);
        AutomaticOrder2 coords(
            ndim,
            sizes,
            ptrs,
            output.data(),
            std::make_unique<knncolle::VptreeBuilder<> >(),
            /* num_neighbors = */ k,
            /* ref_policy = */ choice,
            /* nobs_cap = */ -1,
            /* nthreads = */ 1
        );

        if (choice == mnncorrect::ReferencePolicy::INPUT) {
            EXPECT_EQ(coords.get_order()[0], 0);
        } else if (choice == mnncorrect::ReferencePolicy::MAX_SIZE) {
            auto first = coords.get_order()[0];
            for (auto s : sizes) {
                EXPECT_TRUE(sizes[first] >= s);
            }
        } else if (choice == mnncorrect::ReferencePolicy::MAX_VARIANCE) {
            EXPECT_EQ(coords.get_order()[0], chosen);
        } else if (choice == mnncorrect::ReferencePolicy::MAX_RSS) {
            EXPECT_EQ(coords.get_order()[0], chosen);
        }

        // Same results with parallelization.
        std::vector<double> par_output3(output.size());
        AutomaticOrder2 par_coords3(
            ndim,
            sizes,
            ptrs,
            par_output3.data(),
            std::make_unique<knncolle::VptreeBuilder<> >(),
            /* num_neighbors = */ k,
            /* ref_policy = */ choice,
            /* nobs_cap = */ -1,
            /* nthreads = */ 3 
        );
        EXPECT_EQ(coords.get_order(), par_coords3.get_order());

        std::vector<char> used(sizes.size());
        used[coords.get_order()[0]] = true;

        // Just checking that everything runs to completion under the non-default policies.
        if (choice != AutomaticOrder2::default_policy) {
            for (size_t b = 1; b < sizes.size(); ++b) {
                auto chosen = coords.test_choose();
                EXPECT_FALSE(used[chosen.first]);
                used[chosen.first] = true;

                size_t cursize = sizes[chosen.first];
                auto corrected = scran_tests::simulate_vector(ndim * cursize, [&]{
                    scran_tests::SimulationParameters sparams;
                    sparams.seed = ndim * 1000 + k + b + 69;
                    return sparams;
                }());
                size_t offset = ndim * coords.get_ncorrected();
                std::copy(corrected.begin(), corrected.end(), output.data() + offset);
                coords.test_update(chosen.first);

                auto chosen3 = par_coords3.test_choose();
                std::copy(corrected.begin(), corrected.end(), par_output3.data() + offset);
                EXPECT_EQ(chosen.first, chosen3.first);
                par_coords3.test_update(chosen3.first);
            }

            EXPECT_EQ(coords.get_order(), par_coords3.get_order());
            EXPECT_EQ(output, par_output3);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AutomaticOrder,
    AutomaticOrderTest,
    ::testing::Combine(
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
