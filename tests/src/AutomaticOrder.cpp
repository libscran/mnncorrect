#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "scran_tests/scran_tests.hpp"

#include "mnncorrect/AutomaticOrder.hpp"
#include <random>
#include <algorithm>
#include <cstddef>

TEST(AutomaticOrder, RedistributeCorrectedObservations) {
    constexpr std::size_t num_dim = 5;
    int num_total = 50;

    mnncorrect::internal::FindBatchNeighborsResults<int, double> batch_nns;
    batch_nns.target_ids = std::vector<int>{ 1, 3, 7, 11, 23, 31 };

    mnncorrect::internal::CorrectTargetResults correct_info;
    correct_info.batch.resize(num_total, 123456789);
    correct_info.batch[1] = 2;
    correct_info.batch[3] = 1;
    correct_info.batch[7] = 0;
    correct_info.batch[11] = 1;
    correct_info.batch[23] = 2;
    correct_info.batch[31] = 1;

    auto data = scran_tests::simulate_vector(num_total * num_dim, [&]{
        scran_tests::SimulationParameters params;
        params.seed = 69;
        return params;
    }());

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        std::vector<mnncorrect::internal::BatchInfo<int, double> > batches(3);
        mnncorrect::internal::redistribute_corrected_observations(
            num_dim,
            batch_nns,
            correct_info,
            data.data(),
            builder,
            num_threads,
            batches
        );

        for (int b = 0; b < 3; ++b) {
            const auto& curbatch = batches[b];
            EXPECT_EQ(curbatch.extras.size(), 1);

            std::vector<int> ids;
            if (b == 0) {
                ids = std::vector<int>{ 7 };
            } else if (b == 1) {
                ids = std::vector<int>{ 3, 11, 31 };
            } else {
                ids = std::vector<int>{ 1, 23 };
            }

            EXPECT_EQ(curbatch.extras[0].ids, ids);
            auto searcher = curbatch.extras[0].index->initialize();

            std::vector<int> idx;
            std::vector<double> dist;
            for (decltype(ids.size()) i = 0, end = ids.size(); i < end; ++i) {
                auto id = ids[i];
                searcher->search(data.data() + id * num_dim, 1, &idx, &dist);
                EXPECT_EQ(idx[0], i);
                EXPECT_EQ(dist[0], 0);
            }
        }
    }
}

class AutomaticOrder2 : public mnncorrect::internal::AutomaticOrder<int, double, knncolle::Matrix<int, double> > {
public:
    static constexpr mnncorrect::ReferencePolicy default_policy = mnncorrect::ReferencePolicy::MAX_SIZE;

    template<typename ... Args_>
    AutomaticOrder2(Args_&&... args) : AutomaticOrder<int, double, knncolle::Matrix<int, double> >(std::forward<Args_>(args)...) {}

    const auto& get_batches() const { 
        return my_batches;
    }

    auto advance() {
        return next();
    }
};

class AutomaticOrderTest : public ::testing::Test {};

TEST_F(AutomaticOrderTest, Empty) {
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
    AutomaticOrder2 overlord(
        5,
        std::vector<int>{}, 
        std::vector<const double*>{},
        static_cast<double*>(NULL),
        builder,
        /* num_neighbors = */ 10,
        /* tolerance = */ 3,
        mnncorrect::ReferencePolicy::MAX_RSS,
        /* num_threads = */ 1
    );
    EXPECT_TRUE(overlord.get_batches().empty());

    std::string msg;
    try {
        AutomaticOrder2(
            5,
            std::vector<int>{ 0 }, 
            std::vector<const double*>{},
            static_cast<double*>(NULL),
            builder,
            /* num_neighbors = */ 10,
            /* tolerance = */ 3,
            mnncorrect::ReferencePolicy::MAX_RSS,
            /* num_threads = */ 1
        );
    } catch (std::exception& e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("length of") != std::string::npos);
}

template<typename Index_, typename Float_>
static void check_initialization(
    std::size_t num_dim,
    const std::vector<mnncorrect::internal::BatchInfo<Index_, Float_> >& batches,
    const std::vector<const double*>& sources,
    const double* corrected)
{
    auto nbatches = batches.size();

    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        const auto& curbatch = batches[b];
        auto num_obs = curbatch.num_obs;
        EXPECT_EQ(curbatch.index->num_observations(), num_obs);

        auto searcher = curbatch.index->initialize();
        std::vector<int> indices;
        std::vector<double> distances;

        for (int c = 0; c < num_obs; ++c) {
            auto srcptr = sources[b] + c * num_dim;
            std::vector<double> srcvec(srcptr, srcptr + num_dim);
            auto corptr = corrected + (curbatch.offset + c) * num_dim;
            std::vector<double> corvec(corptr, corptr + num_dim);
            EXPECT_EQ(corvec, srcvec);

            searcher->search(corptr, 1, &indices, &distances);
            EXPECT_EQ(indices.size(), 1);
            EXPECT_EQ(indices[0], c);
            EXPECT_EQ(distances[0], 0);
        }
    }
}

template<typename Float_>
static std::vector<const double*> pointerize(const std::vector<std::vector<Float_> >& data) {
    std::vector<const double*> output;
    output.reserve(data.size());
    for (const auto& batch : data) {
        output.push_back(batch.data());
    }
    return output;
}

TEST_F(AutomaticOrderTest, Input) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 42 + ndim * 10 + sizes[b];
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Testing multiple threads to check for correct parallelization of index construction.
    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        std::vector<double> corrected(ndim * ntotal);
        AutomaticOrder2 overlord(
            ndim,
            sizes,
            ptrs,
            corrected.data(),
            builder,
            /* num_neighbors = */ 10,
            /* tolerance = */ 3,
            mnncorrect::ReferencePolicy::INPUT,
            num_threads
        );

        // Batches should be sorted by input order.
        const auto& batches = overlord.get_batches(); 
        EXPECT_EQ(batches[0].num_obs, 100);
        EXPECT_EQ(batches[0].offset, 0);
        EXPECT_EQ(batches[1].num_obs, 200);
        EXPECT_EQ(batches[1].offset, 100);
        EXPECT_EQ(batches[2].num_obs, 150);
        EXPECT_EQ(batches[2].offset, 300);

        check_initialization(ndim, batches, ptrs, corrected.data());
    }
}

TEST_F(AutomaticOrderTest, MaxSize) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 69 + ndim * 10 + sizes[b];
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> corrected(ndim * ntotal);
    AutomaticOrder2 overlord(
        ndim,
        sizes,
        ptrs,
        corrected.data(),
        builder,
        /* num_neighbors = */ 10,
        /* tolerance = */ 3,
        mnncorrect::ReferencePolicy::MAX_SIZE,
        /* num_threads = */ 1
    );

    // Batches should be sorted by size.
    const auto& batches = overlord.get_batches(); 
    EXPECT_EQ(batches[0].num_obs, 200);
    EXPECT_EQ(batches[0].offset, 100);
    EXPECT_EQ(batches[1].num_obs, 150);
    EXPECT_EQ(batches[1].offset, 300);
    EXPECT_EQ(batches[2].num_obs, 100);
    EXPECT_EQ(batches[2].offset, 0);

    std::vector<const double*> resorted_ptrs { ptrs[1], ptrs[2], ptrs[0] };
    check_initialization(ndim, batches, resorted_ptrs, corrected.data());
}

TEST_F(AutomaticOrderTest, MaxVariance) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 100, 200, 150 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 4269 + ndim * 10 + sizes[b];
            sparams.lower = -3.0 * (b + 1); // later batches are more variable.
            sparams.upper = 3.0 * (b + 1);
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    // Testing multiple threads to check for threading in the variance calculation.
    for (int num_threads = 1; num_threads <= 3; num_threads += 2) {
        std::vector<double> corrected(ndim * ntotal);
        AutomaticOrder2 overlord(
            ndim,
            sizes,
            ptrs,
            corrected.data(),
            builder,
            /* num_neighbors = */ 10,
            /* tolerance = */ 3,
            mnncorrect::ReferencePolicy::MAX_VARIANCE,
            num_threads
        );

        // Batches should be sorted by variance.
        const auto& batches = overlord.get_batches(); 
        EXPECT_EQ(batches[0].num_obs, 150);
        EXPECT_EQ(batches[0].offset, 300);
        EXPECT_EQ(batches[1].num_obs, 200);
        EXPECT_EQ(batches[1].offset, 100);
        EXPECT_EQ(batches[2].num_obs, 100);
        EXPECT_EQ(batches[2].offset, 0);

        std::vector<const double*> resorted_ptrs { ptrs[2], ptrs[1], ptrs[0] };
        check_initialization(ndim, batches, resorted_ptrs, corrected.data());
    }
}

TEST_F(AutomaticOrderTest, MaxRss) {
    constexpr std::size_t ndim = 10;
    std::vector<int> sizes { 50, 500 };
    int ntotal = std::accumulate(sizes.begin(), sizes.end(), 0);

    auto nbatches = sizes.size();
    std::vector<std::vector<double> > data(nbatches);
    for (decltype(nbatches) b = 0; b < nbatches; ++b) {
        data[b] = scran_tests::simulate_vector(sizes[b] * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = 4269 + ndim * 10 + sizes[b];
            sparams.lower = -3.0 / (b + 1); // later batches are less variable.
            sparams.upper = 3.0 / (b + 1);
            return sparams;
        }());
    }

    auto ptrs = pointerize(data);
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> corrected(ndim * ntotal);
    AutomaticOrder2 overlord(
        ndim,
        sizes,
        ptrs,
        corrected.data(),
        builder,
        /* num_neighbors = */ 10,
        /* tolerance = */ 3,
        mnncorrect::ReferencePolicy::MAX_RSS,
        /* num_threads = */ 1
    );

    // Sorting by RSS; so even though the first batch is most variable,
    // it has fewer observations and so the second batch has a higher RSS.
    const auto& batches = overlord.get_batches(); 
    EXPECT_EQ(batches[0].num_obs, 500);
    EXPECT_EQ(batches[0].offset, 50);
    EXPECT_EQ(batches[1].num_obs, 50);
    EXPECT_EQ(batches[1].offset, 0);

    std::vector<const double*> resorted_ptrs { ptrs[1], ptrs[0] };
    check_initialization(ndim, batches, resorted_ptrs, corrected.data());
}

//TEST_P(AutomaticOrderTest, CheckUpdate) {
//    std::vector<AutomaticOrder2> all_coords;
//    all_coords.reserve(3);
//    std::vector<std::vector<double> > all_output(3);
//
//    for (int t = 0; t < 3; ++t) {
//        all_output[t].resize(total_size);
//        all_coords.emplace_back(
//            ndim,
//            sizes,
//            ptrs,
//            all_output[t].data(),
//            builder,
//            /* num_neighbors = */ k,
//            /* ref_policy = */ AutomaticOrder2::default_policy,
//            /* nobs_cap = */ -1,
//            /* nthreads = */ t + 1
//        );
//    }
//
//    std::vector<char> used(sizes.size());
//    used[all_coords.front().get_order()[0]] = true;
//
//    for (std::size_t b = 1; b < sizes.size(); ++b) {
//        auto& coords0 = all_coords[0];
//        int sofar = coords0.get_ncorrected();
//
//        // The parallelized chooser with neighbor re-use is very complicated,
//        // so we test it against the naive serial chooser, just in case.
//        auto simpler = [&]{ 
//            auto corrected = all_output[0].data();
//            auto ref_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, sofar, corrected));
//
//            mnncorrect::internal::MnnPairs<int> output;
//            std::size_t chosen = 0;
//            for (auto r : coords0.get_remaining()) {
//                auto target_to_ref = mnncorrect::internal::quick_find_nns(sizes[r], data[r].data(), *ref_index, /* k = */ k, /* num_threads = */ 1);
//                auto target_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, sizes[r], data[r].data()));
//                auto ref_to_target = mnncorrect::internal::quick_find_nns(sofar, corrected, *target_index, /* k = */ k, /* num_threads = */ 1);
//
//                auto tmp = mnncorrect::internal::find_mutual_nns(ref_to_target, target_to_ref);
//                if (tmp.num_pairs > output.num_pairs) {
//                    output = std::move(tmp);
//                    chosen = r;
//                }
//            }
//
//            return std::make_pair(chosen, std::move(output));
//        }();
//
//        EXPECT_FALSE(used[simpler.first]);
//        used[simpler.first] = true;
//
//        // Double-check that the MNN pair indices are sensible.
//        const auto& m = simpler.second.matches;
//        EXPECT_TRUE(m.size() > 0);
//        for (const auto& x : m) {
//            EXPECT_LT(x.first, sizes[simpler.first]);
//            for (const auto& y : x.second) {
//                EXPECT_LT(y, sofar);
//            }
//        }
//
//        for (std::size_t i = 0; i < all_coords.size(); ++i) {
//            auto chosen = all_coords[i].test_choose();
//            EXPECT_EQ(chosen.first, simpler.first);
//            EXPECT_EQ(chosen.second.num_pairs, simpler.second.num_pairs);
//            EXPECT_EQ(chosen.second.matches, simpler.second.matches);
//        }
//
//        // Applying an update. We mock up some corrected data so that the builders work correctly.
//        std::size_t cursize = sizes[simpler.first];
//        auto corrected = scran_tests::simulate_vector(ndim * cursize, [&]{
//            scran_tests::SimulationParameters sparams;
//            sparams.seed = ndim * 1000 + k + b + 69;
//            return sparams;
//        }());
//
//        std::size_t output_offset = ndim * sofar;
//        for (std::size_t i = 0; i < all_coords.size(); ++i) {
//            std::copy(corrected.begin(), corrected.end(), all_output[i].data() + output_offset);
//            all_coords[i].test_update(simpler.first);
//        }
//
//        // Check that the update works as expected.
//        const auto& remaining = coords0.get_remaining();
//        EXPECT_EQ(remaining.size(), sizes.size() - b - 1);
//        std::size_t new_sofar = coords0.get_ncorrected();
//        EXPECT_EQ(sofar + sizes[simpler.first], new_sofar);
//
//        const auto& ord = coords0.get_order();
//        EXPECT_EQ(ord.size(), b + 1);
//        EXPECT_EQ(ord.back(), simpler.first);
//
//        const auto& rneighbors = coords0.get_neighbors_ref();
//        for (auto r : remaining) {
//            const auto& rcurrent = rneighbors[r];
//            auto target_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, sizes[r], data[r].data()));
//            auto target_search = target_index->initialize();
//            EXPECT_EQ(rcurrent.size(), new_sofar);
//
//            std::vector<int> indices;
//            std::vector<double> distances;
//            for (std::size_t x = sofar; x < new_sofar; ++x) {
//                target_search->search(all_output[0].data() + x * ndim, k, &indices, &distances);
//                compare_to_naive(indices, distances, rcurrent[x]);
//            }
//        }
//
//        auto ref_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, new_sofar, all_output[0].data()));
//        auto ref_search = ref_index->initialize();
//        const auto& tneighbors = coords0.get_neighbors_target();
//        for (auto r : remaining) {
//            const auto& current = data[r];
//            const auto& tcurrent = tneighbors[r];
//            EXPECT_EQ(tcurrent.size(), sizes[r]);
//
//            std::vector<int> indices;
//            std::vector<double> distances;
//            for (int x = 0; x < sizes[r]; ++x) {
//                ref_search->search(current.data() + x * ndim, k, &indices, &distances);
//                compare_to_naive(indices, distances, tcurrent[x]);
//            }
//        }
//    }
//
//    // Same results when run in parallel.
//    EXPECT_EQ(all_output[0], all_output[1]);
//    EXPECT_EQ(all_output[0], all_output[2]);
//}
//
//TEST_P(AutomaticOrderTest, DifferentPolicies) {
//    // Choosing the smallest batch to amplify the variance,
//    // so that it's clear that we're using a different policy.
//    std::size_t chosen = std::min_element(sizes.begin(), sizes.end()) - sizes.begin();
//    for (auto& d : data[chosen]) {
//        d *= 10;
//    }
//
//    for (std::size_t iter = 0; iter < 4; ++iter) {
//        mnncorrect::ReferencePolicy choice = mnncorrect::ReferencePolicy::INPUT;
//        if (iter == 1) {
//            choice = mnncorrect::ReferencePolicy::MAX_SIZE;
//        } else if (iter == 2) {
//            choice = mnncorrect::ReferencePolicy::MAX_VARIANCE;
//        } else if (iter == 3) {
//            choice = mnncorrect::ReferencePolicy::MAX_RSS;
//        }
//
//        std::vector<double> output(total_size);
//        AutomaticOrder2 coords(
//            ndim,
//            sizes,
//            ptrs,
//            output.data(),
//            builder,
//            /* num_neighbors = */ k,
//            /* ref_policy = */ choice,
//            /* nobs_cap = */ -1,
//            /* nthreads = */ 1
//        );
//
//        if (choice == mnncorrect::ReferencePolicy::INPUT) {
//            EXPECT_EQ(coords.get_order()[0], 0);
//        } else if (choice == mnncorrect::ReferencePolicy::MAX_SIZE) {
//            auto first = coords.get_order()[0];
//            for (auto s : sizes) {
//                EXPECT_TRUE(sizes[first] >= s);
//            }
//        } else if (choice == mnncorrect::ReferencePolicy::MAX_VARIANCE) {
//            EXPECT_EQ(coords.get_order()[0], chosen);
//        } else if (choice == mnncorrect::ReferencePolicy::MAX_RSS) {
//            EXPECT_EQ(coords.get_order()[0], chosen);
//        }
//
//        // Same results with parallelization.
//        std::vector<double> par_output3(output.size());
//        AutomaticOrder2 par_coords3(
//            ndim,
//            sizes,
//            ptrs,
//            par_output3.data(),
//            builder,
//            /* num_neighbors = */ k,
//            /* ref_policy = */ choice,
//            /* nobs_cap = */ -1,
//            /* nthreads = */ 3 
//        );
//        EXPECT_EQ(coords.get_order(), par_coords3.get_order());
//
//        std::vector<char> used(sizes.size());
//        used[coords.get_order()[0]] = true;
//
//        // Just checking that everything runs to completion under the non-default policies.
//        if (choice != AutomaticOrder2::default_policy) {
//            for (std::size_t b = 1; b < sizes.size(); ++b) {
//                auto chosen = coords.test_choose();
//                EXPECT_FALSE(used[chosen.first]);
//                used[chosen.first] = true;
//
//                std::size_t cursize = sizes[chosen.first];
//                auto corrected = scran_tests::simulate_vector(ndim * cursize, [&]{
//                    scran_tests::SimulationParameters sparams;
//                    sparams.seed = ndim * 1000 + k + b + 69;
//                    return sparams;
//                }());
//                std::size_t offset = ndim * coords.get_ncorrected();
//                std::copy(corrected.begin(), corrected.end(), output.data() + offset);
//                coords.test_update(chosen.first);
//
//                auto chosen3 = par_coords3.test_choose();
//                std::copy(corrected.begin(), corrected.end(), par_output3.data() + offset);
//                EXPECT_EQ(chosen.first, chosen3.first);
//                par_coords3.test_update(chosen3.first);
//            }
//
//            EXPECT_EQ(coords.get_order(), par_coords3.get_order());
//            EXPECT_EQ(output, par_output3);
//        }
//    }
//}
//
//INSTANTIATE_TEST_SUITE_P(
//    AutomaticOrder,
//    AutomaticOrderTest,
//    ::testing::Combine(
//        ::testing::Values(1, 5, 10), // Number of neighbors
//        ::testing::Values(
//            std::vector<int>{10, 20},        
//            std::vector<int>{10, 20, 30}, 
//            std::vector<int>{100, 50, 80}, 
//            std::vector<int>{50, 30, 100, 90},
//            std::vector<int>{50, 40, 30, 20, 10}
//        )
//    )
//);
