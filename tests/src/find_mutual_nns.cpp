#include <gtest/gtest.h>

#include "mnncorrect/find_mutual_nns.hpp"
#include "knncolle/knncolle.hpp"

#include <random>
#include <vector>

class FindMutualNNsTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    template<class Param>
    void assemble(Param param) {
        nref = std::get<0>(param);
        ntarget = std::get<1>(param);
        k = std::get<2>(param);

        // Simulating values.
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist;

        ref.resize(nref * ndim);
        for (auto& l : ref) {
            l = dist(rng);
        }

        target.resize(ntarget * ndim);
        for (auto& r : target) {
            r = dist(rng);
        }
    }

    template<class Searcher>
    mnncorrect::MnnPairs<size_t> compute_reference(const Searcher& ref_index, const Searcher& target_index) {
        std::set<std::pair<size_t, size_t> > found;
        for (size_t r = 0; r < ntarget; ++r) {
            auto current = ref_index.find_nearest_neighbors(target.data() + r * ndim, 1);
            for (const auto& x : current) {
                found.insert(std::pair<size_t, size_t>(x.first, r));
            }
        }

        mnncorrect::MnnPairs<size_t> output;
        for (size_t l = 0; l < nref; ++l) {
            auto current = target_index.find_nearest_neighbors(ref.data() + l * ndim, k);
            for (const auto& x : current) {
                auto it = found.find(std::pair<size_t, size_t>(l, x.first));
                if (it != found.end()) {
                    output.ref.push_back(l);
                    output.target.push_back(x.first);
                }
            }
        }

        return output;
    }

    int ndim = 5, k;
    size_t nref, ntarget;
    std::vector<double> ref, target;
};

TEST_P(FindMutualNNsTest, Check) {
    assemble(GetParam());

    knncolle::VpTreeEuclidean<> ref_index(ndim, nref, ref.data());
    knncolle::VpTreeEuclidean<> target_index(ndim, ntarget, target.data());
    auto exp = compute_reference(ref_index, target_index);
    EXPECT_TRUE(exp.ref.size() > 0);
    EXPECT_TRUE(exp.target.size() > 0);

    mnncorrect::NeighborSet<int, double> neighbors_of_ref(nref);
    std::vector<int> neighbors_of_target(ntarget);
    auto obs = mnncorrect::find_mutual_nns<int>(ref.data(), target.data(), &ref_index, &target_index, k, neighbors_of_ref, neighbors_of_target.data());

    EXPECT_EQ(
        std::vector<size_t>(exp.ref.begin(), exp.ref.end()), 
        std::vector<size_t>(obs.ref.begin(), obs.ref.end())
    );
    EXPECT_EQ(
        std::vector<size_t>(exp.target.begin(), exp.target.end()), 
        std::vector<size_t>(obs.target.begin(), obs.target.end())
    );
}

INSTANTIATE_TEST_CASE_P(
    FindMutualNNs,
    FindMutualNNsTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // ref
        ::testing::Values(100, 1000), // target
        ::testing::Values(10, 50)  // k
    )
);
