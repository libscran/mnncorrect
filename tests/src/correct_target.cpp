#include <gtest/gtest.h>
#include "mnncorrect/find_mutual_nns.hpp"
#include "mnncorrect/determine_limits.hpp"
#include "mnncorrect/correct_target.hpp"
#include "aarand/aarand.hpp"
#include "knncolle/knncolle.hpp"
#include <random>

class CorrectTargetTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    template<class Param>
    void assemble(Param param) {
        nleft = std::get<0>(param);
        nright = std::get<1>(param);
        k = std::get<2>(param);

        std::mt19937_64 rng(nleft * nright * k);

        left.resize(nleft * ndim);
        for (auto& l : left) {
            l = aarand::standard_normal(rng).first;
        }

        right.resize(nright * ndim);
        for (auto& r : right) {
            r = aarand::standard_normal(rng).first + 5; // throw in a batch effect.
        }

        // Setting up the values for a reasonable comparison.
        knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
        knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
        pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);
    }

    int ndim = 5, k;
    size_t nleft, nright;
    std::vector<double> left, right;
    mnncorrect::MnnPairs<int> pairings;
};

TEST_P(CorrectTargetTest, CenterOfMass) {
    assemble(GetParam());

    // Setting up the values for a reasonable comparison.
    auto right_mnn = mnncorrect::unique(pairings.right);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = mnncorrect::identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

    double limit = mnncorrect::limit_from_closest_distances(self_mnn);
    mnncorrect::compute_center_of_mass(ndim, right_mnn.size(), self_mnn, right.data(), limit, buffer.data());

    // Reference calculation for each MNN.
    std::vector<std::vector<int> > inverted(right_mnn.size());
    for (size_t s = 0; s < self_mnn.size(); ++s) {
        for (const auto& p : self_mnn[s]) {
            if (p.second <= limit) {
                inverted[p.first].push_back(s);
            }
        }
    }

    for (size_t i = 0; i < inverted.size(); ++i) {
        const auto& inv = inverted[i];
        std::vector<double> ref(ndim);

        for (auto x : inv) {
            const double* current = right.data() + x * ndim;
            for (int d = 0; d < ndim; ++d) {
                ref[d] += current[d];
            }
        }

        const double* obs = buffer.data() + i * ndim;
        for (int d = 0; d < ndim; ++d) {
            EXPECT_FLOAT_EQ(ref[d] / inv.size(), obs[d]);
        }
    }
}

TEST_P(CorrectTargetTest, Correction) {
    assemble(GetParam());
    std::vector<double> buffer(nright * ndim);
    mnncorrect::correct_target(ndim, nleft, left.data(), nright, right.data(), pairings, k, buffer.data());

    // Not entirely sure how to check for correctness here; 
    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
    std::vector<double> left_means(ndim), right_means(ndim);
    for (size_t l = 0; l < nleft; ++l) {
        for (int d = 0; d < ndim; ++d) {
            left_means[d] += left[l * ndim + d];
        }
    }
    for (size_t r = 0; r < nright; ++r) {
        for (int d = 0; d < ndim; ++d) {
            right_means[d] += buffer[r * ndim + d];
        }
    }
    for (int d = 0; d < ndim; ++d) {
        left_means[d] /= nleft;
        right_means[d] /= nright;
        double delta = std::abs(left_means[d] - right_means[d]);
        EXPECT_TRUE(delta < 1);
    }
}

INSTANTIATE_TEST_CASE_P(
    CorrectTarget,
    CorrectTargetTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);
