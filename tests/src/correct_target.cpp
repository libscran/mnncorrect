#include <gtest/gtest.h>
#include "mnncorrect/find_mutual_nns.hpp"
#include "mnncorrect/correct_target.hpp"
#include "aarand/aarand.hpp"
#include "knncolle/knncolle.hpp"
#include <random>

TEST(CorrectTarget, IntersectWithSphere) {
    // 2D examples first.
    {
        // This draws a diagonal that cuts across the sphere at the left and top-most points.
        std::vector<double> origin { -2.5, 2 };
        std::vector<double> direction(2, 1/std::sqrt(2));
        std::vector<double> center { -1, 2 };
        auto output = mnncorrect::intersect_with_sphere(2, origin.data(), direction.data(), center.data(), 1.5);
       
        EXPECT_FLOAT_EQ(output.second, output.first);
        EXPECT_FLOAT_EQ(output.first + output.second, 1.5 * std::sqrt(2)); 

        // Changing the direction to be tangential to the sphere.
        direction[0] = 0;
        direction[1] = 1;
        output = mnncorrect::intersect_with_sphere(2, origin.data(), direction.data(), center.data(), 1.5);
        EXPECT_FLOAT_EQ(output.first, 0);
        EXPECT_FLOAT_EQ(output.second, 0);

        // Shifting away from the sphere.
        origin[0] = -3.5;
        output = mnncorrect::intersect_with_sphere(2, origin.data(), direction.data(), center.data(), 1.5);
        EXPECT_FLOAT_EQ(output.first, 0);
        EXPECT_FLOAT_EQ(output.second, -1);
    }

    // Trying a higher-dimensional example now.
    {
        // This draws a diagonal that cuts across the sphere at the left and top-most points.
        std::vector<double> origin { -2.5, 2.3, -1.2, -4 };
        std::vector<double> direction { 1.2, 4.2, 2.3, 1.5 };
        mnncorrect::normalize_vector(direction.size(), direction.data());
        std::vector<double> center { 3.5, 1.2, -4.4, -5.2 };

        auto output = mnncorrect::intersect_with_sphere(origin.size(), origin.data(), direction.data(), center.data(), 10.0);
        EXPECT_TRUE(output.second > 0);

        // Checking that the two points actually do intersect the circle.
        double left = output.first - output.second;
        double right = output.first + output.second;
        double lsum2 = 0, rsum2 = 0;
        for (size_t d = 0; d < origin.size(); ++d) {
            lsum2 += std::pow(origin[d] + left * direction[d] - center[d], 2);
            rsum2 += std::pow(origin[d] + right * direction[d] - center[d], 2);
        }
        EXPECT_FLOAT_EQ(lsum2, 100);
        EXPECT_FLOAT_EQ(rsum2, 100);
    }
}

TEST(CorrectTarget, FindCoverageMax) {
    // Vanilla case.
    {
        std::vector<std::pair<double, bool> > boundaries;
        boundaries.emplace_back(3, true);
        boundaries.emplace_back(4.5, false);
        boundaries.emplace_back(2, true);
        boundaries.emplace_back(5.5, false);
        boundaries.emplace_back(2.5, true);
        boundaries.emplace_back(4, false);

        EXPECT_EQ(mnncorrect::find_coverage_max(boundaries, 10000.0), 4); // no limit
        EXPECT_EQ(mnncorrect::find_coverage_max(boundaries, 3.5), 3.5);
    }

    // Does the right thing with stacked start/end positions.
    {
        std::vector<std::pair<double, bool> > boundaries;
        boundaries.emplace_back(3, true);
        boundaries.emplace_back(4, false);
        boundaries.emplace_back(3, true);
        boundaries.emplace_back(4, false);
        boundaries.emplace_back(4, true);
        boundaries.emplace_back(5, false);
        boundaries.emplace_back(4, true);
        boundaries.emplace_back(5, false);
        EXPECT_EQ(mnncorrect::find_coverage_max(boundaries, 10000.0), 5); // everything has a coverage of 2, so we just max it out.

        boundaries.emplace_back(4.2, true);
        boundaries.emplace_back(4.5, false);
        EXPECT_EQ(mnncorrect::find_coverage_max(boundaries, 10000.0), 4.5); // correctly recognizes the small interval as having max coverage.
    }
}

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
        pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, 1, k);
    }

    int ndim = 5, k;
    size_t nleft, nright;
    std::vector<double> left, right;
    mnncorrect::MnnPairs<int> pairings;
};

TEST_P(CorrectTargetTest, BatchVectors) {
    assemble(GetParam());
    std::vector<double> output(ndim * nleft);
    auto counts = compute_batch_vectors(ndim, nleft, left.data(), pairings, right.data(), output.data());

    // Reference calculation, computing it per left-side point
    // rather than using compute_batch_vector's running calculation.
    std::vector<std::vector<int> > by_left(nleft);
    for (size_t p = 0; p < pairings.size(); ++p) {
        by_left[pairings.left[p]].push_back(pairings.right[p]);
    }

    for (size_t i = 0; i < nleft; ++i) {
        const auto& current = by_left[i];
        EXPECT_EQ(current.size(), counts[i]);

        std::vector<double> expected(ndim);
        for (auto c : current) {
            auto ptr = right.data() + c * ndim;
            for (int d = 0; d < ndim; ++d) {
                expected[d] += ptr[d];
            }
        }

        auto reference = left.data() + i * ndim;
        auto observed = output.data() + i * ndim;
        for (int d = 0; d < ndim; ++d) {
            if (current.size()) {
                expected[d] /= current.size();
                expected[d] -= reference[d];
            }
            EXPECT_FLOAT_EQ(expected[d], observed[d]);
        }
    }
}

//TEST_P(CorrectTargetTest, CenterOfMass) {
//    assemble(GetParam());
//
//    // Setting up the values for a reasonable comparison.
//    auto right_mnn = mnncorrect::unique(pairings.right);
//    std::vector<double> buffer(right_mnn.size() * ndim);
//    auto self_mnn = mnncorrect::identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());
//
//    double limit = mnncorrect::limit_from_closest_distances(self_mnn);
//    mnncorrect::compute_center_of_mass(ndim, right_mnn.size(), self_mnn, right.data(), limit, buffer.data());
//
//    // Reference calculation for each MNN.
//    std::vector<std::vector<int> > inverted(right_mnn.size());
//    for (size_t s = 0; s < self_mnn.size(); ++s) {
//        for (const auto& p : self_mnn[s]) {
//            if (p.second <= limit) {
//                inverted[p.first].push_back(s);
//            }
//        }
//    }
//
//    for (size_t i = 0; i < inverted.size(); ++i) {
//        const auto& inv = inverted[i];
//        std::vector<double> ref(ndim);
//
//        for (auto x : inv) {
//            const double* current = right.data() + x * ndim;
//            for (int d = 0; d < ndim; ++d) {
//                ref[d] += current[d];
//            }
//        }
//
//        const double* obs = buffer.data() + i * ndim;
//        for (int d = 0; d < ndim; ++d) {
//            EXPECT_FLOAT_EQ(ref[d] / inv.size(), obs[d]);
//        }
//    }
//}
//
//TEST_P(CorrectTargetTest, Correction) {
//    assemble(GetParam());
//    std::vector<double> buffer(nright * ndim);
//    mnncorrect::correct_target(ndim, nleft, left.data(), nright, right.data(), pairings, k, 3.0, buffer.data());
//
//    // Not entirely sure how to check for correctness here; 
//    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
//    std::vector<double> left_means(ndim), right_means(ndim);
//    for (size_t l = 0; l < nleft; ++l) {
//        for (int d = 0; d < ndim; ++d) {
//            left_means[d] += left[l * ndim + d];
//        }
//    }
//    for (size_t r = 0; r < nright; ++r) {
//        for (int d = 0; d < ndim; ++d) {
//            right_means[d] += buffer[r * ndim + d];
//        }
//    }
//    for (int d = 0; d < ndim; ++d) {
//        left_means[d] /= nleft;
//        right_means[d] /= nright;
//        double delta = std::abs(left_means[d] - right_means[d]);
//        EXPECT_TRUE(delta < 1);
//    }
//}

INSTANTIATE_TEST_CASE_P(
    CorrectTarget,
    CorrectTargetTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);
