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

    // Does the right thing with empty boundaries.
    {
        std::vector<std::pair<double, bool> > boundaries;
        EXPECT_EQ(mnncorrect::find_coverage_max(boundaries, 10000.0), 10000.0); 
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

        knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
        neighbors_of_right.resize(nright);
        for (size_t r = 0; r < nright; ++r) {
            auto current = right.data() + ndim * r;
            neighbors_of_right[r] = left_index.find_nearest_neighbors(current, 1); 
        }

        knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
        neighbors_of_left.resize(nleft);
        for (size_t l = 0; l < nleft; ++l) {
            auto current = left.data() + ndim * l;
            neighbors_of_left[l] = right_index.find_nearest_neighbors(current, k);
        }

        pairings = mnncorrect::find_mutual_nns<int>(neighbors_of_left, neighbors_of_right);
    }

    int ndim = 5, k;
    size_t nleft, nright;
    std::vector<double> left, right;

    mnncorrect::NeighborSet<int, double> neighbors_of_left;
    mnncorrect::NeighborSet<int, double> neighbors_of_right;
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

TEST_P(CorrectTargetTest, ScaleVectors) {
    assemble(GetParam());

    std::vector<double> output(ndim * nleft);
    auto counts = mnncorrect::compute_batch_vectors(ndim, nleft, left.data(), pairings, right.data(), output.data());
    auto by_left = mnncorrect::observations_by_ref(nleft, neighbors_of_right);

    // Trying with a large radius to test out the coverage-finder.
    {
        auto original = output;
        auto copy = output;
        std::vector<double> radius(nleft, 10);
        mnncorrect::scale_batch_vectors(ndim, nleft, left.data(), radius.data(), by_left, right.data(), copy.data());

        // Checking that the L2 norms are not smaller after scaling, and that some of them have changed.
        int changed = 0;
        for (size_t l = 0; l < nleft; ++l) {
            auto l2old = mnncorrect::l2norm(ndim, original.data() + l * ndim);
            auto l2new = mnncorrect::l2norm(ndim, copy.data() + l * ndim);
            EXPECT_TRUE(l2old <= l2new);
            changed += l2old < l2new;
        }
        EXPECT_TRUE(changed > 0);
    }

    // Trying with a smaller radius.
    {
        auto original = output;
        auto copy = output;
        std::vector<double> radius(nleft, 1);
        mnncorrect::scale_batch_vectors(ndim, nleft, left.data(), radius.data(), by_left, right.data(), copy.data());

        // Checking that the L2 norms are not smaller after scaling.
        int changed = 0;
        for (size_t l = 0; l < nleft; ++l) {
            auto l2old = mnncorrect::l2norm(ndim, original.data() + l * ndim);
            auto l2new = mnncorrect::l2norm(ndim, copy.data() + l * ndim);
            EXPECT_TRUE(l2old <= l2new);
        }
    }

    // Doesn't freak out with all-zero inputs.
    {
        std::vector<double> original(ndim * nleft);
        auto copy = original;
        std::vector<double> radius(nleft, 0.0001);
        mnncorrect::scale_batch_vectors(ndim, nleft, left.data(), radius.data(), by_left, right.data(), copy.data());
        EXPECT_EQ(original, copy);

        // Also testing correct fallback with empty indices.
        by_left[0].clear();
        mnncorrect::scale_batch_vectors(ndim, nleft, left.data(), radius.data(), by_left, right.data(), copy.data());
        EXPECT_EQ(original, copy);
    }
}

TEST_P(CorrectTargetTest, ExtrapolateVectors) {
    assemble(GetParam());
    
    // Mocking up some vectors.
    std::vector<double> output(ndim * nleft);
    std::mt19937_64 rng(ndim * nright);
    for (auto& o : output) {
        o = aarand::standard_normal(rng).first; 
    }

    // Make the first reference missing, and ensuring that it is closest to the second reference.
    {
        auto copy = output;
        std::vector<char> is_ok(nleft, true);
        is_ok[0] = false;
        std::copy(left.begin() + ndim, left.begin() + 2 * ndim, left.begin()); 

        mnncorrect::extrapolate_vectors(ndim, nleft, left.data(), is_ok, output.data());

        // First vector is replaced by the second vector.
        {
            std::vector<double> first(output.begin(), output.begin() + ndim);
            std::vector<double> old(copy.begin(), copy.begin() + ndim);
            EXPECT_NE(first, old);
            std::vector<double> second(output.begin() + ndim, output.begin() + 2 * ndim);
            EXPECT_EQ(first, second);
        }

        // Remaining vectors are the same as the inputs.
        {
            std::vector<double> first(copy.begin() + ndim, copy.end());
            std::vector<double> second(output.begin() + ndim, output.end());
            EXPECT_EQ(first, second);
        }
    }

    // Make the last reference missing, and ensuring that it is closest to the third reference.
    {
        auto copy = output;
        std::vector<char> is_ok(nleft, true);
        is_ok[nleft - 1] = false;
        std::copy(left.begin() + 2 * ndim, left.begin() + 3 * ndim, left.end() - ndim);

        mnncorrect::extrapolate_vectors(ndim, nleft, left.data(), is_ok, output.data());

        // Last vector is replaced by the third vector.
        {
            std::vector<double> first(output.end() - ndim, output.end());
            std::vector<double> old(copy.end() - ndim, copy.end());
            EXPECT_NE(first, old);
            std::vector<double> second(output.begin() + 2 * ndim, output.begin() + 3 * ndim);
            EXPECT_EQ(first, second);
        }

        // Remaining vectors are the same as the inputs.
        {
            std::vector<double> first(copy.begin(), copy.end() - ndim);
            std::vector<double> second(output.begin(), output.end() - ndim);
            EXPECT_EQ(first, second);
        }
    }

    // Throws the exception if nothing is available.
    {
        std::vector<char> is_ok(nleft);
        EXPECT_ANY_THROW({
            try {
                mnncorrect::extrapolate_vectors(ndim, nleft, left.data(), is_ok, output.data());
            } catch (std::exception& e){
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("sufficient MNN") != std::string::npos);
                throw;
            }
        });
    }
}

TEST_P(CorrectTargetTest, AverageBatchVectors) {
    assemble(GetParam());

    std::vector<double> output(ndim * nleft);
    auto counts = mnncorrect::compute_batch_vectors(ndim, nleft, left.data(), pairings, right.data(), output.data());
    auto averaged = mnncorrect::average_batch_vectors(ndim, nleft, counts, output.data());

    std::vector<double> expected(ndim);
    for (size_t p = 0; p < pairings.size(); ++p) {
        auto lptr = left.data() + pairings.left[p] * ndim;
        auto rptr = right.data() + pairings.right[p] * ndim;
        for (int d = 0; d < ndim; ++d) {
            expected[d] += (rptr[d] - lptr[d]);
        }
    }

    for (int d = 0; d < ndim; ++d) {
        expected[d] /= pairings.size();
        EXPECT_FLOAT_EQ(averaged[d], expected[d]);
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
        ::testing::Values(5, 20), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);
