#include <gtest/gtest.h>

#include "mnncorrect/IterativeMerger.hpp"
#include "knncolle/knncolle.hpp"

#include <random>
#include <algorithm>

TEST(IterativeMerger, MedianDistanceFromCenter) {
    // Setting up a scenario.
    int ndim = 3;
    double sqrt_ndim = std::sqrt(static_cast<double>(ndim));

    size_t nobs = 5;
    std::vector<double> data {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    };

    int ncenters = 2;
    std::vector<double> centers {
        2, 3, 4,
        5, 6, 7
    };
    std::vector<int> clusters { 0, 1, 0, 1, 0 };

    // Making sure we get the expected values.
    std::vector<double> meds(ncenters);
    mnncorrect::median_distance_from_center(ndim, nobs, data.data(), ncenters, centers.data(), clusters.data(), meds.data());
    EXPECT_FLOAT_EQ(meds[0], 5 * sqrt_ndim); // points are (1, 7, 13) to (2), so deltas are (1, 5, 11) and median delta is 5.
    EXPECT_FLOAT_EQ(meds[1], 3 * sqrt_ndim); // points are (4, 10) to (5), so deltas are (1, 5) and median delta is 3.

    // Zero distance if no points are assigned.
    std::fill(clusters.begin(), clusters.end(), 1);
    mnncorrect::median_distance_from_center(ndim, nobs, data.data(), ncenters, centers.data(), clusters.data(), meds.data());
    EXPECT_FLOAT_EQ(meds[0], 0);
    EXPECT_FLOAT_EQ(meds[1], 4 * sqrt_ndim);
}

TEST(IterativeMerger, AssignToCluster) {
    int ndim = 3;
    size_t nobs = 5;
    std::vector<double> data {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    };

    int ncenters = 2;
    std::vector<double> centers {
        2, 3, 4,
        6, 7, 8
    };

    knncolle::BruteForceEuclidean<> searcher(ndim, ncenters, centers.data());
    std::vector<int> clusters(nobs);
    mnncorrect::assign_to_cluster(ndim, nobs, data.data(), &searcher, clusters.data());
    
    EXPECT_EQ(clusters[0], 0);
    EXPECT_EQ(clusters[1], 0);
    EXPECT_EQ(clusters[2], 1);
    EXPECT_EQ(clusters[3], 1);
    EXPECT_EQ(clusters[4], 1);
}
