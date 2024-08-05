#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/utils.hpp"
#include <cmath>
#include <vector>

TEST(Utils, InvertNeighbors) {
    mnncorrect::internal::NeighborSet<int, double> nns(4);
    nns[0].emplace_back(4, 0.12); // distances are more-or-less random. 
    nns[0].emplace_back(2, 1);
    nns[0].emplace_back(1, 0.32);

    nns[1].emplace_back(3, 0.5);
    nns[1].emplace_back(4, 1);
    nns[1].emplace_back(2, 0.3);

    nns[2].emplace_back(0, 0.1);
    nns[3].emplace_back(4, 1);

    auto inv = mnncorrect::internal::invert_neighbors(5, nns, 1000.0);
    EXPECT_EQ(inv.size(), 5);
    {
        std::vector<int> exp0 { 2 };
        EXPECT_EQ(inv[0], exp0);
        std::vector<int> exp1 { 0 };
        EXPECT_EQ(inv[1], exp1);
        std::vector<int> exp2 { 0, 1 };
        EXPECT_EQ(inv[2], exp2);
        std::vector<int> exp3 { 1 };
        EXPECT_EQ(inv[3], exp3);
        std::vector<int> exp4 { 0, 1, 3 };
        EXPECT_EQ(inv[4], exp4);
    }

    auto inv2 = mnncorrect::internal::invert_neighbors(5, nns, 0.5);
    EXPECT_EQ(inv2.size(), 5);
    {
        std::vector<int> exp0 { 2 };
        EXPECT_EQ(inv2[0], exp0);
        std::vector<int> exp1 { 0 };
        EXPECT_EQ(inv2[1], exp1);
        std::vector<int> exp2 { 1 };
        EXPECT_EQ(inv2[2], exp2);
        std::vector<int> exp3 { 1 };
        EXPECT_EQ(inv2[3], exp3);
        std::vector<int> exp4 { 0 };
        EXPECT_EQ(inv2[4], exp4);
    }
}

TEST(Utils, InvertIndices) {
    {
        std::vector<int> u { 0, 2, 3, 4};
        auto inv = mnncorrect::internal::invert_indices(5, u);
        std::vector<int> expected{ 0, -1, 1, 2, 3 };
        EXPECT_EQ(inv, expected);
    }

    // Works out of order.
    {
        std::vector<int> u { 5, 2, 8 };
        auto inv = mnncorrect::internal::invert_indices(10, u);
        std::vector<int> expected{ -1, -1, 1, -1, -1, 0, -1, -1, 2, -1 };
        EXPECT_EQ(inv, expected);
    }
}

TEST(Utils, Median) {
    std::vector<double> small { 0.5, 0.1, 0.2, 0.7 };
    {
        auto x = small;
        EXPECT_EQ(mnncorrect::internal::median(x.size(), x.data()), 0.35);
    }
    {
        auto x = small;
        EXPECT_EQ(mnncorrect::internal::median(x.size() - 1, x.data()), 0.2);
    }

    std::vector<double> larger { 0.23, 0.55, 0.62, 0.87, 0.91, 0.17, 0.42, 0.80, 0.14 };
    {
        auto x = larger;
        EXPECT_EQ(mnncorrect::internal::median(x.size(), x.data()), 0.55);
    }
    {
        auto x = larger;
        EXPECT_EQ(mnncorrect::internal::median(x.size() - 1, x.data()), 0.585);
    }

    EXPECT_TRUE(std::isnan(mnncorrect::internal::median(0, static_cast<double*>(NULL))));
}
