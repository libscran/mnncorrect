#include <gtest/gtest.h>
#include "mnncorrect/utils.hpp"
#include <cmath>

TEST(Utils, Unique) {
    {
        auto u = mnncorrect::unique(std::deque<int>{ 1, 1, 2, 3, 4, 4 });
        std::vector<int> ref { 1, 2, 3, 4};
        EXPECT_EQ(u, ref);
    }

    // Trying out of order.
    {
        auto u = mnncorrect::unique(std::deque<int>{ 2, 4, 0, 2, 0, 3, 4});
        std::vector<int> ref { 0, 2, 3, 4};
        EXPECT_EQ(u, ref);
    }
}

TEST(Utils, InvertIndex) {
    {
        std::vector<int> u { 0, 2, 3, 4};
        auto inv = mnncorrect::invert_index(5, u, -1);
        std::vector<int> expected{ 0, -1, 1, 2, 3 };
        EXPECT_EQ(inv, expected);
    }

    // Works out of order.
    {
        std::vector<int> u { 5, 2, 8 };
        auto inv = mnncorrect::invert_index(10, u, -1);
        std::vector<int> expected{ -1, -1, 1, -1, -1, 0, -1, -1, 2, -1 };
        EXPECT_EQ(inv, expected);
    }
}

TEST(Utils, Median) {
    std::vector<double> small { 0.5, 0.1, 0.2, 0.7 };
    {
        auto x = small;
        EXPECT_EQ(mnncorrect::median(x.size(), x.data()), 0.35);
    }
    {
        auto x = small;
        EXPECT_EQ(mnncorrect::median(x.size() - 1, x.data()), 0.2);
    }

    std::vector<double> larger { 0.23, 0.55, 0.62, 0.87, 0.91, 0.17, 0.42, 0.80, 0.14 };
    {
        auto x = larger;
        EXPECT_EQ(mnncorrect::median(x.size(), x.data()), 0.55);
    }
    {
        auto x = larger;
        EXPECT_EQ(mnncorrect::median(x.size() - 1, x.data()), 0.585);
    }

    EXPECT_TRUE(std::isnan(mnncorrect::median(0, static_cast<double*>(NULL))));
}
