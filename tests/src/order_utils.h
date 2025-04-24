#ifndef BUILDER_H
#define BUILDER_H

#include "knncolle/knncolle.hpp"
#include <memory>

template<class V>
inline void compare_to_naive(const V& naive, const V& updated) {
    std::size_t num_naive = naive.size();
    ASSERT_EQ(num_naive, updated.size());
    for (std::size_t i = 0; i < num_naive; ++i) {
        EXPECT_EQ(naive[i].first, updated[i].first);
        EXPECT_EQ(naive[i].second, updated[i].second);
    }
}

#endif 
