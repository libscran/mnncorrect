#ifndef BUILDER_H
#define BUILDER_H

#include "knncolle/knncolle.hpp"
#include <memory>

struct Builder {
    std::shared_ptr<knncolle::Base<int, double> > operator()(int ndim, size_t nobs, const double* stuff) const {
        return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::VpTreeEuclidean<int, double>(ndim, nobs, stuff));
    }
};

template<class V>
inline void compare_to_naive(const V& naive, const V& updated) {
    EXPECT_EQ(naive.size(), updated.size());
    for (size_t i = 0; i < std::min(naive.size(), updated.size()); ++i) {
        EXPECT_EQ(naive[i].first, updated[i].first);
        EXPECT_EQ(naive[i].second, updated[i].second);
    }
}

#endif 
