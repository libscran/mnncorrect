#ifndef HELPER_FIND_MUTUAL_NNS_HPP
#define HELPER_FIND_MUTUAL_NNS_HPP

#include "mnncorrect/find_mutual_nns.hpp"

template<typename Index, typename Float>
mnncorrect::MnnPairs<Index> find_mutual_nns(
    const Float* left, 
    const Float* right, 
    const knncolle::Base<Index, Float>* left_index, 
    const knncolle::Base<Index, Float>* right_index, 
    int k_left, 
    int k_right)
{
    size_t nleft = left_index->nobs();
    size_t nright = right_index->nobs();
    auto neighbors_of_left = mnncorrect::find_nns(nleft, left, right_index, k_left);
    auto neighbors_of_right = mnncorrect::find_nns(nright, right, left_index, k_right);
    return mnncorrect::find_mutual_nns<Index, Float>(neighbors_of_left, neighbors_of_right);
}

#endif
