#ifndef MNNCORRECT_UTILS_HPP
#define MNNCORRECT_UTILS_HPP

#include <deque>
#include <vector>
#include <limits>
#include <type_traits>

namespace mnncorrect {

template<typename Index>
struct MnnPairs {
    size_t size() const { return left.size(); }
    std::deque<Index> left, right;
};

template<typename Index, typename Dist>
using NeighborSet = std::vector<std::vector<std::pair<Index, Dist> > >;

template<class Vector>
auto unique(const Vector& input) {
    typedef typename std::remove_const<typename std::remove_reference<decltype(*input.begin())>::type>::type Value;
    std::set<Value> collected(input.begin(), input.end());
    return std::vector<Value>(collected.begin(), collected.end());
}

template<typename Index>
std::vector<Index> invert_index(size_t n, const std::vector<Index>& uniq, Index placeholder = 0) {
    std::vector<Index> output(n, placeholder);
    for (size_t u = 0; u < uniq.size(); ++u) {
        output[uniq[u]] = u;        
    }
    return output;
}

template<typename Float>
Float median(size_t n, Float* ptr) {
    if (!n) {
        return std::numeric_limits<Float>::quiet_NaN();
    }
    size_t half = n / 2;
    bool is_even = n % 2 == 0;

    std::nth_element(ptr, ptr + half, ptr + n);
    Float mid = *(ptr + half);
    if (!is_even) {
        return mid;
    }

    std::nth_element(ptr, ptr + half - 1, ptr + n);
    return (mid + *(ptr + half - 1)) / 2;
}

constexpr double mad2sigma = 1.4826;

}

#endif
