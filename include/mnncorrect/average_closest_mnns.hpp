#ifndef FIND_CLOSEST_MNNS_HPP
#define FIND_CLOSEST_MNNS_HPP

#include <vector>
#include <deque>
#include <algorithm>
#include "knncolle/knncolle.hpp"

namespace mnncorrect {

template<typename T>
class AverageClosestMNNs {
public:
    template<class Function>
    void find(
        size_t ndim, 
        size_t nobs, 
        const T* data, 
        const std::deque<size_t>& involved, 
        Function indexer, 
        int k,
        T* average)
    {
        in_use.resize(nobs);
        std::fill(in_use.begin(), in_use.end(), 0);
        for (auto i : involved) {
            in_use[i] = 1;
        }

        // Building the index for the observations involved in MNN pairs.
        size_t nmnns = std::accumulate(in_use.begin(), in_use.end(), 0);
        {
            used.resize(nmnns);
            auto uIt = used.begin();
            buffer.resize(nmnns * ndim);
            T* ptr = buffer.data();
            for (size_t m = 0; m < nobs; ++m) {
                if (in_use[m]) {
                    auto start = data + m * ndim;
                    std::copy(start, start + ndim, ptr);
                    *uIt = m;
                    ++uIt;
                    ptr += ndim;
                }
            }
        }
        auto index = indexer(ndim, nmnns, buffer.data());

        #pragma omp parallel for
        for (size_t i = 0; i < nobs; ++i) {
            auto found = index.find_nearest_neighbors(data + i * ndim, k);
            auto output = average + i * ndim;

            if (found.size()) {
                auto first = data + used[found[0].first] * ndim;
                std::copy(first, first + ndim, output);

                for (size_t j = 1; j < found.size(); ++j) {
                    auto next = data + used[found[j].first] * ndim;
                    for (int d = 0; d < ndim; ++d) {
                        output[d] += next[d];
                    }
                }

                for (int d = 0; d < ndim; ++d) {
                    output[d] /= found.size();
                }
            }
        }
    }

    // Default method for testing purposes.
    void find(
        size_t ndim, 
        size_t nobs, 
        const T* data, 
        const std::deque<size_t>& involved, 
        int k,
        T* average)
    {
        find(ndim, nobs, data, involved, 
            [](size_t nd, size_t no, const T* in) -> auto { return knncolle::VpTreeEuclidean<int, T>(nd, no, in); },
            k, average);
    }

private:
    std::vector<size_t> used;
    std::vector<char> in_use;
    std::vector<T> buffer;
};

}

#endif
