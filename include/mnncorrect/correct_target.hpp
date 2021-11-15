#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include "utils.hpp"
#include "knncolle/knncolle.hpp"
#include "determine_limits.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

namespace mnncorrect {

template<typename Index, typename Float>
std::vector<Index> compute_batch_vectors(int ndim, size_t nref, const Float* ref, const MnnPairs<Index>& pairings, const Float* target, Float* output) {
    std::fill(output, output + nref * ndim, 0);
    std::vector<Index> counter(nref);

    for (size_t p = 0; p < pairings.size(); ++p) {
        Float* optr = output + pairings.left[p] * ndim;
        const Float* tptr = target + pairings.right[p] * ndim;
        for (int d = 0; d < ndim; ++d) {
            optr[d] += tptr[d];                    
        }
        ++counter[pairings.left[p]];
    }

    for (size_t r = 0; r < nref; ++r) {
        const Float* rptr = ref + r * ndim;
        Float* optr = output + r * ndim;
        Float n = counter[r];
        if (n) {
            for (int d = 0; d < ndim; ++d) {
                optr[d] /= n;
                optr[d] -= rptr[d];
            }
        }
    }

    return counter;
}

template<typename Float>
std::pair<Float, Float> intersect_with_sphere(int ndim, const Float* origin, const Float* direction, const Float* center, Float radius) {
    Float proj = 0, delta = 0;
    for (int d = 0; d < ndim; ++d) {
        const Float diff = center[d] - origin[d];
        delta += diff * diff;
        proj += diff * direction[d];
    }

    delta = proj * proj - (delta - radius * radius);
    if (delta < 0) {
        delta = -1;
    } else {
        delta = std::sqrt(delta);
    }

    return std::make_pair(proj, delta);
}

template<typename Float>
Float find_coverage_max(std::vector<std::pair<Float, bool> >& boundaries, Float limit) {
    if (boundaries.empty()) {
        return limit;
    }

    std::sort(boundaries.begin(), boundaries.end());
    int coverage = 0, max_coverage = 0;
    Float position_max = 0;

    auto bIt = boundaries.begin();
    while (bIt != boundaries.end()) {
        auto self = bIt->first;
        if (bIt->second) {
            ++coverage;
        } else {
            --coverage;
        }
        ++bIt;

        // Resolving all ties.
        while (bIt != boundaries.end() && bIt->first == self) {
            if (bIt->second) {
                ++coverage;
            } else {
                --coverage;
            }
            ++bIt;
        }

        if (coverage >= max_coverage) {
            coverage = max_coverage;
            if (bIt == boundaries.end()) {
                position_max = limit;
            } else {
                position_max = bIt->first; // yes, this is the _next_ boundary. We take the upper end of the interval.
            }
        }

        if (bIt != boundaries.end() && bIt->first > limit) {
            position_max = limit;
            break;
        }
    }

    return position_max;
}

template<typename Index, typename Float>
std::vector<std::vector<Index> > observations_by_ref(size_t nref, const NeighborSet<Index, Float>& closest_ref) {
    std::vector<std::vector<Index> > by_neighbor(nref);
    for (size_t o = 0; o < closest_ref.size(); ++o) {
        auto r = closest_ref[o].front().first;
        by_neighbor[r].push_back(o);
    }
    return by_neighbor;
}

template<typename Index, typename Float>
void scale_batch_vectors(int ndim, size_t nref, const Float* ref, const Float* radius, const std::vector<std::vector<Index> >& by_ref, const Float* target, Float* vectors) {
    #pragma omp parallel
    {
        std::vector<Float> vbuffer(ndim);
        std::vector<std::pair<Float, bool> > collected;

        #pragma omp for
        for (size_t r = 0; r < nref; ++r) {
            const auto& indices = by_ref[r];

            // If there are no listed neighbor indices, this usually
            // means that some interpolation is to be performed later.
            if (!indices.empty()) {
                Float* vptr = vectors + r * ndim;
                std::copy(vptr, vptr + ndim, vbuffer.data());
                Float l2norm = normalize_vector(ndim, vbuffer.data());

                if (l2norm) {
                    double mean_proj = 0, mean_proj_n = 0;
                    collected.clear();

                    for (auto i : indices) {
                        auto intersection = intersect_with_sphere(ndim, ref + r * ndim, vbuffer.data(), target + i * ndim, radius[r]);
                        Float rootdelta = intersection.second;

                        // Only considering points that lie within 'radius' of
                        // the line. We record the interval along the line
                        // during which that point is in range.
                        if (rootdelta > 0) {
                            auto proj = intersection.first;
                            mean_proj += proj;
                            ++mean_proj_n;

                            Float start = proj - rootdelta, end = proj + rootdelta;
                            if (end > l2norm) {
                                start = std::max(start, l2norm);
                                collected.emplace_back(start, true);
                                collected.emplace_back(end, false);
                            }
                        }
                    }

                    // Only searching for the coverage max if we have observations
                    // and the mean projection is past the MNNs; otherwise we set
                    // the scaling to 1 to use the direct distance to the MNNs. 
                    if (mean_proj_n) { 
                        mean_proj /= mean_proj_n;
                        if (mean_proj > l2norm && collected.size()) {
                            Float at_max = find_coverage_max(collected, mean_proj);
                            Float scale = at_max / l2norm; // Divide by l2norm so that it can directly scale 'vectors'.
                            for (int d = 0; d < ndim; ++d) {
                                vptr[d] *= scale; 
                            }
                        }
                    }
                }
            }
        }
    }

    return;
}

template<typename Float>
void extrapolate_vectors(int ndim, size_t nref, const Float* ref, const std::vector<char>& ok, Float* vectors) {
    for (size_t r = 0; r < nref; ++r) {
        if (!ok[r]) {
            const Float* rptr = ref + r * ndim;
            Float closest = std::numeric_limits<Float>::infinity();
            const Float* closest_ptr = NULL;

            for (size_t r2 = 0; r2 < nref; ++r2) {
                if (ok[r2]) {
                    const Float* rptr2 = ref + r2 * ndim;
                    Float dist = 0;
                    for (int d = 0; d < ndim; ++d) {
                        Float diff = rptr[d] - rptr2[d];
                        dist += diff * diff;
                    }
                    if (dist < closest) {
                        closest = dist;
                        closest_ptr = rptr2;
                    }
                }
            }

            if (closest_ptr == NULL) {
                // This should never be triggered if correct_target does its job properly.
                throw std::runtime_error("no clusters with sufficient MNN pairs");
            } else {
                Float* vptr = vectors + r * ndim;
                const Float* vptr2 = vectors + (closest_ptr - ref);
                std::copy(vptr2, vptr2 + ndim, vptr);
            }
        }
    }
}

template<typename Index, typename Float>
std::vector<Float> average_batch_vectors(int ndim, size_t nref, const std::vector<Index>& counts, const Float* vectors) {
    std::vector<Float> output(ndim);
    Float total = 0;

    for (size_t r = 0; r < nref; ++r) {
        auto vptr = vectors + r * ndim;
        for (int d = 0; d < ndim; ++d) {
            output[d] += vptr[d] * counts[r];
        }
        total += counts[r];
    }

    for (int d = 0; d < ndim; ++d) {
        output[d] /= total;
    }
    return output;
}

template<typename Index, typename Float>
void correct_target(
    int ndim, 
    size_t nref, 
    const Float* ref,
    const Float* radius,
    size_t ntarget, 
    const Float* target, 
    const MnnPairs<Index>& pairings, 
    const NeighborSet<Index, Float>& target_neighbors,
    Float* corrected,
    int min_mnns)
{
    std::vector<Float> vectors(ndim * nref);
    auto counts = compute_batch_vectors (ndim, nref, ref, pairings, target, vectors.data());
    auto by_ref = observations_by_ref(nref, target_neighbors);

    // Deciding whether we need to do some extrapolation of the correction vectors.
    bool needs_filling = false, has_okay = false;
    std::vector<char> is_okay(nref);
    for (size_t r = 0; r < nref; ++r) {
        if (counts[r] < min_mnns) {
            needs_filling = true;
            by_ref[r].clear();
        } else {
            is_okay[r] = true;
            has_okay = true;
        }
    }

    if (has_okay) {
        scale_batch_vectors(ndim, nref, ref, radius, by_ref, target, vectors.data());
        if (needs_filling) {
            extrapolate_vectors(ndim, nref, ref, is_okay, vectors.data());
        }
    } else {
        // Or in the worst case, just using the average, if there aren't enough 
        // MNN pairs for a stable calculation in any cluster.
        auto averaged = average_batch_vectors(ndim, nref, counts, vectors.data());
        for (size_t r = 0; r < nref; ++r) {
            std::copy(averaged.begin(), averaged.end(), vectors.begin() + r * ndim);
        }
    }

    // Applying the correction to each target point.
    for (size_t t = 0; t < ntarget; ++t) {
        auto src = target + t * ndim;
        auto out = corrected + t * ndim;
        auto corr = vectors.data() + target_neighbors[t].front().first * ndim;
        for (int d = 0; d < ndim; ++d) {
            out[d] = src[d] - corr[d];
        }
    }

    return;
}

}

#endif
