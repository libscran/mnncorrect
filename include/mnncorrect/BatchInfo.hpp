#ifndef MNNCORRECT_BATCH_INFO_HPP
#define MNNCORRECT_BATCH_INFO_HPP

#include <memory>
#include <vector>

#include "knncolle/knncolle.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct Corrected {
    Corrected(std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index, std::vector<Index_> ids) : index(std::move(index)), ids(std::move(ids)) {}
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index;
    std::vector<Index_> ids;
};

template<typename Index_, typename Float_>
struct BatchInfo {
    Index_ offset, num_obs;
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index;
    std::vector<Corrected<Index_, Float_> > extras;
};

}

}

#endif
