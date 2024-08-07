#include "Rcpp.h"
#include "mnncorrect/mnncorrect.hpp"

//[[Rcpp::export(rng=false)]]
Rcpp::NumericVector robust_average(Rcpp::NumericMatrix x, int iterations, double trim) {
    mnncorrect::internal::RobustAverageOptions raopt(iterations, trim);
    std::vector<std::pair<double, size_t> > deltas;
    Rcpp::NumericVector output(x.nrow());
    mnncorrect::internal::robust_average(x.nrow(), x.ncol(), static_cast<const double*>(x.begin()), static_cast<double*>(output.begin()), deltas, raopt);
    return output;
}
