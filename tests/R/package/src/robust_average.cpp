#include "Rcpp.h"
#include "mnncorrect/mnncorrect.hpp"

//[[Rcpp::export(rng=false)]]
Rcpp::NumericVector robust_average(Rcpp::NumericMatrix x, int iterations, double trim) {
    mnncorrect::internal::RobustAverageOptions raopt(iterations, trim);
    mnncorrect::internal::RobustAverageWorkspace<double> rawork;
    Rcpp::NumericVector output(x.nrow());
    mnncorrect::internal::robust_average(x.nrow(), x.ncol(), static_cast<const double*>(x.begin()), static_cast<double*>(output.begin()), rawork, raopt);
    return output;
}
