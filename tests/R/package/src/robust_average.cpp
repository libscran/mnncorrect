#include "Rcpp.h"
#include "mnncorrect/RobustAverage.hpp"

//[[Rcpp::export(rng=false)]]
Rcpp::NumericVector robust_average(Rcpp::NumericMatrix x, int iterations, double trim) {
    mnncorrect::RobustAverage<int, double> test(iterations, trim);
    Rcpp::NumericVector output(x.nrow());
    test.run(x.nrow(), x.ncol(), static_cast<const double*>(x.begin()), static_cast<double*>(output.begin()));
    return output;
}
