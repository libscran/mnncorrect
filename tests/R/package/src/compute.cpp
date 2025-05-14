#include "mnncorrect/mnncorrect.hpp"
#include "Rcpp.h"

//[[Rcpp::export(rng=false)]]
Rcpp::RObject compute(
    Rcpp::NumericMatrix x,
    Rcpp::IntegerVector batch,
    int k,
    double tol)
{
    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;
    opt.tolerance = tol;

    Rcpp::NumericMatrix output(x.nrow(), x.ncol());
    mnncorrect::compute(
        x.nrow(),
        x.ncol(), 
        static_cast<const double*>(x.begin()), 
        static_cast<const int*>(batch.begin()), 
        static_cast<double*>(output.begin()),
        opt
    );

    return output;
}
