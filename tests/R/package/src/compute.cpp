#include "mnncorrect/mnncorrect.hpp"
#include "Rcpp.h"

//[[Rcpp::export(rng=false)]]
Rcpp::RObject compute(
    Rcpp::NumericMatrix x,
    Rcpp::IntegerVector batch,
    int k,
    double tol,
    bool input_order)
{
    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;
    opt.tolerance = tol;
    if (input_order) {
        opt.merge_policy = mnncorrect::MergePolicy::INPUT;
    }

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
