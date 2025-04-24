#include "mnncorrect/mnncorrect.hpp"
#include "Rcpp.h"

//[[Rcpp::export(rng=false)]]
Rcpp::List compute(
    Rcpp::NumericMatrix x,
    Rcpp::IntegerVector batch,
    int k,
    double nmads,
    int iterations,
    double trim,
    int mass_cap,
    Rcpp::Nullable<Rcpp::IntegerVector> order,
    bool auto_order)
{
    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;
    opt.num_mads = nmads;
    opt.robust_iterations = iterations;
    opt.robust_trim = trim;
    opt.mass_cap = mass_cap;

    opt.automatic_order = auto_order;
    if (!order.isNull()) {
        Rcpp::IntegerVector iorder(order);
        opt.order.reserve(iorder.size());
        for (auto o : iorder) {
            opt.order.push_back(o -  1); // getting back to 1-based indexing.
        }
    }

    Rcpp::NumericMatrix output(x.nrow(), x.ncol());
    auto res = mnncorrect::compute(
        x.nrow(),
        x.ncol(), 
        static_cast<const double*>(x.begin()), 
        static_cast<const int*>(batch.begin()), 
        static_cast<double*>(output.begin()),
        opt
    );

    return Rcpp::List::create(
        Rcpp::Named("corrected") = output,
        Rcpp::Named("merge.order") = Rcpp::IntegerVector(res.merge_order.begin(), res.merge_order.end()),
        Rcpp::Named("num.pairs") = Rcpp::IntegerVector(res.num_pairs.begin(), res.num_pairs.end())
    );
}
