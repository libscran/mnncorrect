#include "mnncorrect/MnnCorrect.hpp"
#include "Rcpp.h"

//[[Rcpp::export(rng=false)]]
Rcpp::List mnn_correct(Rcpp::NumericMatrix x, Rcpp::IntegerVector batch, int k, double nmads, int iterations, double trim) {
    mnncorrect::MnnCorrect<> runner;
    runner.set_num_neighbors(k).set_num_mads(nmads).set_robust_iterations(iterations).set_robust_trim(trim);

    Rcpp::NumericMatrix output(x.nrow(), x.ncol());
    auto res = runner.run(x.nrow(), x.ncol(), 
        static_cast<const double*>(x.begin()), 
        static_cast<const int*>(batch.begin()), 
        static_cast<double*>(output.begin()));

    return Rcpp::List::create(
        Rcpp::Named("corrected") = output,
        Rcpp::Named("merge.order") = Rcpp::IntegerVector(res.merge_order.begin(), res.merge_order.end()),
        Rcpp::Named("num.pairs") = Rcpp::IntegerVector(res.num_pairs.begin(), res.num_pairs.end())
    );
}
