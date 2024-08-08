#' @export
#' @importFrom Rcpp sourceCpp
#' @useDynLib mnncorrect.ref
mnncorrect.cpp <- function(combined, batch, k=15, nmads=3, iterations=2, trim=0.25, automatic.order=TRUE, order=NULL, mass.cap=-1) {
    compute(
        combined, 
        as.integer(factor(batch)) - 1,
        k=k,
        nmads = nmads,
        iterations = iterations,
        trim = trim,
        mass_cap = mass.cap,
        auto_order = automatic.order,
        order = order # conversion to 0-based ordering is done on the C++ side.
    )
}
