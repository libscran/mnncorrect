#' @export
#' @importFrom Rcpp sourceCpp
#' @useDynLib mnncorrect.ref
mnncorrect.cpp <- function(combined, batch, k=15, nmads=3, iterations=2, trim=0.25) {
    mnn_correct(
        combined, 
        as.integer(factor(batch)) - 1,
        k=k,
        nmads = 3,
        iterations = iterations,
        trim = trim
    )
}
