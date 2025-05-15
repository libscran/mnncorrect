#' @export
#' @importFrom Rcpp sourceCpp
#' @useDynLib mnncorrect.ref
mnncorrect.cpp <- function(combined, batch, k=15, tol=3, input.order=FALSE) {
    compute(
        combined, 
        as.integer(factor(batch)) - 1,
        k=k,
        tol=tol,
        input_order=input.order
    )
}
