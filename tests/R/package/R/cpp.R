#' @export
#' @importFrom Rcpp sourceCpp
#' @useDynLib mnncorrect.ref
mnncorrect.cpp <- function(combined, batch, k=15, steps=1, input.order=FALSE) {
    compute(
        combined, 
        as.integer(factor(batch)) - 1,
        k=k,
        steps=steps,
        input_order=input.order
    )
}
