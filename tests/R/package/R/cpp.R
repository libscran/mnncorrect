#' @export
#' @importFrom Rcpp sourceCpp
#' @useDynLib mnncorrect.ref
mnncorrect.cpp <- function(combined, batch, k=15, steps=1, input.order=FALSE) {
    f <- factor(batch)
    compute(
        combined, 
        as.integer(f) - 1,
        num_batches = nlevels(f),
        k=k,
        steps=steps,
        input_order=input.order
    )
}
