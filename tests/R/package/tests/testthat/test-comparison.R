# Tests that mnncorrect.cpp gives the same results as mnncorrect.ref.
# library(testthat); library(mnncorrect.ref); source("test-comparison.R")

library(mnncorrect.ref)

set.seed(10000)
x <- matrix(rnorm(10000), nrow=10)
b <- rep(0:1, c(600, 400))
trim <- 0.2 # avoid difficult discrepancies due to differences in numerical precision.

test_that("basic comparisons work out", {
    first <- x[,b==0]
    second <- x[,b==1]
    ref <- mnncorrect.ref(first, second, iterations=1, trim=trim)

    cpp <- mnncorrect.cpp(x, b, iterations=1, trim=trim)
    first.cpp <- cpp$corrected[,b==0]
    second.cpp <- cpp$corrected[,b==1]

    expect_identical(first, first.cpp)
    expect_equal(second.cpp, ref)
})

set.seed(100001)
test_that("iterative comparisons work out", {
    b <- sample(3, ncol(x), replace=TRUE) - 1L
    cpp <- mnncorrect.cpp(x, b, iterations=1)
    order <- cpp$merge.order

    first <- x[,b==order[1]]
    second <- x[,b==order[2]]
    third <- x[,b==order[3]]

    ref0 <- mnncorrect.ref(first, second, iterations=1)
    ref <- mnncorrect.ref(cbind(first, ref0), third, iterations=1)
    final <- matrix(0, nrow(x), ncol(x))
    final[,b==order[1]] <- first
    final[,b==order[2]] <- ref0
    final[,b==order[3]] <- ref

    expect_equal(final, cpp$corrected)
})
