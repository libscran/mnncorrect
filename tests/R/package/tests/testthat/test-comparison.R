# Tests that mnncorrect.cpp gives the same results as mnncorrect.ref.
# library(testthat); library(mnncorrect.ref); source("test-comparison.R")

library(mnncorrect.ref)

set.seed(10000)
x <- matrix(rnorm(10000), nrow=10)
b <- rep(0:1, c(600, 400))

test_that("basic comparisons work out", {
    first <- x[,b==0]
    second <- x[,b==1]
    ref <- mnncorrect.ref(first, second)

    cpp <- mnncorrect.cpp(x, b)
    first.cpp <- cpp$corrected[,b==0]
    second.cpp <- cpp$corrected[,b==1]

    expect_identical(first, first.cpp)
})
