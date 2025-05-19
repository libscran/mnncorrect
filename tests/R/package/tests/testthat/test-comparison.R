# Tests that mnncorrect.cpp gives the same results as mnncorrect.ref.
# library(testthat); library(mnncorrect.ref); source("test-comparison.R")

set.seed(10000)
test_that("basic comparisons work out", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(0:1, c(600, 400))

    ref <- mnncorrect.ref(x, b)
    cpp <- mnncorrect.cpp(x, b, input.order=TRUE)

    expect_equal(ref, cpp)
})

set.seed(100001)
test_that("responds correctly to various options", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(0:1, c(200, 800))
    default <- mnncorrect.ref(x, b)

    {
        ref <- mnncorrect.ref(x, b, k=5)
        expect_false(identical(default, ref))

        cpp <- mnncorrect.cpp(x, b, k=5, input.order=TRUE)
        expect_equal(ref, cpp)
    }

    {
        ref <- mnncorrect.ref(x, b, steps=3)
        expect_false(identical(default, ref))

        cpp <- mnncorrect.cpp(x, b, steps=3, input.order=TRUE)
        expect_equal(ref, cpp)
    }
})

set.seed(100003)
test_that("automatic merge order makes sense", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(1:4, 1:4 * 100)
    x <- t(t(x) + b) # injecting a batch effect to make it interesting.

    cpp <- mnncorrect.cpp(x, b)
    ref <- mnncorrect.ref(x, 5L - b) # batches are sorted by their batch number, so the biggest batch is now first and so on.
    expect_equal(ref, cpp)
})
