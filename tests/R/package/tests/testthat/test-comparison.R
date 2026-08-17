# Tests that mnncorrect.cpp gives the same results as mnncorrect.ref.
# library(testthat); library(mnncorrect.ref); source("test-comparison.R")

set.seed(10000)
test_that("basic comparisons work out", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(0:1, c(600, 400))

    ref <- mnncorrect.ref(x, b)
    cpp <- mnncorrect.cpp(x, b, input.order=TRUE)
    expect_equal(ref, cpp)

    # Inverting the order.
    ref <- mnncorrect.ref(x, 1 - b)
    cpp <- mnncorrect.cpp(x, 1 - b, input.order=TRUE)
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
test_that("automatic merge order makes sense (simple)", {
    b <- rep(1:4, 1:4 * 100)
    x <- matrix(rnorm(10 * length(b)), nrow=10)
    x <- t(t(x) + b) # injecting a batch effect to make it interesting.

    cpp <- mnncorrect.cpp(x, b)
    ref <- mnncorrect.ref(x, 5L - b) # batches are sorted by their batch number, so the biggest batch has the highest RSS and should now be the first.
    expect_equal(ref, cpp)

    # Same results with some shuffling.
    o <- sample(length(b))
    cpp <- mnncorrect.cpp(x[,o], b[o])
    ref <- mnncorrect.ref(x[,o], 5L - b[o])
    expect_equal(ref, cpp)
})

set.seed(100004)
test_that("automatic merge order makes sense (complex)", {
    b <- rep(1:5, 100)
    x <- matrix(rnorm(20 * length(b)), nrow=20)

    for (curb in 1:5) {
        curcells <- curb == b
        x[curb,curcells] <- x[curb,curcells] + 5 # injecting a batch effect to make it interesting.
        halfcells <- which(curcells)
        halfcells <- halfcells[halfcells %% 2 == 0]
        x[10 + curb,halfcells] <- x[10 + curb,halfcells] + (6 - curb) * 10 # injecting some population structure in half of the cells.
    }

    cpp <- mnncorrect.cpp(x, b)
    ref <- mnncorrect.ref(x, b) # batch with the lowest index should have the highest RSS.
    expect_equal(ref, cpp)

    # Same results with some shuffling.
    o <- sample(length(b))
    cpp <- mnncorrect.cpp(x[,o], b[o])
    ref <- mnncorrect.ref(x[,o], b[o])
    expect_equal(ref, cpp)
})
