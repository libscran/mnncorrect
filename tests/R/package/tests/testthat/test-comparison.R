# Tests that mnncorrect.cpp gives the same results as mnncorrect.ref.
# library(testthat); library(mnncorrect.ref); source("test-comparison.R")

set.seed(10000)
test_that("basic comparisons work out", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(0:1, c(600, 400))

    first <- x[,b==0]
    second <- x[,b==1]
    ref <- mnncorrect.ref(first, second)

    cpp <- mnncorrect.cpp(x, b, automatic.order=FALSE)
    first.cpp <- cpp$corrected[,b==0]
    second.cpp <- cpp$corrected[,b==1]

    expect_identical(first, first.cpp)
    expect_equal(second.cpp, ref)
})

set.seed(100001)
test_that("responds correctly to various options", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(0:1, c(200, 800))

    first <- x[,b==0]
    second <- x[,b==1]
    ref <- mnncorrect.ref(first, second)

    {
        out <- mnncorrect.ref(first, second, trim = 0.5)
        expect_false(identical(ref, out))

        cpp <- mnncorrect.cpp(x, b, trim = 0.5, automatic.order=FALSE)
        first.cpp <- cpp$corrected[,b==0]
        second.cpp <- cpp$corrected[,b==1]
        expect_identical(first, first.cpp)
        expect_equal(second.cpp, out)
    }

    {
        out <- mnncorrect.ref(first, second, k = 5)
        expect_false(identical(ref, out))

        cpp <- mnncorrect.cpp(x, b, k = 5, automatic.order=FALSE)
        first.cpp <- cpp$corrected[,b==0]
        second.cpp <- cpp$corrected[,b==1]
        expect_identical(first, first.cpp)
        expect_equal(second.cpp, out)
    }

    {
        out <- mnncorrect.ref(first, second, iterations = 0)
        expect_false(identical(ref, out))

        cpp <- mnncorrect.cpp(x, b, iterations = 0, automatic.order=FALSE)
        first.cpp <- cpp$corrected[,b==0]
        second.cpp <- cpp$corrected[,b==1]
        expect_identical(first, first.cpp)
        expect_equal(second.cpp, out)
    }

    {
        out <- mnncorrect.ref(first, second, nmads = 1)
        expect_false(identical(ref, out))

        cpp <- mnncorrect.cpp(x, b, nmads = 1, automatic.order=FALSE)
        first.cpp <- cpp$corrected[,b==0]
        second.cpp <- cpp$corrected[,b==1]
        expect_identical(first, first.cpp)
        expect_equal(second.cpp, out)
    }

    {
        out <- mnncorrect.ref(first, second, mass.cap = 20)
        expect_false(identical(ref, out))

        cpp <- mnncorrect.cpp(x, b, mass.cap = 20, automatic.order=FALSE)
        first.cpp <- cpp$corrected[,b==0]
        second.cpp <- cpp$corrected[,b==1]
        expect_identical(first, first.cpp)
        expect_equal(second.cpp, out)
    }
})

set.seed(100002)
test_that("iterative comparisons work out", {
    x <- matrix(rnorm(10000), nrow=10)
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

set.seed(100003)
test_that("automatic merge order makes sense", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(1:4, 1:4 * 100)
    x <- t(t(x) + b)

    cpp <- mnncorrect.cpp(x, b, automatic.order=TRUE)
    order <- cpp$merge.order
    expect_identical(order, 3:0) # largest to smallest, as more cells => more MNNs.

    ref <- x[,b==4]
    for (batch in 3:1) {
        current <- x[,b==batch,drop=FALSE]
        ref2 <- mnncorrect.ref(ref, current)
        ref <- cbind(ref2, ref)
    }

    expect_equal(ref, cpp$corrected)
})

set.seed(100004)
test_that("custom merge order makes sense", {
    x <- matrix(rnorm(10000), nrow=10)
    b <- rep(1:4, 4:1 * 100)
    x <- t(t(x) + b)

    cpp <- mnncorrect.cpp(x, b, order=4:1)
    order <- cpp$merge.order
    expect_identical(order, 3:0) # largest to smallest, as more cells => more MNNs.

    ref <- x[,b==4]
    for (batch in 3:1) {
        current <- x[,b==batch,drop=FALSE]
        ref2 <- mnncorrect.ref(ref, current)
        ref <- cbind(ref2, ref)
    }

    expect_equal(ref, cpp$corrected)
})
