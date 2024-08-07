# library(testthat); library(mnncorrect.ref); source("test-robust_average.R")

set.seed(1000)

test_that("robust average works the same in R and C++", {
    for (it in c(0, 1, 2, 5, 10)) {
        for (trim in c(0.1, 0.2, 0.5)) {
            y <- matrix(rnorm(1000), ncol=100)
            ref <- mnncorrect.ref:::robust_centroid(y, trim=trim, iterations=it)
            cpp <- mnncorrect.ref:::robust_average(y, trim=trim, iterations=it)
            expect_equal(ref, cpp)
        }
    }
})

