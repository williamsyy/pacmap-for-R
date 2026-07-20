test_that("bad distance metric errors", {
  X <- as.matrix(iris[, 1:4])
  expect_error(pacmap(X, distance = "cosine", num_iters = c(5L, 5L, 5L)))
})

test_that("n <= 1 errors", {
  expect_error(pacmap(matrix(1, 1, 3), num_iters = c(5L, 5L, 5L)),
               "Sample size")
})

test_that("lr <= 0 errors", {
  X <- as.matrix(iris[, 1:4])
  expect_error(pacmap(X, lr = 0, num_iters = c(5L, 5L, 5L)))
  expect_error(pacmap(X, lr = -1, num_iters = c(5L, 5L, 5L)))
})

test_that("n_components < 1 errors", {
  X <- as.matrix(iris[, 1:4])
  expect_error(pacmap(X, n_components = 0L, num_iters = c(5L, 5L, 5L)))
})
