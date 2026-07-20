test_that("high-dim > 100 triggers the PCA-to-100 preprocessing branch", {
  set.seed(1)
  X <- matrix(stats::rnorm(200L * 150L), 200L, 150L)
  emb <- pacmap(X, num_iters = c(20L, 20L, 40L), random_state = 1L, verbose = FALSE)
  expect_equal(dim(emb$embedding), c(200L, 2L))
  expect_true(all(is.finite(emb$embedding)))
  expect_equal(ncol(emb$preprocess$X), 100L)  # PCA-to-100 applied
})

test_that("small dataset with high dim (n=40, d=60) still works", {
  set.seed(2)
  X <- matrix(stats::rnorm(40L * 60L), 40L, 60L)
  emb <- pacmap(X, num_iters = c(20L, 20L, 40L), random_state = 2L)
  expect_equal(dim(emb$embedding), c(40L, 2L))
})

test_that("2D -> 2D and 3D -> 3D both return the requested shape", {
  set.seed(3)
  X2 <- matrix(stats::rnorm(200L * 2L), 200L, 2L)
  e2 <- pacmap(X2, n_components = 2L, num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_equal(dim(e2$embedding), c(200L, 2L))

  X3 <- matrix(stats::rnorm(200L * 3L), 200L, 3L)
  e3 <- pacmap(X3, n_components = 3L, num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_equal(dim(e3$embedding), c(200L, 3L))
})

test_that("num_iters accepts a scalar and expands to c(100, 100, x)", {
  X <- as.matrix(iris[, 1:4])
  emb <- pacmap(X, num_iters = 50L, random_state = 1L)
  expect_length(emb$loss, 100L + 100L + 50L)
  expect_equal(emb$params$num_iters, c(100L, 100L, 50L))
})

test_that("data.frame input is coerced to matrix", {
  X <- iris[, 1:4]
  emb <- pacmap(X, num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
})
