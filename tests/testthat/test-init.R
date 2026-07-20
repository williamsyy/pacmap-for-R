test_that("init = 'pca' works", {
  X <- as.matrix(iris[, 1:4])
  emb <- pacmap(X, init = "pca", num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_true(all(is.finite(emb$embedding)))
})

test_that("init = 'random' works and is different from pca init", {
  X <- as.matrix(iris[, 1:4])
  emb_pca <- pacmap(X, init = "pca",    num_iters = c(10L, 10L, 20L), random_state = 1L)
  emb_rnd <- pacmap(X, init = "random", num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_true(all(is.finite(emb_rnd$embedding)))
  expect_false(isTRUE(all.equal(emb_pca$embedding, emb_rnd$embedding)))
})

test_that("user-matrix init is accepted", {
  X <- as.matrix(iris[, 1:4])
  Y0 <- matrix(stats::rnorm(nrow(X) * 2L), nrow(X), 2L)
  emb <- pacmap(X, init = Y0, num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_true(all(is.finite(emb$embedding)))
})

test_that("user-matrix init with wrong shape errors", {
  X <- as.matrix(iris[, 1:4])
  Y0 <- matrix(0, nrow(X) + 1L, 2L)
  expect_error(pacmap(X, init = Y0, num_iters = c(5L, 5L, 5L)))
})
