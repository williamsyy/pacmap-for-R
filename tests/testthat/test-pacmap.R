test_that("pacmap runs on iris and returns the right shape", {
  X <- as.matrix(iris[, 1:4])
  emb <- pacmap(X, n_components = 2L, num_iters = c(50L, 50L, 100L),
                random_state = 42L, verbose = FALSE)
  expect_s3_class(emb, "pacmap")
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
  expect_true(all(is.finite(emb$embedding)))
  expect_length(emb$loss, 200L)
  # Loss should decrease from start to end.
  expect_lt(mean(utils::tail(emb$loss, 20)), mean(utils::head(emb$loss, 20)))
})

test_that("print method works", {
  X <- as.matrix(iris[, 1:4])
  emb <- pacmap(X, num_iters = c(10L, 10L, 20L), random_state = 1L)
  expect_output(print(emb), "pacmap embedding")
  expect_output(print(emb), "distance=euclidean")
})

test_that("embedding separates iris species better than the raw data on any 2 axes", {
  X <- as.matrix(iris[, 1:4])
  y <- as.integer(iris$Species)
  emb <- pacmap(X, num_iters = c(50L, 50L, 100L), random_state = 3L)
  # Silhouette-like: mean within-cluster dist / mean between-cluster dist in Y
  d <- as.matrix(dist(emb$embedding))
  within <- mean(vapply(seq_len(3L), function(cls) {
    idx <- which(y == cls); mean(d[idx, idx][upper.tri(d[idx, idx])])
  }, numeric(1L)))
  between <- mean(d[y == 1L, y != 1L])
  expect_lt(within, between)
})
