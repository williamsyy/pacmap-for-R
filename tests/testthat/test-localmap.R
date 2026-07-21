test_that("localmap runs on iris and returns the right shape", {
  X <- as.matrix(iris[, 1:4])
  emb <- localmap(X, num_iters = c(50L, 50L, 100L), random_state = 1L)
  expect_s3_class(emb, "localmap")
  expect_s3_class(emb, "pacmap")
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
  expect_true(all(is.finite(emb$embedding)))
  expect_true(!is.null(emb$pair_FP_final))
  expect_equal(dim(emb$pair_FP_final), dim(emb$pair_FP))
  # LocalMAP's FP resampling in phase 3 should shift at least some partners.
  expect_false(identical(emb$pair_FP, emb$pair_FP_final))
})

test_that("localmap separates iris species", {
  X <- as.matrix(iris[, 1:4])
  y <- as.integer(iris$Species)
  emb <- localmap(X, num_iters = c(100L, 100L, 250L), random_state = 3L)
  d <- as.matrix(dist(emb$embedding))
  within  <- mean(vapply(seq_len(3L), function(cls) {
    idx <- which(y == cls); mean(d[idx, idx][upper.tri(d[idx, idx])])
  }, numeric(1L)))
  between <- mean(d[y == 1L, y != 1L])
  expect_lt(within, between)
})

test_that("bad low_dist_thres errors", {
  X <- as.matrix(iris[, 1:4])
  expect_error(localmap(X, low_dist_thres = 0, num_iters = c(5L, 5L, 5L)))
  expect_error(localmap(X, low_dist_thres = -1, num_iters = c(5L, 5L, 5L)))
})

test_that("localmap works with distance='precomputed'", {
  X <- as.matrix(iris[, 1:4])
  D <- as.matrix(dist(X))
  emb <- localmap(D, distance = "precomputed",
                  num_iters = c(30L, 30L, 60L), random_state = 1L)
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
  expect_true(all(is.finite(emb$embedding)))
})
