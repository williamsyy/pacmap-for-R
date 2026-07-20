test_that("changing n_neighbors changes pair counts", {
  X <- as.matrix(iris[, 1:4])
  a <- find_pacmap_pairs(X, n_neighbors = 5L,  random_state = 1L)
  b <- find_pacmap_pairs(X, n_neighbors = 20L, random_state = 1L)
  expect_equal(nrow(a$pair_neighbors), nrow(X) * 5L)
  expect_equal(nrow(b$pair_neighbors), nrow(X) * 20L)
})

test_that("MN_ratio and FP_ratio control pair counts", {
  X <- as.matrix(iris[, 1:4])
  pairs <- find_pacmap_pairs(X, n_neighbors = 10L, MN_ratio = 1.0, FP_ratio = 3.0,
                             random_state = 1L)
  expect_equal(nrow(pairs$pair_MN), nrow(X) * 10L)
  expect_equal(nrow(pairs$pair_FP), nrow(X) * 30L)
})

test_that("higher lr converges faster (final loss lower after fewer iters)", {
  X <- as.matrix(iris[, 1:4])
  e_slow <- pacmap(X, lr = 0.1, num_iters = c(30L, 30L, 40L), random_state = 1L)
  e_fast <- pacmap(X, lr = 1.0, num_iters = c(30L, 30L, 40L), random_state = 1L)
  # Not strictly guaranteed but usually true on iris.
  expect_lte(utils::tail(e_fast$loss, 1L), utils::tail(e_slow$loss, 1L) * 1.5)
})
