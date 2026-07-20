test_that("transform on the original data lands near the fitted embedding", {
  X <- as.matrix(iris[, 1:4])
  fit <- pacmap(X, num_iters = c(100L, 100L, 250L), random_state = 42L)
  # Re-embedding the original data should land close to the fit's embedding.
  # (Not identical -- transform uses random FP-free optimization from a fresh init.)
  tr <- transform(fit, X, num_iters = c(50L, 50L, 100L))
  expect_equal(dim(tr), dim(fit$embedding))
  expect_true(all(is.finite(tr)))
  # Each transformed point should land closer to its original fit position
  # than to a random other fit position.
  d_self  <- sqrt(rowSums((tr - fit$embedding)^2))
  perm    <- sample(nrow(X))
  d_other <- sqrt(rowSums((tr - fit$embedding[perm, ])^2))
  expect_gt(mean(d_other) / mean(d_self), 3)
})

test_that("transform of a held-out point lands in the correct species cluster", {
  X <- as.matrix(iris[, 1:4])
  y <- as.integer(iris$Species)
  set.seed(1)
  holdout <- sample(nrow(X), 30L)
  fit <- pacmap(X[-holdout, ], num_iters = c(100L, 100L, 250L), random_state = 7L)
  tr  <- transform(fit, X[holdout, ], num_iters = c(50L, 50L, 100L))

  # For each held-out point, its 5 nearest neighbors in the fitted embedding
  # should mostly share its species.
  hits <- vapply(seq_along(holdout), function(i) {
    d <- sqrt(rowSums(sweep(fit$embedding, 2L, tr[i, ], "-")^2))
    nn <- order(d)[1:5]
    mean(y[-holdout][nn] == y[holdout[i]])
  }, numeric(1L))
  expect_gt(mean(hits), 0.7)  # loose bound: iris is easy
})
