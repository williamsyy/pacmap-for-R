test_that("distance='precomputed' errors on non-square input", {
  X <- as.matrix(iris[, 1:4])
  expect_error(pacmap(X, distance = "precomputed", num_iters = c(5L, 5L, 5L)),
               "square")
})

test_that("distance='precomputed' with L2 distance matrix matches 'euclidean' quality", {
  X <- as.matrix(iris[, 1:4])
  y <- as.integer(iris$Species)
  D <- as.matrix(dist(X))          # L2 distance matrix

  fit_pre <- pacmap(D, distance = "precomputed",
                    num_iters = c(100L, 100L, 250L), random_state = 1L)
  fit_euc <- pacmap(X, distance = "euclidean",
                    num_iters = c(100L, 100L, 250L), random_state = 1L,
                    apply_pca = FALSE)  # same preprocessing as precomputed

  same_class_pre <- mean(vapply(seq_len(nrow(X)), function(i) {
    nn <- order(sqrt(rowSums(sweep(fit_pre$embedding, 2L, fit_pre$embedding[i,], "-")^2)))[2:11]
    mean(y[nn] == y[i])
  }, numeric(1L)))
  same_class_euc <- mean(vapply(seq_len(nrow(X)), function(i) {
    nn <- order(sqrt(rowSums(sweep(fit_euc$embedding, 2L, fit_euc$embedding[i,], "-")^2)))[2:11]
    mean(y[nn] == y[i])
  }, numeric(1L)))
  # Both should recover species neighborhoods well; precomputed should be
  # within ~15% of the feature-based fit.
  expect_gt(same_class_pre, 0.85)
  expect_gt(same_class_pre, same_class_euc - 0.15)
})

test_that("distance='precomputed' works with an asymmetric distance (cosine)", {
  X <- as.matrix(iris[, 1:4])
  # Cosine distance is symmetric but the branch docs say the matrix need
  # not be perfectly symmetric -- we just want to exercise the path.
  Xn <- X / sqrt(rowSums(X^2))
  D  <- 1 - Xn %*% t(Xn)
  D[D < 0] <- 0
  emb <- pacmap(D, distance = "precomputed",
                num_iters = c(50L, 50L, 100L), random_state = 2L)
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
  expect_true(all(is.finite(emb$embedding)))
})

test_that("transform() errors on a model fit with precomputed distances", {
  X <- as.matrix(iris[, 1:4])
  D <- as.matrix(dist(X))
  fit <- pacmap(D, distance = "precomputed", num_iters = c(10L, 10L, 10L),
                random_state = 1L)
  expect_error(transform(fit, D[1:5, ]), "precomputed")
})

test_that("apply_pca is silently disabled when precomputed", {
  X <- as.matrix(iris[, 1:4])
  D <- as.matrix(dist(X))
  emb <- pacmap(D, distance = "precomputed", apply_pca = TRUE,
                num_iters = c(5L, 5L, 5L), random_state = 1L)
  expect_false(emb$params$apply_pca)
})
