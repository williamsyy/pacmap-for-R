# Test helpers.

make_blobs <- function(n_per = 60, d = 4, k = 3, sd = 0.5, seed = 1L) {
  set.seed(seed)
  centers <- matrix(stats::rnorm(k * d, sd = 4), nrow = k)
  X <- do.call(rbind, lapply(seq_len(k), function(i) {
    matrix(stats::rnorm(n_per * d, sd = sd), n_per, d) +
      matrix(centers[i, ], n_per, d, byrow = TRUE)
  }))
  y <- rep(seq_len(k), each = n_per)
  list(X = X, y = y)
}

# Simple KNN-preservation metric: for each point, fraction of its k-NN in the
# original space that also appear in its k-NN in the embedding. 1.0 = perfect
# local structure preservation.
knn_preservation <- function(X, Y, k = 10L) {
  X_nn <- RcppHNSW::hnsw_knn(X, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  Y_nn <- RcppHNSW::hnsw_knn(Y, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  n <- nrow(X)
  mean(vapply(seq_len(n), function(i) {
    length(intersect(X_nn[i, ], Y_nn[i, ])) / k
  }, numeric(1L)))
}
