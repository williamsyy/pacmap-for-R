# Support for distance = "precomputed": X is an n x n distance matrix
# instead of a feature matrix. Mirrors the LocalMAP branch feature
# (williamsyy/LocalMAP@feature/precomputed-distance-matrix).

.knn_from_distance_matrix <- function(D, k) {
  n <- nrow(D)
  if (ncol(D) != n) stop("Precomputed distance matrix must be square")
  if (k > n - 1L) k <- n - 1L
  idx   <- matrix(0L,  n, k)
  dists <- matrix(0.0, n, k)
  for (i in seq_len(n)) {
    row <- D[i, ]
    row[i] <- Inf                       # exclude self
    ord <- order(row)[seq_len(k)]
    idx[i, ]   <- ord - 1L              # 0-indexed for C++
    dists[i, ] <- row[ord]
  }
  list(indices = idx, distances = dists)
}

# Precomputed MN sampling. Python's numba version picks 6 random candidates,
# computes distances from features, keeps the 2nd-closest. Here we just look
# up distances in the row of the precomputed matrix.
.sample_mn_pairs_precomputed <- function(D, n_MN, random_state = NULL) {
  n <- nrow(D)
  if (!is.null(random_state)) set.seed(as.integer(random_state))
  pair_MN <- matrix(0L, n * n_MN, 2)
  for (i in seq_len(n)) {
    rejected <- integer(0)
    for (j in seq_len(n_MN)) {
      # Sample 6 unique candidates, excluding self and previously-picked MN partners for i.
      candidates <- integer(0)
      forbidden  <- c(i, rejected + 1L)
      while (length(candidates) < 6L) {
        pool <- setdiff(seq_len(n), c(forbidden, candidates))
        if (length(pool) == 0L) break
        pick <- sample(pool, min(6L - length(candidates), length(pool)))
        candidates <- c(candidates, pick)
      }
      if (length(candidates) < 2L) next
      d <- D[i, candidates]
      # Drop argmin, then argmin of what remains -> 2nd-closest.
      am <- which.min(d)
      d2 <- d[-am]; c2 <- candidates[-am]
      picked <- c2[which.min(d2)]
      pair_MN[(i - 1L) * n_MN + j, 1L] <- i - 1L
      pair_MN[(i - 1L) * n_MN + j, 2L] <- picked - 1L
      rejected <- c(rejected, picked - 1L)
    }
  }
  pair_MN
}
