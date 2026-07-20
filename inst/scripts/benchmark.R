#!/usr/bin/env Rscript
# pacmapr benchmark harness.
#
# Measures:
#   1. Wall-clock time as n grows (fixed d)
#   2. Wall-clock time as d grows (fixed n)
#   3. KNN-preservation quality on labeled blobs
#   4. Determinism check with a fixed seed
#
# Run:
#   Rscript inst/scripts/benchmark.R                     # default sizes
#   Rscript inst/scripts/benchmark.R --quick             # tiny grid
#   Rscript inst/scripts/benchmark.R --out bench.csv
#
# Requires:  pacmapr installed, plus RcppHNSW (for KNN preservation)

suppressPackageStartupMessages({
  library(pacmapr)
})

args <- commandArgs(trailingOnly = TRUE)
quick <- "--quick" %in% args
out_arg <- which(args == "--out")
out_csv <- if (length(out_arg)) args[out_arg + 1L] else "pacmapr_benchmark.csv"

ns_by_n <- if (quick) c(300L, 1000L)          else c(500L, 2000L, 5000L, 10000L)
ds_by_d <- if (quick) c(10L, 50L)             else c(10L, 50L, 200L, 500L)
n_fixed_for_d <- if (quick) 500L               else 2000L
d_fixed_for_n <- 20L
n_iters       <- c(100L, 100L, 250L)
seed          <- 42L

make_blobs <- function(n, d, k = 5L, sd = 0.6, seed = 1L) {
  set.seed(seed)
  n_per <- ceiling(n / k)
  centers <- matrix(stats::rnorm(k * d, sd = 4), nrow = k)
  X <- do.call(rbind, lapply(seq_len(k), function(i) {
    matrix(stats::rnorm(n_per * d, sd = sd), n_per, d) +
      matrix(centers[i, ], n_per, d, byrow = TRUE)
  }))[seq_len(n), , drop = FALSE]
  y <- rep(seq_len(k), each = n_per)[seq_len(n)]
  list(X = X, y = y)
}

knn_preservation <- function(X, Y, k = 10L) {
  X_nn <- RcppHNSW::hnsw_knn(X, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  Y_nn <- RcppHNSW::hnsw_knn(Y, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  mean(vapply(seq_len(nrow(X)), function(i) {
    length(intersect(X_nn[i, ], Y_nn[i, ])) / k
  }, numeric(1L)))
}

# Fraction of a point's k-NN in Y that share its ground-truth label.
# 1.0 = each cluster is a connected neighborhood in the embedding.
label_preservation <- function(Y, y, k = 10L) {
  Y_nn <- RcppHNSW::hnsw_knn(Y, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  mean(vapply(seq_len(nrow(Y)), function(i) mean(y[Y_nn[i, ]] == y[i]), numeric(1L)))
}

time_one <- function(X, ...) {
  t0 <- Sys.time()
  emb <- pacmap(X, num_iters = n_iters, random_state = seed, verbose = FALSE, ...)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  list(embedding = emb$embedding, elapsed = elapsed, final_loss = utils::tail(emb$loss, 1L))
}

rows <- list()
cat("--- Scaling with n (d =", d_fixed_for_n, ") ---\n")
for (n in ns_by_n) {
  bl <- make_blobs(n, d_fixed_for_n)
  r  <- time_one(bl$X)
  q  <- knn_preservation(bl$X, r$embedding, k = 10L)
  lp <- label_preservation(r$embedding, bl$y, k = 10L)
  cat(sprintf("  n=%6d  time=%7.2fs  loss=%9.2f  knn_pres=%.3f  label_pres=%.3f\n",
              n, r$elapsed, r$final_loss, q, lp))
  rows[[length(rows) + 1L]] <- data.frame(
    axis = "n", n = n, d = d_fixed_for_n,
    time_s = r$elapsed, final_loss = r$final_loss,
    knn_pres = q, label_pres = lp
  )
}

cat("\n--- Scaling with d (n =", n_fixed_for_d, ") ---\n")
for (d in ds_by_d) {
  bl <- make_blobs(n_fixed_for_d, d)
  r  <- time_one(bl$X)
  q  <- knn_preservation(bl$X, r$embedding, k = 10L)
  lp <- label_preservation(r$embedding, bl$y, k = 10L)
  cat(sprintf("  d=%6d  time=%7.2fs  loss=%9.2f  knn_pres=%.3f  label_pres=%.3f\n",
              d, r$elapsed, r$final_loss, q, lp))
  rows[[length(rows) + 1L]] <- data.frame(
    axis = "d", n = n_fixed_for_d, d = d,
    time_s = r$elapsed, final_loss = r$final_loss,
    knn_pres = q, label_pres = lp
  )
}

cat("\n--- Determinism (n =", n_fixed_for_d, ", d =", d_fixed_for_n, ") ---\n")
bl <- make_blobs(n_fixed_for_d, d_fixed_for_n)
a <- pacmap(bl$X, num_iters = n_iters, random_state = 1L)$embedding
b <- pacmap(bl$X, num_iters = n_iters, random_state = 1L)$embedding
det_delta <- max(abs(a - b))
cat(sprintf("  max|a - b| with same seed: %.3e  (expect 0)\n", det_delta))

df <- do.call(rbind, rows)
write.csv(df, out_csv, row.names = FALSE)
cat("\nWrote", out_csv, "\n")

# Small ASCII summary
cat("\nSummary:\n")
print(df, row.names = FALSE)
