#!/usr/bin/env Rscript
# MNIST benchmark for pacmapr.
#
# Downloads MNIST (70k images x 784 features) from a backup mirror,
# caches locally, runs pacmap on subsets of increasing size, reports
# wall-clock + label preservation, saves colored scatter plots.
#
# Run:  Rscript inst/scripts/mnist.R
# Args: --cache-dir DIR    where to cache the IDX files (default ~/.pacmapr_mnist)
#       --sizes "1000,5000,10000,70000"   subset sizes to test
#       --out PREFIX       prefix for output CSV/PNGs (default pacmapr_mnist)

suppressPackageStartupMessages({
  library(pacmapr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  i <- which(args == paste0("--", name))
  if (length(i)) args[i + 1L] else default
}
cache_dir <- get_arg("cache-dir", file.path(path.expand("~"), ".pacmapr_mnist"))
sizes     <- as.integer(strsplit(get_arg("sizes", "1000,5000,10000,70000"), ",")[[1L]])
out_pref  <- get_arg("out", "pacmapr_mnist")

dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1. Download MNIST (cached) --------------------------------------------

MIRROR <- "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES <- c(
  train_x = "train-images-idx3-ubyte.gz",
  train_y = "train-labels-idx1-ubyte.gz",
  test_x  = "t10k-images-idx3-ubyte.gz",
  test_y  = "t10k-labels-idx1-ubyte.gz"
)
for (nm in names(FILES)) {
  dest <- file.path(cache_dir, FILES[nm])
  if (!file.exists(dest)) {
    cat("Downloading", FILES[nm], "...\n")
    utils::download.file(paste0(MIRROR, FILES[nm]), dest, mode = "wb", quiet = TRUE)
  } else {
    cat("Cached:", FILES[nm], "\n")
  }
}

# ---- 2. Parse IDX format ---------------------------------------------------

read_idx_images <- function(path) {
  con <- gzfile(path, "rb"); on.exit(close(con))
  magic <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  stopifnot(magic == 2051L)
  n    <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  rows <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  cols <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  raw  <- readBin(con, "raw", n = n * rows * cols)
  m    <- matrix(as.integer(raw), nrow = n, ncol = rows * cols, byrow = TRUE)
  storage.mode(m) <- "double"
  m / 255  # scale to [0, 1]
}
read_idx_labels <- function(path) {
  con <- gzfile(path, "rb"); on.exit(close(con))
  magic <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  stopifnot(magic == 2049L)
  n     <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  as.integer(readBin(con, "raw", n = n))
}

cat("\nParsing IDX files...\n")
Xtr <- read_idx_images(file.path(cache_dir, FILES["train_x"]))
ytr <- read_idx_labels(file.path(cache_dir, FILES["train_y"]))
Xte <- read_idx_images(file.path(cache_dir, FILES["test_x"]))
yte <- read_idx_labels(file.path(cache_dir, FILES["test_y"]))
X_all <- rbind(Xtr, Xte)
y_all <- c(ytr, yte)
cat(sprintf("MNIST loaded: %d images, %d features (0..1 scaled).\n",
            nrow(X_all), ncol(X_all)))
stopifnot(nrow(X_all) == 70000L, ncol(X_all) == 784L)

# ---- 3. Label preservation helper ------------------------------------------

label_preservation <- function(Y, y, k = 10L) {
  Y_nn <- RcppHNSW::hnsw_knn(Y, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  mean(vapply(seq_len(nrow(Y)), function(i) mean(y[Y_nn[i, ]] == y[i]), numeric(1L)))
}

# ---- 4. Run pacmap on each size --------------------------------------------

set.seed(42)
idx_shuf <- sample(nrow(X_all))
rows <- list()

for (n in sizes) {
  n <- min(n, nrow(X_all))
  cat(sprintf("\n=== n = %d ===\n", n))
  ii <- idx_shuf[seq_len(n)]
  X  <- X_all[ii, , drop = FALSE]
  y  <- y_all[ii]

  t0 <- Sys.time()
  emb <- pacmap(X, n_components = 2L,
                num_iters = c(100L, 100L, 250L),
                random_state = 42L, verbose = FALSE)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  lp <- label_preservation(emb$embedding, y, k = 10L)
  cat(sprintf("time = %.1fs   final loss = %.1f   label_preservation@10 = %.3f\n",
              elapsed, utils::tail(emb$loss, 1L), lp))

  # Save scatter plot
  fname <- sprintf("%s_n%d.png", out_pref, n)
  png(fname, width = 900, height = 900, res = 120)
  palette10 <- c("#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
                 "#911eb4","#42d4f4","#f032e6","#469990","#9a6324")
  plot(emb$embedding[, 1], emb$embedding[, 2],
       col = palette10[y + 1L], pch = 19, cex = 0.4,
       xlab = "PaCMAP 1", ylab = "PaCMAP 2",
       main = sprintf("pacmapr on MNIST (n=%d, %.1fs, label_pres=%.3f)",
                      n, elapsed, lp))
  legend("topright", legend = 0:9, col = palette10, pch = 19, cex = 0.7,
         bg = rgb(1, 1, 1, 0.8))
  dev.off()
  cat("Saved", fname, "\n")

  rows[[length(rows) + 1L]] <- data.frame(
    n = n, time_s = elapsed,
    final_loss = utils::tail(emb$loss, 1L),
    label_pres = lp
  )
}

df <- do.call(rbind, rows)
csv <- paste0(out_pref, ".csv")
write.csv(df, csv, row.names = FALSE)
cat("\n=== Summary ===\n"); print(df, row.names = FALSE)
cat("\nWrote", csv, "\n")
