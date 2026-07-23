#!/usr/bin/env Rscript
# Side-by-side visualization: pacmap() vs localmap() on the same MNIST subset,
# same random_state. Saves a single PNG showing both embeddings, plus per-panel
# label-preservation numbers so you can see LocalMAP's phase-3 refinement
# tightening cluster boundaries.

suppressPackageStartupMessages(library(pacmapr))

n      <- 30000L
seed   <- 42L
iters  <- c(100L, 100L, 250L)
CACHE  <- file.path(path.expand("~"), ".pacmapr_mnist")
FILES  <- c(x = "train-images-idx3-ubyte.gz", y = "train-labels-idx1-ubyte.gz")

read_idx_images <- function(path) {
  con <- gzfile(path, "rb"); on.exit(close(con))
  m <- readBin(con, "integer", n = 1, size = 4, endian = "big"); stopifnot(m == 2051L)
  n <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  r <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  c <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  raw <- readBin(con, "raw", n = n * r * c)
  M   <- matrix(as.integer(raw), nrow = n, ncol = r * c, byrow = TRUE)
  storage.mode(M) <- "double"; M / 255
}
read_idx_labels <- function(path) {
  con <- gzfile(path, "rb"); on.exit(close(con))
  m <- readBin(con, "integer", n = 1, size = 4, endian = "big"); stopifnot(m == 2049L)
  n <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  as.integer(readBin(con, "raw", n = n))
}

X_all <- read_idx_images(file.path(CACHE, FILES["x"]))
y_all <- read_idx_labels(file.path(CACHE, FILES["y"]))
set.seed(seed); ii <- sample(nrow(X_all), n)
X <- X_all[ii, , drop = FALSE]; y <- y_all[ii]

label_pres <- function(Y, y, k = 10L) {
  Y_nn <- RcppHNSW::hnsw_knn(Y, k = as.integer(k + 1L), distance = "l2")$idx[, -1L, drop = FALSE]
  mean(vapply(seq_len(nrow(Y)), function(i) mean(y[Y_nn[i, ]] == y[i]), numeric(1L)))
}

cat("Fitting pacmap...\n")
t_p <- Sys.time()
emb_p <- pacmap(X, num_iters = iters, random_state = seed, verbose = FALSE)
elapsed_p <- as.numeric(difftime(Sys.time(), t_p, units = "secs"))
lp_p <- label_pres(emb_p$embedding, y)
cat(sprintf("  %.1fs, label_pres@10 = %.3f\n", elapsed_p, lp_p))

cat("Fitting localmap (low_dist_thres = 10)...\n")
t_l <- Sys.time()
emb_l <- localmap(X, num_iters = iters, random_state = seed,
                  low_dist_thres = 10, verbose = FALSE)
elapsed_l <- as.numeric(difftime(Sys.time(), t_l, units = "secs"))
lp_l <- label_pres(emb_l$embedding, y)
cat(sprintf("  %.1fs, label_pres@10 = %.3f\n", elapsed_l, lp_l))

out <- "pacmap_vs_localmap.png"
png(out, width = 1600, height = 850, res = 130)
op <- par(mfrow = c(1, 2), mar = c(3.5, 3.5, 3, 1), oma = c(0, 0, 2, 0))
palette10 <- c("#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
               "#911eb4","#42d4f4","#f032e6","#469990","#9a6324")
plot(emb_p$embedding[, 1], emb_p$embedding[, 2],
     col = palette10[y + 1L], pch = 19, cex = 0.35,
     xlab = "", ylab = "", axes = FALSE,
     main = sprintf("pacmap()  (%.1fs, label_pres = %.3f)", elapsed_p, lp_p))
box()
plot(emb_l$embedding[, 1], emb_l$embedding[, 2],
     col = palette10[y + 1L], pch = 19, cex = 0.35,
     xlab = "", ylab = "", axes = FALSE,
     main = sprintf("localmap()  (%.1fs, label_pres = %.3f)", elapsed_l, lp_l))
box()
mtext(sprintf("MNIST n = %d, same seed = %d, num_iters = c(%s)",
              n, seed, paste(iters, collapse = ",")),
      outer = TRUE, cex = 1.05)
par(op); dev.off()
cat("\nSaved", out, "\n")
