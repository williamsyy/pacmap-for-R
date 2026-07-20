#!/usr/bin/env Rscript
# Synthetic-blobs example. Shows that PaCMAP recovers cluster structure from a
# high-dim mixture of Gaussians.

library(pacmapr)

set.seed(1)
n_per <- 200L
k     <- 5L
d     <- 50L
centers <- matrix(rnorm(k * d, sd = 4), nrow = k)
X <- do.call(rbind, lapply(seq_len(k), function(i) {
  matrix(rnorm(n_per * d, sd = 0.7), n_per, d) +
    matrix(centers[i, ], n_per, d, byrow = TRUE)
}))
y <- rep(seq_len(k), each = n_per)

t0 <- Sys.time()
emb <- pacmap(X, num_iters = c(100L, 100L, 250L), random_state = 42L, verbose = TRUE)
cat(sprintf("Total elapsed: %.2fs\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

png("pacmap_blobs.png", width = 700, height = 700)
plot(emb$embedding, col = y, pch = 19, cex = 0.6,
     xlab = "PaCMAP 1", ylab = "PaCMAP 2",
     main = sprintf("PaCMAP of %d blobs in %d-D", k, d))
dev.off()
cat("Saved pacmap_blobs.png\n")
