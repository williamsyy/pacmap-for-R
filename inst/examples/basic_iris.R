#!/usr/bin/env Rscript
# Minimal pacmapr example -- iris.

library(pacmapr)

X <- as.matrix(iris[, 1:4])
y <- iris$Species

emb <- pacmap(X, n_components = 2L, random_state = 42L, verbose = TRUE)
print(emb)

png("pacmap_iris.png", width = 600, height = 600)
plot(emb$embedding, col = as.integer(y), pch = 19,
     xlab = "PaCMAP 1", ylab = "PaCMAP 2",
     main = "PaCMAP embedding of iris")
legend("topright", legend = levels(y), col = seq_along(levels(y)), pch = 19)
dev.off()
cat("Saved pacmap_iris.png\n")
