.distance_to_int <- function(distance) {
  switch(distance,
    "euclidean" = 0L,
    "manhattan" = 1L,
    "angular"   = 2L,
    "hamming"   = 3L,
    stop("distance must be one of: euclidean, manhattan, angular, hamming")
  )
}

# Randomized SVD (Halko, Martinsson & Tropp 2011). For tall dense matrices
# with narrow k this is 2-3x faster than RSpectra::svds while typically
# indistinguishable in accuracy when the spectrum decays (as with any real
# data -- PCA-projected features, image pixels, expression counts, etc).
# Two power iterations tighten accuracy on flat spectra.
.randomized_svd <- function(X, k, oversampling = 10L, n_iter = 2L, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  d <- ncol(X)
  p <- as.integer(k + oversampling)
  Omega <- matrix(stats::rnorm(d * p), d, p)
  Y <- X %*% Omega                          # n x p
  for (i in seq_len(n_iter)) {
    Y <- X %*% crossprod(X, Y)              # (X %*% t(X)) %*% Y
    Y <- qr.Q(qr(Y))
  }
  Q  <- qr.Q(qr(Y))
  B  <- crossprod(Q, X)                     # p x d
  sv <- svd(B, nu = k, nv = k)
  list(u = Q %*% sv$u,
       d = sv$d[seq_len(k)],
       v = sv$v)
}

# Match Python preprocess_X: PCA-to-100 if high-dim>100 (and distance != hamming),
# else min-max scale + mean-center. Returns preprocessed X plus everything the
# transform() step needs to preprocess new data the same way.
.preprocess_X <- function(X, distance, apply_pca, n_components, verbose) {
  n <- nrow(X); high_dim <- ncol(X)
  tsvd <- NULL
  pca_solution <- FALSE

  if (distance != "hamming" && high_dim > 100L && apply_pca) {
    xmean <- colMeans(X)
    X <- sweep(X, 2L, xmean, "-")
    k <- 100L
    # Thick-restart Lanczos. On MNIST-shape data (60k x 784, k=100) this
    # beats irlba (~5x) and a hand-rolled randomized SVD (empirically
    # slower here despite the paper claim, presumably due to R's memory
    # overhead on the power iterations at large n).
    sv <- RSpectra::svds(X, k = k)
    X <- sv$u %*% diag(sv$d, k, k)
    tsvd <- list(v = sv$v, d = sv$d, k = k)
    pca_solution <- TRUE
    xmin <- 0; xmax <- 0
    if (verbose) message("Applied PCA, dimensionality reduced to ", k)
  } else {
    xmin <- min(X); X <- X - xmin
    xmax <- max(X); if (xmax > 0) X <- X / xmax
    xmean <- colMeans(X)
    X <- sweep(X, 2L, xmean, "-")
    # Init-only PCA. Prefer RSpectra (fast partial SVD) when we can, but its
    # `k` must be strictly < min(dim(X)); otherwise fall back to base svd()
    # which returns the full decomposition and lets us keep n_components cols
    # even when d == n_components.
    k <- as.integer(n_components)
    lim <- min(dim(X)) - 1L
    tsvd <- tryCatch({
      if (k <= lim) {
        sv <- RSpectra::svds(X, k = k)
        list(v = sv$v, d = sv$d, k = k)
      } else {
        sv <- svd(X, nu = 0L, nv = k)
        list(v = sv$v[, seq_len(k), drop = FALSE],
             d = sv$d[seq_len(k)], k = k)
      }
    }, error = function(e) NULL)
    if (verbose) message("X normalized (min-max + mean-center)")
  }
  list(X = X, pca_solution = pca_solution, tsvd = tsvd,
       xmin = xmin, xmax = xmax, xmean = xmean)
}

# Init the low-D embedding from PCA/SVD unless user supplied one.
.init_embedding <- function(X, n_components, init, pca_solution, tsvd, seed) {
  if (is.matrix(init)) {
    stopifnot(nrow(init) == nrow(X), ncol(init) == n_components)
    Yinit <- scale(init, center = TRUE, scale = TRUE)
    Yinit[is.na(Yinit)] <- 0
    return(Yinit * 1e-4)
  }
  if (is.null(init) || identical(init, "pca")) {
    if (pca_solution) {
      return(0.01 * X[, seq_len(n_components), drop = FALSE])
    }
    if (!is.null(tsvd)) {
      Y <- (X %*% tsvd$v) * 0.01
      return(Y[, seq_len(n_components), drop = FALSE])
    }
    # fallback
    init <- "random"
  }
  if (identical(init, "random")) {
    if (!is.null(seed)) set.seed(seed)
    return(matrix(stats::rnorm(nrow(X) * n_components) * 1e-4,
                  nrow(X), n_components))
  }
  stop("init must be one of 'pca', 'random', or a matrix")
}
