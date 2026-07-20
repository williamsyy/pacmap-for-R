#' Fit a PaCMAP embedding
#'
#' PaCMAP (Pairwise Controlled Manifold Approximation Projection) maps a
#' high-dimensional dataset to a low-dimensional embedding by simultaneously
#' optimizing over three types of pairs: nearest-neighbour, mid-near, and
#' further pairs.
#'
#' This is a native R + Rcpp implementation. No Python required.
#'
#' @param X a numeric matrix (rows = observations, cols = features).
#' @param n_components dimension of the output embedding (default 2).
#' @param n_neighbors number of nearest neighbours for the NN pair term. If
#'   \code{NULL}, uses the heuristic from Wang et al. (10 for n<=10000, else
#'   \code{round(10 + 15 * (log10(n) - 4))}).
#' @param MN_ratio ratio of mid-near pairs to NN pairs (default 0.5).
#' @param FP_ratio ratio of further pairs to NN pairs (default 2.0).
#' @param distance one of "euclidean", "manhattan", "angular", "hamming", or
#'   "precomputed". With "precomputed", \code{X} is treated as a square
#'   distance matrix (n x n); \code{apply_pca} is ignored and \code{init}
#'   defaults to "random". \code{transform()} is not supported in this mode.
#' @param lr Adam learning rate (default 1.0).
#' @param num_iters length-3 integer vector or single integer. If single, uses
#'   \code{c(100, 100, num_iters)} to mirror the Python default.
#' @param init "pca" (default), "random", or a numeric matrix of shape
#'   \code{c(nrow(X), n_components)}.
#' @param apply_pca logical, apply PCA-to-100 preprocessing when ncol(X)>100.
#' @param ann_backend ANN backend for neighbour search. Currently "hnsw".
#' @param n_threads threads for the ANN build/query step.
#' @param random_state integer seed for deterministic pair sampling. If NULL,
#'   pair sampling uses R's RNG.
#' @param verbose print progress.
#'
#' @return An S3 object of class "pacmap" with fields
#'   \code{embedding}, \code{loss}, \code{pair_neighbors}, \code{pair_MN},
#'   \code{pair_FP}, and \code{params}.
#'
#' @examples
#' \dontrun{
#'   emb <- pacmap(as.matrix(iris[, 1:4]), random_state = 42L)
#'   plot(emb$embedding, col = iris$Species)
#' }
#' @export
pacmap <- function(X,
                   n_components = 2L,
                   n_neighbors  = NULL,
                   MN_ratio     = 0.5,
                   FP_ratio     = 2.0,
                   distance     = c("euclidean", "manhattan", "angular", "hamming", "precomputed"),
                   lr           = 1.0,
                   num_iters    = c(100L, 100L, 250L),
                   init         = "pca",
                   apply_pca    = TRUE,
                   ann_backend  = c("hnsw", "faiss", "auto"),
                   n_threads    = 1L,
                   random_state = NULL,
                   verbose      = FALSE) {

  distance <- match.arg(distance)
  ann_backend <- match.arg(ann_backend)
  precomputed <- identical(distance, "precomputed")
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  n <- nrow(X)
  if (n <= 1L) stop("Sample size must be > 1")
  if (n_components < 1L) stop("n_components must be >= 1")
  if (lr <= 0) stop("lr must be > 0")
  if (precomputed) {
    if (ncol(X) != n) stop("distance='precomputed' requires a square n x n matrix, got ", n, " x ", ncol(X))
    if (isTRUE(apply_pca))  # silently disable (matches Python)
      apply_pca <- FALSE
    if (identical(init, "pca")) init <- "random"
  }

  if (length(num_iters) == 1L) num_iters <- c(100L, 100L, as.integer(num_iters))
  num_iters <- as.integer(num_iters)
  stopifnot(length(num_iters) == 3L)

  # Neighbor / pair counts ---------------------------------------------------
  if (is.null(n_neighbors)) {
    n_neighbors <- if (n <= 10000L) 10L else as.integer(round(10 + 15 * (log10(n) - 4)))
  }
  n_neighbors <- as.integer(min(n_neighbors, n - 1L))
  n_MN        <- as.integer(round(n_neighbors * MN_ratio))
  n_FP        <- as.integer(round(n_neighbors * FP_ratio))
  n_MN        <- min(n_MN, n - 1L)
  n_FP        <- min(n_FP, n - 1L - n_neighbors)
  if (n_neighbors < 1L) stop("n_neighbors < 1")
  if (n_FP        < 1L) stop("n_FP < 1")
  if (n_neighbors + n_MN + n_FP >= n) {
    r <- 1 + MN_ratio + FP_ratio
    n_neighbors <- as.integer(n / r)
    n_MN        <- as.integer(n / r * MN_ratio)
    n_FP        <- as.integer(n / r * FP_ratio)
  }

  rs_int <- if (is.null(random_state)) NULL else as.integer(random_state)

  if (precomputed) {
    # Precomputed distance path: X is the distance matrix; no preprocessing,
    # no ANN, MN sampling looks up distances directly.
    if (verbose) message("Using precomputed distance matrix (n=", n, ")")
    pp <- list(X = X, pca_solution = FALSE, tsvd = NULL,
               xmin = 0, xmax = 0, xmean = rep(0, n),
               precomputed = TRUE)
    k_extra <- as.integer(min(n_neighbors + 50L, n - 1L))
    nn <- .knn_from_distance_matrix(X, k_extra)
    pair_nb <- sample_neighbor_pairs_cpp(nn$indices, nn$distances, n_neighbors)
    pair_MN <- .sample_mn_pairs_precomputed(X, n_MN, rs_int)
    pair_FP <- sample_fp_pairs_cpp(pair_nb, n, n_neighbors, n_FP, rs_int)
    # No features -> no PCA init available; random.
    if (!is.null(rs_int)) set.seed(rs_int)
    Y0 <- matrix(stats::rnorm(n * n_components) * 1e-4, n, n_components)
    if (is.matrix(init)) {
      stopifnot(nrow(init) == n, ncol(init) == n_components)
      Y0 <- scale(init, center = TRUE, scale = TRUE) * 1e-4
    }
  } else {
    # Preprocess ------------------------------------------------------------
    if (verbose) message("Preprocessing X")
    pp <- .preprocess_X(X, distance, apply_pca, n_components, verbose)
    pp$precomputed <- FALSE
    Xp <- pp$X

    # Neighbor search -------------------------------------------------------
    k_extra <- as.integer(min(n_neighbors + 50L, n - 1L))
    if (verbose) message("Neighbor search (k=", k_extra, ")")
    nn <- .knn_search(Xp, k = k_extra, distance = distance,
                      backend = ann_backend, n_threads = n_threads,
                      verbose = verbose)

    # Pair sampling ---------------------------------------------------------
    if (verbose) message("Sampling ", n_neighbors, " NN + ", n_MN, " MN + ", n_FP, " FP pairs")
    pair_nb <- sample_neighbor_pairs_cpp(nn$indices, nn$distances, n_neighbors)
    dopt <- .distance_to_int(distance)
    pair_MN <- sample_mn_pairs_cpp(Xp, n_MN, dopt, rs_int)
    pair_FP <- sample_fp_pairs_cpp(pair_nb, n, n_neighbors, n_FP, rs_int)

    Y0 <- .init_embedding(Xp, n_components, init, pp$pca_solution, pp$tsvd,
                          seed = rs_int)
  }

  if (verbose) message("Optimizing (", sum(num_iters), " iters)")
  t0 <- Sys.time()
  res <- pacmap_optimize_cpp(Y0, pair_nb, pair_MN, pair_FP, lr, num_iters, verbose)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (verbose) message(sprintf("Optimization done in %.2fs", elapsed))

  structure(
    list(
      embedding      = res$embedding,
      loss           = res$loss,
      pair_neighbors = pair_nb,
      pair_MN        = pair_MN,
      pair_FP        = pair_FP,
      params = list(
        n_components = n_components, n_neighbors = n_neighbors,
        n_MN = n_MN, n_FP = n_FP, MN_ratio = MN_ratio, FP_ratio = FP_ratio,
        distance = distance, lr = lr, num_iters = num_iters,
        apply_pca = apply_pca, ann_backend = ann_backend,
        random_state = random_state
      ),
      preprocess = pp,
      elapsed    = elapsed
    ),
    class = "pacmap"
  )
}

#' Sample PaCMAP pairs without optimizing
#'
#' Exposes the pair-sampling step so users can reuse pairs across runs or
#' inspect them. Returns a list with matrices \code{pair_neighbors},
#' \code{pair_MN}, \code{pair_FP} (each 2-column, 0-indexed into X's rows).
#'
#' @inheritParams pacmap
#' @export
find_pacmap_pairs <- function(X,
                              n_neighbors  = 10L,
                              MN_ratio     = 0.5,
                              FP_ratio     = 2.0,
                              distance     = "euclidean",
                              ann_backend  = c("hnsw", "faiss", "auto"),
                              n_threads    = 1L,
                              random_state = NULL) {
  ann_backend <- match.arg(ann_backend)
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  n <- nrow(X)
  n_neighbors <- as.integer(min(n_neighbors, n - 1L))
  n_MN <- as.integer(round(n_neighbors * MN_ratio))
  n_FP <- as.integer(round(n_neighbors * FP_ratio))
  k_extra <- as.integer(min(n_neighbors + 50L, n - 1L))
  nn <- .knn_search(X, k = k_extra, distance = distance,
                    backend = ann_backend, n_threads = n_threads)
  pair_nb <- sample_neighbor_pairs_cpp(nn$indices, nn$distances, n_neighbors)
  dopt <- .distance_to_int(distance)
  rs_int <- if (is.null(random_state)) NULL else as.integer(random_state)
  pair_MN <- sample_mn_pairs_cpp(X, n_MN, dopt, rs_int)
  pair_FP <- sample_fp_pairs_cpp(pair_nb, n, n_neighbors, n_FP, rs_int)
  list(pair_neighbors = pair_nb, pair_MN = pair_MN, pair_FP = pair_FP)
}

#' @export
print.pacmap <- function(x, ...) {
  p <- x$params
  cat("<pacmap embedding>\n")
  cat(sprintf("  n = %d, d_in preproc = %d, d_out = %d\n",
              nrow(x$embedding), ncol(x$preprocess$X), p$n_components))
  cat(sprintf("  n_neighbors=%d, n_MN=%d, n_FP=%d, distance=%s\n",
              p$n_neighbors, p$n_MN, p$n_FP, p$distance))
  cat(sprintf("  iters=(%s), lr=%.3g, backend=%s\n",
              paste(p$num_iters, collapse = ","), p$lr, p$ann_backend))
  cat(sprintf("  final loss = %.4f, elapsed = %.2fs\n",
              utils::tail(x$loss, 1), x$elapsed))
  invisible(x)
}
