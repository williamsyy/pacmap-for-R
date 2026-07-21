#' Fit a LocalMAP embedding
#'
#' LocalMAP (Wang, Rudin & Shaposhnik) extends PaCMAP with a local graph-
#' adjustment stage in phase 3: it resamples further-pair partners to points
#' that are already close in the low-D embedding (within \code{low_dist_thres})
#' and multiplies the NN-term gradient by \code{low_dist_thres / (2 sqrt(d_ij))}.
#' This tightens local structure vs base PaCMAP.
#'
#' All arguments other than \code{low_dist_thres} match \code{\link{pacmap}}.
#'
#' @inheritParams pacmap
#' @param low_dist_thres low-D distance threshold used for FP resampling and
#'   as the LocalMAP NN-gradient coefficient (Wang et al. default: 10).
#' @return An S3 object of class "localmap" (also inherits "pacmap") with the
#'   same fields as \code{\link{pacmap}} plus \code{pair_FP_final} (the FPs
#'   after phase-3 resampling).
#' @export
localmap <- function(X,
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
                     n_threads    = NULL,
                     random_state = NULL,
                     low_dist_thres = 10,
                     verbose      = FALSE) {
  distance    <- match.arg(distance)
  ann_backend <- match.arg(ann_backend)
  if (is.null(n_threads)) n_threads <- max(1L, parallel::detectCores() - 1L)
  n_threads <- as.integer(n_threads)
  precomputed <- identical(distance, "precomputed")
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  n <- nrow(X)
  if (n <= 1L) stop("Sample size must be > 1")
  if (n_components < 1L) stop("n_components must be >= 1")
  if (lr <= 0) stop("lr must be > 0")
  if (low_dist_thres <= 0) stop("low_dist_thres must be > 0")
  if (precomputed) {
    if (ncol(X) != n) stop("distance='precomputed' requires a square n x n matrix")
    if (isTRUE(apply_pca)) apply_pca <- FALSE
    if (identical(init, "pca")) init <- "random"
  }
  if (length(num_iters) == 1L) num_iters <- c(100L, 100L, as.integer(num_iters))
  num_iters <- as.integer(num_iters)
  stopifnot(length(num_iters) == 3L)

  # Neighbor / pair counts (same heuristic as pacmap)
  if (is.null(n_neighbors)) {
    n_neighbors <- if (n <= 10000L) 10L else as.integer(round(10 + 15 * (log10(n) - 4)))
  }
  n_neighbors <- as.integer(min(n_neighbors, n - 1L))
  n_MN <- as.integer(round(n_neighbors * MN_ratio))
  n_FP <- as.integer(round(n_neighbors * FP_ratio))
  n_MN <- min(n_MN, n - 1L)
  n_FP <- min(n_FP, n - 1L - n_neighbors)
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
    pp <- list(X = X, pca_solution = FALSE, tsvd = NULL,
               xmin = 0, xmax = 0, xmean = rep(0, n), precomputed = TRUE)
    k_extra <- as.integer(min(n_neighbors + 50L, n - 1L))
    nn <- .knn_from_distance_matrix(X, k_extra)
    pair_nb <- sample_neighbor_pairs_cpp(nn$indices, nn$distances, n_neighbors)
    pair_MN <- .sample_mn_pairs_precomputed(X, n_MN, rs_int)
    pair_FP <- sample_fp_pairs_cpp(pair_nb, n, n_neighbors, n_FP, rs_int)
    if (!is.null(rs_int)) set.seed(rs_int)
    Y0 <- matrix(stats::rnorm(n * n_components) * 1e-4, n, n_components)
  } else {
    pp <- .preprocess_X(X, distance, apply_pca, n_components, verbose)
    pp$precomputed <- FALSE
    Xp <- pp$X
    k_extra <- as.integer(min(n_neighbors + 50L, n - 1L))
    nn <- .knn_search(Xp, k = k_extra, distance = distance,
                      backend = ann_backend, n_threads = n_threads,
                      verbose = verbose)
    pair_nb <- sample_neighbor_pairs_cpp(nn$indices, nn$distances, n_neighbors)
    dopt <- .distance_to_int(distance)
    pair_MN <- sample_mn_pairs_cpp(Xp, n_MN, dopt, rs_int)
    pair_FP <- sample_fp_pairs_cpp(pair_nb, n, n_neighbors, n_FP, rs_int)
    Y0 <- .init_embedding(Xp, n_components, init, pp$pca_solution, pp$tsvd,
                          seed = rs_int)
  }

  if (verbose) message("LocalMAP optimizing (", sum(num_iters), " iters, ",
                       n_threads, " threads, low_dist_thres=", low_dist_thres, ")")
  t0 <- Sys.time()
  res <- localmap_optimize_cpp(Y0, pair_nb, pair_MN, pair_FP, lr, num_iters,
                               low_dist_thres, n_threads, rs_int, verbose)
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  structure(
    list(
      embedding      = res$embedding,
      loss           = res$loss,
      pair_neighbors = pair_nb,
      pair_MN        = pair_MN,
      pair_FP        = pair_FP,
      pair_FP_final  = res$pair_FP_final,
      params = list(
        n_components = n_components, n_neighbors = n_neighbors,
        n_MN = n_MN, n_FP = n_FP, MN_ratio = MN_ratio, FP_ratio = FP_ratio,
        distance = distance, lr = lr, num_iters = num_iters,
        apply_pca = apply_pca, ann_backend = ann_backend,
        low_dist_thres = low_dist_thres, random_state = random_state
      ),
      preprocess = pp,
      elapsed    = elapsed
    ),
    class = c("localmap", "pacmap")
  )
}
