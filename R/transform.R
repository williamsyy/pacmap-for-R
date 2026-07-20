# Preprocess new data using the stats saved from the fit.
.preprocess_new <- function(X, pp) {
  if (!is.null(pp$tsvd) && isTRUE(pp$pca_solution)) {
    X <- sweep(X, 2L, pp$xmean, "-")
    return(X %*% pp$tsvd$v)
  }
  X <- X - pp$xmin
  if (pp$xmax > 0) X <- X / pp$xmax
  sweep(X, 2L, pp$xmean, "-")
}

# Sample n_neighbors basis points for each new point using the ANN backend.
# Returns pair_XP as (new_row_i_in_Y, basis_j) in 0-indexed form. new_row_i
# starts at n_basis (the new points occupy rows n_basis..(n_basis+npr-1) in the
# concatenated Y matrix).
.generate_extra_pair_basis <- function(X_new, X_basis, n_neighbors,
                                       distance, backend, n_threads) {
  npr <- nrow(X_new); nb <- nrow(X_basis)
  n_neighbors <- as.integer(min(n_neighbors, nb - 1L))

  hnsw_metric <- switch(distance,
    "euclidean" = "l2", "angular" = "cosine",
    "manhattan" = "l2", "hamming" = "l2"
  )
  ann <- RcppHNSW::hnsw_build(X_basis, distance = hnsw_metric,
                              M = 16L, ef = 200L,
                              n_threads = n_threads, verbose = FALSE)
  res <- RcppHNSW::hnsw_search(X_new, ann, k = n_neighbors,
                               ef = 200L, n_threads = n_threads,
                               verbose = FALSE)
  # Build pair_XP: for each new point i, emit n_neighbors rows
  # (nb + i, basis_j) -- so the new point sits at row (nb + i) in the
  # concatenated Y.
  new_row <- rep(seq(0L, npr - 1L) + nb, each = n_neighbors)
  basis_j <- as.integer(t(res$idx - 1L))  # transpose then flatten by row
  cbind(new_row, basis_j)
}

#' Embed new points into an existing PaCMAP model
#'
#' Preserves the model's coordinate system: new points are placed relative to
#' the frozen basis embedding, matching the Python reference's
#' \code{PaCMAP.transform()}.
#'
#' @param _data a fitted "pacmap" object (named this way to match the
#'   S3 generic \code{transform()}).
#' @param newdata numeric matrix of new points (same number of columns as the
#'   original input).
#' @param num_iters length-3 int vector (default \code{c(50, 50, 100)}, ~half
#'   of fit).
#' @param init "pca" (default), "random", or a numeric matrix
#'   \code{c(nrow(newdata), n_components)}.
#' @param n_threads threads for the ANN step.
#' @param verbose print progress.
#' @param ... unused.
#' @return numeric matrix, one row per input point.
#' @export
transform.pacmap <- function(`_data`, newdata,
                             num_iters = c(50L, 50L, 100L),
                             init = "pca",
                             n_threads = 1L,
                             verbose = FALSE, ...) {
  object <- `_data`
  if (isTRUE(object$preprocess$precomputed))
    stop("transform() is not supported when the model was fit with distance='precomputed'")
  if (!is.matrix(newdata)) newdata <- as.matrix(newdata)
  storage.mode(newdata) <- "double"
  if (ncol(newdata) != ncol(object$preprocess$X) &&
      !isTRUE(object$preprocess$pca_solution)) {
    # Raw input width should match the original raw X (before preprocessing).
    # In the non-PCA branch xmean was computed on the raw d; length(pp$xmean)
    # is the original d.
    if (ncol(newdata) != length(object$preprocess$xmean)) {
      stop("newdata has ", ncol(newdata), " columns; model expects ",
           length(object$preprocess$xmean))
    }
  }
  p <- object$params
  num_iters <- as.integer(num_iters); stopifnot(length(num_iters) == 3L)

  X_new    <- .preprocess_new(newdata, object$preprocess)
  X_basis  <- object$preprocess$X

  if (verbose) message("Building/querying ANN against basis (n_basis=",
                       nrow(X_basis), ")")
  pair_XP <- .generate_extra_pair_basis(X_new, X_basis,
                                        n_neighbors = p$n_neighbors,
                                        distance = p$distance,
                                        backend = p$ann_backend,
                                        n_threads = n_threads)

  # Init new-point embedding, then concatenate with basis embedding.
  Y_new <- .init_embedding(X_new, p$n_components, init,
                           object$preprocess$pca_solution,
                           object$preprocess$tsvd,
                           seed = p$random_state)
  Y0 <- rbind(object$embedding, Y_new)

  if (verbose) message("Optimizing (", sum(num_iters), " iters)")
  res <- pacmap_fit_optimize_cpp(Y0, pair_XP,
                                 n_basis = nrow(object$embedding),
                                 lr = p$lr, num_iters = num_iters,
                                 verbose = verbose)
  # Return only the new-point rows.
  res$embedding[(nrow(object$embedding) + 1L):nrow(res$embedding), , drop = FALSE]
}
