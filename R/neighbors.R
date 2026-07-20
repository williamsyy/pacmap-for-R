# ANN backend abstraction.
#   "hnsw"  -- RcppHNSW (default; CRAN-clean, works on all platforms)
#   "faiss" -- faissR (tkcaccia/faissR; UNIX ONLY, requires libfaiss)
#   "auto"  -- pick faiss if faissR is loadable, else hnsw

.knn_search <- function(X, k, distance = "euclidean",
                        backend = "hnsw", n_threads = 1L,
                        verbose = FALSE) {
  backend <- match.arg(backend, choices = c("auto", "hnsw", "faiss"))
  if (backend == "auto") {
    backend <- if (requireNamespace("faissR", quietly = TRUE)) "faiss" else "hnsw"
  }
  if (backend == "faiss") return(.knn_faissR(X, k, distance, n_threads, verbose))

  hnsw_metric <- switch(distance,
    "euclidean" = "l2",
    "angular"   = "cosine",
    # RcppHNSW has no L1/Hamming index; fall back to l2 as an approximation.
    "manhattan" = { warning("HNSW backend does not support 'manhattan'; using l2."); "l2" },
    "hamming"   = { warning("HNSW backend does not support 'hamming'; using l2."); "l2" }
  )
  if (verbose) message("Building HNSW index (metric=", hnsw_metric, ")")
  ann <- RcppHNSW::hnsw_build(X, distance = hnsw_metric,
                              M = 16L, ef = 200L,
                              n_threads = n_threads, verbose = FALSE)
  # Ask for k+1 to allow one self-hit; HNSW is approximate so self can appear
  # anywhere (or not at all) -- we filter it below rather than trusting the
  # first column.
  res <- RcppHNSW::hnsw_search(X, ann, k = as.integer(k + 1L),
                               ef = 200L, n_threads = n_threads,
                               verbose = FALSE)
  dists <- res$dist
  if (hnsw_metric == "l2")     dists <- sqrt(pmax(dists, 0))
  if (hnsw_metric == "cosine") dists <- sqrt(pmax(2 * dists, 0))

  n <- nrow(X)
  # Drop self from each row and take the top-k of what remains. Vectorized.
  idx_out  <- matrix(0L,  n, k)
  dist_out <- matrix(0.0, n, k)
  for (i in seq_len(n)) {
    keep <- res$idx[i, ] != i
    row_idx  <- res$idx[i, keep]
    row_dist <- dists[i, keep]
    if (length(row_idx) < k) {
      # HNSW under-recall: pad with -1 sentinels (rare, small n_neighbors)
      pad <- k - length(row_idx)
      row_idx  <- c(row_idx,  rep(row_idx[length(row_idx)], pad))
      row_dist <- c(row_dist, rep(row_dist[length(row_dist)], pad))
    }
    idx_out[i, ]  <- row_idx[seq_len(k)]
    dist_out[i, ] <- row_dist[seq_len(k)]
  }
  list(indices = idx_out - 1L, distances = dist_out)  # 0-indexed for C++
}

# faissR backend. faissR is Unix-only (see docs/installation.md in
# tkcaccia/faissR); on Windows the install itself refuses. We probe for the
# package here rather than at load time so pacmapr still installs cleanly
# everywhere.
.knn_faissR <- function(X, k, distance = "euclidean",
                        n_threads = 1L, verbose = FALSE) {
  if (!requireNamespace("faissR", quietly = TRUE)) {
    stop("ann_backend = 'faiss' requested but faissR is not installed. ",
         "Install with `remotes::install_github(\"tkcaccia/faissR\")` on ",
         "macOS/Linux (Unix-only; not available on Windows).")
  }
  # faissR's exact API surface: search for whichever knn function it exposes.
  ns  <- asNamespace("faissR")
  fns <- ls(ns)
  # Try likely names in order.
  knn_fn <- NULL
  for (nm in c("faiss_knn", "knn", "faiss_kNN", "faiss_knn_search",
               "kNN", "run_knn")) {
    if (nm %in% fns) { knn_fn <- get(nm, envir = ns); break }
  }
  if (is.null(knn_fn)) {
    stop("faissR is installed but no known kNN function was found. ",
         "Please open an issue with the output of `ls(asNamespace('faissR'))`.")
  }
  metric <- switch(distance,
    "euclidean" = "l2", "angular" = "ip", "manhattan" = "l1",
    { warning("faiss backend: unknown distance '", distance,
              "'; falling back to l2"); "l2" }
  )
  if (verbose) message("Calling faissR::", environmentName(environment(knn_fn)),
                       " (metric=", metric, ")")
  # faissR functions typically take (X, k) and return list(idx, dist).
  # Try common signature variants defensively.
  res <- tryCatch(knn_fn(X, k = as.integer(k + 1L), metric = metric),
                  error = function(e) knn_fn(X, X, as.integer(k + 1L)))
  # Normalize output
  idx  <- if (!is.null(res$idx))       res$idx
          else if (!is.null(res$I))    res$I
          else if (!is.null(res$knn))  res$knn
          else res[[1L]]
  dst  <- if (!is.null(res$dist))      res$dist
          else if (!is.null(res$D))    res$D
          else res[[2L]]
  # If 1-indexed, convert to 0-indexed at the drop-self step below.
  # Heuristic: FAISS returns 0-indexed; R wrappers sometimes bump to 1-indexed.
  one_indexed <- min(idx) == 1L
  n <- nrow(X)
  idx_out  <- matrix(0L,  n, k)
  dist_out <- matrix(0.0, n, k)
  for (i in seq_len(n)) {
    row_idx  <- if (one_indexed) idx[i, ] - 1L else idx[i, ]
    row_dist <- dst[i, ]
    keep <- row_idx != (i - 1L)
    row_idx <- row_idx[keep]; row_dist <- row_dist[keep]
    if (length(row_idx) < k) {  # rare under-recall
      pad <- k - length(row_idx)
      row_idx  <- c(row_idx,  rep(row_idx[length(row_idx)], pad))
      row_dist <- c(row_dist, rep(row_dist[length(row_dist)], pad))
    }
    idx_out[i, ]  <- row_idx[seq_len(k)]
    dist_out[i, ] <- row_dist[seq_len(k)]
  }
  list(indices = idx_out, distances = dist_out)
}
