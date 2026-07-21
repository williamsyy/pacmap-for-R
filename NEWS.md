# pacmapr 0.1.1

## New

- **`localmap()`** — LocalMAP variant (Wang, Rudin & Shaposhnik). Same
  pipeline as `pacmap()` for phases 1-2; phase 3 resamples further-pair
  partners restricted to points within `low_dist_thres` in the current
  embedding and switches the NN-term gradient to the LocalMAP form
  `w * NN_coef_recip / sqrt(d_ij)`. New `low_dist_thres` hyperparameter
  (default 10).

## Speed

- **Parallel gradient** — the 3-term gradient loop now runs over OpenMP
  threads with per-thread grad buffers reduced at the end. Same math as
  serial (mod float-add ordering, ~1e-12 differences).
- **Multi-threaded HNSW** — `n_threads` now defaults to
  `parallel::detectCores() - 1L` (was `1L`) across `pacmap()`,
  `localmap()`, `transform.pacmap()`, and `find_pacmap_pairs()`.
- **Full MNIST 70k benchmark:** 105s → **77s** (1.36× vs 0.1.0), same
  embedding quality (`label_preservation@10 = 0.952`).

## Fixed

- `transform.pacmap()` first argument renamed to `` `_data` `` to match the
  S3 generic (was tripping the `checking S3 generic/method consistency`
  WARNING in `R CMD check`).

# pacmapr 0.1.0

Initial release. Native R + Rcpp port of
[PaCMAP](https://github.com/YingfanWang/PaCMAP)
(Pairwise Controlled Manifold Approximation Projection), following the
architectural pattern of [`uwot`](https://github.com/jlmelville/uwot). No
Python dependency, no `reticulate`.

## Features

- `pacmap()` — fit an embedding (equivalent of Python's `fit_transform`).
- `transform()` (S3 method) — embed new points into an existing model.
- `save_pacmap()` / `load_pacmap()` — one-file `.rds` round-trip.
- `find_pacmap_pairs()` — expose the pair-sampling step for inspection or reuse.
- Distances: `euclidean`, `manhattan`, `angular`, `hamming`, and
  `precomputed` (pass an `n x n` distance matrix as `X`; mirrors
  [`williamsyy/LocalMAP@feature/precomputed-distance-matrix`](https://github.com/williamsyy/LocalMAP/tree/feature/precomputed-distance-matrix)).
- ANN backends via `ann_backend =`:
  - `"hnsw"` (default) via `RcppHNSW` — works everywhere.
  - `"faiss"` via [`faissR`](https://github.com/tkcaccia/faissR) — Unix only.
  - `"auto"` — faiss if installed, else hnsw.
- Adam optimizer with the 3-phase weight schedule from Wang et al.
- Exact reproducibility via `random_state` (identical embedding across runs).

## Verified

- 39 testthat tests, 107 assertions, all pass (1 skipped when faissR absent).
- `R CMD check --as-cran`: **0 ERRORs, 0 WARNINGs, 1 NOTE** (the standard
  "new submission" + optional-dep note for faissR).
- Full benchmark: `label_preservation = 1.000` on synthetic blobs at
  `n = 500 ... 10000`.
