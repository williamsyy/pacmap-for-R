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
