# pacmapr

Native R port of [PaCMAP](https://github.com/YingfanWang/PaCMAP)
(Pairwise Controlled Manifold Approximation Projection) — no Python
dependency, no `reticulate`. Layout modelled on [`uwot`](https://github.com/jlmelville/uwot).

## Status

**Working (39 tests, 107 assertions, 0 failures; faissR test skipped on Windows):**

- `pacmap()` — fit an embedding (equivalent of Python's `fit_transform`)
- `transform()` — embed new points into an existing model (S3 method)
- `save_pacmap()` / `load_pacmap()` — round-trip to a single .rds
- `find_pacmap_pairs()` — expose pair sampling for inspection/reuse
- **5 distance modes**: euclidean, manhattan, angular, hamming, **`precomputed`**
  (pass an n×n distance matrix as `X`; mirrors
  [`williamsyy/LocalMAP@feature/precomputed-distance-matrix`](https://github.com/williamsyy/LocalMAP/tree/feature/precomputed-distance-matrix))
- PCA-to-100 preprocessing when `ncol(X) > 100`
- PCA / random / user-matrix initialization
- Adam optimizer with the 3-phase weight schedule
- Deterministic pair sampling via `random_state` — exact reproducibility (max|a-b| = 0)
- **ANN backends** via `ann_backend =`:
  - `"hnsw"` (default) — RcppHNSW, works everywhere
  - `"faiss"` — [tkcaccia/faissR](https://github.com/tkcaccia/faissR), **Unix-only**
    (macOS / Linux); Windows install errors out with `Unix-only package`.
    Install with `remotes::install_github("tkcaccia/faissR")` after
    `brew install faiss libomp` (macOS) or an equivalent Linux install.
  - `"auto"` — faiss if available, else hnsw

**Deferred:**

- LocalMAP variant (subclass with `sample_FP_nearby`)
- Parallel gradient with thread-local buffers (currently serial gradient
  + parallel Adam; fast enough for ≤ 100k in benchmark)

## Speed

Wall-clock on 8-core Windows laptop (blobs, 5 clusters, seed=42, num_iters=(100,100,250)):

| n     | d  | time  | label_pres |
|-------|----|-------|------------|
| 500   | 20 | 1.1s  | 1.000      |
| 2000  | 20 | 0.5s  | 1.000      |
| 5000  | 20 | 1.5s  | 1.000      |
| 10000 | 20 | 3.7s  | 1.000      |
| 2000  | 500| 1.7s  | 1.000      |

`label_pres` = fraction of a point's 10 embedding-neighbors that share its true cluster label. 1.0 = perfect cluster recovery.

## Install

Modelled on `uwot`'s install flow. Pick whichever line fits your situation:

```r
# 1. From GitHub (once the repo is public):
remotes::install_github("YiyangSun/pacmapr")

# 2. From a local tarball (offline / air-gapped):
install.packages("pacmapr_0.1.0.tar.gz", repos = NULL, type = "source")

# 3. From a source checkout:
install.packages("pacmapr", repos = NULL, type = "source")

# 4. (Future) from CRAN once published:
install.packages("pacmapr")
```

**Prerequisites.** A working C++17 compiler.
- **Windows:** [Rtools 4.5](https://cran.r-project.org/bin/windows/Rtools/) — `winget install RProject.Rtools`
- **macOS:** Xcode command-line tools — `xcode-select --install`
- **Linux:** `build-essential` (Debian/Ubuntu) or equivalent

CRAN dependencies (`Rcpp`, `RcppHNSW`, `RSpectra`) are pulled in automatically by any of the four commands above.

**Verified on Windows 11 + R 4.5.1 + Rtools 4.5:**
- Tarball install: ~8s
- Fresh-session load + `pacmap()` + `transform()` + save/load round-trip: works out of the box
- `R CMD check --as-cran`: **0 ERRORs, 0 WARNINGs, 1 NOTE** (the standard "new submission" + optional-dep note for the GitHub-only FAISS backend)

**Optional FAISS backend (macOS / Linux only):**
```r
# macOS
system("brew install faiss libomp")
remotes::install_github("tkcaccia/faissR")
# then in R: pacmap(X, ann_backend = "faiss")
```
On Windows, `faissR` refuses to install (`Unix-only package`); the default HNSW backend works fine and produces identical embeddings.

## Usage

```r
library(pacmapr)

# Fit on a feature matrix
emb <- pacmap(as.matrix(iris[, 1:4]), n_components = 2L, random_state = 42L)
plot(emb$embedding, col = iris$Species, pch = 19)

# Fit on a precomputed distance matrix (new)
D <- as.matrix(dist(iris[, 1:4]))
emb2 <- pacmap(D, distance = "precomputed", random_state = 42L)

# Use FAISS if you're on macOS/Linux and want a bigger index
emb3 <- pacmap(X, ann_backend = "faiss")     # Unix only
emb3 <- pacmap(X, ann_backend = "auto")      # faiss if present, else hnsw
```

## Design notes

- **ANN backend.** HNSW via `RcppHNSW` is the default (CRAN-clean, fast). The
  Python reference defaults to FAISS; no maintained CRAN binding exists yet,
  so FAISS is deferred. The `.knn_search()` dispatcher in `R/neighbors.R`
  matches the pattern used by `uwot`, so adding backends is local.
- **Gradient parallelism.** The gradient is computed serially; the Adam
  update is parallelized with OpenMP. The Python `pacmap_grad` uses
  `numba.prange` and races on `grad[i]/grad[j]` — it works out because
  float32 races on well-conditioned arithmetic rarely blow up, but we avoid
  the correctness question here. If profiling shows this as the bottleneck,
  the next step is thread-local grad buffers with a reduction.
- **Determinism.** Python seeds NumPy per-point (`np.random.seed(base + i*n_MN + j)`).
  We reproduce the same per-point stream pattern with an LCG; the *values*
  won't match Python exactly (different RNGs), but seeded runs in R are
  reproducible.
```
