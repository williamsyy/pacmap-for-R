# pacmapr

Native R implementation of **PaCMAP** — *Pairwise Controlled Manifold Approximation Projection*, a fast dimensionality-reduction method that preserves both local and global structure (Wang, Huang, Rudin & Shaposhnik, *JMLR* 2021).

## Quick start

```r
# install.packages("remotes")
remotes::install_github("williamsyy/pacmap-for-R")

library(pacmapr)
X   <- as.matrix(iris[, 1:4])
emb <- pacmap(X, random_state = 42)
plot(emb$embedding, col = iris$Species, pch = 19)
```

## Installation

You need a C++17 compiler and the standard R build toolchain.

|         | one-time setup                                            |
|---------|-----------------------------------------------------------|
| Windows | `winget install RProject.Rtools`                          |
| macOS   | `xcode-select --install`                                  |
| Linux   | `sudo apt install build-essential` (or equivalent)        |

Then:

```r
remotes::install_github("williamsyy/pacmap-for-R")
```

CRAN dependencies (`Rcpp`, `RcppHNSW`, `RSpectra`) are pulled in automatically.

## Features

- **`pacmap()`** — fit a PaCMAP embedding.
- **`localmap()`** — fit a LocalMAP embedding (adds phase-3 local FP resampling + modified NN gradient; extra hyperparameter `low_dist_thres`).
- **`transform()`** — embed new points into an existing model.
- **`save_pacmap()` / `load_pacmap()`** — one-file `.rds` persistence.
- **`find_pacmap_pairs()`** — expose the pair-sampling step for inspection or reuse.
- **Distance metrics** — `euclidean`, `manhattan`, `angular`, `hamming`, and `precomputed` (pass an *n × n* distance matrix instead of a feature matrix; mirrors the [LocalMAP precomputed-distance branch](https://github.com/williamsyy/LocalMAP/tree/feature/precomputed-distance-matrix)).
- **ANN backends** — [`RcppHNSW`](https://github.com/jlmelville/rcpphnsw) by default (all platforms), or [`faissR`](https://github.com/tkcaccia/faissR) via `ann_backend = "faiss"`. `faissR`'s default install refuses on Windows (`OS_type: unix`); Windows users need one of the paths described in [faissR's installation guide](https://github.com/tkcaccia/faissR/blob/main/docs/installation.md#windows-installation-for-faissr) (WSL2 or a manual FAISS+Rtools build).
- **Deterministic** — same `random_state` → identical embedding.

## Benchmark: MNIST

Full MNIST, 70 000 images × 784 pixels → 2D:

| n         | wall-clock  | label preservation @ 10 |
|----------:|------------:|------------------------:|
|    10 000 |     7.4 s   | 0.903                   |
|    30 000 |    28.0 s   | 0.937                   |
| **70 000** | **77.4 s**  | **0.952**              |

Windows 11 laptop, 21 threads (parallel gradient + multi-threaded HNSW), default hyperparameters (`num_iters = c(100, 100, 250)`). Reproduce with

```r
Rscript inst/scripts/mnist.R
```

*Label preservation @ 10 = fraction of a point's 10 nearest neighbours in the embedding that share its true digit label; 0.952 means 9½ out of 10 on average.*

![PaCMAP embedding of MNIST (n = 70 000, 2D)](man/figures/README-mnist.png)

## PaCMAP or LocalMAP — which to use?

Both algorithms are implemented and share the same input, output, and hyperparameters. LocalMAP adds a phase-3 refinement that resamples further-pair partners restricted to points already close in the current embedding, plus a modified NN gradient. The trade-off:

|                       | `pacmap()`                                                            | `localmap()`                                                                             |
|-----------------------|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Use when**          | You want a robust, well-tested default; global + local structure both matter | Cluster boundaries matter more than continuous structure; input distances are noisy      |
| **Cluster separation** | Good                                                                  | Tighter — the local FP resampling sharpens boundaries in phase 3                          |
| **Continuous manifolds / trajectories** | Preserved smoothly                                    | May over-tighten; prefer `pacmap()`                                                       |
| **Extra hyperparameter** | —                                                                   | `low_dist_thres` (default 10; controls the phase-3 FP resampling radius)                 |
| **Cost**              | Baseline (MNIST 70k ≈ 77 s)                                          | ~10-20% slower (extra FP resampling every 10 iters in phase 3)                            |
| **Paper**             | Wang, Huang, Rudin & Shaposhnik, [*JMLR* 2021](https://www.jmlr.org/papers/v22/20-1061.html) | Wang, Sun, Huang & Rudin, [*AAAI* 2025](https://doi.org/10.1609/aaai.v39i20.35436)       |

```r
# Same call surface — swap one word.
emb_p <- pacmap(  X, random_state = 42)
emb_l <- localmap(X, random_state = 42, low_dist_thres = 10)
```

If you're unsure, start with `pacmap()`. If your embedding shows blurred cluster boundaries and cluster identification is the goal, switch to `localmap()`.

**Side-by-side on MNIST (n = 30 000, same seed):**

![pacmap() vs localmap() on MNIST (n = 30 000)](man/figures/README-pacmap-vs-localmap.png)

Both recover the ten digit classes. `localmap()`'s phase-3 refinement tightens the cluster interiors and thins the bridges between adjacent digits (e.g. 4/7/9, 3/5/8), lifting `label_preservation@10` from **0.930 → 0.943**. Reproduce with

```r
Rscript inst/scripts/pacmap_vs_localmap.R
```

## Advanced usage

### Embed new points into a fitted model

```r
train <- as.matrix(iris[1:100, 1:4])
new   <- as.matrix(iris[101:150, 1:4])

fit <- pacmap(train, random_state = 42)
tr  <- transform(fit, new)          # 50 x 2 embedding for the held-out points
```

### Precomputed distance matrix

Useful when you already have a custom distance (kernel, biological similarity, etc.):

```r
D   <- as.matrix(dist(X, method = "manhattan"))
emb <- pacmap(D, distance = "precomputed", random_state = 42)
```

`apply_pca` is disabled automatically, `init` defaults to `"random"`, and `transform()` is not supported in this mode.

### FAISS backend

An alternative ANN backend via [tkcaccia/faissR](https://github.com/tkcaccia/faissR), useful for very large *n* if you're willing to pull in a system dependency. Once `faissR` is installed:

```r
emb <- pacmap(X, ann_backend = "faiss")       # explicit
emb <- pacmap(X, ann_backend = "auto")        # faiss if installed, else hnsw
```

Installing `faissR` itself depends on your OS. The authoritative guide is [`faissR`'s installation.md](https://github.com/tkcaccia/faissR/blob/main/docs/installation.md); the short version:

**macOS:**
```sh
brew install faiss libomp
```
```r
remotes::install_github("tkcaccia/faissR")
```

**Linux:** install `libfaiss-dev` (Debian/Ubuntu: `sudo apt install libfaiss-dev`) or build FAISS from source with CMake, then `remotes::install_github("tkcaccia/faissR")`.

**Windows:** `remotes::install_github("tkcaccia/faissR")` will fail with `ERROR: Unix-only package` because faissR's `DESCRIPTION` sets `OS_type: unix` — the automated builders can't produce a Windows-compatible FAISS. Two supported workarounds, both non-trivial (see the [Windows section of the guide](https://github.com/tkcaccia/faissR/blob/main/docs/installation.md#windows-installation-for-faissr)):

1. **WSL2** (recommended) — install a Linux distribution via `wsl --install`, then follow the Linux path inside it. Your work has to happen inside WSL2, not native Windows R.
2. **Native Rtools + FAISS from source** — build FAISS with CMake using the same mingw-w64 toolchain as Rtools, set `FAISS_HOME` to the install prefix, patch/override `OS_type: unix`, then `R CMD INSTALL .` from a `faissR` checkout. Expect 1-3 hours of CMake fiddling for a first attempt.

The default HNSW backend works everywhere with zero extra deps and produces indistinguishable embedding quality; unless you specifically need FAISS's ANN implementation, you can skip this section.

### Save and reload a model

```r
save_pacmap(fit, "fit.rds")
fit2 <- load_pacmap("fit.rds")
identical(fit$embedding, fit2$embedding)      # TRUE
```

## Citation

```r
citation("pacmapr")
```

Cite the R package plus the paper for the algorithm you used:

- **`pacmap()`** — Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). *Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization.* Journal of Machine Learning Research, 22(201), 1–73. <https://www.jmlr.org/papers/v22/20-1061.html>
- **`localmap()`** — Wang, Y., Sun, Y., Huang, H., & Rudin, C. (2025). *Dimension Reduction with Locally Adjusted Graphs.* Proceedings of the AAAI Conference on Artificial Intelligence, 39(20), 21357–21365. <https://doi.org/10.1609/aaai.v39i20.35436>

## License

MIT — see [`LICENSE`](LICENSE). The package is an independent R re-implementation; the underlying algorithms are the work of the authors above.
