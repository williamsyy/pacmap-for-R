# pacmapr

Native R implementation of **PaCMAP** — *Pairwise Controlled Manifold Approximation Projection*, a fast dimensionality-reduction method that preserves both local and global structure (Wang, Huang, Rudin & Shaposhnik, *JMLR* 2021).

No Python required. No `reticulate`. Rcpp + OpenMP under the hood. The design mirrors [`uwot`](https://github.com/jlmelville/uwot), the native R port of UMAP.

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

- **`pacmap()`** — fit an embedding.
- **`transform()`** — embed new points into an existing model.
- **`save_pacmap()` / `load_pacmap()`** — one-file `.rds` persistence.
- **`find_pacmap_pairs()`** — expose the pair-sampling step for inspection or reuse.
- **Distance metrics** — `euclidean`, `manhattan`, `angular`, `hamming`, and `precomputed` (pass an *n × n* distance matrix instead of a feature matrix; mirrors the [LocalMAP precomputed-distance branch](https://github.com/williamsyy/LocalMAP/tree/feature/precomputed-distance-matrix)).
- **ANN backends** — [`RcppHNSW`](https://github.com/jlmelville/rcpphnsw) by default (all platforms), or [`faissR`](https://github.com/tkcaccia/faissR) on macOS / Linux via `ann_backend = "faiss"`.
- **Deterministic** — same `random_state` → identical embedding.

## Benchmark: MNIST

Full MNIST, 70 000 images × 784 pixels → 2D:

| n         | wall-clock  | label preservation @ 10 |
|----------:|------------:|------------------------:|
|    10 000 |     9.4 s   | 0.903                   |
|    30 000 |    36.0 s   | 0.937                   |
| **70 000** | **104.9 s** | **0.952**              |

Single-threaded on a Windows 11 laptop, default hyperparameters (`num_iters = c(100, 100, 250)`). Reproduce with

```r
Rscript inst/scripts/mnist.R
```

*Label preservation @ 10 = fraction of a point's 10 nearest neighbours in the embedding that share its true digit label; 0.952 means 9½ out of 10 on average.*

![PaCMAP embedding of MNIST (n = 70 000, 2D)](man/figures/README-mnist.png)

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

### FAISS backend (macOS / Linux)

Faster ANN for very large *n* if you're willing to pull in a system dependency:

```r
system("brew install faiss libomp")           # macOS
remotes::install_github("tkcaccia/faissR")

emb <- pacmap(X, ann_backend = "faiss")       # explicit
emb <- pacmap(X, ann_backend = "auto")        # faiss if installed, else hnsw
```

`faissR` is [Unix-only](https://github.com/tkcaccia/faissR/blob/main/docs/installation.md). On Windows the default HNSW backend is fast and gives indistinguishable results.

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

Please cite both the R package and the [original PaCMAP paper](https://www.jmlr.org/papers/v22/20-1061.html) (Wang et al., *JMLR* 2021).

## License

MIT. See [`LICENSE`](LICENSE).
