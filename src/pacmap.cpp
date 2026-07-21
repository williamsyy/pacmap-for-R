// PaCMAP core: pair sampling, gradient, and Adam optimizer.
// Port of source/pacmap/pacmap.py from Wang et al.'s reference implementation.

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ---------- distance ----------

enum DistanceKind { DIST_EUCLIDEAN = 0, DIST_MANHATTAN = 1, DIST_ANGULAR = 2, DIST_HAMMING = 3 };

static inline double euclid_dist(const double* a, const double* b, int d) {
  double s = 0.0;
  for (int k = 0; k < d; ++k) { double diff = a[k] - b[k]; s += diff * diff; }
  return std::sqrt(s);
}
static inline double manhattan_dist(const double* a, const double* b, int d) {
  double s = 0.0;
  for (int k = 0; k < d; ++k) s += std::abs(a[k] - b[k]);
  return s;
}
static inline double angular_dist(const double* a, const double* b, int d) {
  double na = 0.0, nb = 0.0, dot = 0.0;
  for (int k = 0; k < d; ++k) { na += a[k]*a[k]; nb += b[k]*b[k]; dot += a[k]*b[k]; }
  na = std::max(std::sqrt(na), 1e-20); nb = std::max(std::sqrt(nb), 1e-20);
  double v = 2.0 - 2.0 * dot / (na * nb);
  return std::sqrt(std::max(v, 0.0));
}
static inline double hamming_dist(const double* a, const double* b, int d) {
  double s = 0.0;
  for (int k = 0; k < d; ++k) if (a[k] != b[k]) s += 1.0;
  return s;
}
static inline double dist_by_kind(const double* a, const double* b, int d, int kind) {
  switch (kind) {
    case DIST_EUCLIDEAN: return euclid_dist(a, b, d);
    case DIST_MANHATTAN: return manhattan_dist(a, b, d);
    case DIST_ANGULAR:   return angular_dist(a, b, d);
    case DIST_HAMMING:   return hamming_dist(a, b, d);
    default:             return euclid_dist(a, b, d);
  }
}

// ---------- RNG (deterministic per-point, mirroring Python's np.random.seed pattern) ----------

// Simple LCG for reproducible per-point sampling. Uses a base seed so different
// (base, offset) combinations produce different independent streams -- the
// same trick pacmap.py's deterministic variants use.
struct LCG {
  uint64_t state;
  LCG(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  inline uint32_t next_u32() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state >> 32);
  }
  inline int32_t next_below(int32_t bound) {
    return static_cast<int32_t>(next_u32() % static_cast<uint32_t>(bound));
  }
};

// ---------- sample_FP: sample n_samples from [0,maximum) excluding self_ind and reject_ind ----------

static void sample_fp_indices(int n_samples, int maximum,
                              const int* reject_ind, int n_reject,
                              int self_ind, int* out, LCG& rng) {
  for (int i = 0; i < n_samples; ++i) {
    while (true) {
      int j = rng.next_below(maximum);
      if (j == self_ind) continue;
      bool dup = false;
      for (int k = 0; k < i; ++k) if (out[k] == j) { dup = true; break; }
      if (dup) continue;
      bool rejected = false;
      for (int k = 0; k < n_reject; ++k) if (reject_ind[k] == j) { rejected = true; break; }
      if (rejected) continue;
      out[i] = j;
      break;
    }
  }
}

// ---------- neighbor-pair sampling with sigma-scaled distances ----------
// Input: nbrs (n x n_neighbors_extra), knn_distances (same shape).
// Output: pair_neighbors (n*n_neighbors x 2), each row (i, nbrs[i, sortIdx[j]]).

// [[Rcpp::export]]
IntegerMatrix sample_neighbor_pairs_cpp(IntegerMatrix nbrs, NumericMatrix knn_distances,
                                        int n_neighbors) {
  int n = nbrs.nrow();
  int k_extra = nbrs.ncol();

  // sig[i] = mean(knn_distances[i, 3:6])  (Python uses columns 3..5, i.e. 3rd..5th nearest)
  std::vector<double> sig(n);
  for (int i = 0; i < n; ++i) {
    double s = 0.0; int cnt = 0;
    for (int c = 3; c < 6 && c < k_extra; ++c) { s += knn_distances(i, c); ++cnt; }
    sig[i] = std::max(cnt > 0 ? s / cnt : 1e-10, 1e-10);
  }

  IntegerMatrix pair_neighbors(n * n_neighbors, 2);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < n; ++i) {
    // Compute scaled distances for row i.
    std::vector<std::pair<double,int>> scored(k_extra);
    for (int c = 0; c < k_extra; ++c) {
      int nb = nbrs(i, c);
      double d = knn_distances(i, c);
      double sd = (d * d) / (sig[i] * sig[nb]);
      scored[c] = {sd, c};
    }
    std::sort(scored.begin(), scored.end(),
              [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                return a.first < b.first;
              });
    for (int j = 0; j < n_neighbors; ++j) {
      pair_neighbors(i * n_neighbors + j, 0) = i;
      pair_neighbors(i * n_neighbors + j, 1) = nbrs(i, scored[j].second);
    }
  }
  return pair_neighbors;
}

// ---------- mid-near pair sampling ----------
// For each i: sample 6 candidates, keep the 2nd-closest (Python: drop argmin, then argmin remaining).

// [[Rcpp::export]]
IntegerMatrix sample_mn_pairs_cpp(NumericMatrix X, int n_MN, int distance_kind,
                                  Nullable<int> random_state = R_NilValue) {
  int n = X.nrow();
  int d = X.ncol();
  bool det = random_state.isNotNull();
  int base_seed = det ? as<int>(random_state) : 0;

  IntegerMatrix pair_MN(n * n_MN, 2);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(!det)
  #endif
  for (int i = 0; i < n; ++i) {
    std::vector<int> reject_buf;
    reject_buf.reserve(n_MN);
    for (int j = 0; j < n_MN; ++j) {
      LCG rng(det ? static_cast<uint64_t>(base_seed + i * n_MN + j + 1)
                  : static_cast<uint64_t>((uint32_t)R::runif(0, 1e9) + 1));
      int sampled[6];
      sample_fp_indices(6, n, reject_buf.empty() ? nullptr : reject_buf.data(),
                        static_cast<int>(reject_buf.size()), i, sampled, rng);
      double dists[6];
      for (int t = 0; t < 6; ++t) {
        const double* xi = &X(i, 0);
        const double* xs = &X(sampled[t], 0);
        dists[t] = dist_by_kind(xi, xs, d, distance_kind);
      }
      // Drop the argmin, then argmin of what remains -> the 2nd-closest.
      int argmin = 0;
      for (int t = 1; t < 6; ++t) if (dists[t] < dists[argmin]) argmin = t;
      int second = -1; double best = std::numeric_limits<double>::infinity();
      for (int t = 0; t < 6; ++t) {
        if (t == argmin) continue;
        if (dists[t] < best) { best = dists[t]; second = t; }
      }
      int picked = sampled[second];
      pair_MN(i * n_MN + j, 0) = i;
      pair_MN(i * n_MN + j, 1) = picked;
      reject_buf.push_back(picked);
    }
  }
  return pair_MN;
}

// ---------- further pair sampling ----------

// [[Rcpp::export]]
IntegerMatrix sample_fp_pairs_cpp(IntegerMatrix pair_neighbors, int n_points,
                                  int n_neighbors, int n_FP,
                                  Nullable<int> random_state = R_NilValue) {
  int n = n_points;
  bool det = random_state.isNotNull();
  int base_seed = det ? as<int>(random_state) : 0;

  IntegerMatrix pair_FP(n * n_FP, 2);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(!det)
  #endif
  for (int i = 0; i < n; ++i) {
    // Reject the neighbor indices of point i.
    std::vector<int> reject(n_neighbors);
    for (int k = 0; k < n_neighbors; ++k) reject[k] = pair_neighbors(i * n_neighbors + k, 1);

    LCG rng(det ? static_cast<uint64_t>(base_seed + i + 1)
                : static_cast<uint64_t>((uint32_t)R::runif(0, 1e9) + 1));
    std::vector<int> out(n_FP);
    sample_fp_indices(n_FP, n, reject.data(), n_neighbors, i, out.data(), rng);
    for (int k = 0; k < n_FP; ++k) {
      pair_FP(i * n_FP + k, 0) = i;
      pair_FP(i * n_FP + k, 1) = out[k];
    }
  }
  return pair_FP;
}

// ---------- weight schedule ----------

static inline void find_weight(double w_MN_init, int itr,
                               int phase1, int phase2,
                               double& w_MN, double& w_neighbors, double& w_FP) {
  if (itr < phase1) {
    double t = static_cast<double>(itr) / phase1;
    w_MN = (1.0 - t) * w_MN_init + t * 3.0;
    w_neighbors = 2.0;
    w_FP = 1.0;
  } else if (itr < phase1 + phase2) {
    w_MN = 3.0;
    w_neighbors = 3.0;
    w_FP = 1.0;
  } else {
    w_MN = 0.0;
    w_neighbors = 1.0;
    w_FP = 1.0;
  }
}

// ---------- gradient ----------
// Each OpenMP thread accumulates into its own row-major grad buffer, then
// reduces at the end. Same math as the serial version modulo float-add
// ordering (~1e-12 differences on typical inputs; runs are still fully
// deterministic for a fixed n_threads).

static double pacmap_grad_parallel(const NumericMatrix& Y,
                                   const IntegerMatrix& pair_nb,
                                   const IntegerMatrix& pair_MN,
                                   const IntegerMatrix& pair_FP,
                                   double w_nb, double w_mn, double w_fp,
                                   int n_threads,
                                   NumericMatrix& grad) {
  const int n   = Y.nrow();
  const int dim = Y.ncol();
  // zero grad up front
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) grad(i, d) = 0.0;

  double loss = 0.0;
  const int nnp = pair_nb.nrow();
  const int nmn = pair_MN.nrow();
  const int nfp = pair_FP.nrow();

#ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads) reduction(+:loss)
#endif
  {
    // Per-thread row-major grad buffer.
    std::vector<double> lgrad(static_cast<size_t>(n) * dim, 0.0);
    std::vector<double> yij(dim);

    // NN attractive term: L = w_nb * d/(10+d)
#ifdef _OPENMP
    #pragma omp for schedule(static) nowait
#endif
    for (int t = 0; t < nnp; ++t) {
      const int i = pair_nb(t, 0), j = pair_nb(t, 1);
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_nb * (d_ij / (10.0 + d_ij));
      const double w1 = w_nb * (20.0 / std::pow(10.0 + d_ij, 2));
      const size_t ib = static_cast<size_t>(i) * dim;
      const size_t jb = static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        lgrad[ib + d] += w1 * yij[d];
        lgrad[jb + d] -= w1 * yij[d];
      }
    }
    // MN attractive term
#ifdef _OPENMP
    #pragma omp for schedule(static) nowait
#endif
    for (int t = 0; t < nmn; ++t) {
      const int i = pair_MN(t, 0), j = pair_MN(t, 1);
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_mn * (d_ij / (10000.0 + d_ij));
      const double w1 = w_mn * (20000.0 / std::pow(10000.0 + d_ij, 2));
      const size_t ib = static_cast<size_t>(i) * dim;
      const size_t jb = static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        lgrad[ib + d] += w1 * yij[d];
        lgrad[jb + d] -= w1 * yij[d];
      }
    }
    // FP repulsive term
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int t = 0; t < nfp; ++t) {
      const int i = pair_FP(t, 0), j = pair_FP(t, 1);
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_fp * (1.0 / (1.0 + d_ij));
      const double w1 = w_fp * (2.0 / std::pow(1.0 + d_ij, 2));
      const size_t ib = static_cast<size_t>(i) * dim;
      const size_t jb = static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        lgrad[ib + d] -= w1 * yij[d];
        lgrad[jb + d] += w1 * yij[d];
      }
    }

    // Reduce this thread's lgrad into the shared grad. Rcpp NumericMatrix is
    // column-major, so we index (i, d) rather than a flat offset.
#ifdef _OPENMP
    #pragma omp critical
#endif
    for (int i = 0; i < n; ++i) {
      const size_t ib = static_cast<size_t>(i) * dim;
      for (int d = 0; d < dim; ++d) grad(i, d) += lgrad[ib + d];
    }
  }
  return loss;
}

// ---------- gradient for transform/fit: only updates row i (new points) ----------
// pair_XP holds (new_i, basis_j) pairs. new_i indexes into Y starting at n_basis.
// Only the attractive NN-style term is used (see pacmap_grad_fit in pacmap.py).

static double pacmap_grad_fit_parallel(const NumericMatrix& Y,
                                       const IntegerMatrix& pair_XP,
                                       double w_nb,
                                       int n_basis,
                                       int n_threads,
                                       NumericMatrix& grad) {
  const int n = Y.nrow();
  const int dim = Y.ncol();
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) grad(i, d) = 0.0;
  double loss = 0.0;
  const int npairs = pair_XP.nrow();

#ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads) reduction(+:loss)
#endif
  {
    std::vector<double> lgrad(static_cast<size_t>(n) * dim, 0.0);
    std::vector<double> yij(dim);
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int t = 0; t < npairs; ++t) {
      const int i = pair_XP(t, 0), j = pair_XP(t, 1);
      if (i < n_basis) continue;  // safety: only new points move
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_nb * (d_ij / (10.0 + d_ij));
      const double w1 = w_nb * (20.0 / std::pow(10.0 + d_ij, 2));
      const size_t ib = static_cast<size_t>(i) * dim;
      for (int d = 0; d < dim; ++d) lgrad[ib + d] += w1 * yij[d];
    }
#ifdef _OPENMP
    #pragma omp critical
#endif
    for (int i = 0; i < n; ++i) {
      const size_t ib = static_cast<size_t>(i) * dim;
      for (int d = 0; d < dim; ++d) grad(i, d) += lgrad[ib + d];
    }
  }
  return loss;
}

// ---------- Adam step (safe to parallelize over rows) ----------

static void adam_step(NumericMatrix& Y, const NumericMatrix& grad,
                      NumericMatrix& m, NumericMatrix& v,
                      double beta1, double beta2, double lr, int itr) {
  int n = Y.nrow(), dim = Y.ncol();
  double lr_t = lr * std::sqrt(1.0 - std::pow(beta2, itr + 1)) /
                (1.0 - std::pow(beta1, itr + 1));
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < dim; ++d) {
      m(i, d) += (1 - beta1) * (grad(i, d) - m(i, d));
      v(i, d) += (1 - beta2) * (grad(i, d) * grad(i, d) - v(i, d));
      Y(i, d) -= lr_t * m(i, d) / (std::sqrt(v(i, d)) + 1e-7);
    }
  }
}

// ---------- main optimization loop ----------

// [[Rcpp::export]]
List pacmap_optimize_cpp(NumericMatrix Y_init,
                         IntegerMatrix pair_nb,
                         IntegerMatrix pair_MN,
                         IntegerMatrix pair_FP,
                         double lr,
                         IntegerVector num_iters,   // length-3: phase1, phase2, phase3
                         int n_threads,
                         bool verbose) {
  if (num_iters.size() != 3) stop("num_iters must have length 3");
  if (n_threads < 1) n_threads = 1;
  int phase1 = num_iters[0], phase2 = num_iters[1], phase3 = num_iters[2];
  int total = phase1 + phase2 + phase3;

  int n = Y_init.nrow(), dim = Y_init.ncol();
  NumericMatrix Y(n, dim);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) Y(i, d) = Y_init(i, d);

  NumericMatrix grad(n, dim), m(n, dim), v(n, dim);
  const double w_MN_init = 1000.0;
  const double beta1 = 0.9, beta2 = 0.999;

  NumericVector losses(total);

  for (int itr = 0; itr < total; ++itr) {
    double w_mn, w_nb, w_fp;
    find_weight(w_MN_init, itr, phase1, phase2, w_mn, w_nb, w_fp);
    double loss = pacmap_grad_parallel(Y, pair_nb, pair_MN, pair_FP,
                                       w_nb, w_mn, w_fp, n_threads, grad);
    losses[itr] = loss;
    adam_step(Y, grad, m, v, beta1, beta2, lr, itr);
    if (verbose && ((itr + 1) % 25 == 0 || itr == 0))
      Rcpp::Rcout << "Iteration: " << (itr + 1) << ", Loss: " << loss << std::endl;
    Rcpp::checkUserInterrupt();
  }

  return List::create(_["embedding"] = Y, _["loss"] = losses);
}

// ---------- LocalMAP: FP resampling restricted to low-D nearby points ----------
// Mirrors sample_FP_pair_nearby in pacmap.py. For each anchor i, resample n_FP
// further-pair partners uniformly from points whose current low-D distance to i
// is <= low_dist_thres, excluding self and neighbor partners. If sampling
// exhausts >100 tries (crowded neighborhood), keep the old FP partner.

// [[Rcpp::export]]
IntegerMatrix sample_fp_pairs_nearby_cpp(NumericMatrix Y,
                                         IntegerMatrix pair_neighbors,
                                         IntegerMatrix old_pair_FP,
                                         double low_dist_thres,
                                         Nullable<int> random_state = R_NilValue) {
  const int n   = Y.nrow();
  const int dim = Y.ncol();
  const int n_neighbors = pair_neighbors.nrow() / n;
  const int n_FP        = old_pair_FP.nrow() / n;
  const double thres2   = low_dist_thres * low_dist_thres;
  const bool det        = random_state.isNotNull();
  const int base_seed   = det ? as<int>(random_state) : 0;

  IntegerMatrix pair_FP(n * n_FP, 2);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(!det)
  #endif
  for (int i = 0; i < n; ++i) {
    // Build the reject set: point i itself + its NN partners + already-picked FPs
    std::vector<int> reject(n_neighbors);
    for (int k = 0; k < n_neighbors; ++k) reject[k] = pair_neighbors(i * n_neighbors + k, 1);

    LCG rng(det ? static_cast<uint64_t>(base_seed + i + 1)
                : static_cast<uint64_t>((uint32_t)R::runif(0, 1e9) + 1));

    std::vector<int> picked; picked.reserve(n_FP);
    for (int s = 0; s < n_FP; ++s) {
      int j_out = -1;
      int tries = 0;
      while (tries++ <= 100) {
        int j = rng.next_below(n);
        if (j == i) continue;
        bool dup = false;
        for (int k : reject)  if (k == j) { dup = true; break; }
        if (dup) continue;
        for (int k : picked)  if (k == j) { dup = true; break; }
        if (dup) continue;
        // Squared low-D distance i-j
        double d2 = 0.0;
        for (int d = 0; d < dim; ++d) {
          double diff = Y(i, d) - Y(j, d);
          d2 += diff * diff;
        }
        if (d2 > thres2) continue;
        j_out = j;
        break;
      }
      pair_FP(i * n_FP + s, 0) = i;
      if (j_out >= 0) {
        pair_FP(i * n_FP + s, 1) = j_out;
        picked.push_back(j_out);
      } else {
        // Fallback to previous FP partner (Python behaviour).
        pair_FP(i * n_FP + s, 1) = old_pair_FP(i * n_FP + s, 1);
      }
    }
  }
  return pair_FP;
}

// ---------- LocalMAP gradient: adds NN_coef_recip/sqrt(d_ij) factor to NN term ----------

static double localmap_grad_parallel(const NumericMatrix& Y,
                                     const IntegerMatrix& pair_nb,
                                     const IntegerMatrix& pair_MN,
                                     const IntegerMatrix& pair_FP,
                                     double w_nb, double w_mn, double w_fp,
                                     double NN_coef_recip,
                                     int n_threads,
                                     NumericMatrix& grad) {
  const int n = Y.nrow();
  const int dim = Y.ncol();
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) grad(i, d) = 0.0;

  double loss = 0.0;
  const int nnp = pair_nb.nrow();
  const int nmn = pair_MN.nrow();
  const int nfp = pair_FP.nrow();

#ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads) reduction(+:loss)
#endif
  {
    std::vector<double> lgrad(static_cast<size_t>(n) * dim, 0.0);
    std::vector<double> yij(dim);
#ifdef _OPENMP
    #pragma omp for schedule(static) nowait
#endif
    for (int t = 0; t < nnp; ++t) {
      const int i = pair_nb(t, 0), j = pair_nb(t, 1);
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_nb * (d_ij / (10.0 + d_ij));
      double w1 = w_nb * (20.0 / std::pow(10.0 + d_ij, 2));
      w1 *= NN_coef_recip / std::sqrt(d_ij);  // LocalMAP modification
      const size_t ib = static_cast<size_t>(i) * dim, jb = static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        lgrad[ib + d] += w1 * yij[d];
        lgrad[jb + d] -= w1 * yij[d];
      }
    }
#ifdef _OPENMP
    #pragma omp for schedule(static) nowait
#endif
    for (int t = 0; t < nmn; ++t) {
      const int i = pair_MN(t, 0), j = pair_MN(t, 1);
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_mn * (d_ij / (10000.0 + d_ij));
      const double w1 = w_mn * (20000.0 / std::pow(10000.0 + d_ij, 2));
      const size_t ib = static_cast<size_t>(i) * dim, jb = static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        lgrad[ib + d] += w1 * yij[d];
        lgrad[jb + d] -= w1 * yij[d];
      }
    }
#ifdef _OPENMP
    #pragma omp for schedule(static)
#endif
    for (int t = 0; t < nfp; ++t) {
      const int i = pair_FP(t, 0), j = pair_FP(t, 1);
      double d_ij = 1.0;
      for (int d = 0; d < dim; ++d) { yij[d] = Y(i, d) - Y(j, d); d_ij += yij[d] * yij[d]; }
      loss += w_fp * (1.0 / (1.0 + d_ij));
      const double w1 = w_fp * (2.0 / std::pow(1.0 + d_ij, 2));
      const size_t ib = static_cast<size_t>(i) * dim, jb = static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        lgrad[ib + d] -= w1 * yij[d];
        lgrad[jb + d] += w1 * yij[d];
      }
    }
#ifdef _OPENMP
    #pragma omp critical
#endif
    for (int i = 0; i < n; ++i) {
      const size_t ib = static_cast<size_t>(i) * dim;
      for (int d = 0; d < dim; ++d) grad(i, d) += lgrad[ib + d];
    }
  }
  return loss;
}

// ---------- LocalMAP main optimization loop ----------
// Phases 1 & 2: identical to pacmap. Phase 3: switches to localmap_grad and
// resamples pair_FP every 10 iters using sample_fp_pairs_nearby_cpp.

// [[Rcpp::export]]
List localmap_optimize_cpp(NumericMatrix Y_init,
                           IntegerMatrix pair_nb,
                           IntegerMatrix pair_MN,
                           IntegerMatrix pair_FP,
                           double lr,
                           IntegerVector num_iters,
                           double low_dist_thres,
                           int n_threads,
                           Nullable<int> random_state,
                           bool verbose) {
  if (num_iters.size() != 3) stop("num_iters must have length 3");
  if (n_threads < 1) n_threads = 1;
  int phase1 = num_iters[0], phase2 = num_iters[1], phase3 = num_iters[2];
  int total = phase1 + phase2 + phase3;
  int n = Y_init.nrow(), dim = Y_init.ncol();

  NumericMatrix Y(n, dim);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) Y(i, d) = Y_init(i, d);

  IntegerMatrix pair_FP_curr = clone(pair_FP);  // resampled in phase 3

  NumericMatrix grad(n, dim), m(n, dim), v(n, dim);
  const double w_MN_init = 1000.0;
  const double beta1 = 0.9, beta2 = 0.999;
  const double NN_coef_recip = low_dist_thres / 2.0;
  NumericVector losses(total);

  for (int itr = 0; itr < total; ++itr) {
    double w_mn, w_nb, w_fp;
    find_weight(w_MN_init, itr, phase1, phase2, w_mn, w_nb, w_fp);
    double loss;
    if (itr > phase1 + phase2) {
      loss = localmap_grad_parallel(Y, pair_nb, pair_MN, pair_FP_curr,
                                    w_nb, w_mn, w_fp, NN_coef_recip,
                                    n_threads, grad);
    } else {
      loss = pacmap_grad_parallel(Y, pair_nb, pair_MN, pair_FP_curr,
                                  w_nb, w_mn, w_fp, n_threads, grad);
    }
    losses[itr] = loss;
    adam_step(Y, grad, m, v, beta1, beta2, lr, itr);
    if (itr > phase1 + phase2 && itr % 10 == 0) {
      pair_FP_curr = sample_fp_pairs_nearby_cpp(Y, pair_nb, pair_FP_curr,
                                                low_dist_thres, random_state);
    }
    if (verbose && ((itr + 1) % 25 == 0 || itr == 0))
      Rcpp::Rcout << "LocalMAP iter " << (itr + 1) << ", Loss: " << loss << std::endl;
    Rcpp::checkUserInterrupt();
  }
  return List::create(_["embedding"] = Y, _["loss"] = losses,
                      _["pair_FP_final"] = pair_FP_curr);
}

// ---------- fit-optimize (transform) ----------
// Only new-point rows (indices >= n_basis) receive updates.

// [[Rcpp::export]]
List pacmap_fit_optimize_cpp(NumericMatrix Y_init,
                             IntegerMatrix pair_XP,
                             int n_basis,
                             double lr,
                             IntegerVector num_iters,
                             int n_threads,
                             bool verbose) {
  if (num_iters.size() != 3) stop("num_iters must have length 3");
  if (n_threads < 1) n_threads = 1;
  int phase1 = num_iters[0], phase2 = num_iters[1], phase3 = num_iters[2];
  int total = phase1 + phase2 + phase3;

  int n = Y_init.nrow(), dim = Y_init.ncol();
  NumericMatrix Y(n, dim);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) Y(i, d) = Y_init(i, d);

  NumericMatrix grad(n, dim), m(n, dim), v(n, dim);
  const double beta1 = 0.9, beta2 = 0.999;
  NumericVector losses(total);

  for (int itr = 0; itr < total; ++itr) {
    // Weight schedule: only the NN weight matters here (MN/FP terms omitted for the
    // transform case, matching the Python pacmap_grad_fit).
    double w_mn, w_nb, w_fp;
    find_weight(1000.0, itr, phase1, phase2, w_mn, w_nb, w_fp);
    double loss = pacmap_grad_fit_parallel(Y, pair_XP, w_nb, n_basis,
                                           n_threads, grad);
    losses[itr] = loss;
    adam_step(Y, grad, m, v, beta1, beta2, lr, itr);
    if (verbose && ((itr + 1) % 25 == 0 || itr == 0))
      Rcpp::Rcout << "transform iter " << (itr + 1) << ", Loss: " << loss << std::endl;
    Rcpp::checkUserInterrupt();
  }
  return List::create(_["embedding"] = Y, _["loss"] = losses);
}
