test_that("find_pacmap_pairs returns pairs of the right shape", {
  X <- as.matrix(iris[, 1:4])
  pairs <- find_pacmap_pairs(X, n_neighbors = 10L, random_state = 1L)
  n <- nrow(X)
  expect_equal(nrow(pairs$pair_neighbors), n * 10L)
  expect_equal(ncol(pairs$pair_neighbors), 2L)
  expect_equal(nrow(pairs$pair_MN), n * 5L)   # MN_ratio 0.5
  expect_equal(nrow(pairs$pair_FP), n * 20L)  # FP_ratio 2.0
})

test_that("pair indices are in [0, n) and 0-indexed", {
  X <- as.matrix(iris[, 1:4])
  pairs <- find_pacmap_pairs(X, random_state = 1L)
  n <- nrow(X)
  for (nm in c("pair_neighbors", "pair_MN", "pair_FP")) {
    m <- pairs[[nm]]
    expect_true(min(m) >= 0L, info = nm)
    expect_true(max(m) <  n,  info = nm)
    expect_true(all(m[, 1L] != m[, 2L]), info = paste(nm, "self-pair"))
  }
})

test_that("neighbor pairs come from row 1 = point 0, row 2 = point 0, ... (grouped by i)", {
  X <- as.matrix(iris[, 1:4])
  n_neighbors <- 10L
  pairs <- find_pacmap_pairs(X, n_neighbors = n_neighbors, random_state = 1L)
  # First column should be 0..(n-1), each repeated n_neighbors times
  expected_i <- rep(seq(0L, nrow(X) - 1L), each = n_neighbors)
  expect_identical(as.integer(pairs$pair_neighbors[, 1L]), expected_i)
})

test_that("FP rejects the neighbor set for the same anchor", {
  X <- as.matrix(iris[, 1:4])
  n_neighbors <- 10L; n <- nrow(X)
  pairs <- find_pacmap_pairs(X, n_neighbors = n_neighbors, random_state = 1L)
  # For a random sample of anchors, verify pair_FP's j is not in pair_neighbors' j for the same i.
  for (i in sample(seq_len(n), 30L)) {
    nb_j <- pairs$pair_neighbors[pairs$pair_neighbors[, 1L] == (i - 1L), 2L]
    fp_j <- pairs$pair_FP[pairs$pair_FP[, 1L] == (i - 1L), 2L]
    expect_length(intersect(nb_j, fp_j), 0L)
  }
})

test_that("deterministic sampling: same random_state -> identical pairs", {
  X <- as.matrix(iris[, 1:4])
  a <- find_pacmap_pairs(X, random_state = 7L)
  b <- find_pacmap_pairs(X, random_state = 7L)
  expect_identical(a$pair_MN, b$pair_MN)
  expect_identical(a$pair_FP, b$pair_FP)
  expect_identical(a$pair_neighbors, b$pair_neighbors)
})

test_that("different random_state -> different MN/FP pairs", {
  X <- as.matrix(iris[, 1:4])
  a <- find_pacmap_pairs(X, random_state = 1L)
  b <- find_pacmap_pairs(X, random_state = 2L)
  expect_false(identical(a$pair_MN, b$pair_MN))
  expect_false(identical(a$pair_FP, b$pair_FP))
})
