# faissR is Unix-only; on Windows the install itself refuses. These tests
# skip cleanly on any platform where faissR is not installed.

test_that("ann_backend='faiss' errors gracefully when faissR is missing", {
  skip_if(requireNamespace("faissR", quietly = TRUE),
          "faissR is installed; the missing-package branch cannot be tested")
  X <- as.matrix(iris[, 1:4])
  expect_error(
    pacmap(X, ann_backend = "faiss", num_iters = c(5L, 5L, 5L),
           random_state = 1L),
    "faissR"
  )
})

test_that("ann_backend='auto' falls back to hnsw when faissR is missing", {
  skip_if(requireNamespace("faissR", quietly = TRUE),
          "faissR is installed; the auto path would pick faiss instead")
  X <- as.matrix(iris[, 1:4])
  emb <- pacmap(X, ann_backend = "auto", num_iters = c(10L, 10L, 20L),
                random_state = 1L)
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
})

test_that("ann_backend='faiss' produces an embedding when faissR IS installed", {
  skip_if_not(requireNamespace("faissR", quietly = TRUE),
              "faissR not installed (Unix-only)")
  X <- as.matrix(iris[, 1:4])
  emb <- pacmap(X, ann_backend = "faiss", num_iters = c(50L, 50L, 100L),
                random_state = 1L)
  expect_equal(dim(emb$embedding), c(nrow(X), 2L))
  expect_true(all(is.finite(emb$embedding)))
})
