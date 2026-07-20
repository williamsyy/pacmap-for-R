test_that("each distance metric runs to completion on blobs", {
  bl <- make_blobs(n_per = 30L, d = 5L, k = 3L)
  for (dm in c("euclidean", "manhattan", "angular", "hamming")) {
    withCallingHandlers({
      emb <- pacmap(bl$X, distance = dm, num_iters = c(20L, 20L, 40L),
                    random_state = 1L)
      expect_equal(dim(emb$embedding), c(nrow(bl$X), 2L), info = dm)
      expect_true(all(is.finite(emb$embedding)), info = dm)
    }, warning = function(w) invokeRestart("muffleWarning"))
  }
})

test_that("euclidean gives high KNN preservation on blobs", {
  bl <- make_blobs(n_per = 60L, d = 4L, k = 3L)
  emb <- pacmap(bl$X, distance = "euclidean", num_iters = c(50L, 50L, 100L),
                random_state = 1L)
  score <- knn_preservation(bl$X, emb$embedding, k = 10L)
  # Random baseline for 3 blobs of 60 pts in a 2D output is well under 0.5.
  expect_gt(score, 0.35)
})
