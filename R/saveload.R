#' Save a fitted PaCMAP model to disk
#'
#' Writes the model as an .rds. Everything needed for \code{predict()}/
#' \code{transform()} is included; the ANN index is rebuilt on load (cheap for
#' most workloads). No pickle-style separate index file like Python's --
#' \code{saveRDS} handles the whole model in one go.
#'
#' @param object a fitted "pacmap" object.
#' @param file path to write the .rds to.
#' @export
save_pacmap <- function(object, file) {
  stopifnot(inherits(object, "pacmap"))
  saveRDS(object, file = file)
  invisible(file)
}

#' Load a saved PaCMAP model
#' @param file path to a .rds written by \code{save_pacmap()}.
#' @export
load_pacmap <- function(file) {
  obj <- readRDS(file)
  if (!inherits(obj, "pacmap"))
    stop("File does not contain a pacmap object.")
  obj
}
