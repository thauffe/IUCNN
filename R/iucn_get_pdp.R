#' Get partial dependence probabilities for IUCN Categories
#'
#'@param x iucnn_model object, as produced as output
#'when running \code{\link{iucnn_train_model}}
#'@param dropout_reps integer, (default = 100). The number of how often the
#'predictions are to be repeated (only for dropout models). A value of 100 is
#'recommended to capture the stochasticity of the predictions, lower values
#'speed up the prediction time.
#'@param feature_blocks a list.
#'@param provide_indices logical. Set to TRUE if custom \code{feature_blocks}
#'are provided as indices. Default is FALSE.

#' @export
#' @importFrom reticulate import source_python
#' @importFrom checkmate assert_class assert_numeric assert_character
#'   assert_logical
#'
iucnn_get_pdp <- function(x,
                          dropout_reps,
                          feature_blocks = list(),
                          provide_indices = FALSE){

  if (!any(file.exists(x$trained_model_path))) {
    stop("Model path doesn't exists.
         Please check if you saved it in a temporary directory.")
  }

  # assertions
  assert_class(x, "iucnn_model")
  assert_numeric(dropout_reps)
  assert_class(feature_blocks, "list")
  assert_logical(provide_indices)

  dropout_reps <- as.integer(dropout_reps)

  # features for which to obtain PDP
  fb <- make_feature_block(x = x,
                           feature_blocks = feature_blocks,
                           include_all_features = FALSE,
                           provide_indices = provide_indices,
                           unlink_features_within_block = unlink_features_within_block)
  feature_block_indices <- fb$feature_block_indices

  if (x$model == 'bnn-class') {
    placeholder <- NULL
  }
  else {
    reticulate::source_python(system.file("python", "IUCNN_pdp.py",
                                          package = "IUCNN"))

    model_dir <- x$trained_model_path
    iucnn_mode <- x$model
    dropout <- x$mc_dropout
    rescale_factor <- x$label_rescaling_factor
    min_max_label <- x$min_max_label_rescaled
    stretch_factor_rescaled_labels <- x$label_stretch_factor

    data_pdp <- rbind(x$input_data$data, x$input_data$test_data)


    pdp <- vector(mode = "list", length = 6)
    for (i in 5:6) {
      ii <- as.list(as.integer(i))
      pdp[[i]] <- iucnn_pdp(input_features = data_pdp,
                            focal_features = ii,
                            model_dir = model_dir,
                            iucnn_mode = iucnn_mode,
                            dropout = dropout,
                            dropout_reps = dropout_reps,
                            rescale_factor = rescale_factor,
                            min_max_label = min_max_label,
                            stretch_factor_rescaled_labels = stretch_factor_rescaled_labels)
    }

  }
  return(pdp)
}
