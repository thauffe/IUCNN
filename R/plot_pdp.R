#' Plot partial dependence probabilities
#'
#' @export
#'
plot.iucnn_pdp <- function(x,
                           ask = FALSE,
                           features = NULL,
                           uncertainty = TRUE,
                           order_categorical = TRUE,
                           col = NULL, ...) {
  if (is.null(features)) {
    features <- 1:length(x)
  }
  feature_names <- names(x)

  num_cats <- ncol(x[[features[1]]]$pdp)

  if (is.null(col)) {
    col <- c("#468351", "#BBD25B", "#F4EB5A", "#EDAA4C", "#DA4741")
    if (num_cats == 2) {
      col <- col[c(2, 5)]
    }
  }

  if (uncertainty) {
    uncertainty <- length(x[[1]]) == 4
  }

  for (fe in 1:length(features)) {
    cont_feature <- is.numeric(x[[features[fe]]]$feature[, 1])
    x_fe <- x[[fe]]

    if (cont_feature) {
      plot(0, 0, type = "n",
           ylim = c(0, 1), xlim = range(x_fe$feature),
           xlab = feature_names[fe],
           ylab = "Partial dependence probability",
           xaxs = "i", yaxs = "i")

      if (uncertainty) {
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(x_fe$lwr[, 1], rep(0, nrow(x_fe$pdp))),
                border = NA, col = col[1])
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(x_fe$lwr[, 1], rev(x_fe$upr[, 1])),
                border = NA, col = "grey")
        if (num_cats == 5) {
          for (i in 2:(num_cats - 1)) {
            polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                    y = c(x_fe$lwr[, i], rev(x_fe$upr[, i - 1])),
                    border = NA, col = col[i])
            polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                    y = c(x_fe$lwr[, i], rev(x_fe$upr[, i])),
                    border = NA, col = "grey")
          }
        }
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(rep(1, nrow(x_fe$pdp)), rev(x_fe$upr[, num_cats - 1])),
                border = NA, col = col[num_cats])
        for (i in 1:(num_cats - 1)) {
          lines(x_fe$feature, x_fe$pdp[, i], col = col[i + 1], lwd = 3)
        }
      }

      else {
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(x_fe$pdp[, 1], rep(0, nrow(x_fe$pdp))),
                border = NA, col = col[1])
        for (i in 2:num_cats) {
          polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                  y = c(x_fe$pdp[, i], rev(x_fe$pdp[, i - 1])),
                  border = NA, col = col[i])
        }
      }
    }

    else {

      if (order_categorical) {
        x_fe <- order_cat_pdp(x_fe, num_cats, uncertainty)
      }
      plot(0, 0, type = "n",
           ylim = c(0, 1), xlim = c(0, nrow(x_fe$feature)),
           xlab = "", ylab = "Partial dependence probability",
           xaxs = "i", yaxs = "i", xaxt = "n")
      par(las = 2)
      axis(side = 1, at = 1:nrow(x_fe$feature) - 0.5,
           labels = x_fe$feature[, 1])

      if (uncertainty) {
        placeholder <- NULL
      }
      else {
        for (j in 1:nrow(x_fe$feature)) {
          p <- c(0, x_fe$pdp[j, ])
          for (k in 2:(num_cats + 1)) {
            rect(xleft = j - 1, xright = j, ybottom = p[k - 1], ytop = p[k],
                 border = NA, col = col[k - 1])
          }
        }
      }

    }
  }
}


order_cat_pdp <- function(x_fe, num_cats, uncertainty) {
  if (num_cats == 5) {
    pca1 <- prcomp(scale(x_fe$pdp[, 1:4]))$x[, 1]
    ord <- order(pca1)
    a <- x_fe$pdp[ord, 1]
    if (a[1] > a[length(a)]) {
      ord <- rev(ord)
    }
  } else {
    ord <- order(x_fe$pdp[, 1])
  }
  x_fe$feature <- x_fe$feature[ord, , drop = FALSE]
  x_fe$pdp <- x_fe$pdp[ord, ]
  if (uncertainty) {
    x_fe$lwr <- x_fe$lwr[ord, ]
    x_fe$upr <- x_fe$upr[ord, ]
  }
  return(x_fe)
}
