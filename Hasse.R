haveHasseDiagram <- "hasseDiagram" %in% rownames(installed.packages())
haveRgraphviz <- "Rgraphviz" %in% rownames(installed.packages())
haveBiocManager <- "BiocManager" %in% rownames(installed.packages())
if (!haveHasseDiagram) {
  if (!haveRgraphviz) {
    if (!haveBiocManager) {
      install.packages("BiocManager")
    }
    BiocManager::install("Rgraphviz")
  }
  install.packages("hasseDiagram")
}

library(hasseDiagram)

addPair <- function(x, y, labels, M) {
  stopifnot(x %in% labels)
  stopifnot(y %in% labels)
  # x precedes y in the ordering
  xIndex <- match(x, labels)
  yIndex <- match(y, labels)
  M[xIndex, yIndex] <- TRUE  # x precedes y
  M[yIndex, xIndex] <- FALSE  # y does not precede x
  return(M)
}


# main

labels <- c(
  "Ditone Spreading (T-prime subcase)",
  "Ditone Spreading (T-double-prime subcase)",
  "5 Deletion",
  "Default 2",
  "214 Spreading",
  "214-35 Replacement"
)

M <- matrix(data=FALSE, nrow=length(labels), ncol=length(labels))

M <- addPair("214-35 Replacement", "Ditone Spreading (T-double-prime subcase)", labels, M)
M <- addPair("Ditone Spreading (T-prime subcase)", "5 Deletion", labels, M)
M <- addPair("Ditone Spreading (T-double-prime subcase)", "5 Deletion", labels, M)
M <- addPair("5 Deletion", "Default 2", labels, M)
M <- addPair("214 Spreading", "Default 2", labels, M)

hasse(M, labels)
