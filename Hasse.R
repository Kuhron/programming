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
  "*1", "*2", "*3", "*4", "*5"
)

M <- matrix(data=FALSE, nrow=length(labels), ncol=length(labels))

M <- addPair("*5", "*1", labels, M)
M <- addPair("*5", "*3", labels, M)
M <- addPair("*1", "*2", labels, M)
M <- addPair("*3", "*2", labels, M)
M <- addPair("*4", "*2", labels, M)
M <- addPair("*5", "*2", labels, M)
M <- addPair("*4", "*1", labels, M)
M <- addPair("*4", "*2", labels, M)

hasse(M, labels)
