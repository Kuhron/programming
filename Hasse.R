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
  # "Umlaut",
  # "H Loss",
  # "O Creation",
  "Weak Final I Loss",
  # "Strong Final I Loss",
  "M Degemination",
  # "Stop Voicing",
  # "Weak Final A Loss",
  "Strong Final A Loss",
  "Superheavy Prohibition",
  "I Creation",
  # "Progressive Harmony",
  # "Palatalization",
  "Ash Merger"
)

M <- matrix(data=FALSE, nrow=length(labels), ncol=length(labels))

M <- addPair("Weak Final I Loss", "M Degemination", labels, M)
M <- addPair("Superheavy Prohibition", "I Creation", labels, M)
M <- addPair("I Creation", "Strong Final A Loss", labels, M)

hasse(M, labels)
