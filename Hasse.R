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
  "Develarization",
  "Grimm's Law",
  "Fricative Voicing",
  "Stress Shift",
  "Vowel Deletion",
  "Epenthesis",
  "RS Reduction",
  "Nasal Assimilation",
  "Lowering",
  "Great Gothic Vowel Shift",
  "Long A Changes"
)

M <- matrix(data=FALSE, nrow=length(labels), ncol=length(labels))

M <- addPair("Stress Shift", "Vowel Deletion", labels, M)
# M <- addPair("Vowel Deletion", "Great Gothic Vowel Shift", labels, M)  # WRONG
M <- addPair("Great Gothic Vowel Shift", "Stress Shift", labels, M)
M <- addPair("Grimm's Law", "Fricative Voicing", labels, M)
M <- addPair("Develarization", "Nasal Assimilation", labels, M)
M <- addPair("Vowel Deletion", "RS Reduction", labels, M)
M <- addPair("Great Gothic Vowel Shift", "Long A Changes", labels, M)
M <- addPair("Vowel Deletion", "Long A Changes", labels, M)
M <- addPair("Fricative Voicing", "Stress Shift", labels, M)
M <- addPair("Great Gothic Vowel Shift", "Lowering", labels, M)

hasse(M, labels)
