library(scRNAseq)
sceG <- GrunPancreasData()
sceM <- MuraroPancreasData()

common <- intersect(rownames(sceG), rownames(sceM))
combined <- cbind(assay(sceG)[common,], assay(sceM)[common,])
batch <- rep(c("Grun", "Muraro"), c(ncol(sceG), ncol(sceM)))

saveRDS(list(combined, batch), file="args.rds")
#X <- readRDS("args.rds"); x0 <- X[[1]]; block <- X[[2]]

########################
# Preamble of scran.chan::quickMergedAnalysis

library(scran.chan)
library(Matrix)
x <- initializeSparseMatrix(x0, num.threads=1)

qc.metrics <- perCellQCMetrics.chan(x, subsets=list(), num.threads=1)
qc.filters <- perCellQCFilters.chan(qc.metrics$sums, batch=block, qc.metrics$detected, qc.metrics$subsets, nmads=3)
qc.discard <- qc.filters$filters$overall
x <- filterCells.chan(x, qc.discard)

sf <- qc.metrics$sums[!qc.discard]
x <- logNormCounts.chan(x, sf, batch=block)

variances <- modelGeneVar.chan(x, batch=block, span = 0.4, num.threads=1)
keep <- rank(-variances$statistics$residuals, ties.method="first") <= 4000

pca <- runPCA.chan(x, num.comp=25, subset=keep, num.threads=1, batch=block, batch.method="weight")
pcs <- pca$components

#######################
# Merging method starts here.

plock <- block[!qc.discard]
y.g <- pcs[,plock == "Grun"]
y.m <- pcs[,plock == "Muraro"]

library(BiocNeighbors)
pairings <- findMutualNN(t(y.g), t(y.m), k1=15)

# Computing centers of mass for each MNN-involved cell.
mnn.g <- unique(pairings$first)
mnn.m <- unique(pairings$second)

closest.g <- queryKNN(query=t(y.g), X=t(y.g[,mnn.g]), k=15)
limit.g <- median(closest.g$distance) + mad(closest.g$distance) * 3
keep <- closest.g$index
keep[closest.g$distance > limit.g] <- NA

neighbors.g <- split(rep(seq_len(ncol(y.g)), 15), keep)
centers.g <- matrix(0, nrow(y.g), length(neighbors.g))
for (i in seq_along(neighbors.g)) {
    candidates <- y.g[,neighbors.g[[i]],drop=FALSE]
    centers.g[,i] <- rowMeans(candidates)
}

closest.m <- queryKNN(query=t(y.m), X=t(y.m[,mnn.m]), k=15)
limit.m <- median(closest.m$distance) + mad(closest.m$distance) * 3
keep <- closest.m$index
keep[closest.m$distance > limit.m] <- NA

neighbors.m <- split(rep(seq_len(ncol(y.m)), 15), keep)
centers.m <- matrix(0, nrow(y.m), length(neighbors.m))
for (i in seq_along(neighbors.m)) {
    candidates <- y.m[,neighbors.m[[i]],drop=FALSE]
    centers.m[,i] <- rowMeans(candidates)
}

# Finishing it off.
corrected.g <- y.g
for (x in seq_len(ncol(y.g))) {
    closest <- closest.g$index[x,]

    keep <- pairings$first %in% mnn.g[closest]
    pfirst <- pairings$first[keep]
    psecond <- pairings$second[keep]
    candidates <- centers.m[,match(psecond, mnn.m),drop=FALSE] - centers.g[,match(pfirst, mnn.g),drop=FALSE]

    correction <- rowMeans(candidates)
    d <- dist(t(candidates))
    med <- candidates[,which.max(-colSums(as.matrix(d)))]

    corrected.g[,x] <- corrected.g[,x] + (correction * 9 + med) / 10
}

total <- cbind(corrected.g, y.m)
out <- runTSNE.chan(total)

plot(out[,1], out[,2], col=factor(plock))
segments(
    out[pairings$first,1], 
    out[pairings$first,2],
    out[ncol(y.g) + pairings$second,1],
    out[ncol(y.g) + pairings$second,2],
    col="dodgerblue"
)
