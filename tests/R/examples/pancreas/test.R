library(scRNAseq)
sceG <- GrunPancreasData()
sceM <- MuraroPancreasData()

common <- intersect(rownames(sceG), rownames(sceM))
combined <- cbind(assay(sceG)[common,], assay(sceM)[common,])
batch <- rep(c("Grun", "Muraro"), c(ncol(sceG), ncol(sceM)))

saveRDS(list(combined, batch), file="args.rds")
# X <- readRDS("args.rds"); combined <- X[[1]]; block <- X[[2]]

########################
# Preamble of scran.chan::quickMergedAnalysis

library(scran.chan)
library(Matrix)
x <- initializeSparseMatrix(combined, num.threads=1)

qc.metrics <- perCellQCMetrics.chan(x, subsets=list(), num.threads=1)
qc.filters <- perCellQCFilters.chan(qc.metrics$sums, batch=block, qc.metrics$detected, qc.metrics$subsets, nmads=3)
qc.discard <- qc.filters$filters$overall
x <- filterCells.chan(x, qc.discard)

sf <- qc.metrics$sums[!qc.discard]
plock <- block[!qc.discard]
x <- logNormCounts.chan(x, sf, batch=plock)

variances <- modelGeneVar.chan(x, batch=plock, span = 0.4, num.threads=1)
keep <- rank(-variances$statistics$residuals, ties.method="first") <= 4000

pca <- runPCA.chan(x, num.comp=25, subset=keep, num.threads=1, batch=plock, batch.method="weight")
pcs <- pca$components

#######################
# Merging method starts here.

y.g <- pcs[,plock == "Grun"]
y.m <- pcs[,plock == "Muraro"]

library(mnncorrect.ref)
corrected.g <- mnncorrect.ref(y.m, y.g)
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
