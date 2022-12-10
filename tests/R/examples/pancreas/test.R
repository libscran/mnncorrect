library(scRNAseq)
sceG <- GrunPancreasData()
sceM <- MuraroPancreasData()

common <- intersect(rownames(sceG), rownames(sceM))
combined <- cbind(assay(sceG)[common,], assay(sceM)[common,])
block <- rep(c("Grun", "Muraro"), c(ncol(sceG), ncol(sceM)))

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
#corrected.g <- mnncorrect.ref(y.m, y.g)
#total <- cbind(corrected.g, y.m)
total <- mnncorrect.cpp(pcs, plock)$corrected
out <- runTSNE.chan(total)

before <- runTSNE.chan(pcs) # for comparison's sake.

png("output.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
plot(before[,1], before[,2], col=factor(plock), xlab="TSNE1", ylab="TSNE2", main="Before")
plot(out[,1], out[,2], col=factor(plock), xlab="TSNE1", ylab="TSNE2", main="After")
legend("topright", c("Grun", "Muraro"), col=1:2, pch=1)
dev.off()

#segments(
#    out[pairings$first,1], 
#    out[pairings$first,2],
#    out[ncol(y.g) + pairings$second,1],
#    out[ncol(y.g) + pairings$second,2],
#    col="dodgerblue"
#)
