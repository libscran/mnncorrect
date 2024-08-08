########################
# Setting up the datasets.

library(TENxPBMCData)

sce3k <- TENxPBMCData("pbmc3k")
sce4k <- TENxPBMCData("pbmc4k")
common <- intersect(rownames(sce3k), rownames(sce4k))

sce3k <- sce3k[common,]
sce4k <- sce4k[common,]
x0 <- cbind(assay(sce3k), assay(sce4k))
x0 <- as(x0, "dgCMatrix")
block <- rep(c("3k", "4k"), c(ncol(sce3k), ncol(sce4k)))

saveRDS(list(x0, block), file="whee.rds")
# X <- readRDS("whee.rds"); x0 <- X[[1]]; block <- X[[2]]

########################
# Preamble of scran.chan::quickMergedAnalysis

library(scran.chan)
library(HDF5Array)
x <- initializeSparseMatrix(x0, num.threads=1)

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

y3k <- pcs[,plock == "3k"]
y4k <- pcs[,plock == "4k"]

library(mnncorrect.ref)
#corrected.3k <- mnncorrect.ref(y4k, y3k, k=15)
#total <- cbind(corrected.3k, y4k)
total <- mnncorrect.cpp(pcs, plock)$corrected
out <- runTSNE.chan(total)

before <- runTSNE.chan(pcs) # for comparison's sake.

png("output.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
plot(before[,1], before[,2], col=factor(plock), xlab="TSNE1", ylab="TSNE2", main="Before")
plot(out[,1], out[,2], col=factor(plock), xlab="TSNE1", ylab="TSNE2", main="After")
legend("topright", c("3k", "4k"), col=1:2, pch=1)
dev.off()
