########################
# Setting up the datasets.

library(scRNAseq)

sce.z <- ZeiselBrainData()
sce.t <- TasicBrainData()
common <- intersect(rownames(sce.z), rownames(sce.t))

sce.z <- sce.z[common,]
sce.t <- sce.t[common,]
x0 <- cbind(assay(sce.z), assay(sce.t))
block <- rep(c("zeisel", "tasic"), c(ncol(sce.z), ncol(sce.t)))

saveRDS(list(x0, block), file="whee.rds")
# X <- readRDS("whee.rds"); x0 <- X[[1]]; block <- X[[2]]

########################
# Preamble of scran.chan::quickMergedAnalysis

library(scran.chan)
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

yz <- pcs[,plock == "zeisel"]
yt <- pcs[,plock == "tasic"]

library(mnncorrect.ref)
corrected.t <- mnncorrect.ref(yz, yt, k=50)
total <- cbind(yz, corrected.t)
out <- runTSNE.chan(total)

plot(out[,1], out[,2], col=factor(plock))


clusters <- clusterKmeans.chan(yz, 20)
plot(out[plock=="zeisel",1], out[plock=="zeisel",2], col=clusters$clusters)
