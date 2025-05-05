########################
# Setting up the datasets.

library(scRNAseq)

sce.z <- ZeiselBrainData()
sce.t <- TasicBrainData()
common <- intersect(rownames(sce.z), rownames(sce.t))

sce.z <- sce.z[common,]
sce.t <- sce.t[common,]
x <- cbind(assay(sce.z), assay(sce.t))
block <- rep(c("zeisel", "tasic"), c(ncol(sce.z), ncol(sce.t)))

saveRDS(list(x, block), file="whee.rds")
# X <- readRDS("whee.rds"); x <- X[[1]]; block <- X[[2]]

########################
# Preamble of scran.chan::quickMergedAnalysis

library(scrapper)
qc.metrics <- computeRnaQcMetrics(x, subsets=list(), num.threads=1)
qc.filters <- suggestRnaQcThresholds(qc.metrics, block=block, num.mads=3)
keep <- filterRnaQcMetrics(qc.filters, qc.metrics, block=block)

filtered <- x[,keep]
plock <- block[keep]
sf <- centerSizeFactors(qc.metrics$sum[keep], block=plock)
norm <- normalizeCounts(filtered, sf)

variances <- modelGeneVariances(norm, block=plock, span = 0.4, num.threads=1)
hvgs <- chooseHighlyVariableGenes(variances$statistics$residual)

pca <- runPca(norm[hvgs,], number=25, num.threads=1, block=plock)
pcs <- pca$components

#######################
# Merging method starts here.

yz <- pcs[,plock == "zeisel"]
yt <- pcs[,plock == "tasic"]

library(mnncorrect.ref)
#corrected.t <- mnncorrect.ref(yz, yt)
#total <- cbind(yz, corrected.t)
total <- mnncorrect.cpp(pcs, plock)$corrected
out <- runTSNE.chan(total)

before <- runTSNE.chan(pcs) # for comparison's sake.

png("output.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
plot(before[,1], before[,2], col=factor(plock), xlab="TSNE1", ylab="TSNE2", main="Before")
plot(out[,1], out[,2], col=factor(plock), xlab="TSNE1", ylab="TSNE2", main="After")
legend("topright", c("Tasic", "Zeisel"), col=1:2, pch=1)
dev.off()
