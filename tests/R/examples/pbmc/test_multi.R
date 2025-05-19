########################
# Setting up the datasets.

library(TENxPBMCData)

sce3k <- TENxPBMCData("pbmc3k")
sce4k <- TENxPBMCData("pbmc4k")
sce8k <- TENxPBMCData("pbmc8k")
common <- Reduce(intersect, list(rownames(sce3k), rownames(sce4k), rownames(sce8k)))

sce3k <- sce3k[common,]
sce4k <- sce4k[common,]
sce8k <- sce8k[common,]
x <- cbind(assay(sce3k), assay(sce4k), assay(sce8k))
x <- as(x, "dgCMatrix")
block <- rep(c("3k", "4k", "8k"), c(ncol(sce3k), ncol(sce4k), ncol(sce8k)))

########################
# Preamble of scrapper::analyze()

library(scrapper)

qc.metrics <- computeRnaQcMetrics(x, subsets=list(), num.threads=1)
qc.filters <- suggestRnaQcThresholds(qc.metrics, block=block, num.mads=3)
keep <- filterRnaQcMetrics(qc.filters, qc.metrics, block=block)

filtered <- x[,keep]
f.block <- block[keep]
sf <- centerSizeFactors(qc.metrics$sum[keep], block=f.block)
normalized <- normalizeCounts(filtered, sf)

variances <- modelGeneVariances(normalized, block=f.block, span=0.4, num.threads=1)
hvgs <- chooseHighlyVariableGenes(variances$statistics$residual, top=4000)

pca <- runPca(normalized[hvgs,], number=25, num.threads=1, block=f.block)
pcs <- pca$components

#######################
# Merging method starts here.

library(mnncorrect.ref)
total <- mnncorrect.cpp(pcs, f.block)
out <- runTsne(total)

before <- runTsne(pcs) # for comparison's sake.

png("output_multi.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
f <- factor(f.block)
plot(before[,1], before[,2], col=f, xlab="TSNE1", ylab="TSNE2", main="Before", cex=0.3, pch=16)
plot(out[,1], out[,2], col=f, xlab="TSNE1", ylab="TSNE2", main="After", cex=0.3, pch=16)
legend("topright", levels(f), col=seq_len(nlevels(f)), pch=16, cex=0.8)
dev.off()
