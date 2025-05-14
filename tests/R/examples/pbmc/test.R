########################
# Setting up the datasets.

library(TENxPBMCData)

sce3k <- TENxPBMCData("pbmc3k")
sce4k <- TENxPBMCData("pbmc4k")
common <- intersect(rownames(sce3k), rownames(sce4k))

sce3k <- sce3k[common,]
sce4k <- sce4k[common,]
x <- cbind(assay(sce3k), assay(sce4k))
x <- as(x, "dgCMatrix")
block <- rep(c("3k", "4k"), c(ncol(sce3k), ncol(sce4k)))

saveRDS(list(x, block), file="whee.rds")
# reloaded <- readRDS("whee.rds"); x <- reloaded[[1]]; block <- reloaded[[2]]

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

y3k <- pcs[,f.block == "3k"]
y4k <- pcs[,f.block == "4k"]

library(mnncorrect.ref)
#corrected.3k <- mnncorrect.ref(y4k, y3k, k=15)
#total <- cbind(corrected.3k, y4k)
total <- mnncorrect.cpp(pcs, f.block)
out <- runTsne(total)

before <- runTsne(pcs) # for comparison's sake.

png("output.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
plot(before[,1], before[,2], col=factor(f.block), xlab="TSNE1", ylab="TSNE2", main="Before")
plot(out[,1], out[,2], col=factor(f.block), xlab="TSNE1", ylab="TSNE2", main="After")
legend("topright", c("3k", "4k"), col=1:2, pch=1)
dev.off()
