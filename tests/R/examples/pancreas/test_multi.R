########################
# Setting up the datasets.

library(scRNAseq)

sceG <- GrunPancreasData()
sceM <- MuraroPancreasData()
sceB <- BaronPancreasData(which="mouse")

common <- Reduce(intersect, list(rowData(sceG)$symbol, rowData(sceM)$symbol, toupper(rownames(sceB))))
x <- cbind(assay(sceG)[match(common, rowData(sceG)$symbol),], assay(sceM)[match(common, rowData(sceM)$symbol),], assay(sceB)[match(common, toupper(rownames(sceB))),])
block <- rep(c("Grun", "Muraro", "Baron"), c(ncol(sceG), ncol(sceM), ncol(sceB)))

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
plot(before[,1], before[,2], col=f, xlab="TSNE1", ylab="TSNE2", main="Before", pch=16, cex=0.3)
plot(out[,1], out[,2], col=f, xlab="TSNE1", ylab="TSNE2", main="After", pch=16, cex=0.3)
legend("topright", levels(f), col=seq_len(nlevels(f)), pch=16, cex=0.8)
dev.off()
