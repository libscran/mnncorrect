########################
# Setting up the datasets.

library(scRNAseq)

sceG <- GrunPancreasData()
sceM <- MuraroPancreasData()

common <- intersect(rownames(sceG), rownames(sceM))
x <- cbind(assay(sceG)[common,], assay(sceM)[common,])
block <- rep(c("Grun", "Muraro"), c(ncol(sceG), ncol(sceM)))

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

y.g <- pcs[,f.block == "Grun"]
y.m <- pcs[,f.block == "Muraro"]

library(mnncorrect.ref)
#corrected.g <- mnncorrect.ref(y.m, y.g)
#total <- cbind(corrected.g, y.m)
total <- mnncorrect.cpp(pcs, f.block)
out <- runTsne(total)

before <- runTsne(pcs) # for comparison's sake.

png("output.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
plot(before[,1], before[,2], col=factor(f.block), xlab="TSNE1", ylab="TSNE2", main="Before")
plot(out[,1], out[,2], col=factor(f.block), xlab="TSNE1", ylab="TSNE2", main="After")
legend("topright", c("Grun", "Muraro"), col=1:2, pch=1)
dev.off()

#segments(
#    out[pairings$first,1], 
#    out[pairings$first,2],
#    out[ncol(y.g) + pairings$second,1],
#    out[ncol(y.g) + pairings$second,2],
#    col="dodgerblue"
#)
