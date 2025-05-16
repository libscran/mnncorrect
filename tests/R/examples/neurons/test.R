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

yz <- pcs[,f.block == "zeisel"]
yt <- pcs[,f.block == "tasic"]

library(mnncorrect.ref)
#corrected.t <- mnncorrect.ref(yz, yt)
#total <- cbind(yz, corrected.t)
total <- mnncorrect.cpp(pcs, f.block)
out <- runTsne(total)

before <- runTsne(pcs) # for comparison's sake.

png("output.png", res=120, width=10, height=6, units="in")
par(mfrow=c(1,2))
plot(before[,1], before[,2], col=factor(f.block), xlab="TSNE1", ylab="TSNE2", main="Before")
plot(out[,1], out[,2], col=factor(f.block), xlab="TSNE1", ylab="TSNE2", main="After")
legend("topright", c("Tasic", "Zeisel"), col=1:2, pch=1)
dev.off()
