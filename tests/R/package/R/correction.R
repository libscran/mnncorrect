#' @export
#' @importFrom utils head tail
mnncorrect.ref <- function(x, batch, k=15, steps=1) {
    stopifnot(ncol(x) == length(batch))
    batches <- split(seq_along(batch), batch)
    for (i in seq_along(batches)) {
        chosen <- batches[[i]]
        batches[[i]] <- list(
            value=x[,chosen,drop=FALSE],
            id=chosen
        )
    }

    while (length(batches) > 1L) {
        target <- tail(batches, 1)[[1]]
        ref.batches <- head(batches, -1)
        corout <- corrector(ref.batches, target$value, k=k, steps=steps)

        reassignments <- split(seq_along(corout$batch), corout$batch)
        for (r0 in names(reassignments)) {
            r <- as.integer(r0)
            idx <- reassignments[[r0]]
            ref.batches[[r]]$value <- cbind(ref.batches[[r]]$value, corout$corrected[,idx,drop=FALSE])
            ref.batches[[r]]$id <- c(ref.batches[[r]]$id, target$id[idx])
        }

        batches <- ref.batches
    }

    final <- matrix(0, nrow(x), ncol(x))
    final[,batches[[1]]$id] <- batches[[1]]$value
    final
}

#' @importFrom BiocNeighbors findMutualNN queryKNN
corrector <- function(ref.batches, target, k, steps) {
    ref <- do.call(cbind, lapply(ref.batches, function(x) x$value))
    pairings <- findMutualNN(t(ref), t(target), k1=k)

    # Find the closest reference partner for each MNN-involved cell in the target. 
    by.target <- split(pairings$first, pairings$second)
    mnn.target <- as.integer(names(by.target))
    mnn.ref <- integer(length(mnn.target))
    for (i in seq_along(by.target)) {
        ref.current <- by.target[[i]]
        target.current <- mnn.target[i]
        distances <- colSums((ref[,ref.current,drop=FALSE] - target[,target.current])^2)
        mnn.ref[i] <- ref.current[which.min(distances)]
    }

    batch.sizes <- vapply(ref.batches, function(x) ncol(x$value), 0L)
    batch.id <- rep(seq_along(batch.sizes), batch.sizes)
    unique.mnn.ref <- unique(mnn.ref)
    unique.mnn.ref.by.batch <- split(unique.mnn.ref, batch.id[unique.mnn.ref])
    batch.offsets <- c(0L, cumsum(batch.sizes))

    centers.r <- matrix(0, nrow(ref), length(unique.mnn.ref))
    for (r0 in names(unique.mnn.ref.by.batch)) {
        r <- as.integer(r0)
        curref.batch <- ref.batches[[r]]
        curref.mnn <- unique.mnn.ref.by.batch[[r0]]
        centers.r[,match(curref.mnn, unique.mnn.ref)] <- center_of_mass(curref.batch$value, curref.mnn - batch.offsets[r], k=k, steps=steps)
    }

    centers.t <- center_of_mass(target, mnn.target, k=k, steps=steps)
    correction <- centers.r[,match(mnn.ref, unique.mnn.ref),drop=FALSE] - centers.t

    neighbors.to.mnn <- queryKNN(query=target, X=target[,mnn.target,drop=FALSE], k=1, transposed=TRUE, get.distance=FALSE)
    closest.t <- neighbors.to.mnn$index[,1]
    corrected <- target + correction[,closest.t,drop=FALSE]

    list(corrected=corrected, batch=batch.id[mnn.ref[closest.t]])
}

#' @importFrom BiocNeighbors queryKNN
center_of_mass <- function(y, mnn, k, steps) {
    neighbors <- queryKNN(query=y, X=y, k=k, transposed=TRUE, get.distance=FALSE) # use 'query' to ensure we detect self.
    centers <- matrix(0, nrow(y), length(mnn))
    for (i in seq_along(mnn)) {
        collected <- to.check <- neighbors$index[mnn[i],]
        for (s in seq_len(steps)) {
            latest <- unique(neighbors$index[to.check,,drop=FALSE])
            collected <- union(collected, latest)
            to.check <- latest
        }
        centers[,i] <- rowMeans(y[,collected,drop=FALSE])
    }
    centers
}
