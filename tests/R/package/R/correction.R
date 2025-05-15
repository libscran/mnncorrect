#' @export
#' @importFrom utils head tail
mnncorrect.ref <- function(x, batch, k=15, tolerance=3) {
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
        ref.batch.id <- rep(seq_along(ref.batches), vapply(ref.batches, function(x) ncol(x$value), 0L))
        ref.combined <- do.call(cbind, lapply(ref.batches, function(x) x$value))
        corout <- corrector(ref.combined, ref.batch.id, target$value, k=k, tolerance=tolerance)

        reassignments <- split(seq_along(corout$batch), factor(corout$batch, seq_along(ref.batches)))
        for (r in seq_along(reassignments)) {
            idx <- reassignments[[r]]
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
corrector <- function(ref, batch.id, target, k, tolerance) {
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

    unique.ref <- unique(mnn.ref)
    r.out <- center_of_mass(ref, unique.ref, k=k, tolerance=tolerance)
    centers.r <- r.out$centers

    t.out <- center_of_mass(target, mnn.target, k=k, tolerance=tolerance)
    closest.t <- t.out$closest # already indexes into 'mnn.target'.
    centers.t <- t.out$centers

    correction <- centers.r[,match(mnn.ref, unique.ref),drop=FALSE] - centers.t
    corrected <- target + correction[,closest.t,drop=FALSE]

    list(corrected=corrected, batch=batch.id[mnn.ref[closest.t]])
}

#' @importFrom BiocNeighbors queryKNN
center_of_mass <- function(y, mnn, k, tolerance) {
    ty <- t(y)
    mnn.y <- ty[mnn,,drop=FALSE]
    neighbors.from.mnn <- queryKNN(query=mnn.y, X=ty, k=k, get.distance=FALSE)
    neighbors.to.mnn <- queryKNN(query=ty, X=mnn.y, k=k, get.distance=FALSE)
    inverted.neighbors.to <- split(rep(seq_len(ncol(y)), k), factor(neighbors.to.mnn$index, seq_along(mnn)))

    centers <- matrix(0, nrow(y), length(mnn))
    for (i in seq_along(mnn)) {
        # Computing the seed.
        curneighbors.from <- neighbors.from.mnn$index[i,]
        for_sure <- y[,curneighbors.from,drop=FALSE]
        curmean <- rowMeans(for_sure)
        curss2 <- rowSums((for_sure - curmean) ^ 2)

        # Iteratively adding more neighbors.
        curneighbors.to <- inverted.neighbors.to[[i]]
        curneighbors.to <- setdiff(curneighbors.to, curneighbors.from)
        candidates <- y[,curneighbors.to,drop=FALSE]
        dist2i <- colSums((candidates - y[,mnn[i]])^2)

        counter <- ncol(for_sure)
        for (x in order(dist2i)) {
            curval <- candidates[,x]
            delta <- curval - curmean
            if (any(abs(delta) > tolerance * sqrt(curss2 / (counter - 1))))  {
                next
            }
            counter <- counter + 1L
            curmean <- curmean + delta / counter
            curss2 <- curss2 + delta * (curval - curmean)
        }

        centers[,i] <- curmean
    }

    list(closest=neighbors.to.mnn$index[,1], centers=centers)
}
