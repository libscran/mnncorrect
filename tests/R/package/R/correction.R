#' @export
#' @importFrom BiocNeighbors findMutualNN queryKNN
mnncorrect.ref <- function(ref, target, k=15, nmads=3, iterations=2, trim=0.25, mass.cap=0) {
    pairings <- findMutualNN(t(ref), t(target), k1=k)
    mnn.r <- unique(pairings$first)
    mnn.t <- unique(pairings$second)

    # Computing centers of mass for each MNN-involved cell.
    r.out <- center_of_mass(ref, mnn.r, k=k, nmads=nmads, mass.cap=mass.cap)
    centers.r <- r.out$centers

    t.out <- center_of_mass(target, mnn.t, k=k, nmads=nmads, mass.cap=0)
    closest.t <- t.out$closest
    centers.t <- t.out$centers

    # Finishing it off.
    for (x in seq_len(ncol(target))) {
        closest <- closest.t$index[x,]

        keep <- pairings$second %in% mnn.t[closest]
        used.ref <- pairings$first[keep]
        used.target <- pairings$second[keep]

        candidates <- centers.r[,match(used.ref, mnn.r),drop=FALSE] - centers.t[,match(used.target, mnn.t),drop=FALSE]
        correction <- robust_centroid(candidates, iterations=iterations, trim=trim) 

        target[,x] <- target[,x] + correction
    }

    target 
}

#' @importFrom utils head tail
#' @importFrom stats median mad
#' @importFrom BiocNeighbors queryKNN
center_of_mass <- function(y, mnn, k, nmads, mass.cap) {
    ty <- t(y)

    if (mass.cap <= 0 || ncol(y) <= mass.cap) {
        chosen <- seq_len(ncol(y))
        closest <- queryKNN(query=ty, X=ty[mnn,,drop=FALSE], k=k)
    } else {
        chosen <- floor(head(seq(from=1, to=ncol(y)+1, length.out=mass.cap+1), -1))
        closest <- queryKNN(query=ty[chosen,,drop=FALSE], X=ty[mnn,,drop=FALSE], k=k)
        closest$index[] <- chosen[closest$index]
    }

    idx <- closest$index
    neighbors <- split(rep(chosen, k), idx)
    min.required <- k

    centers <- y[,mnn,drop=FALSE] # the center of mass for any MNN-involved point without neighbors is just itself.
    for (i in names(neighbors)) {
        curneighbors <- neighbors[[i]]
        candidates <- y[,curneighbors,drop=FALSE]

        dist2i <- colSums((candidates - centers[,as.integer(i)])^2)
        o <- order(dist2i)

        for_sure <- candidates[,head(o, min.required),drop=FALSE]
        curmean <- rowMeans(for_sure)
        curss2 <- rowSums((for_sure - curmean) ^ 2)

        counter <- ncol(for_sure)
        for (x in tail(o, -min.required)) {
            curval <- candidates[,x]
            delta <- curval - curmean
            if (any(abs(delta) > nmads * sqrt(curss2 / (counter - 1))))  {
                next
            }
            counter <- counter + 1L
            curmean <- curmean + delta / counter
            curss2 <- curss2 + delta * (curval - curmean)
        }

        centers[,as.integer(i)] <- curmean
    }

    list(closest=closest, centers=centers)
}

#' @importFrom stats quantile
robust_centroid <- function(y, iterations, trim) {
    center <- rowMeans(y)
    for (i in seq_len(iterations)) {
        delta <- sqrt(colSums((y - center)^2))

        # We give it a bit of a bump to avoid problems with numerical precision and ties.
        keep <- delta <= quantile(delta, 1-trim) * 1.00000001 

        center <- rowMeans(y[,keep,drop=FALSE])
    }

    center
}
