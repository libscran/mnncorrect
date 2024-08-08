#' @export
#' @importFrom BiocNeighbors findMutualNN queryKNN
mnncorrect.ref <- function(ref, target, k=15, nmads=3, iterations=2, trim=0.25, mass.cap=-1) {
    pairings <- findMutualNN(t(ref), t(target), k1=k)
    mnn.r <- unique(pairings$first)
    mnn.t <- unique(pairings$second)

    # Computing centers of mass for each MNN-involved cell.
    r.out <- center_of_mass(ref, mnn.r, k=k, nmads=nmads, iterations=iterations, trim=trim, mass.cap=mass.cap)
    centers.r <- r.out$centers

    t.out <- center_of_mass(target, mnn.t, k=k, nmads=nmads, iterations=iterations, trim=trim, mass.cap=-1)
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

#' @importFrom stats median mad
#' @importFrom BiocNeighbors queryKNN
center_of_mass <- function(y, mnn, k, nmads, iterations, trim, mass.cap) {
    ty <- t(y)

    if (mass.cap < 0 || ncol(y) <= mass.cap) {
        chosen <- seq_len(ncol(y))
        closest <- queryKNN(query=ty, X=ty[mnn,,drop=FALSE], k=k)
    } else {
        chosen <- floor(head(seq(from=1, to=ncol(y)+1, length.out=mass.cap+1), -1))
        closest <- queryKNN(query=ty[chosen,,drop=FALSE], X=ty[mnn,,drop=FALSE], k=k)
    }

    limit <- median(closest$distance) + mad(closest$distance) * nmads

    idx <- closest$index
    idx[closest$distance > limit] <- NA
    neighbors <- split(rep(chosen, k), idx)

    centers <- y[,mnn,drop=FALSE] # the center of mass for any MNN-involved point without neighbors is just itself.
    for (i in names(neighbors)) {
        curneighbors <- neighbors[[i]]
        candidates <- y[,curneighbors,drop=FALSE]
        centers[,as.integer(i)] <- robust_centroid(candidates, iterations=iterations, trim=trim)
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
