#' @export
#' @importFrom BiocNeighbors findMutualNN queryKNN
mnncorrect.ref <- function(ref, target, k=15, nmads=3, iterations=2, trim=0.25) {
    pairings <- findMutualNN(t(ref), t(target), k1=k)
    mnn.r <- unique(pairings$first)
    mnn.t <- unique(pairings$second)

    # Computing centers of mass for each MNN-involved cell.
    r.out <- center_of_mass(ref, mnn.r, k=k, nmads=nmads, iterations=iterations, trim=trim)
    closest.r <- r.out$closest
    centers.r <- r.out$centers

    t.out <- center_of_mass(target, mnn.t, k=k, nmads=nmads, iterations=iterations, trim=trim)
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
center_of_mass <- function(y, mnn, k, nmads, iterations, trim) {
    ty <- t(y)
    closest <- queryKNN(query=ty, X=ty[mnn,,drop=FALSE], k=k)
    limit <- median(closest$distance) + mad(closest$distance) * nmads

    idx <- closest$index
    idx[closest$distance > limit] <- NA
    neighbors <- split(rep(seq_len(ncol(y)), k), idx)

    centers <- matrix(0, nrow(y), length(neighbors))
    for (i in seq_along(neighbors)) {
        candidates <- y[,neighbors[[i]],drop=FALSE]
        centers[,i] <- robust_centroid(candidates, iterations=iterations, trim=trim)
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

#    ref <- robust_average(y, iterations, trim)
#    if (!isTRUE(all.equal(center, ref))) {
#        cat("DELTA IS:\n")
#        print(sort(delta))
#        print(sum(keep))
#        print(quantile(delta, 1-trim) * 1.00000001)
#        cat("REF IS:\n")
#        print(ref)
#        cat("CENTER IS:\n")
#        print(center)
#        stop("ARGGHH")
#    }

    center
}
