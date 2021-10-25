# C++ library for MNN correction

## Overview

This library provides functionality for batch correction of arbitrary data via the use of mutual nearest neighbors (MNNs).
MNN correction was initially described in the context of single-cell RNA sequencing data analysis (see [Haghverdi et al., 2018](https://doi.org/10.1038/nbt.4091))
but the same methodology can be applied for any high-dimensional data containing shared populations across multiple batches.
The MNN implementation here is based on the `fastMNN()` function in the [**batchelor** package](https://bioconductor.org/packages/batchelor),
which provides a number of improvements and speed-ups over the original method in the Haghverdi paper.
