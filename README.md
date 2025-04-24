# C++ library for MNN correction

![Unit tests](https://github.com/libscran/mnncorrect/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/libscran/mnncorrect/actions/workflows/doxygenate.yaml/badge.svg)
![R comparison](https://github.com/libscran/mnncorrect/actions/workflows/compare-R.yaml/badge.svg)
[![codecov](https://codecov.io/gh/libscran/mnncorrect/branch/master/graph/badge.svg?token=J3dxS3MtT1)](https://codecov.io/gh/libscran/mnncorrect)

## Overview

This library provides functionality for batch correction of arbitrary data via the use of mutual nearest neighbors (MNNs).
MNN correction was initially described in the context of single-cell RNA sequencing data analysis (see [Haghverdi et al., 2018](https://doi.org/10.1038/nbt.4091))
but the same methodology can be applied for any high-dimensional data containing shared populations across multiple batches.
The MNN implementation here is based on the `fastMNN()` function in the [**batchelor** package](https://bioconductor.org/packages/batchelor),
which provides a number of improvements and speed-ups over the original method in the Haghverdi paper.

## Quick start

Consider a dense matrix in column-major format where rows are dimensions (e.g., principal components) and cells are columns,
and a vector of integers specifying the batch of origin for each cell.
These are supplied to the `mnncorrect::compute()` function to compute corrected values:

```cpp
#include "mnncorrect/MnnCorrect.hpp"

std::vector<double> matrix(ndim * nobs); // fill with values...
std::vector<int> batch(nobs) // fill with batch IDs from [0, num_batches)

mnncorrect::Options<int, double> opt;
std::vector<double> output(ndim * nobs);
mnncorrect::compute(ndim, nobs, matrix.data(), batch.data(), output.data(), opt);
```

We also support batches in separate arrays, storing the corrected values for all batches in a single output array:

```cpp
int nbatches = 3;
std::vector<int> batch_size;
std::vector<std::vector<double> > batches;
for (int b = 0; b < 3; ++b) { // mocking up three batches of different size.
    batch_size.push_back((b + 1) * 100);
    batch.resize(ndim * batch_size.back()); // fill with values...
}

std::size_t total_size = std::accumulate(batch_size.begin(), batch_size.end(), 0);
std::vector<double> output(ndim * total_size);
mnncorrect::compute(ndim, batch_size, batch_ptrs, output.data(), opt);
```

Advanced users can also fiddle with the options: 

```cpp
opt.num_neighbors = 10;
opt.num_threads = 3;

// Manually specify your own merge order:
opt.order = std::vector<std::size_t>{ 3, 1, 0, 2 };

// Change the nearest-neighbor search algorithm:
opt.builder.reset(new knncolle_annoy::AnnoyBuilder<Annoy::Euclidean>);

// Approximate the center-of-mass calculations for greater speed.
opt.mass_cap = 100000;
```

See the [reference documentation](https://libscran.github.io/mnncorrect) for more details.

## Theoretical details 

We assume that (i) our batches share some subpopulations, and (ii) even after the addition of an arbitrary batch effect,
the cells from one subpopulation in one batch are still closer to the cells in the corresponding subpopulation in the other batch (and vice versa) when compared to cells in different subpopulations.
Thus, by identifying pairs of cells that are MNNs, we can determine which subpopulations are shared across batches. 
Any differences in location between batches for the shared subpopulations are attributed to batch effects and targeted for removal.
In contrast, a subpopulation unique to a single batch will not contain any MNNs (and thus will not interfere with correction), 
as it will not have a corresponding subpopulation in the other batch for which it can be the closest neighbor.

To remove batch effects, we consider one batch to be the "reference" and another to be the "target".
For each MNN pair, we compute a correction vector that moves the target batch towards the reference.
For each cell in the target batch, we identify the closest $k$ cells in the same batch that are part of a MNN pair (i.e., "MNN-involved cells").
We then compute a robust average across all of the correction vectors associated with those closest cells.
This average is used as the correction vector for that cell, allowing the correction to adjust to local variations in the magnitude and direction of the batch effect.

The correction vector for each MNN pair is not directly computed from its two paired cells.
Rather, for each cell, we compute a "center of mass" using neighboring points from the same batch.
Specifically:

- Given a cell $v$ that is part of an MNN pair, we find the set $\mathbf{S}$ of cells in the same batch for which $v$ is one of the $k$ closest neighbors.
  We take the robust average of the coordinates in $\mathbf{S}$, which is defined as the center of mass location for $v$.
  The correction vector is then computed between the two center of mass locations across batches.
  This aims to eliminate "kissing" effects where the correction only brings the surfaces of the batches into contact.
- The center of mass calculation excludes all cells that are more than a threshold distance away from $v$.
  We compute the distances from each cell in $\mathbf{S}$ to $v$, pool these distances across all $v$ involved in MNN pairs, and define the threshold as "median + $x$ MADS" on the distances.
  The aim is to ensure that the threshold is large enough to capture cells from the same subpopulation without including cells from distinct subpopulations.
  Our assumption is that most cells in each batch belong to a shared subpopulation.
- We can "cap" the size of the reference dataset when computing the center of mass for each reference cell involved in an MNN pair.
  This is done by only considering every $n$-th cell for inclusion in $\mathbf{S}$ where $n > 1$.
  The aim is to avoid an increase in the computational cost of each merge step as the reference dataset grows after the second batch (see below).

In the case of >2 batches, we progressively merge the batches to a reference, i.e., after each merge step, the merged dataset is used as the new reference to merge the next batch, and so on.
By default, we pick the batch with the largest residual sum of squares as the first reference, though this can be changed to, e.g., use the largest or most variable batch.
At each step, we choose the batch with the most MNN pairs to merge, which ensures that we have a plentiful number of MNNs for a stable correction.
This strategy allows us to eventually merge batches that share no subpopulations as long as there is an intervening batch that can "plug the gap", so to speak.

## Examples

The `tests/R/examples` directory contains a few examples using the C++ code on some real datasets (namely, single-cell RNA-seq datasets).
To run these, install the package at `tests/R/package` (this requires the [**scran.chan**](https://github.com/LTLA/scran.chan) package, which also wraps this C++ library in a more complete package).

`pbmc`: mergesthe PBMC 3K and 4K datasets from 10X Genomics.
These are technical replicates (I think) so a complete merge is to be expected.

![pbmc-output](https://raw.githubusercontent.com/libscran/mnncorrect/images/tests/R/examples/pbmc/output.png)

`pancreas`: merges the [Grun et al. (2016)](https://dx.doi.org/10.1016%2Fj.stem.2016.05.010) and [Muraro et al. (2016)](https://doi.org/10.1016/j.cels.2016.09.002) datasets.
I believe this involves data from different patients but using the same-ish technology.

![pancreas-output](https://raw.githubusercontent.com/libscran/mnncorrect/images/tests/R/examples/pancreas/output.png)

`neurons`: merges the [Zeisel et al. (2015)](https://doi.org/10.1126/science.aaa1934) and [Tasic et al. (2016)](https://doi.org/10.1038/nn.4216) datasets.
This involves different technologies and different cell populations.

![neurons-output](https://raw.githubusercontent.com/libscran/mnncorrect/images/tests/R/examples/neurons/output.png)

## Building projects

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  mnncorrect
  GIT_REPOSITORY https://github.com/libscran/mnncorrect
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(mnncorrect)
```

Then you can link to **libscran** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe mnncorrect)

# For libaries
target_link_libraries(mylib INTERFACE mnncorrect)
```

### CMake with `find_package()`

```cmake
find_package(libscran_mnncorrect CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE libscran::mnncorrect)
```

To install the library, use:

```sh
mkdir build && cd build
cmake .. -DMNNCORRECT_TESTS=OFF
cmake --build . --target install
```

By default, this will use `FetchContent` to fetch all external dependencies.
If you want to install them manually, use `-DMNNCORRECT_FETCH_EXTERN=OFF`.
See [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find compatible versions of each dependency.

### Manual

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt), which also need to be made available during compilation.

## References

Haghverdi L, Lun ATL, Morgan MD, Marioni JC (2018).
Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors.
_Nat. Biotechnol._ 36(5):421-427
