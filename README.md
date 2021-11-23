# C++ library for MNN correction

![Unit tests](https://github.com/LTLA/CppMnnCorrect/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/CppMnnCorrect/actions/workflows/doxygenate.yaml/badge.svg)
![R comparison](https://github.com/LTLA/CppMnnCorrect/actions/workflows/compare-R.yaml/badge.svg)
[![codecov](https://codecov.io/gh/LTLA/CppMnnCorrect/branch/master/graph/badge.svg?token=J3dxS3MtT1)](https://codecov.io/gh/LTLA/CppMnnCorrect)

## Overview

This library provides functionality for batch correction of arbitrary data via the use of mutual nearest neighbors (MNNs).
MNN correction was initially described in the context of single-cell RNA sequencing data analysis (see [Haghverdi et al., 2018](https://doi.org/10.1038/nbt.4091))
but the same methodology can be applied for any high-dimensional data containing shared populations across multiple batches.
The MNN implementation here is based on the `fastMNN()` function in the [**batchelor** package](https://bioconductor.org/packages/batchelor),
which provides a number of improvements and speed-ups over the original method in the Haghverdi paper.

## Quick start

Given a dense feature-by-observation matrix in column-major format and a batch assignment vector for each observation (i.e., column), we can compute corrected values:

```cpp
#include "mnncorrect/MnnCorrect.hpp"

std::vector<double> matrix(ndim * nobs); // fill with values...
std::vector<int> batch(nobs) // fill with values...

mnncorrect::MnnCorrect<> runner;
std::vector<double> output(ndim * nobs);
runner.run(ndim, nobs, matrix.data(), batch.data(), output.data());
```

See the [reference documentation](https://ltla.github.io/CppMnnCorrect) for more details.

## Theoretical details 

We assume that (i) our batches share some subpopulations, and (ii) even after the addition of an arbitrary batch effect,
the cells from one subpopulation in one batch are still closer to the cells in the corresponding subpopulation in the other batch (and vice versa) when compared to cells in different subpopulations.
Thus, by identifying pairs of cells that are MNNs, we can determine which subpopulations are shared across batches. 
Any differences in location between batches for the shared subpopulations are attributed to batch effects and targeted for removal.
In contrast, a subpopulation unique to a single batch will not contain any MNNs (and thus will not interfere with correction), 
as it will not have a corresponding subpopulation in the other batch for which it can be the closest neighbor.

To remove batch effects, we consider one batch to be the "reference" and another to be the "target".
For each MNN pair, we compute a correction vector that moves the target batch towards the reference.
For each observation in the target batch, we identify the closest `k` observations in the same batch that are part of a MNN pair (i.e., "MNN-involved observations").
We then compute a robust average across all of the correction vectors associated with those closest observations.
This average is used as the correction vector for that observation, allowing the correction to adjust to local variations in the magnitude and direction of the batch effect.

The correction vector for each MNN pair is not directly computed from the two paired observations.
Rather, for each observation, we compute a "center of mass" using neighboring points from the same batch.
Specifically, given an observation **v** that is part of an MNN pair, we find the set **S** of observations in the same batch for which **v** is one of the `k` closest neighbors.
We take the robust average of the coordinates in **S**, which is defined as the center of mass location for **v**.
The correction vector is then computed between the two center of mass locations across batches.
This aims to eliminate "kissing" effects where the correction only brings the surfaces of the batches into contact.

As an additional note, the center of mass calculation excludes all observations that are more than a threshold distance away from **v**.
To choose this threshold, we compute the distances from each observation in **S** to **v**, and we pool this across all **v** involved in MNN pairs.
We then use the "median + 3 MAD" approach on the resulting distribution of distances to define a threshold.
The aim is to ensure that the threshold is large enough to capture cells from the same subpopulation without including cells from distinct subpopulations.
Our assumption is that most cells in each batch belong to a shared subpopulation.

In the case of >2 batches, we progressively merge the batches to a reference, i.e., after each merge step, the merged dataset is used as the new reference to merge the next batch, and so on.
By default, we use the largest batch as the reference and we choose the batch with the most MNN pairs to merge at each step.
This ensures that we have a plentiful number of MNNs for a stable correction at each step.
Indeed, it allows us to merge together batches that share no subpopulations as long as there is an intervening batch that can "plug the gap", so to speak.

## Examples

The `tests/R/examples` directory contains a few examples using the C++ code on some real datasets (namely, single-cell RNA-seq datasets).
To run these, install the package at `tests/R/package` (this requires the [**scran.chan**](https://github.com/LTLA/scran.chan) package, which also wraps this C++ library in a more complete package).

`pbmc`: mergesthe PBMC 3K and 4K datasets from 10X Genomics.
These are technical replicates (I think) so a complete merge is to be expected.

![pbmc-output](https://raw.githubusercontent.com/LTLA/CppMnnCorrect/images/tests/R/examples/pbmc/output.png)

`pancreas`: merges the [Grun et al. (2016)](https://dx.doi.org/10.1016%2Fj.stem.2016.05.010) and [Muraro et al. (2016)](https://doi.org/10.1016/j.cels.2016.09.002) datasets.
I believe this involves data from different patients but using the same-ish technology.

![pancreas-output](https://raw.githubusercontent.com/LTLA/CppMnnCorrect/images/tests/R/examples/pancreas/output.png)

`neurons`: merges the [Zeisel et al. (2015)](https://doi.org/10.1126/science.aaa1934) and [Tasic et al. (2016)](https://doi.org/10.1038/nn.4216) datasets.
This involves different technologies and different cell populations.

![neurons-output](https://raw.githubusercontent.com/LTLA/CppMnnCorrect/images/tests/R/examples/neurons/output.png)

## Building projects

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```
include(FetchContent)

FetchContent_Declare(
  libscran
  GIT_REPOSITORY https://github.com/LTLA/libscran
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(libscran)
```

Then you can link to **libscran** to make the headers available during compilation:

```
# For executables:
target_link_libraries(myexe libscran)

# For libaries
target_link_libraries(mylib INTERFACE libscran)
```

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
Note that this requires manual management of the [**knncolle**](https://github.com/LTLA/knncolle) library for k-nearest neighbor detection.
This in turn has a suite of its own dependencies, see the link for details.

## References

Haghverdi L, Lun ATL, Morgan MD, Marioni JC (2018).
Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors.
_Nat. Biotechnol._ 36(5):421-427
