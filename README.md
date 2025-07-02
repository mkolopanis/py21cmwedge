# py21cmwedge
[![Run Tests](https://github.com/mkolopanis/py21cmwedge/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/mkolopanis/py21cmwedge/actions/workflows/continuous-integration.yml)
[![codecov](https://codecov.io/gh/mkolopanis/py21cmwedge/graph/badge.svg?token=vxXsXrWa2j)](https://codecov.io/gh/mkolopanis/py21cmwedge)

py21cmwedge allows for quick computation of the footprint of a radio array in the (u,v) plane. This is especially useful to
determine the amount of foreground leakage in image based power spectrum analysis due to "Multi-Baseline Mode Mixing."

This package's outputs can compares directly to  FHD ([Fast Holographic Deconvolution](https://www.github.com/EorImaging/FHD))
and Eppsilon ([error propagated power spectrum with interleaved observed noise](https://github.com/EoRImaging/eppsilon))

# Package Details
## Analysis Verification
The comparison between outputs made by this package and FHD, and FHD+Eppsilon are contained in the IPython notebooks
in the notebooks subfolder and viewable directly on GitHub.

A list of the notebooks and what information they include:

* DFT_Kernel_Effects

   An analysis of the kernel of a Healpix Pixel when performing a DFT from the sky to the (u,v) plane.

* Beam_Gridding

   A comparison of how this package and FHD grid the Beam, a single baseline and two baselines onto the UV plane.

* Rotation_Synthesis

   A demonstration of how rotation synthesis is handled at the [u,v,w] level with this package

*  UVF_Uniform_Comparison

   A comparison of how this package and FHD+Eppsilon grid entire arrays in uniform weighting


# Installation

## Installing py21cmwedge
The reccommended python packaging manager to use with this repository is [uv](https://docs.astral.sh/uv/) via `uv pip install .`

However this package is compliant and installable with `pip` via `pip install .`

## Dependencies for conda users
For anaconda users, prerequisites can be installed using conda to install astropy, numpy and scipy and conda-forge
for healpy (`conda install -c conda-forge healpy`).

This package requires the following packages :

* numpy > 2
* scipy >= 1.16
* astropy >= 7
* healpy >= 1.18