"""Beam Handling and DFT Modules."""

import healpy as hp
import numpy as np


def hpx_to_uv(map_in, uv_delta):
    """Perform Discrete Fourier Transform from the healpix map to uv plane.

    UV plane size determined by healpix pixel size measured in wavelengths
    uv pixel size determined by uv_delta.
    """
    # Get info of the input map
    nside = hp.get_nside(map_in)
    pix_size = 1.0 / hp.nside2resol(nside)  # 1./pix_resol to get wavelengths
    # Only create a grid as large as the +/- pixel_size/2
    uv_size = np.ceil(pix_size / 2.0)  # in wavelengths
    # Cut the size in half so only extends the amount of 1 Healpix
    # pixel in wavelengths
    # The corners will be slightly longer but that should be okay.
    # Perhaps the best way to fix this would be to dynamically
    # calculate when the main lobe ends and only go out to there
    uv_size /= uv_delta  # in pixels
    uv_size = int(uv_size)
    # we want to make sure the uv_size is always odd
    if uv_size % 2 == 0:
        uv_size += 1
    # Create a the _u,_v grid and baselines vectors
    _range = np.arange(uv_size).astype(np.float64)
    center = (uv_size - 1) / 2.0
    _range -= center
    _range *= uv_delta
    _u, _v = np.meshgrid(_range, _range)

    uv_beam = np.zeros_like(_u, dtype=complex).ravel()

    # Before DFT, get all pixels above the horizon
    # and stack the unit vectors (x,y) into array
    _xyz = hp.pix2vec(nside, np.arange(map_in.size))
    pix_above_horizon = np.where(_xyz[2] >= 0)[0]
    s_ = np.array([_xyz[0], _xyz[1]])  # stack x,y into array
    _u = _u.ravel()
    _v = _v.ravel()
    _uv = np.array([_u, _v])

    b_dot_s = np.einsum("ij,ik -> jk", _uv, s_)
    phases = np.exp(-2j * np.pi * b_dot_s[:, pix_above_horizon])
    uv_beam = np.mean(map_in[pix_above_horizon] * phases, axis=1)

    return uv_beam.reshape((uv_size, uv_size))


def uv_to_hpx(uv_beam, nside, uv_delta):
    """Perform Discrete Fourier Transform from the uv plane to the sky.

    Choose Nside parameter for Healpix map resolution.
    Provide uv_delta, pixel size of uv plane
    """
    uv_size = uv_beam.shape[0]
    uv_beam = uv_beam.ravel()
    uv_beam.shape += (1,)

    _range = np.arange(uv_size).astype(np.float64)
    center = (uv_size - 1) / 2.0
    _range -= center
    _range *= uv_delta
    _u, _v = np.meshgrid(_range, _range)

    sky_beam = np.zeros(hp.nside2npix(nside), dtype=complex)
    # Before DFT, get all pixels above the horizon
    # and stack the unit vectors (x,y) into array
    _xyz = hp.pix2vec(nside, np.arange(sky_beam.size))
    pix_above_horizon = np.where(_xyz[2] >= 0)[0]
    s_ = np.array([_xyz[0], _xyz[1]])[:, pix_above_horizon]  # stack x,y into array
    _uv = np.array([_u.ravel(), _v.ravel()])

    # Perform the DFT for each sky pixel
    b_dot_s = np.einsum("ij,ik-> jk", _uv, s_)
    phases = np.exp(2j * np.pi * b_dot_s)
    sky_beam[pix_above_horizon] = np.sum(uv_beam * phases, axis=0)

    return sky_beam
