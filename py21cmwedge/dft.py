"""Beam Handling and DFT Modules."""
import healpy as hp
import numpy as np


def hpx_to_uv(map_in, uv_delta):
    """Perform Discrete Fourier Transform from the healpix map to uv plane.

    UV plane size determined by healpix pixel size measued in wavelengths
    uv pixel size determined by uv_delta.
    """
    # Get info of the input map
    nside = hp.get_nside(map_in)
    pix_size = 1./hp.nside2resol(nside)  # 1./pix_resol to get wavelengths
    uv_size = pix_size / 2.  # Only create a grid as large as the pixel size/2

    # Create a the _u,_v grid and baselines vectors
    _range = np.arange(uv_size).astype(np.float64)
    center = (uv_size - 1)/2
    _range -= center
    _range *= uv_delta
    _v, _u = np.meshgrid(_range, _range)

    uv_beam = np.zeros_like(_u, type=np.complex)

    # Before DFT, get all pixels above the horizon
    # and stack the unit vectors (x,y) into array
    _xyz = hp.pix2vec(nside, np.arange(map_in.size))
    pix_above_horizon = np.where(_xyz[2] >= 0)[0]
    s_ = np.array([_xyz[0], _xyz[1]])  # stack x,y into array

    for cnt in xrange(_u.ravel().size):
        __u, __v = _u.ravel()[cnt], _v.ravel()[cnt]
        b_dot_s = np.einsum('i,i...', [__u, __v], s_)
        phases = np.exp(-2j * np.pi * b_dot_s[pix_above_horizon])
        uv_plane.ravel()[cnt] = np.mean(map_in[pix_above_horizon] * phases)

    return uv_beam


def uv_to_hpx(uv_beam, nside, uv_delta):
    """Perform Discrete Fourier Transform from the uv plane to the sky.

    Choose Nsize parameter for Healpix map resolution.
    Provide uv_delta, pixel size of uv plane
    """
    uv_size = uv_beam.shape[0]
    _range = np.arange(uv_size).astype(np.float64)
    center = (uv_size - 1)/2
    _range -= center
    _range *= uv_delta
    _v, _u = np.meshgrid(_range, _range)

    sky_beam = np.zeros(hp.nside2npix(nside), type=np.complex)
    # Before DFT, get all pixels above the horizon
    # and stack the unit vectors (x,y) into array
    _xyz = hp.pix2vec(nside, np.arange(sky_beam.size))
    pix_above_horizon = np.where(_xyz[2] >= 0)[0]
    s_ = np.array([_xyz[0], _xyz[1]])  # stack x,y into array

    # Perform the DFT for each sky pixel
    for pix in pix_above_horizon:
        b_dot_s = np.einsum('i...,i', [_u, _v], s_.T[pix])
        phases = np.exp(2j * np.pi * b_dot_s)
        sky_beam[pix] = np.sum(uv_beam * phases)

    return sky_beam
