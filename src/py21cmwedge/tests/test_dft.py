"""Test for DFT Module."""

import numpy as np
import pytest

from py21cmwedge import dft

threshold = 1e-2


def test_dft_idft():
    """Test dft and idft return a delta function at desired location."""
    test_uv = np.zeros((19, 19))
    delta_coords = [0, 1]
    test_uv[9 + delta_coords[0], 9 + delta_coords[1]] += 1
    test_sky = dft.uv_to_hpx(test_uv, 32, 0.5)
    test_new_uv = dft.hpx_to_uv(test_sky, 0.5)
    size_diff = (test_new_uv.shape[0] - test_uv.shape[0]) // 2
    test_uv = np.pad(test_uv, (size_diff, size_diff), "edge")
    center = (test_new_uv.shape[0] - 1) // 2
    np.testing.assert_allclose(
        test_new_uv[center + delta_coords[0], center + delta_coords[1]], 1
    )


def test_rms():
    """Test the rms of the dft and idft is below threshold."""
    test_uv = np.zeros((19, 19))
    delta_coords = [0, 1]
    test_uv[9 + delta_coords[0], 9 + delta_coords[1]] += 1
    test_sky = dft.uv_to_hpx(test_uv, 32, 0.5)
    test_new_uv = dft.hpx_to_uv(test_sky, 0.5)
    size_diff = (test_new_uv.shape[0] - test_uv.shape[0]) // 2
    test_uv = np.pad(test_uv, (size_diff, size_diff), "edge")
    rms = np.sqrt(np.mean((np.abs(test_uv) - np.abs(test_new_uv)) ** 2))
    with pytest.raises(AssertionError):
        np.testing.assert_array_less(rms, threshold)
