"""Test for DFT Module."""
import nose.tools as nt
import nose
import os
import copy
import numpy as np
from py21cmwedge import dft

testdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
threshold = 1e-2


def test_dft_idft():
    """Test dft and idft return a delta function at desired location."""
    test_uv = np.zeros((19, 19))
    delta_coords = [0, 1]
    test_uv[9 + delta_coords[0], 9 + delta_coords[1]] += 1
    test_sky = dft.uv_to_hpx(test_uv, 32, .5)
    test_new_uv = dft.hpx_to_uv(test_sky, .5)
    size_diff = (test_new_uv.shape[0] - test_uv.shape[0])/2
    test_uv = np.pad(test_uv, (size_diff, size_diff), 'edge')
    center = (test_new_uv.shape[0] - 1)/2
    nt.assert_equal(test_new_uv[center + delta_coords[0],
                                center + delta_coords[1]], 1)


@nt.raises(AssertionError)
def test_rms():
    """Test the rms of the dft and idft is below threshold."""
    test_uv = np.zeros((19, 19))
    delta_coords = [0, 1]
    test_uv[9 + delta_coords[0], 9 + delta_coords[1]] += 1
    test_sky = dft.uv_to_hpx(test_uv, 32, .5)
    test_new_uv = dft.hpx_to_uv(test_sky, .5)
    size_diff = (test_new_uv.shape[0] - test_uv.shape[0])/2
    test_uv = np.pad(test_uv, (size_diff, size_diff), 'edge')
    rms = np.sqrt(np.mean((np.abs(test_uv) - np.abs(test_new_uv))**2))
    nt.assert_less(rms, threshold)
