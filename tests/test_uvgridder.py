"""Test for Gridder Object."""

import copy
import os

import numpy as np
import pytest

from py21cmwedge import UVGridder

from .data import TEST_DATA_PATH


def test_equality():
    """Test base object."""
    test_obj = UVGridder()
    assert test_obj == test_obj


def test_un_equality():
    """Test that inequality is handled."""
    test_obj = UVGridder()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.freqs = 1.5e6
    assert test_obj != test_obj2


def test_read_antpos():
    """Antenna Position file to uvws test.

    Read in antenna positions, convert to uvws,
    check equality with predefined uvw file.
    """
    test_obj = UVGridder()
    test_obj.read_antpos(
        os.path.join(TEST_DATA_PATH, "test_antpos.txt"), skiprows=1, delimiter=","
    )
    test_antpos = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
        ]
    ).T
    # min baseline length 1 unit
    np.testing.assert_allclose(test_obj.bl_len_min, 1.0)
    # min baseline length 2* sqrt(2) units
    np.testing.assert_allclose(test_obj.bl_len_max, 2 * np.sqrt(2))
    np.testing.assert_allclose(test_antpos, test_obj.antpos)


def test_freq_from_int():
    """Create frequency array from integer input."""
    test_obj = UVGridder()
    test_freq = 150000000
    test_obj.set_freqs(test_freq)
    assert type(test_obj.freqs) is np.ndarray


def test_freq_from_float():
    """Create frequency array from float input."""
    test_obj = UVGridder()
    test_freq = 150000000.0
    test_obj.set_freqs(test_freq)
    assert type(test_obj.freqs) is np.ndarray


def test_freq_from_list():
    """Create frequency array from list input."""
    test_obj = UVGridder()
    test_freq = [150000000, 160000000]
    test_obj.set_freqs(test_freq)
    assert type(test_obj.freqs) is np.ndarray


def test_freq_from_set():
    """Create frequency array from set input."""
    test_obj = UVGridder()
    test_freq = set([170000000, 200000000])
    test_obj.set_freqs(test_freq)
    assert type(test_obj.freqs) is np.ndarray


def test_freq_from_array():
    """Create frequency array from numpy array input."""
    test_obj = UVGridder()
    test_freq = np.array([100000000.0, 160000000.0])
    test_obj.set_freqs(test_freq)
    assert type(test_obj.freqs) is np.ndarray


def test_set_uvw_transpose():
    """Test uvw transposes objects like (N_uvw x 3)."""
    test_obj = UVGridder()
    test_uvw = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    test_obj.set_uvw_array(test_uvw)

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, test_uvw, test_obj.uvw_array
    )
    np.testing.assert_array_equal(test_uvw.T, test_obj.uvw_array)


def test_set_fwhm():
    """Test setting fwhm."""
    test_obj = UVGridder()
    test_fwhm = 2.0
    test_obj.set_fwhm(test_fwhm)
    assert test_fwhm == test_obj.fwhm


def test_sigma_from_fwhm():
    """Test setting fwhm to get the sigma."""
    test_obj = UVGridder()
    test_fwhm = 2.0
    test_sigma = test_fwhm / np.sqrt(4.0 * np.log(2))
    test_obj.set_fwhm(test_fwhm)
    np.testing.assert_allclose(test_sigma, test_obj.sigma_beam)
    np.testing.assert_allclose(test_obj.fwhm, 2.0)


def test_set_sigma():
    """Test setting sigma."""
    test_obj = UVGridder()
    test_fwhm = 2.0
    test_sigma = test_fwhm / np.sqrt(4.0 * np.log(2))
    test_obj.set_sigma_beam(test_sigma)
    np.testing.assert_allclose(test_sigma, test_obj.sigma_beam)
    np.testing.assert_allclose(test_obj.fwhm, 2.0)


def test_zero_uvbin():
    """Test setting all uvw's to zero."""
    test_obj = UVGridder()
    test_uvw = np.zeros((3, 200))
    test_uvbin = {}
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()
    assert test_obj.uvbins == test_uvbin


def test_uvbin_is_dict():
    """Test uvbins get saved as dict."""
    test_obj = UVGridder()
    test_uvw = np.zeros((3, 200)) + np.array([[14.6], [0], [0]])
    test_uvbin = {"14.600,0.000": 200 * ["14.600,0.000"]}
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()
    assert test_obj.uvbins == test_uvbin


def test_gauss_sum():
    """Test Gaussian sum is unity."""
    test_obj = UVGridder()
    test_fwhm = 3
    test_sigma = test_fwhm / np.sqrt(4.0 * np.log(2))
    test_obj.set_sigma_beam(test_sigma)
    test_obj.uv_size = 101
    g = test_obj.gauss()

    np.testing.assert_allclose(g.sum(), 1)


def test_uv_delta():
    """Test setting uv delta."""
    test_obj = UVGridder()
    test_delta = 0.125
    test_obj.set_uv_delta(0.125)
    assert test_obj.uv_delta == test_delta


def test_t_int():
    """Test setting integration time."""
    test_obj = UVGridder()
    test_t_int = 11
    test_obj.set_t_int(11)
    assert test_obj.t_int == test_t_int


def test_set_omega():
    """Test setting rotation omega."""
    test_obj = UVGridder()
    test_omega = 5.5e-5
    test_obj.set_omega(5.5e-5)
    assert test_obj.omega == test_omega


def test_set_latitude():
    """Test setting latitude of array."""
    test_obj = UVGridder()
    test_lat = -40.377
    test_obj.set_latitude(-40.377 * np.pi / 180)
    np.testing.assert_allclose(test_obj.latitude, test_lat * np.pi / 180)


def test_n_obs():
    """Test setting N_obs."""
    test_obj = UVGridder()
    test_n_obs = 530
    test_obj.set_n_obs(530)
    assert test_obj.n_obs == test_n_obs


def test_set_beam_type():
    """Test the type uv_beam are complex."""
    test_obj = UVGridder()
    test_fwhm = 3
    test_obj.set_fwhm(test_fwhm)
    test_obj.uv_size = 13
    # put a delta function on the sky
    test_beam = np.zeros(12 * 128**2)
    test_beam[0] += 1
    test_obj.set_beam(test_beam)

    assert isinstance(test_obj.get_uv_beam().flatten()[0], complex)


def test_bad_beam_pix():
    """Test that module refuses beams of the wrong size."""
    test_obj = UVGridder()
    test_beam = np.zeros(12 * 128**2 - 1)
    with pytest.raises(ValueError):
        test_obj.set_beam(test_beam)
        test_obj.set_uv_beam(test_beam)


def test_set_uv_beam():
    """Test the set_uv_beam is same as input."""
    test_obj = UVGridder()
    test_beam = np.zeros((5, 5), dtype=complex)
    test_beam[1, 2] += 1
    test_obj.set_uv_beam(test_beam)
    np.testing.assert_allclose(test_obj.get_uv_beam()[0], test_beam)


def test_set_uv_beam_good_dims():
    """Test the set_uv_beam is same as input with n dims=3."""
    test_obj = UVGridder()
    test_beam = np.zeros((1, 5, 5), dtype=complex)
    test_beam[0, 1, 2] += 1
    test_obj.set_uv_beam(test_beam)
    np.testing.assert_allclose(test_obj.get_uv_beam(), test_beam)


def test_set_uv_beam_bad_dims():
    """Test the set_uv_beam raises exception for bad ndim."""
    test_obj = UVGridder()
    test_beam = np.zeros((1, 1, 5, 5), dtype=complex)
    test_beam[0, 0, 1, 2] += 1
    with pytest.raises(ValueError):
        test_obj.set_uv_beam(test_beam)


def test_no_set_beam():
    """Test returns gauss when no beam set."""
    test_obj = UVGridder()
    test_obj.uv_size = 13
    test_obj.set_freqs([150e6])
    test_shape = (1, 13, 13)
    beam_shape = test_obj.get_uv_beam().shape
    assert test_shape == beam_shape


def test_observation():
    """Test that simulate_observation returns correct size array."""
    test_obj = UVGridder()
    test_obj.set_t_int(60)
    test_obj.set_n_obs(12)
    test_uvw = np.zeros((3, 100)) + np.array([[14.6], [0], [0]])
    test_obj.set_uvw_array(test_uvw)
    test_obj.simulate_observation()
    assert np.shape(test_obj.uvw_array)[-1] == 12 * 100


def test_weights_sum():
    """Test the uv_weights are unity normalized."""
    test_obj = UVGridder()
    test_obj.set_uv_delta(0.5)
    test_obj.uv_size = 13
    test_weights = test_obj.uv_weights(1, 1)
    assert test_weights.sum() == 1


def test_sum_uv():
    """Test sum_uv errors if grid_uvw not called."""
    test_obj = UVGridder()
    test_obj.uv_size = 51
    test_obj.uv_delta = 0.5
    test_obj.set_freqs(150e6)
    test_uvw = np.zeros((3, 10)) + np.array([[14.6], [0], [0]])
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()

    with pytest.raises(TypeError):
        test_obj.__sum_uv__(test_obj.uvbins.keys()[0])


def test_grid_uv():
    """Test grid_uv sets up complex type."""
    test_obj = UVGridder()
    test_obj.uv_delta = 0.5
    test_obj.set_freqs(150e6)
    test_uvw = np.zeros((3, 10)) + np.array([[14.6], [0], [0]])
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()
    test_obj.grid_uvw()
    assert isinstance(test_obj.uvf_cube.flatten()[0], complex)


def test_grid_uv_deltas():
    """Test grid_uv sets up complex type."""
    test_obj = UVGridder()
    test_obj.uv_delta = 0.5
    test_obj.set_freqs(150e6)
    test_uvw = np.zeros((3, 10)) + np.array([[14.6], [0], [0]])
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()
    test_obj.grid_uvw(convolve_beam=False, spatial_function="nearest")

    np.testing.assert_allclose(test_obj.uvf_cube[0, 33, 18], 10)
