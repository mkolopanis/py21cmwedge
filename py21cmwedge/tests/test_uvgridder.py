"""Test for Gridder Object."""
import nose.tools as nt
import os
import copy
from py21cmwedge import UVGridder
import numpy as np

testdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class UVTest(UVGridder):
    """UVGridder test object."""

    def __init__(self):
        """Create test object."""
        # comment
        self.path = testdir
        super(UVTest, self).__init__()


def test_equality():
    """Test base object."""
    test_obj = UVTest()
    nt.assert_equal(test_obj, test_obj)


def test_unequality():
    """Test that inequality is handled."""
    test_obj = UVTest()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.freqs = 1.5e6
    nt.assert_not_equal(test_obj, test_obj2)


def test_readantpos():
    """Antenna Position file to uvws test.

    Read in antenna positions, convert to uvws,
    check equality with predefined uvw file.
    """
    test_obj = UVTest()
    test_obj.read_antpos(os.path.join(test_obj.path, 'test_antpos.txt'),
                         skiprows=1, delimiter=',')
    test_antpos = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [1, 1, 0],
                            [-1, -1, 0],
                            [1, -1, 0],
                            [-1, 1, 0]]).T
    nt.assert_true(np.allclose(test_antpos, test_obj.antpos))


def test_freq_from_int():
    """Create frequency array from integer input."""
    test_obj = UVTest()
    test_freq = 150000000
    test_obj.set_freqs(test_freq)
    nt.assert_equal(np.ndarray, type(test_obj.freqs))


def test_freq_from_float():
    """Create frequency array from float input."""
    test_obj = UVTest()
    test_freq = 150000000.
    test_obj.set_freqs(test_freq)
    nt.assert_equal(np.ndarray, type(test_obj.freqs))


def test_freq_from_list():
    """Create frequency array from list input."""
    test_obj = UVTest()
    test_freq = [150000000, 160000000]
    test_obj.set_freqs(test_freq)
    nt.assert_equal(np.ndarray, type(test_obj.freqs))


def test_freq_from_set():
    """Create frequency array from set input."""
    test_obj = UVTest()
    test_freq = set([170000000, 200000000])
    test_obj.set_freqs(test_freq)
    nt.assert_equal(np.ndarray, type(test_obj.freqs))


def test_freq_from_array():
    """Create frequency array from numpy array input."""
    test_obj = UVTest()
    test_freq = np.array([100000000., 160000000.])
    test_obj.set_freqs(test_freq)
    nt.assert_equal(np.ndarray, type(test_obj.freqs))


def test_set_uvw_traspose():
    """Test uvw transposes objects like (N_uvw x 3)."""
    test_obj = UVTest()
    test_uvw = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    test_obj.set_uvw_array(test_uvw)
    nt.assert_not_equal(test_uvw, test_obj.uvw_array)
    nt.assert_true(np.all(test_uvw.T == test_obj.uvw_array))


def test_set_fwhm():
    """Test setting fwhm."""
    test_obj = UVTest()
    test_fwhm = 2.0
    test_obj.set_fwhm(test_fwhm)
    nt.assert_equal(test_fwhm, test_obj.fwhm)


def test_sigma_from_fwhm():
    """Test setting fwhm to get the sigma."""
    test_obj = UVTest()
    test_fwhm = 2.0
    test_sigma = test_fwhm / np.sqrt(4. * np.log(2))
    test_obj.set_fwhm(test_fwhm)
    nt.assert_equal(test_sigma, test_obj.sigma_beam)


def test_set_sigma():
    """Test setting sigma."""
    test_obj = UVTest()
    test_fwhm = 2.0
    test_sigma = test_fwhm / np.sqrt(4. * np.log(2))
    test_obj.set_sigma_beam(test_sigma)
    nt.assert_equal(test_sigma, test_obj.sigma_beam)


def test_zero_uvbin():
    """Test setting all uvw's to zero."""
    test_obj = UVTest()
    test_uvw = np.zeros((3, 200))
    test_uvbin = {}
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()
    nt.assert_equal(test_obj.uvbins, test_uvbin)


def test_uvbin_is_dict():
    """Test uvbins get saved as dict."""
    test_obj = UVTest()
    test_uvw = np.zeros((3, 200)) + np.array([[14.6], [0], [0]])
    test_uvbin = {"14.600,0.000": 200*["14.600,0.000"]}
    test_obj.set_uvw_array(test_uvw)
    test_obj.uvw_to_dict()
    nt.assert_equal(test_obj.uvbins, test_uvbin)


def test_gauss_sum():
    """Test Gaussian sum is unity."""
    test_obj = UVTest()
    test_fwhm = 3
    test_sigma = test_fwhm / np.sqrt(4. * np.log(2))
    test_obj.set_sigma_beam(test_sigma)
    test_obj.uv_size = 101
    g = test_obj.gauss()
    nt.assert_equal(g.sum(), 1)


def test_uv_delta():
    """Test setting uv delta."""
    test_obj = UVTest()
    test_delta = .125
    test_obj.set_uv_delta(.125)
    nt.assert_equal(test_obj.uv_delta, test_delta)


def test_t_int():
    """Test setting integration time."""
    test_obj = UVTest()
    test_t_int = 11
    test_obj.set_t_int(11)
    nt.assert_equal(test_obj.t_int, test_t_int)


def test_set_omega():
    """Test setting rotation omega."""
    test_obj = UVTest()
    test_omega = 5.5e-5
    test_obj.set_omega(5.5e-5)
    nt.assert_equal(test_obj.omega, test_omega)


def test_set_latitude():
    """Test setting latitude of array."""
    test_obj = UVTest()
    test_lat = -40.377
    test_obj.set_latitude(-40.377 * np.pi/180)
    nt.assert_equal(test_obj.latitude, test_lat * np.pi / 180)


def test_n_obs():
    """Test setting N_obs."""
    test_obj = UVTest()
    test_n_obs = 530
    test_obj.set_n_obs(530)
    nt.assert_equal(test_obj.n_obs, test_n_obs)


def test_set_beam_type():
    """Test the type uv_beam are complex."""
    test_obj = UVTest()
    test_fwhm = 3
    test_obj.set_fwhm(test_fwhm)
    test_obj.uv_size = 13
    # put a delta function on the sky
    test_beam = np.zeros(12*128l**2)
    test_beam[0] += 1
    test_obj.set_beam(test_beam)
    nt.assert_true(isinstance(test_obj.get_uv_beam().flatten()[0], complex))


@nt.raises(ValueError)
def test_bad_beam_pix():
    """Test that module refuses beams of the wrong size."""
    test_obj = UVTest()
    test_beam = np.zeros(12 * 128l**2 - 1)
    test_obj.set_beam(test_beam)
    test_obj.set_uv_beam(test_beam)


def test_set_uv_beam():
    """Test the set_uv_beam is same as input."""
    test_obj = UVTest()
    test_beam = np.zeros((5, 5), dtype=np.complex)
    test_beam[1, 2] += 1
    test_obj.set_uv_beam(test_beam)
    nt.assert_true(np.allclose(test_obj.get_uv_beam(), test_beam))


def test_set_uv_beam_dims():
    """Test the set_uv_beam is same as input with ndims=3."""
    test_obj = UVTest()
    test_beam = np.zeros((1, 5, 5), dtype=np.complex)
    test_beam[0, 1, 2] += 1
    test_obj.set_uv_beam(test_beam)
    nt.assert_true(np.allclose(test_obj.get_uv_beam(), test_beam))


@nt.raises(ValueError)
def test_set_uv_beam_dims():
    """Test the set_uv_beam raises exception for bad ndim."""
    test_obj = UVTest()
    test_beam = np.zeros((1, 1, 5, 5), dtype=np.complex)
    test_beam[0, 0, 1, 2] += 1
    test_obj.set_uv_beam(test_beam)


def test_no_set_beam():
    """Test returns gauss when no beam set."""
    test_obj = UVTest()
    test_obj.uv_size = 13
    test_obj.set_freqs([150e6])
    test_shape = (1, 13, 13)
    beam_shape = test_obj.get_uv_beam().shape
    nt.assert_equal(test_shape, beam_shape)


def test_observation():
    """Test that simulate_observation returns correct size array."""
    test_obj = UVTest()
    test_obj.set_t_int(60)
    test_obj.set_n_obs(12)
    test_uvw = np.zeros((3, 100)) + np.array([[14.6], [0], [0]])
    test_obj.set_uvw_array(test_uvw)
    new_uvw_array = test_obj.simulate_observation()
    nt.assert_equal(np.shape(new_uvw_array)[-1], 12 * 100)


def test_weights_sum():
    """Test the uv_weights are unity normalized."""
    test_obj = UVTest()
    test_obj.set_uv_delta(.5)
    test_obj.uv_size = 13
    test_weights = test_obj.uv_weights(1, 1)
    nt.assert_equal(test_weights.sum(), 1)
