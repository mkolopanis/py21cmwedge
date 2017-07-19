"""Primary UV Gridder."""
import numpy as np
import os
from astropy import constants as const
from scipy.ndimage import filters
from scipy.signal import fftconvolve
from py21cmwedge import cosmo, dft
import healpy as hp


class UVGridder(object):
    """Base uvgridder object."""

    def __init__(self):
        """Create new UVGridder object."""
        self.freqs = None
        self.uv_sum = None
        self.uvbins = {}
        self.bl_len_max = 0
        self.bl_len_min = np.Inf
        self.beam = None
        self.uvw_array = None
        self.antpos = None
        self.uvf_cube = None
        self.uv_size = None
        self.uv_delta = 1  # default 1 wavelength pixels
        self.fwhm = 1.0
        self.sigma_beam = self.fwhm / np.sqrt(4. * np.log(2.))
        self.uv_beam_array = None
        self.beam_sky = None
        self.omega = 2 * np.pi / (23. * 3600. + 56 * 60. + 4.09)
        self.t_int = 0  # integration or snapshot time of array
        self.latitude = 0  # set default array at the equator
        self.ra = None
        self.n_obs = 1  # Default to a single snapshot

    def set_uv_delta(self, delta):
        """Set grid sampling size."""
        self.uv_delta = delta

    def set_t_int(self, t_int):
        """Set the integration time of array."""
        self.t_int = t_int

    def set_omega(self, omega):
        """Manually set rotation speed of planet (rad/s)."""
        self.omega = omega

    def set_latitude(self, latitude):
        """Set latitude of array."""
        self.latitude = latitude

    def set_n_obs(self, n_obs):
        """Set number of time samples."""
        self.n_obs = n_obs

    def read_antpos(self, filename, **kwargs):
        """Read antenna position file and set positions to object.

        Provides functionality wrapper around numpy.loadtxt
        Please provide keywords:
        usecols = column numbers of East, North and Up as list (e.g [1,2,3])
        skiprows = number of row numbers to skip before data
        delimiter = string of delimiter used to parse file (e.g. ',')
        """
        antpos = np.loadtxt(filename, unpack=True, **kwargs)
        self.set_antpos(antpos)

    def set_antpos(self, antpos):
        """Manually set antenna positions.

        Antpos must be of the shape 3 x N_ants
        have the form East, North, Altitude
        """
        self.antpos = antpos
        self.uvw_array = self.__createuv__()
        self.uvw_stats()

    def set_uvw_array(self, uvw):
        """Manually set uvw array from outside source.

        Should be in the form 3 x N_uvws
        """
        if np.shape(uvw[0]) != 3 and np.shape(uvw)[1] == 3:
            uvw = np.transpose(uvw, [1, 0])
        self.uvw_array = uvw
        self.uvw_stats()

    def set_freqs(self, freq):
        """Set Frequency or Array of Frequencies."""
        if type(freq) not in [list, np.ndarray, set]:
            freq = np.array([freq])
        else:
            freq = np.asarray(list(freq))
        self.freqs = freq
        self.wavelength = const.c.to('m/s').value / self.freqs

    def set_fwhm(self, fwhm):
        """Set the FWHM of a Gaussian Beam."""
        self.fwhm = fwhm
        self.sigma_beam = self.fwhm / np.sqrt(4. * np.log(2))

    def set_sigma_beam(self, sigma):
        """Manually Set Gaussian standard deviation for Beam."""
        self.sigma_beam = sigma
        self.fwhm = self.sigma_beam * np.sqrt(4 * np.log(2))

    def gauss(self):
        """Return simple 2-d Gaussian."""
        _range = np.arange(self.uv_size)
        y, x = np.meshgrid(_range, _range)
        cen = (self.uv_size-1)/2.  # correction for centering
        y = -1 * y + cen
        x = x - cen
        dist = np.linalg.norm([x, y], axis=0)
        g = np.exp(- dist**2/(2.*self.sigma_beam**2))
        g /= g.sum()
        return g

    def set_beam(self, beam_in):
        """Set beam from outside source.

        Can be single beam or an array of beams.
        Input beam must be in Healpix Format and
        Ordered from lowest to highest frequency
        """
        # wrap single beams in a list, we wish to end with an array
        # with shape [n_freqs, uv_beam_size, uv_beam_size]
        if np.ndim(beam_in) == 1:
            beam_in = [beam_in]
        self.beam_sky = np.array(beam_in)
        beam_list = []

        for beam in beam_in:
            # check that beam is healpix array:
            if not hp.isnpixok(beam.size):
                print 'Input image is not in Healpix format'
                print 'Replacing with Gaussian Beam'
                beam_list.append(self.gauss())
            else:
                # make sure beam integrate to unity:
                _beam = dft.hpx_to_uv(beam, self.uv_delta)
                _beam /= _beam.sum()  # * self.uv_delta**2
                beam_list.append(_beam)

        beam_list = np.array(beam_list)
        self.uv_beam_array = beam_list

    def set_uv_beam(self, beam_in):
        """Manually set Beam in the uv plane.

        Input should have shape [Npix, Npix] or [Nfreqs, Npix, Npix]
        """
        if np.ndim(beam_in) == 2:
            self.uv_beam_array = np.array([beam_in])
        elif np.ndim(beam_in) == 3:
            self.uv_beam_array = np.array(beam_in)
        else:
            print ("Beams of the shape {0} "
                   "are not supported".format(np.shape(beam_in)))

    def get_uv_beam(self):
        """Return beam in the UV plane.

        If no beam set, returns a gaussian.
        """
        if self.uv_beam_array is None:
            return np.tile(self.gauss(), (self.freqs.size, 1, 1))
        else:
            return self.uv_beam_array

    def __createuv__(self):
        """Create Matrix of UVs from antenna positions."""
        u_rows1 = np.tile(self.antpos[0], (self.antpos.shape[1], 1))
        u_rows2 = np.tile(self.antpos[0], (self.antpos.shape[1], 1)).T
        v_rows1 = np.tile(self.antpos[1], (self.antpos.shape[1], 1))
        v_rows2 = np.tile(self.antpos[1], (self.antpos.shape[1], 1)).T
        w_rows1 = np.tile(self.antpos[2], (self.antpos.shape[1], 1))
        w_rows2 = np.tile(self.antpos[2], (self.antpos.shape[1], 1)).T

        u = u_rows1 - u_rows2
        v = v_rows1 - v_rows2
        w = w_rows1 - w_rows2
        return np.array([u.ravel(), v.ravel(), w.ravel()])

    def uvw_stats(self):
        """Compute the bl_len_max, and bl_len_min."""
        norms = np.linalg.norm(self.uvw_array, axis=0)
        self.bl_len_max = np.max(norms)
        self.bl_len_min = np.min(norms[norms > 0])

    def simulate_observation(self, t_int=None, n_obs=None, ra=None):
        """Simulate the sky moving over the array."""
        # obnoxiously precise rotation speed of the Earth.
        if t_int is None:
            t_int = self.t_int
        if n_obs is None:
            n_obs = self.n_obs
        if ra is None:
            ra = self.latitude
        hour_angles = np.arange(n_obs) * t_int * self.omega

        # delta is hte this should be the latitude of the array
        # this is used to transform u,v,w to XYZ
        delta = np.repeat(self.latitude, n_obs)
        # delta_prime is the ra of the observed object
        # drift is used for an object which will move over zenith
        cH = np.cos(hour_angles)
        sH = np.sin(hour_angles)
        cd = np.cos(delta)
        sd = np.sin(delta)
        cr = np.cos(ra)
        sr = np.sin(ra)
        rotation_matrix = np.array([
            [cH, -sd * sH, sH * cd],
            [sr * sH, sr * sd * cH + cr * cd, -sr * cd * cH + cr * sd],
            [-cr * sH, -sd * cr * cH + sr * cd, cr * cd * cH + sr * sd]])
        new_uvw_array = []
        for uvw in self.uvw_array.T:
            _uvw = np.einsum('jik,i', rotation_matrix, uvw)
            new_uvw_array.extend(np.transpose(_uvw, [1, 0]))
        new_uvw_array = np.array(new_uvw_array)
        all_zero = np.all(new_uvw_array == 0, axis=1)
        non_zero = np.logical_not(all_zero)
        return new_uvw_array.T

    def uvw_to_dict(self, uvw_array=None):
        """Convert UVWs array into a dictionary.

        Assumes W term is zero or very very small.
        Elemetns of dictionary are lists of bls keyed by uv lengths
        """
        if uvw_array is None:
            uvw_array = np.copy(self.uvw_array)
        for _u, _v in uvw_array[:2].T:
            if np.linalg.norm([_u, _v]) == 0:
                continue
            uv = '{0:.3f},{1:.3f}'.format(_u, _v)
            if uv in self.uvbins.keys():
                self.uvbins[uv].append(uv)
            else:
                self.uvbins[uv] = [uv]

    def uv_weights(self, u, v):
        """Compute weights for arbitrary baseline on a gridded UV plane.

        uv must be in units of pixels.
        """
        # weights = 1. - np.abs(uv - grid)/np.diff(grid)[0]
        #     weights = 1. - (np.abs(uv - grid)/np.diff(grid)[0])**2
        #     weights = np.exp( - (uv - grid)**2/(2*np.diff(grid)[0]**2))
        #     weights = np.exp( - abs(uv - grid)/(np.diff(grid)[0]))
        _range = (np.arange(self.uv_size) - (self.uv_size - 1)/2.)
        _range *= self.uv_delta
        x, y = np.meshgrid(_range, _range)
        x = u - x
        y = v - y
        weights = (1. -
                   np.linalg.norm([x, y], axis=0)/self.uv_delta)
        weights = np.ma.masked_less_equal(weights, 1e-4).filled(0)
        weights /= np.sum(weights)
        return weights

    def beamgridder(self, u, v):
        """Grid Gaussian Beam."""
        beam = np.zeros((self.freqs.size, self.uv_size, self.uv_size))
        inds = np.logical_and(v <= self.uv_size - 1,
                              u <= self.uv_size - 1)
        for _fq in xrange(self.freqs.size):
            # Create interpolation weights based on grid size and sampling
            beam[_fq] += self.uv_weights(u[_fq], v[_fq])
            # filters.gaussian_filter(beam[_fq], self.sigma_beam,
            #                   output=beam[_fq])
        return beam

    def sum_uv(self, uv_key):
        """Convert uvbin dictionary to a UV-plane."""
        uvbin = self.uvbins[uv_key]
        nbls = len(uvbin)
        u, v = np.array(map(float, uv_key.split(',')))
        u /= self.wavelength
        v /= self.wavelength
        _beam = self.beamgridder(u=u, v=v)
        self.uvf_cube += nbls * _beam

    def grid_uvw(self):
        """Create UV coverage from object data."""
        self.uv_size = int(np.round(self.bl_len_max
                                    / self.wavelength
                                    / self.uv_delta).max()) * 2 + 5
        self.uvf_cube = np.zeros(
            (self.freqs.size, self.uv_size, self.uv_size))
        for uv_key in self.uvbins.keys():
            self.sum_uv(uv_key)
        beam_array = self.get_uv_beam()
        # if only one beam was given, use that beam for all freqs
        if np.shape(beam_array)[0] < self.freqs.size:
            beam_array = np.tile(beam_array[0], (self.freqs.size, 1, 1))
        for _fq in xrange(self.freqs.size):
            beam = beam_array[_fq]
            self.uvf_cube[_fq] = fftconvolve(self.uvf_cube[_fq],
                                             beam, mode='same')

    def calc_all(self, refresh_all=True):
        """Calculate all necessary info.

        Perform All calculations:
        Convert uvw_array to dict (uvw_to_dict())
        Grid uvw to plane (grid_uvw())
        refresh_all : boolean, if true, recalculate the uvbins
        """
        if refresh_all:
            self.uvbins = {}
            self.uvf_cube = None
        observed_uvw = self.simulate_observation()
        self.uvw_to_dict(uvw_array=observed_uvw)
        self.grid_uvw()
