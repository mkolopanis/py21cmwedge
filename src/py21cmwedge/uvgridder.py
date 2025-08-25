"""Primary UV Gridder."""

import healpy as hp
import numpy as np
from astropy import constants as const
from scipy.signal import fftconvolve
from tqdm import tqdm

from . import dft


class UVGridder(object):
    """Base uvgridder object."""

    def __init__(self):
        """Create new UVGridder object."""
        self.freqs = None
        self.uv_sum = None
        self.uvbins = {}
        self.bl_len_max = 0
        self.bl_len_min = np.inf
        self.beam = None
        self.uvw_array = None
        self.antpos = None
        self.uvf_cube = None
        self.uv_size = None
        self.uv_delta = 1  # default 1 wavelength pixels
        self.wavelength_scale = 2.0  # Max wavelength scale of antenna
        self.fwhm = 1.0
        self.uv_beam_array = None
        self.beam_sky = None
        self.omega = 2 * np.pi / (23.0 * 3600.0 + 56 * 60.0 + 4.09)
        self.t_int = 0  # integration or snapshot time of array
        self.latitude = 0  # set default array at the equator
        self.ra = None
        self.n_obs = 1  # Default to a single snapshot

    @property
    def sigma_beam(self):
        return self.fwhm / np.sqrt(4.0 * np.log(2.0))

    @sigma_beam.setter
    def sigma_beam(self, sigma):
        self.fwhm = sigma * np.sqrt(4.0 * np.log(2.0))

    @property
    def wavelength(self):
        return const.c.to_value("m/s") / self.freqs

    @wavelength.setter
    def wavelength(self, wavelength):
        self.freqs = const.c.to_value("m/s") / wavelength

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

    def set_fwhm(self, fwhm):
        """Set the FWHM of a Gaussian Beam."""
        self.fwhm = fwhm

    def set_sigma_beam(self, sigma):
        """Manually Set Gaussian standard deviation for Beam."""
        self.sigma_beam = sigma

    def gauss(self):
        """Return simple 2-d Gaussian."""
        _range = np.arange(self.uv_size)
        y, x = np.meshgrid(_range, _range)
        cen = (self.uv_size - 1) / 2.0  # correction for centering
        y = -1 * y + cen
        x = x - cen
        dist = np.linalg.norm([x, y], axis=0)
        g = np.exp(-(dist**2) / (2.0 * self.sigma_beam**2))
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
                raise ValueError(
                    "Input image is not in Healpix format. "
                    "Input image only has {0}"
                    " pixels".format(beam.size)
                )
            else:
                # make sure beam integrate to unity:
                _beam = dft.hpx_to_uv(beam, self.uv_delta)
                # We are most interested in the central lobe of the
                # main beam. We can mask out everything less that 0
                # to get around this for now, but there may be
                # a better way to handle this
                _beam = np.ma.masked_less(_beam, 0).filled(0)
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
            raise ValueError(
                "Beams of the shape {0}".format(np.shape(beam_in))
                + " are not supported"
            )

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
        v_rows1 = np.tile(self.antpos[1], (self.antpos.shape[1], 1))
        w_rows1 = np.tile(self.antpos[2], (self.antpos.shape[1], 1))

        u = u_rows1 - u_rows1.T
        v = v_rows1 - v_rows1.T
        w = w_rows1 - w_rows1.T
        return np.array([u.ravel(), v.ravel(), w.ravel()])

    def uvw_stats(self):
        """Compute the bl_len_max, and bl_len_min."""
        norms = np.linalg.norm(self.uvw_array, axis=0)
        self.bl_len_max = np.max(norms)
        if self.bl_len_max != 0:
            self.bl_len_min = np.min(norms[norms > 0])

    def simulate_observation(self):
        """Simulate the sky moving over the array."""
        # obnoxiously precise rotation speed of the Earth.
        ra = self.latitude
        hour_angles = np.arange(self.n_obs) * self.t_int * self.omega

        # delta is hte this should be the latitude of the array
        # this is used to transform u,v,w to XYZ
        delta = np.repeat(self.latitude, self.n_obs)
        # delta_prime is the ra of the observed object
        # drift is used for an object which will move over zenith
        cH = np.cos(hour_angles)
        sH = np.sin(hour_angles)
        cd = np.cos(delta)
        sd = np.sin(delta)
        cr = np.cos(ra)
        sr = np.sin(ra)
        rotation_matrix = np.array(
            [
                [cH, -sd * sH, sH * cd],
                [sr * sH, sr * sd * cH + cr * cd, -sr * cd * cH + cr * sd],
                [-cr * sH, -sd * cr * cH + sr * cd, cr * cd * cH + sr * sd],
            ]
        )
        # Using a tensordot here is faster than einsum
        # Specifying the axes is like using two transposes then numpy.dot
        new_uvw_array = np.tensordot(rotation_matrix, self.uvw_array, axes=[[1], [0]])
        new_uvw_array = new_uvw_array.reshape(new_uvw_array.shape[0], -1)
        self.uvw_array = new_uvw_array

    def uvw_to_dict(self):
        """Convert UVWs array into a dictionary.

        Assumes W term is zero or very very small.
        Elements of dictionary are lists of bls keyed by uv lengths
        """

        def to_str(arr):
            return np.array(f"{arr[0]:.3f},{arr[1]:.3f}", dtype=object)

        uv_bins, counts = np.unique(
            np.apply_along_axis(
                to_str,
                0,
                self.uvw_array,
            ),
            return_counts=True,
        )

        for uv_bin, count in zip(uv_bins, counts):
            if uv_bin == "0.000,0.000":
                continue
            self.uvbins[uv_bin] = count

    def uv_weights(self, u, v, spatial_function="triangle"):
        """Compute weights for arbitrary baseline on a gridded UV plane.

        uv must be in units of pixels.

        Parameters
        ----------
        convolve_beam: bool
            when set to true, perform an FFT convolution with the supplied beam
        spatial_function: string
            must be one of ["nearest", "triangle"].
            Nearest modes performs delta function like assignment into a uv-bin
            triangle performs simple distance based weighting of uv-bins based
              on self.wavelength_scale slope
        """
        match spatial_function.casefold():
            case "triangle":
                _range = np.arange(self.uv_size) - (self.uv_size - 1) / 2.0
                _range *= self.uv_delta
                x, y = np.meshgrid(_range, _range)
                x.shape += (1,)
                y.shape += (1,)
                x = u - x
                y = v - y
                dists = np.linalg.norm([x, y], axis=0)
                weights = 1.0 - dists / self.wavelength_scale
                weights = np.ma.masked_less_equal(weights, 0).filled(0)
                weights /= np.sum(weights, axis=(0, 1))
                weights = np.transpose(weights, [2, 0, 1])
            case "nearest":
                _range = np.arange(self.uv_size) - (self.uv_size - 1) / 2.0
                _range *= self.uv_delta
                x, y = _range, _range
                x.shape += (1,)
                y.shape += (1,)
                x = u - x
                y = v - y
                u_index = np.argmin(np.abs(x), axis=0).squeeze()
                v_index = np.argmin(np.abs(y), axis=0).squeeze()

                weights = np.zeros(
                    (x.shape[-1], self.uv_size, self.uv_size), dtype=complex
                )

                weights[range(weights.shape[0]), u_index, v_index] = 1.0
            case _:
                raise ValueError(
                    f"Unknown value for 'spatial_function': {spatial_function}"
                )

        return weights

    def __sum_uv__(self, uv_key, spatial_function="triangle"):
        """Convert uvbin dictionary to a UV-plane.

        Parameters
        ----------
        spatial_function: string
            must be one of ["nearest", "triangle"].
            Nearest modes performs delta function like assignment into a uv-bin
            triangle performs simple distance based weighting of uv-bins based
            on self.wavelength_scale slope
        """
        nbls = self.uvbins[uv_key]
        u, v = np.array(list(map(float, uv_key.split(","))))
        u /= self.wavelength
        v /= self.wavelength
        _beam = np.zeros((self.freqs.size, self.uv_size, self.uv_size), dtype=complex)
        # Create interpolation weights based on grid size and sampling
        _beam += self.uv_weights(u, v, spatial_function=spatial_function)
        self.uvf_cube += nbls * _beam

    def grid_uvw(self, convolve_beam=True, spatial_function="triangle"):
        """Create UV coverage from object data.

        Parameters
        ----------
        convolve_beam: bool
            when set to true, perform an FFT convolution with the supplied beam
        spatial_function: string
            must be one of ["nearest", "triangle"].
            Nearest modes performs delta function like assignment into a uv-bin
            triangle performs simple distance based weighting of uv-bins based
              on self.wavelength_scale slope
        """
        self.uv_size = (
            int(np.round(self.bl_len_max / self.wavelength / self.uv_delta).max() * 1.1)
            * 2
            + 5
        )
        self.uvf_cube = np.zeros(
            (self.freqs.size, self.uv_size, self.uv_size), dtype=complex
        )
        for uv_key in tqdm(self.uvbins.keys(), unit="Baseline"):
            self.__sum_uv__(uv_key, spatial_function=spatial_function)

        if convolve_beam:
            beam_array = self.get_uv_beam()
            # if only one beam was given, use that beam for all freqs
            if np.shape(beam_array)[0] < self.freqs.size:
                beam_array = np.tile(beam_array[0], (self.freqs.size, 1, 1))
            for _fq in range(self.freqs.size):
                beam = beam_array[_fq]
                self.uvf_cube[_fq] = fftconvolve(self.uvf_cube[_fq], beam, mode="same")

    def calc_all(
        self, refresh_all=True, convolve_beam=True, spatial_function="triangle"
    ):
        """Calculate all necessary info.

        Perform All calculations:
        Convert uvw_array to dict (uvw_to_dict())
        Grid uvw to plane (grid_uvw())

        Parameters
        -----------
        refresh_all : boolean,
            if true, recalculate the uvbins
        convolve_beam: bool
            when set to true, perform an FFT convolution with the supplied beam
        spatial_function: string
            must be one of ["nearest", "triangle"].
            Nearest modes performs delta function like assignment into a uv-bin
            triangle performs simple distance based weighting of uv-bins based
              on self.wavelength_scale slope
        """
        if refresh_all:
            self.uvbins = {}
            self.uvf_cube = None
        if self.n_obs > 1:
            self.simulate_observation()
        self.uvw_to_dict()
        self.grid_uvw(convolve_beam=convolve_beam, spatial_function=spatial_function)
