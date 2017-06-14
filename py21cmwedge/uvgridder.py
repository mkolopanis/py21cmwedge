"""Primary UV Gridder."""
import numpy as np
import os
from astropy import constants as const
from scipy.ndimage import filters


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
        self.uvws = None
        self.antpos = None
        self.uv_grid = None
        self.grid_size = None
        self.grid_delta = None
        self.fwhm = None

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
        self.uvws = self.__createuv__()

    def set_uvw(self, uvw):
        """Manually set uvw array from outside source.

        Should be in the form 3 x N_uvws
        """
        if np.shape(uvw[0]) != 3 and np.shape(uvw)[1] == 3:
            uvw = np.transpose(uvw, [1, 0])
        self.uvws = uvw

    def set_freqs(self, freq):
        """Set Frequency or Array of Frequencies."""
        self.freqs = np.array([freq])
        self.wavelength = const.c.to('m/s').value / self.freqs

    def set_beam(self, beam):
        """Set beam from outside source."""
        # This doesn't actually do anything

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
        norms = np.linalg.norm([u.ravel(), v.ravel(), w.ravel()], axis=0)
        self.bl_len_max = np.max(norms)
        self.bl_len_min = np.min(norms[norms > 0])

        return np.array([u.ravel(), v.ravel(), w.ravel()])

    def uvw_to_dict(self):
        """Convert UVWs array into a dictionary.

        Assumes W term is zero or very very small.
        Elemetns of dictionary are lists of bls keyed by uv lengths
        """
        for _u, _v in self.uvws[:2].T:
            if np.linalg.norm([_u, _v]) == 0:
                continue
            uv = '{0:.3f},{1:.3f}'.format(_u, _v)
            if uv in self.uvbins.keys():
                self.uvbins[uv].append(uv)
            else:
                self.uvbins[uv] = [uv]

    def beamgridder(self, xcen, ycen):
        """Grid Gaussian Beam."""
        cen = self.grid_size/2 + 0.5  # correction for centering
        xcen += cen
        ycen = -1 * ycen + cen
        beam = np.zeros((self.grid_size, self.grid_size))
        inds = np.logical_and(np.round(ycen) <= self.grid_size - 1,
                              np.round(xcen) <= self.grid_size - 1)
        ycen = map(int, np.round(ycen[inds]))
        xcen = map(int, np.round(xcen[inds]))
        beam[ycen, xcen] += 1.  # single pixel gridder
        beam = filters.gaussian_filter(beam, self.fwhm)
        return beam

    def sum_uv(self):
        """Convert uvbin dictionary to a UV-plane."""
        self.uv_grid = np.zeros((self.grid_size, self.grid_size))
        for uv_key in self.uvbins.keys():
            uvbin = self.uvbins[uv_key]
            nbls = len(uvbin)
            u, v = np.array(map(float, uv_key.split(',')))
            u /= self.wavelength
            v /= self.wavelength
            if u == 0 and v == 0:
                print uv_key

            _beam = self.beamgridder(xcen=u/self.grid_delta,
                                     ycen=v/self.grid_delta)
            self.uv_grid += nbls * _beam

    def get_uvcoverage(self):
        """Create UV coverage from object data."""
        self.grid_delta = np.amin(np.concatenate([self.wavelength/4.,
                                                  [self.bl_len_min/2.]]))
        self.grid_size = int(np.round(self.bl_len_max
                                      / self.wavelength
                                      / self.grid_delta).max()) * 4 + 1
        self.uvw_to_dict()
        self.sum_uv()
