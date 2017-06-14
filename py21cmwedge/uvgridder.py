"""Primary UV Gridder."""
import numpy as np
import os
from astropy import constants as const


class UVGridder(object):
    """Base uvgridder object."""

    def __init__(self):
        """Create new UVGridder object."""
        self.freqs = None
        self.uv_sum = None
        self.uv_bins = None
        self.bl_len_max = 0
        self.bl_len_min = np.Inf
        self.beam = None
        self.uvws = None
        self.antpos = None

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
        have the form East, North, Up
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
