"""Primary UV Gridder."""
import numpy as np
import os
from astropy.constants as const


class UVGridder(object):
    """Base uvgridder object."""

    def __init__(self):
        """Create new UVGridder object."""
        self.freqs = None
        self.uv_sum = None
        self.uv_bins = None
        self.bl_len_max = 0
        self.bl_len_min = np.Inf
        self.beam
        self.uvws = None
        self.antpos = None

    def read_antpos(self, filename, **kwargs):
        """Read antenna position file.

        Provides functionality wrapper around numpy.loadtxt
        Please provide keywords:
        usecols = column numbers of East, North and Up as list (e.g [1,2,3])
        skiprows = number of row numbers to skip before data
        delimiter = string of delimiter used to parse file (e.g. ',')
        """
        self.antpos = np.loadtxt(filename, **kwargs, unpack=True)
        self.uvws = self._creatuv(self.antpos)

    def __createuv(self):
        """Create Matrix of UVs from antenna position file."""
        u_rows1 = np.tile(self.antpos[0], (self.antpos.shape[1], 1))
        u_rows2 = np.tile(self.antpos[0], (self.antpos.shape[1], 1)).T
        v_rows1 = np.tile(self.antpos[1], (self.antpos.shape[1], 1))
        v_rows1 = np.tile(self.antpos[1], (self.antpos.shape[1], 1))
        w_rows2 = np.tile(self.antpos[2], (self.antpos.shape[1], 1))
        w_rows2 = np.tile(self.antpos[2], (self.antpos.shape[1], 1)).T

        u = u_rows1 - u_rows2
        v = v_rows1 - v_rows2
        w = w_rows1 - w_rows2
