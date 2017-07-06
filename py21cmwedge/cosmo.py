"""Ancillary Cosmology Support."""
import numpy as np
from scipy.integrate import quad


def E_inv(redshift=None, omega_matter=None,
          omega_lambda=None, omega_curv=None):
    """Compute co-moving distance by integrating this function."""
    return 1. / np.sqrt(omega_matter * (1. + redshift)**3
                        + omega_curv * (1. + redshift)**2 + omega_lambda)


def comoving_distance(redshift=None, hubble_param=1, omega_matter=.27,
                      omega_lambda=.73, omega_curv=0.0):
        """Calculate comoving distance for a Lambda_CDM cosmology."""
        hubble_dist = 3e2 / hubble_param
        integral, err = quad(E_inv, 0., redshift,
                             args=(omega_matter, omega_lambda, omega_curv))
        return hubble_dist * integral


def kperp2u(kperp, z):
    """Convert k_perpendicular to interferometric |u|."""
    return kperp * comoving_distance(z) / (1. + z) / (2. * np.pi)


def u2kperp(u, z):
    """Convert interferometric |u| to k_perpendicular."""
    return u * 2. * np.pi * (1. + z) / comoving_distance(z)
