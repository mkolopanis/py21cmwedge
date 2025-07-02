"""Ancillary Cosmology Support."""

import numpy as np
from astropy import constants as const
from scipy.integrate import quad

# Define global variables for cosmology
# WMAP 5 year (in the abstract of astro-ph 0803.0586v2)
omega_lambda = 0.742
omega_matter = 0.25656
hubble_param = 0.719
Ho = 100.0 * hubble_param
omega_curve = 1 - omega_lambda - omega_matter

# other constants
c = const.c.to("m/s").value
ckm = c / 1000.0  # km/s
f21 = 1.420e9  # GHz


# Refer to (Liu et al 2014a) Appendix A for formula references
def Ez(
    redshift=None,
    omega_matter=omega_matter,
    omega_lambda=omega_lambda,
    omega_curve=omega_curve,
):
    """Compute co-moving distance by integrating this function."""
    return np.sqrt(
        omega_matter * (1.0 + redshift) ** 3
        + omega_curve * (1.0 + redshift) ** 2
        + omega_lambda
    )


def comoving_distance(
    redshift=None,
    mega_matter=omega_matter,
    omega_lambda=omega_lambda,
    omega_curve=omega_curve,
):
    """Calculate comoving distance for a Lambda_CDM cosmology."""
    hubble_dist = 3e3 / hubble_param
    integral, err = quad(lambda z: 1 / Ez(z), 0.0, redshift)
    return hubble_dist * integral


def kperp2u(kperp, z):
    """Convert k_perpendicular to interferometric |u|."""
    return kperp * comoving_distance(z) / (2.0 * np.pi)


def u2kperp(u, z):
    """Convert interferometric |u| to k_perpendicular."""
    return u * 2.0 * np.pi / comoving_distance(z)


def eta2kpar(eta, z):
    """Convert delay eta to k_parallel."""
    return eta * 2 * np.pi * Ho * f21 / ckm * Ez(z) / (1 + z) ** 2


def kpar2eta(kpar, z):
    """Convert k_parallel to delay."""
    return kpar * (1 + z) ** 2 * ckm / (2 * np.pi * Ho * f21 * Ez(z))
