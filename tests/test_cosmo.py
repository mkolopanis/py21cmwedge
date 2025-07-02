"""Test for Cosmo Module."""

import numpy as np

from py21cmwedge import cosmo


def test_u_to_k_to_u():
    """Convert a |U| value to k and back again."""
    test_u = 1.5
    redshift = 8.5
    u_to_k = cosmo.u2kperp(test_u, redshift)
    u_to_k_to_u = cosmo.kperp2u(u_to_k, redshift)
    np.testing.assert_allclose(test_u, u_to_k_to_u)


def test_eta_to_kpar_to_u():
    """Convert a delay, eta, to k_par and back."""
    test_eta = 1.0 / 50e6  # 50MHz test bandwidth
    redshift = 8.5
    eta_to_kpar = cosmo.eta2kpar(test_eta, redshift)
    eta_to_kpar_to_eta = cosmo.kpar2eta(eta_to_kpar, redshift)
    np.testing.assert_allclose(test_eta, eta_to_kpar_to_eta)
