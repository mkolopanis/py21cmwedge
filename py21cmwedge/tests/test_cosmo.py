"""Test for Cosmo Module."""
import nose.tools as nt
import nose
import os
import copy
import numpy as np
from py21cmwedge import cosmo

testdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def test_u_to_k_to_u():
    """Convert a |U| value to k and back again."""
    test_u = 1.5
    redshift = 8.5
    u_to_k = cosmo.u2kperp(test_u, redshift)
    u_to_k_to_u = cosmo.kperp2u(u_to_k, redshift)
    nt.assert_true(np.isclose(test_u, u_to_k_to_u))


def test_eta_to_kpar_to_u():
    """Convert a delay, eta, to k_par and back."""
    test_eta = 1./50e6  # 50MHz test bandwidth
    redshift = 8.5
    eta_to_kpar = cosmo.eta2kpar(test_eta, redshift)
    eta_to_kpar_to_eta = cosmo.kpar2eta(eta_to_kpar, redshift)
    nt.assert_true(np.isclose(test_eta, eta_to_kpar_to_eta))
