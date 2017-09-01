"""Test for Cosmo Module."""
import nose.tools as nt
import nose
import os
import copy
import numpy as np
from py21cmwedge import cosmology

testdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def test_u_to_k():
    """Test |U| to k against a known value."""
    test_u = 1.5
    redshift = 8.5
    u_to_k = cosmo.u2kperp(test_u, redshift)
    nt.assert_equal(u_to_k, 0.0014275626347673188)


def test_k_to_u():
    """Test k to |u| against known value."""
    test_k = 0014275626347673188
    redshift = 8.5
    k_to_u = cosmo.kperp2u(test_k, redshfit)
    nt.assert_equal(k_to_u, 1.5)


def test_u_to_k_to_u():
    """Convert a |U| value to k and back again."""
    test_u = 1.5
    redshift = 8.5
    u_to_k = cosmo.u2kperp(test_u, redshift)
    u_to_k_to_u = cosmo.kperp2u(u_to_k, redshift)
    nt.assert_equal(test_u, u_to_k_to_u)
