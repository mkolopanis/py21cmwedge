"""Test for Gridder Object."""
import nose.tools as nt
import os
from 21cm_wedge import UVGridder

testdir = os.path.dirname(os.path.realpath(__file__))


def test_readantpos():
    """Antenna Position file to uvws test.

    Read in antenna positions, convert to uvws,
    check equality with predefined uvw file.
    """
