"""Test for Gridder Object."""
import nose.tools as nt
import os
from py21cm_wedge import UVGridder

testdir = os.path.dirname(os.path.realpath(__file__))


class UVTest(UVGridder):
    def __init__(self):
        """UVGridder test object."""
        # comment

        super(UVTest, self).__init()


def test_equality():
    """Test base object."""
    test_obj = UVGridder()
    nt.assert_equal(test_obj, test_obj)


def test_readantpos():
    """Antenna Position file to uvws test.

    Read in antenna positions, convert to uvws,
    check equality with predefined uvw file.
    """
