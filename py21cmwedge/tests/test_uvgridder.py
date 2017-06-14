"""Test for Gridder Object."""
import nose.tools as nt
import os
import copy
from py21cmwedge import UVGridder
import numpy as np

testdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class UVTest(UVGridder):
    def __init__(self):
        """UVGridder test object."""
        # comment
        self.path = testdir
        super(UVTest, self).__init__()


def test_equality():
    """Test base object."""
    test_obj = UVGridder()
    nt.assert_equal(test_obj, test_obj)


def test_unequality():
    """Test that inequality is handled."""
    test_obj = UVGridder()
    test_obj2 = copy.deepcopy(test_obj)
    test_obj2.freqs = 1.5e6
    nt.assert_not_equal(test_obj, test_obj2)


def test_readantpos():
    """Antenna Position file to uvws test.

    Read in antenna positions, convert to uvws,
    check equality with predefined uvw file.
    """
    test_obj = UVTest()
    test_obj.read_antpos(os.path.join(test_obj.path, 'test_antpos.txt'),
                         skiprows=1, delimiter=',')
    test_antpos = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [1, 1, 0],
                            [-1, -1, 0],
                            [1, -1, 0],
                            [-1, 1, 0]]).T
    nt.assert_true(np.allclose(test_antpos, test_obj.antpos))
