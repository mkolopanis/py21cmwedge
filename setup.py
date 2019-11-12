"""Setup modules py21cmwedge."""
from setuptools import setup
import glob
import os.path as op
from os import listdir
__version__ = '0.0.2'

data_files = [op.join('./py21cmwedge/data', f)
              for f in listdir('./py21cmwedge/data')
              if op.isfile(op.join('./py21cmwedge/data', f))]

test_files = [op.join('./py21cmwedge/tests/data', f)
              for f in listdir('./py21cmwedge/tests/data')
              if op.isfile(op.join('./py21cmwedge/tests/data', f))]

test_files += glob.glob('./py21cmwedge/tests/*.py')

all_files = data_files + test_files

setup_args = {
    'name': 'py21cmwedge',
    'author': 'Matthew Kolopanis',
    'license': 'BSD',
    'description': ('an utility to predict the theoretical wedge leakage '
                    'from a radio intereferometric array'),
    'package_dir': {'py21cmwedge': 'py21cmwedge'},
    'packages': ['py21cmwedge'],
    'version': __version__,
    'package_data': {'py21cmwedge': all_files},
    'install_requires': ['numpy>1.10', 'scipy', 'astropy>1.2', 'nose',
                         'healpy', 'six'],
    'test_suite': 'nose'
}

if __name__ == '__main__':
    setup(**setup_args)
