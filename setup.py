from setuptools import setup
import glob
import os.path as op
from os import listdir

__version__ = '0.0.1'

setup_args = {
    'name': 'py21cmwedge',
    'author': 'Matthew Kolopanis',
    'licsense': 'BSD',
    'description': 'an utility to predict the theoretical wedge leakage from a radio intereferometric array',
    'package_dir': {'py21cmwedge': 'py21cmwedge'},
    'packages': ['py21cmwedge'],
    'version': __version__,
    'package_data': {'py21cmwedge': [f for f in listdir('./py21cmwedge/data') if op.isfile(op.join('./py21cmwedge/data', f))] + [f for f in listdir('./py21cmwedge/data') if op.isfile(op.join('./py21cmwedge/tests/data', f))] },
    'install_requires': ['numpy>1.10', 'scipy', 'astropy>1.2', 'nose'],
    'test_suite': 'py21cmwedge'
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
