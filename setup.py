from setuptools import setup, find_packages

from acumencompressor import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='acumencompressor',
    version=__version__,
    license="MIT",
    url='https://github.com/cosmaadrian/acumen-compressor',
    author='Adrian Cosma',
    author_email='cosma.i.adrian@gmail.com',
    packages = ['acumencompressor'],
    install_requires = requirements,
)