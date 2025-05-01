"""
Minimal setup.py to work with versioneer.
Most configuration is in pyproject.toml.
"""
from setuptools import setup, find_packages
import versioneer

setup(
    name="flyem-snapshot",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
) 