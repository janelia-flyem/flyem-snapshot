"""
flyem-snapshot/setup.py
"""
from setuptools import find_packages, setup
import versioneer

# For now, requirements are only specified in the conda recipe, not here.

setup(
    name='flyem-snapshot',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.',
    url='https://github.com/janelia-flyem/flyem-snapshot',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'flyem-snapshot = flyem_snapshot.bin.flyem_snapshot_entrypoint:main'
        ]
    }
)
