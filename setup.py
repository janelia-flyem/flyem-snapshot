"""
flyem-snapshot/setup.py
"""
from setuptools import find_packages, setup
import versioneer

# For now, requirements are only specified in the conda recipe, not here.

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.strip().startswith('#')]

setup(
    name='flyem-snapshot',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.',
    url='https://github.com/janelia-flyem/flyem-snapshot',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'flyem-snapshot = flyem_snapshot.bin.flyem_snapshot_entrypoint:main',
            'ingest-neuprint-snapshot-using-apptainer = flyem_snapshot.bin.ingest_neuprint_snapshot_using_apptainer:main',
            'inspect-neuprint-snapshot = flyem_snapshot.bin.inspect_neuprint_snapshot:main',
            'parse-neuprint-log = flyem_snapshot.bin.parse_neuprint_log:main',
            'update-neuprint-annotations = flyem_snapshot.bin.update_neuprint_annotations:main',
        ]
    }
)
