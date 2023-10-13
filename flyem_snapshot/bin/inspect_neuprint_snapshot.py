"""
Spins up a neo4j container using apptainer (singularity)
to allow manual inspection of a neuprint database.
This will launch neo4j and start a bash shell within the container.
(This Python script is a thin wrapper around a bash script.)

Usage:

    inspect-neuprint-snapshot <neo4j-export-dir>

... where <neo4j-export-dir> contains:
        conf/  data/  logs/  plugins/

Note:
    The neo4j-export-dir should be on a local hard disk, NOT on network storage such as Janelia's prfs.
    If you try to use network storage, you may encounter the following error:

        Unable to get a routing table for database 'data' because this database is unavailable

Example commands to try in the cypher shell:

    @data> SHOW DATABASES;
    @data> SHOW INDEXES;
    @data> MATCH (m:manc_Meta) RETURN m.dataset;
"""
import os
import sys
import argparse
import subprocess

import flyem_snapshot


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('neo4j_export_dir',
                        help='The exported neo4j directory tree, as produced by ingest-neuprint-snapshot-using-apptainer')
    args = parser.parse_args()

    package_dir = os.path.dirname(flyem_snapshot.__file__)
    os.path.abspath(package_dir)
    script = f"{package_dir}/outputs/neuprint/scripts/inspect-neuprint-snapshot.sh"
    p = subprocess.run([script, args.neo4j_export_dir], check=False)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
