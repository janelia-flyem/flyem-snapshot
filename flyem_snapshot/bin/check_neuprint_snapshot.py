"""
Validate an ingested neuprint database.

Spins up the snapshot's neo4j database in a container, runs a suite of checks
against it, and shuts it down again.  Exits 0 if every check passed and 1
otherwise, so this is usable as a gate in a pipeline script or from CI.
(This Python script is a thin wrapper around a bash script.)

Usage:

    check-neuprint-snapshot <neo4j-export-dir>

... where <neo4j-export-dir> contains:
        conf/  data/  logs/  plugins/

The dataset name is read from the :Meta node, so this works on any dataset
(wasp, hemibrain, fish2, ...) without configuration.

What it checks:

    - node and relationship counts, and that every node carries a known label
    - bodyId integrity (no duplicates, no nulls) and uniqueness constraints
    - every index is ONLINE and fully populated
    - every index refers to a property that actually exists
    - every ROI in Meta.roiInfo has both a matching property and an index
    - every ROI index is usable when forced via an index hint
    - node and relationship totals match what the importer reported, and
      the import skipped no bad entries
    - Segment/Synapse/SynapseSet counts match the exported CSV row counts
    - the database name and pinned Cypher language version

Environment overrides, all optional:

    NEO4J_DB     Database to check (default 'data').  Use 'neo4j' for a
                 pre-upgrade 4.4-era database.
    NEO4J_IMAGE  Container image (default docker://neo4j:2026.07.1).
    CHECK_CSV_COUNTS
                 Set to 0 to skip reconciling label counts against the
                 exported CSV row counts.  On by default; it is the
                 strongest check here, but reads every CSV, which is slow
                 on a large dataset over network storage.
    HEAP_SIZE    Override the database's own neo4j.conf memory sizing, which
    MAX_MEMORY   is otherwise respected as-is.  Needed only when checking a
                 cluster-sized snapshot on a smaller machine.

Example:

    check-neuprint-snapshot 2026-05-12-32c9ac/neo4j
    HEAP_SIZE=4G MAX_MEMORY=8G check-neuprint-snapshot 2026-05-12-32c9ac/neo4j
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
    package_dir = os.path.abspath(package_dir)
    script = f"{package_dir}/outputs/neuprint/scripts/check-neuprint-snapshot.sh"
    p = subprocess.run([script, args.neo4j_export_dir], check=False)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
