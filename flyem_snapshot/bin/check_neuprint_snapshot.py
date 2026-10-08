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
    - every index refers to a property that actually exists, and to a
      label that at least one node carries
    - the uniqueness constraints are UNIQUENESS on Segment/Neuron bodyId,
      not merely present
    - the neo4j.conf persisted beside the database carries explicit
      memory sizing and the Cypher language pin, so the snapshot starts
      on a machine other than the one that built it
    - every ROI in Meta.roiInfo has both a matching property and an index
    - every ROI index is usable when forced via an index hint
    - node and relationship totals match what the importer reported, and
      the import skipped no bad entries
    - Segment/Synapse/SynapseSet counts match the exported CSV row counts,
      as do the ConnectsTo/SynapsesTo/Contains/CloseTo relationship counts
    - the two tables exported as both feather and CSV (Neuprint_Neurons and
      Neuprint_Neuron_Connections) have matching row counts.  Both are
      written from one DataFrame, so this checks the CSV batching rather
      than the data -- an error upstream of that DataFrame appears in both
      and passes
    - every Neuron's type agrees with tables/body-annotations-*.feather,
      the annotations as fetched from DVID before they were merged into the
      neuron table.  Row counts are not comparable (the table covers every
      annotated body, most of which are not exported as Neurons), so this is
      a subset check over the bodies the graph carries a type for.  A type
      with no row in that table is a WARNING, not a failure: a dataset may
      draw annotations from a config-supplied table or from point
      annotations as well as from DVID.
      These three need pyarrow importable on the host, so run via 'pixi run';
      without it they are skipped with a notice and the total drops by three
    - every relationship carries one of the known types
    - Meta.totalPreCount/totalPostCount do not exceed the synapses that
      exist (they may legitimately be smaller, when the config restricts
      the dataset totals to in-bounds ROIs)
    - the store format is reported, since neo4j has no downgrade path
    - neuPrintExplorer's search queries (both the label-scanning one and
      the fulltext-index one) execute, with their timings reported and
      their row counts compared; inputs are derived deterministically, so
      runs are comparable
    - the FULLTEXT index is ONLINE, and whether it covers every search
      property the dataset actually populates.  A gap here is reported as
      a WARNING rather than a failure: the default indexes three
      properties, and a dataset that annotates more is expected to list
      them under 'find-neurons-fulltext-index-properties' in its own
      config.  Where it does not, the fast search query silently returns
      fewer rows than the slow one, so the warning is worth acting on --
      but the snapshot itself is sound.
    - the database name and pinned Cypher language version

Environment overrides, all optional:

    NEO4J_DB     Database to check (default 'data').  Use 'neo4j' for a
                 pre-upgrade 4.4-era database.
    NEO4J_IMAGE  Container image (default docker://neo4j:2026.08.1).
    CHECK_CSV_COUNTS
                 Set to 0 to skip reconciling label counts against the
                 exported CSV row counts.  On by default; it is the
                 strongest check here, but reads every CSV, which is slow
                 on a large dataset over network storage.
    MAX_QUERY_MS
                 If set, the complex-query timing becomes an assertion
                 rather than an informational line.  Left unset by default,
                 since a threshold chosen without baselines fails for
                 environmental reasons more often than for regressions.
                 Milliseconds, because the whole plausible range on a
                 snapshot-sized dataset sits well under one second.
    QUERY_SEARCH_TERM
                 Search term for the complex query.  This, not the bodyId,
                 is what determines how long that query takes: every neuron
                 is scanned and tested against eleven CONTAINS predicates,
                 then everything matched is sorted, so cost tracks how large
                 a fraction of the label the term matches.  A short or
                 common term is expensive, a long one is cheap.  Defaults to
                 the commonest letter in the dataset's type names, which is
                 the closest thing to a worst case.
    QUERY_SEARCH_TERMS
                 Comma-separated list of terms to time in one run, e.g.
                 'a,in,LC,SNxx'.  Takes precedence over QUERY_SEARCH_TERM.
                 Use this to hunt for a term heavy enough to be worth
                 asserting on: neo4j is booted once and each term timed in
                 turn, where a term per invocation would cost minutes each.
                 Pair it with CHECK_CSV_COUNTS=0 to skip the slow checks.
    SHOW_QUERY   Set to 1 to print the generated Cypher.  It is printed
                 automatically when the query errors or exceeds
                 MAX_QUERY_MS, so this is only needed to see it on a
                 passing run.
    QUERY_BODY_ID
                 bodyId for the complex query (default: the lowest bodyId
                 carrying a type).  Pin it for reproducibility; it has
                 almost no effect on the timing.
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
