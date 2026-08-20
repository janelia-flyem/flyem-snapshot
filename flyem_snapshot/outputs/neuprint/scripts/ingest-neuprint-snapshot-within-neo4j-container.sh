#!/bin/bash

##
## This script is not meant to be invoked directly.
## It is invoked from ingest-neuprint-snapshot-using-apptainer.sh
## (which itself is usually invoked via a Python wrapper script).
##

##
## This script is meant to be run from WITHIN the neo4j container.
## (At the time of this writing, we use neo4j:2026.06.0.)
## This ingests ALL of the CSV files from a neuprint snapshot via the
## neo4j-admin tool in ONE STEP.
## (In neo4j v5, incremental import is supported, but only in the Enterprise edition.)
##
## Note that the neo4j-admin tool constructs a neo4j database WITHOUT using neo4j itself.
## (The neo4j server need not be running.)
## After we load the CSV files, we launch neo4j with the newly constructed database and
## send the appropriate cypher commands to create indexes for segment properties.
##

##
## To summarize, the steps are:
##
##  1. BEFORE launching neo4j, use neo4j-admin to ingest all the CSV files (nodes/relationships).
##  2. Launch neo4j with the new database files.
##  3. Use cypher-shell to create indexes on the ingested data.
##  4. Stop neo4j when this script exits (using a trap).
##

# This is optionally set via the calling script, when you use
# ingest-neuprint-snapshot-using-apptainer <snapshot-dir> --debug-shell
DEBUG_SHELL=$1

set -e

# If we used the normal docker entrypoint, then we could leave the config in /conf.
# But since we bypass the docker entrypoint, then /conf is ignored, apparently.
# We must overwrite the default config file.
cp /conf/neo4j.conf ${NEO4J_HOME}/conf/neo4j.conf

# We don't actually use plugins during ingestion,
# but it's convenient to have access to them when debugging.
# Install them by copying into NEO4J_HOME.
ls /plugins/* > /dev/null 2>&1 && cp /plugins/* ${NEO4J_HOME}/plugins/

# The snapshot's CSV files are bind-mounted here by the calling script.
SNAPSHOT_DIR=/snapshot

cd ${SNAPSHOT_DIR}

##
## Import CSVs for nodes/relationships
##

# Note: every path handed to neo4j-admin below MUST include a directory
# component. A bare filename (e.g. 'Neuprint_Meta.csv') makes neo4j-admin fail
# with "Unable to find the parent of the path", because a single-component
# relative path has no parent for it to resolve. We therefore anchor every
# path on the absolute ${SNAPSHOT_DIR} rather than relying on the cwd.

# Node arguments.
# There may be hundreds of thousands of node CSV files, which is why we supply
# these arguments to neo4j-admin via a special arguments file.  (See below.)
META_ARG=--nodes=${SNAPSHOT_DIR}/Neuprint_Meta.csv
SYNSET_ARG=--nodes=${SNAPSHOT_DIR}/Neuprint_SynapseSet.csv
NEURON_ARGS=$(for f in $(find ${SNAPSHOT_DIR}/Neuprint_Neurons/ -name "*.csv"); do printf -- "--nodes=$f "; done)
SYNAPSE_ARGS=$(for f in $(find ${SNAPSHOT_DIR}/Neuprint_Synapses/ -name "*.csv"); do printf -- "--nodes=$f "; done)
if [[ -d ${SNAPSHOT_DIR}/Neuprint_Elements ]]; then
    ELEMENT_ARGS=$(for f in $(find ${SNAPSHOT_DIR}/Neuprint_Elements/ -name "*.csv"); do printf -- "--nodes=$f "; done)
    ELMSET_ARGS=$(for f in $(find ${SNAPSHOT_DIR}/Neuprint_ElementSets/ -name "*.csv"); do printf -- "--nodes=$f "; done)
fi

if [[ -z "${NEURON_ARGS}" ]]
then
    echo "Didn't find any Neuron csv files!" 1>&2
    exit 1
fi

if [[ -z "${SYNAPSE_ARGS}" ]]
then
    echo "Didn't find any Synapse csv files!" 1>&2
    exit 1
fi

if [[ -z "${LSB_MAX_NUM_PROCESSORS}" ]]
then
    # Note: The neo4j container image doesn't ship python, so we can't
    # use multiprocessing.cpu_count() here as we did with neo4j:4.4.
    TOTAL_CPUS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
    CPU_COUNT=$((TOTAL_CPUS / 2))
    # --threads=0 is invalid, so don't go below 1 on a single-core machine.
    [[ ${CPU_COUNT} -lt 1 ]] && CPU_COUNT=1
else
    CPU_COUNT=${LSB_MAX_NUM_PROCESSORS}
fi

# Relationship arguments.
# (As above, these must all carry a directory component.)
NEURON_CONNECTSTO_ARG=--relationships=ConnectsTo=${SNAPSHOT_DIR}/Neuprint_Neuron_Connections.csv
SYNAPSE_SYNAPSESTO_ARG=--relationships=SynapsesTo=${SNAPSHOT_DIR}/Neuprint_Synapse_Connections.csv
ELEMENT_CLOSETO_ARGS=$(for f in $(find ${SNAPSHOT_DIR} -name "Neuprint_Elements_CloseTo_*.csv"); do printf -- "--relationships=CloseTo=$f "; done)

NEURON_CONTAINS_SYNSET_ARG=--relationships=Contains=${SNAPSHOT_DIR}/Neuprint_Neuron_to_SynapseSet.csv
SYNSET_CONTAINS_SYNAPSE_ARG=--relationships=Contains=${SNAPSHOT_DIR}/Neuprint_SynapseSet_to_Synapses.csv
SYNSET_CONNECTSTO_ARG=--relationships=ConnectsTo=${SNAPSHOT_DIR}/Neuprint_SynapseSet_to_SynapseSet.csv

NEURON_CONTAINS_ELMSET_ARGS=$(for f in $(find ${SNAPSHOT_DIR} -name "Neuprint_Neuron_to_ElementSet_*.csv"); do printf -- "--relationships=Contains=$f "; done)
ELMSET_CONTAINS_ELEMENT_ARGS=$(for f in $(find ${SNAPSHOT_DIR} -name "Neuprint_ElementSet_to_Element_*.csv"); do printf -- "--relationships=Contains=$f "; done)


# The neo4j docs say this about the HEAP_SIZE variable:
# "If doing imports in the order of magnitude of 100 billion entities, 20G will be an appropriate value."
# (We have ~0.5B entities)
#
# The defaults here are sized for a large cluster node. On a smaller machine
# (e.g. a VM), the JVM will fail to start or be OOM-killed, so both values can
# be overridden from the calling environment, e.g.
#
#   HEAP_SIZE=4G MAX_MEMORY=8G ingest-neuprint-snapshot-using-apptainer <snapshot-dir>
#
export HEAP_SIZE=${HEAP_SIZE:-31G}

# TODO: Should we use this option?
# --cache-on-heap=true

MAX_MEMORY=${MAX_MEMORY:-150G}

# HEAP_SIZE and MAX_MEMORY above only apply to the 'neo4j-admin import' step.
# The neo4j SERVER we launch further below (to create indexes) takes its memory
# settings from neo4j.conf instead, which is likewise sized for a big cluster
# node. Keep the two in sync so that one pair of env vars sizes both phases,
# otherwise the import succeeds and then the server fails to start.
# (The defaults substituted here are identical to the values already in
# neo4j.conf, so this is a no-op unless the caller overrode them.)
#
# Note: we rewrite BOTH copies of neo4j.conf:
#   - ${NEO4J_HOME}/conf/neo4j.conf is the one this container's server reads
#   - /conf/neo4j.conf is the bind-mounted workspace copy, which gets persisted
#     next to the database and is later read by inspect-neuprint-snapshot
# If we only did the former, the conf shipped alongside the database would
# still claim 31G/150G and inspect-neuprint-snapshot would fail to start on
# any machine smaller than a cluster node.
#
for conf in ${NEO4J_HOME}/conf/neo4j.conf /conf/neo4j.conf
do
    [[ -w "${conf}" ]] || continue
    sed -i \
        -e "s|^server\.memory\.heap\.initial_size=.*|server.memory.heap.initial_size=${HEAP_SIZE}|" \
        -e "s|^server\.memory\.heap\.max_size=.*|server.memory.heap.max_size=${HEAP_SIZE}|" \
        -e "s|^server\.memory\.pagecache\.size=.*|server.memory.pagecache.size=${MAX_MEMORY}|" \
        ${conf}
done

echo "[$(date)] Using HEAP_SIZE=${HEAP_SIZE}, MAX_MEMORY=${MAX_MEMORY}, threads=${CPU_COUNT}"
echo "[$(date)] Server memory settings in effect:"
grep -E '^server\.memory\.' ${NEO4J_HOME}/conf/neo4j.conf | sed 's/^/    /'

# Neo4j 2025.12 changed the default --bad-tolerance from 1000 to -1 (unlimited),
# which means a malformed CSV row would be skipped and logged instead of failing
# the import. For a connectome export we'd rather not silently lose rows, so pin
# it to the pre-CalVer default. Set to 0 to fail on the very first bad record.
BAD_TOLERANCE=${BAD_TOLERANCE:-1000}

cat > ingestion-args.txt << EOF
--overwrite-destination=true
--normalize-types=false
--high-parallel-io=on
--bad-tolerance=${BAD_TOLERANCE}
--max-off-heap-memory=${MAX_MEMORY}
--threads=${CPU_COUNT}
${META_ARG}
${NEURON_ARGS}
${SYNAPSE_ARGS}
${ELEMENT_ARGS}
${SYNSET_ARG}
${ELMSET_ARGS}
${NEURON_CONNECTSTO_ARG}
${SYNSET_CONNECTSTO_ARG}
${SYNAPSE_SYNAPSESTO_ARG}
${NEURON_CONTAINS_SYNSET_ARG}
${SYNSET_CONTAINS_SYNAPSE_ARG}
${ELEMENT_CLOSETO_ARGS}
${NEURON_CONTAINS_ELMSET_ARGS}
${ELMSET_CONTAINS_ELEMENT_ARGS}
EOF

if [[ ! -z "${DEBUG_SHELL}" ]]
then
    # Drop the user into a bash shell instead of running the ingestion.
    /bin/bash
    exit $?
fi

start=$(date +%s)
echo "[$(date)] Ingesting nodes and relationships"
echo "[$(date)] (There may be a LONG pause after the next line of output.)"

# Our argument list would be way too long to supply on the command line.
# (Error: Argument list too long)
# Luckily, we can supply the arguments via a file!
# https://github.com/neo4j/neo4j/issues/7333#issuecomment-1746238765
# Note more note: Our 'meta' node includes multiline fields, hence the option used here.
/var/lib/neo4j/bin/neo4j-admin database import full data --multiline-fields=true @ingestion-args.txt > >(tee /logs/import.out.log) 2> >(tee /logs/import.err.log)
end=$(date +%s)

if grep -i 'import failed' /logs/import.*.log > /dev/null;
then
    echo "[$(date)] Node/relationship ingest FAILED. See /logs/import.*.log"
    exit 1
fi

echo "[$(date)] Node/relationship ingest completed."
echo "Duration: $(date -d@$((end-start)) -u +%H:%M:%S)"

##
## Create indexes
##

# neo4j creates indexes IN THE BACKGROUND.
# Those queued background operations prevent neo4j from
# shutting down if they haven't completed yet.
# We certainly don't want to interrupt that, so we give a ridiculously
# long amount of time to shut down if it needs it.
export NEO4J_SHUTDOWN_TIMEOUT=86400

start=$(date +%s)
echo "[$(date)] Launching neo4j..."
# Note: We used 'set -e' above, which means the trap won't hide the exit code.
# https://unix.stackexchange.com/questions/667368/bash-change-exit-status-in-trap#comment1444973_667384
trap 'neo4j stop && echo "[$(date)] Indexes created (unless an error occured). Duration: $(date -d@$((end-start)) -u +%H:%M:%S)"' EXIT
neo4j start --verbose

# Wait for neo4j to start. (Wait for the "Started." in the log file.)
grep -q 'Started\.' <(tail -n1 -f /logs/neo4j.log)

/var/lib/neo4j/bin/cypher-shell \
    -d data \
    --format verbose \
    -f /snapshot/create-indexes.cypher \
    > >(tee /logs/create-indexes.out.log) \
    2> >(tee /logs/create-indexes.err.log) \
##

if grep -i 'database is unavailable' /logs/create-indexes.*.log > /dev/null || [ ! -s /logs/create-indexes.out.log ];
then
    echo "[$(date)] Index generation FAILED. See /logs/create-indexes.*.log"
    exit 1
fi

end=$(date +%s)
