#!/bin/bash

##
## This script is not meant to be invoked directly.
## It is invoked from ingest-neuprint-snapshot-using-apptainer.sh
## (which itself is usually invoked via a Python wrapper script).
##

##
## This script is meant to be run from WITHIN the neo4j:4.4 container.
## (At the time of this writing, we use neo4j:4.4.16.)
## This ingests ALL of the CSV files from a neuprint snapshot via the
## neo4j-admin tool in ONE STEP.
## (In neo4j v4.4, only a full import is supported.  In newer versions of neo4j,
## incremental import is supported, but only in the Enterprise edition.)
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

cd /snapshot

##
## Import CSVs for nodes/relationships
##

# Node arguments.
# There may be hundreds of thousands of node CSV files, which is why we supply
# these arguments to neo4j-admin via a special arguments file.  (See below.)
META_ARG=--nodes=Neuprint_Meta.csv
SYNSET_ARG=--nodes=Neuprint_SynapseSet.csv
NEURON_ARGS=$(for f in $(find Neuprint_Neurons/ -name "*.csv"); do printf -- "--nodes=$f "; done)
SYNAPSE_ARGS=$(for f in $(find Neuprint_Synapses/ -name "*.csv"); do printf -- "--nodes=$f "; done)
if [[ -d Neuprint_Elements ]]; then
    ELEMENT_ARGS=$(for f in $(find Neuprint_Elements/ -name "*.csv"); do printf -- "--nodes=$f "; done)
    ELMSET_ARGS=$(for f in $(find Neuprint_ElementSets/ -name "*.csv"); do printf -- "--nodes=$f "; done)
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
    CPU_COUNT=$(python -c 'import multiprocessing; print(multiprocessing.cpu_count()//2)')
else
    CPU_COUNT=${LSB_MAX_NUM_PROCESSORS}
fi

# Relationship arguments.
NEURON_CONNECTSTO_ARG=--relationships=ConnectsTo=Neuprint_Neuron_Connections.csv
SYNAPSE_SYNAPSESTO_ARG=--relationships=SynapsesTo=Neuprint_Synapse_Connections.csv
ELEMENT_CLOSETO_ARGS=$(for f in $(find . -name "Neuprint_Elements_CloseTo_*.csv"); do printf -- "--relationships=CloseTo=$f "; done)

NEURON_CONTAINS_SYNSET_ARG=--relationships=Contains=Neuprint_Neuron_to_SynapseSet.csv
SYNSET_CONTAINS_SYNAPSE_ARG=--relationships=Contains=Neuprint_SynapseSet_to_Synapses.csv
SYNSET_CONNECTSTO_ARG=--relationships=ConnectsTo=Neuprint_SynapseSet_to_SynapseSet.csv

NEURON_CONTAINS_ELMSET_ARGS=$(for f in $(find . -name "Neuprint_Neuron_to_ElementSet_*.csv"); do printf -- "--relationships=Contains=$f "; done)
ELMSET_CONTAINS_ELEMENT_ARGS=$(for f in $(find . -name "Neuprint_ElementSet_to_Element_*.csv"); do printf -- "--relationships=Contains=$f "; done)


# The v4.4 docs say this about the HEAP_SIZE variable:
# "If doing imports in the order of magnitude of 100 billion entities, 20G will be an appropriate value."
# (We have ~0.5B entities)
export HEAP_SIZE='31G'

# TODO: Should we use this option?
# --cache-on-heap=true

# NOTE: This is a hard-coded value for --max-memory!
# (BTW, in neo4j v5, it will be renamed to --max-off-heap-memory)
MAX_MEMORY='150G'

cat > ingestion-args.txt << EOF
--force=true
--database=data
--normalize-types=false
--high-io=true
--max-memory=${MAX_MEMORY}
--processors=${CPU_COUNT}
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

# Our argument list would be way too long to supply on the command line.
# (Error: Argument list too long)
# Luckily, we can supply the arguments via a file!
# https://github.com/neo4j/neo4j/issues/7333#issuecomment-1746238765
/var/lib/neo4j/bin/neo4j-admin import @ingestion-args.txt > >(tee /logs/import.out.log) 2> >(tee /logs/import.err.log)
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
