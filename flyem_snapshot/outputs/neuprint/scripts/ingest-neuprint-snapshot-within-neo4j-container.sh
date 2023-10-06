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

# This is optionally set via the calling script, when you use
# ingest-neuprint-snapshot-using-apptainer <snapshot-dir> --debug-shell
DEBUG_SHELL=$1

set -e

# If we used the normal docker entrypoint, then we could leave the config in /conf.
# But since we bypass the docker entrypoint, then /conf is ignored, apparently.
# We must overwrite the default config file.
cp /conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf

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

# Relationship arguments.
NEURON_CONNECTSTO_ARG=--relationships=ConnectsTo=Neuprint_Neuron_Connections.csv
SYNSET_CONNECTSTO_ARG=--relationships=ConnectsTo=Neuprint_SynapseSet_to_SynapseSet.csv
SYNAPSE_SYNAPSESTO_ARG=--relationships=SynapsesTo=Neuprint_Synapse_Connections.csv
NEURON_CONTAINS_SYNSET_ARG=--relationships=Contains=Neuprint_Neuron_to_SynapseSet.csv
SYNSET_CONTAINS_SYNAPSE_ARG=--relationships=Contains=Neuprint_SynapseSet_to_Synapses.csv


cat > ingestion-args.txt << EOF
--force=true \
--database=data \
--normalize-types=false \
${META_ARG} \
${NEURON_ARGS} \
${SYNAPSE_ARGS} \
${SYNSET_ARG} \
${NEURON_CONNECTSTO_ARG} \
${SYNSET_CONNECTSTO_ARG} \
${SYNAPSE_SYNAPSESTO_ARG} \
${NEURON_CONTAINS_SYNSET_ARG} \
${SYNSET_CONTAINS_SYNAPSE_ARG} \

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
/var/lib/neo4j/bin/neo4j-admin import @ingestion-args.txt | tee /logs/import.log
end=$(date +%s)
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
trap "neo4j stop" EXIT
neo4j start --verbose

# Wait for neo4j to start. (Wait for the "Started." in the log file.)
grep -q 'Started\.' <(tail -n1 -f /logs/neo4j.log)

/var/lib/neo4j/bin/cypher-shell \
    -d data \
    --format verbose \
    -f /snapshot/create-indexes.cypher \
    | tee /logs/create-indexes.log
##

end=$(date +%s)
echo "[$(date)] Index creation completed."
echo "Duration: $(date -d@$((end-start)) -u +%H:%M:%S)"
