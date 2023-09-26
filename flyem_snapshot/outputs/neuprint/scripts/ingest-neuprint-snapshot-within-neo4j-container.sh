#!/bin/bash

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

set -e

# If we used the normal docker entrypoint, then we could leave the config in /conf.
# But since we bypass the docker entrypoint, then /conf is ignored, apparently.
# We must overwrite the default config file.
cp /conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf

##
## Import CSVs for nodes/relationships
##

META_ARG=--nodes=/snapshot/Neuprint_Meta.csv
SYNSET_ARG=--nodes=/snapshot/Neuprint_SynapseSet.csv

# These may each result in thousands (or hundreds of thousands) of arguments,
# but that's our only option without incremental import.
NEURON_ARGS=$(for f in $(find /snapshot/Neuprint_Neurons -name "*.csv"); do printf -- "--nodes=$f "; done)
SYNAPSE_ARGS=$(for f in $(find /snapshot/Neuprint_Synapses -name "*.csv"); do printf -- "--nodes=$f "; done)

NEURON_CONNECTSTO_ARG=--relationships=ConnectsTo=/snapshot/Neuprint_Neuron_Connections.csv
SYNSET_CONNECTSTO_ARG=--relationships=ConnectsTo=/snapshot/Neuprint_SynapseSet_to_SynapseSet.csv
SYNAPSE_SYNAPSESTO_ARG=--relationships=SynapsesTo=/snapshot/Neuprint_Synapse_Connections.csv

NEURON_CONTAINS_SYNSET_ARG=--relationships=Contains=/snapshot/Neuprint_Neuron_to_SynapseSet.csv
SYNSET_CONTAINS_SYNAPSE_ARG=--relationships=Contains=/snapshot/Neuprint_SynapseSet_to_Synapses.csv

echo "Ingesting nodes and relationships"
start=$(date +%s)
/var/lib/neo4j/bin/neo4j-admin import \
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
    | tee /logs/import.log \
##

end=$(date +%s)
echo "Node/relationship ingest completed at: $(date)"
echo "Duration: $(date -d@$((end-start)) -u +%H:%M:%S)"

##
## Create indexes
##

# neo4j seems to require a lot of time to shut down.
# Let's give it a ton of time:
# Half of the time it took to run the ingestion.
export NEO4J_SHUTDOWN_TIMEOUT=$(((end-start)/2))

start=$(date +%s)
echo "Launching neo4j..."
trap "neo4j stop" EXIT
neo4j start --verbose

# Give neo4j a few seconds to start
sleep 10
tail -n1 /logs/neo4j.log

/var/lib/neo4j/bin/cypher-shell \
    -d data \
    --format verbose \
    -f /snapshot/create-indexes.cypher \
    | tee /logs/create-indexes.log
##

end=$(date +%s)
echo "Indexes creation completed at: $(date)"
echo "Duration: $(date -d@$((end-start)) -u +%H:%M:%S)"
