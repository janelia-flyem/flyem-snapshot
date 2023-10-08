#!/bin/bash

##
## Spins up a neo4j container using apptainer (singularity)
## to allow manual inspection of a neuprint database.
## This will launch neo4j and start a bash shell within the container.
## Users typicallly call this script indirectly via a Python wrapper script.
## See inspect_neuprint_snapshot.py
##

set -e

if [[ -z "$1" ]]; then
    echo "Usage:" 1>&2
    echo "" 1>&2
    echo "  inspect-neuprint-snapshot <neo4j-export-dir>" 1>&2
    echo "" 1>&2
    echo "where <neo4j-export-dir> contains: conf  data  logs  plugins" 1>&2
    echo "Note: It should be on a local drive, not network storage" 1>&2
    exit 1
fi

SNAPSHOT_NEO4J_ROOT=$1

export APPTAINER_BIND="${SNAPSHOT_NEO4J_ROOT}/conf:/conf,${APPTAINER_BIND}"
export APPTAINER_BIND="${SNAPSHOT_NEO4J_ROOT}/data:/data,${APPTAINER_BIND}"
export APPTAINER_BIND="${SNAPSHOT_NEO4J_ROOT}/logs:/logs,${APPTAINER_BIND}"
export APPTAINER_BIND="${SNAPSHOT_NEO4J_ROOT}/plugins:/plugins,${APPTAINER_BIND}"

# The directory in which this bash script resides.
# https://stackoverflow.com/questions/59895
SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p /tmp/scripts
cp -R "${SCRIPTS_DIR}"/* /tmp/scripts/
export APPTAINER_BIND="/tmp/scripts:/scripts,${APPTAINER_BIND}"

# I can't figure out how to make plugins work in cypher-shell
# The following doesn't do it.
# Fortunately, we don't really need apoc functions during the ingestion procedure.
# cp /groups/flyem/data/neo4j-plugins/apoc-4.4.0.7-all.jar ${WORKSPACE_DIR}/plugins/

# Note: By default, the container's networking is the same as the host,
# so there's no need to map ports explicitly unless we want to use
# different ports within the container and host.

# We use --writable-tmpfs since neo4j needs a writable filesystem.
# https://github.com/apptainer/singularity/issues/4546#issuecomment-537152617

# singularity run --writable-tmpfs docker://neo4j:4.4.16
singularity exec --writable-tmpfs docker://neo4j:4.4.16 /scripts/_launch_snapshot_and_bash_shell.sh

