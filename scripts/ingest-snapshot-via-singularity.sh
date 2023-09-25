#!/bin/bash

set -e

if [[ -z "$1" ]]; then
    echo "Usage:" 1>&2
    echo "  ingest-snapshot-via-singularity.sh <snapshot-dir>" 1>&2
    echo "where <snapshot-dir> contains a 'neuprint' subdirectory containing CSV files and scripts to use for neuprint ingestion." 1>&2
    exit 1
fi

SNAPSHOT_DIR=$1

# The directory in which this bash script resides.
# https://stackoverflow.com/questions/59895
SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPTS_DIR}

# debug values
#SNAPSHOT_DIR=/groups/flyem/data/scratchspace/flyemflows/cns-full/tmp/small-test
#SCRIPTS_DIR=/groups/flyem/data/scratchspace/flyemflows/cns-full/tmp/scripts

# We export the database to /scratch.
# Assuming I'm on a cluster node, this directory is available.
WORKSPACE_DIR=${WORKSPACE_DIR-/scratch/${USER}/neo4j}
export APPTAINER_BIND="${SNAPSHOT_DIR}/neuprint:/snapshot"

# Create these directories in our workspace and
# mount them into the container.
mount_dirs=(data logs scripts conf plugins)
for d in ${mount_dirs[@]}
do
    mkdir -p ${WORKSPACE_DIR}/${d}
    chmod a+rw ${WORKSPACE_DIR}/${d}
    rm -rf ${WORKSPACE_DIR}/${d}/*

    # https://docs.sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html#
    export APPTAINER_BIND="${APPTAINER_BIND},${WORKSPACE_DIR}/${d}:/${d}"
done

# I have no idea why neo4j balks if the log file
# doesn't already exist when the server is launched.
touch ${WORKSPACE_DIR}/logs/neo4j.log

# I can't figure out how to make plugins work in cypher-shell
# The following doesn't do it.
# Fortunately, we don't really need apoc functions during the ingestion procedure.
# cp /groups/flyem/data/neo4j-plugins/apoc-4.4.0.7-all.jar ${WORKSPACE_DIR}/plugins/

cp * ${WORKSPACE_DIR}/scripts/
cp ${SNAPSHOT_DIR}/neuprint/create-indexes.cypher ${WORKSPACE_DIR}/scripts/
cp neo4j.conf ${WORKSPACE_DIR}/conf/

# Note: By default, the container's networking is the same as the host,
# so there's no need to map ports explicitly unless we want to use
# different ports within the container and host.

# We use --writable-tmpfs since neo4j needs a writable filesystem.
# https://github.com/apptainer/singularity/issues/4546#issuecomment-537152617

singularity exec --writable-tmpfs docker://neo4j:4.4.16 /scripts/ingest-snapshot.sh

# Now copy the database files from /scratch to the snapshot directory
echo "$(date '+%Y-%m-%d %H:%M:%S') Copying database to ${SNAPSHOT_DIR}"
cp -R ${WORKSPACE_DIR} ${SNAPSHOT_DIR}/
echo "$(date '+%Y-%m-%d %H:%M:%S') DONE"
