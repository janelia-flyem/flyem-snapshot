#!/bin/bash

set -e

if [[ -z "$1" ]]; then
    echo "Usage:" 1>&2
    echo "" 1>&2
    echo "  ingest-neuprint-snapshot-using-apptainer.sh <snapshot-dir> [--debug-shell]" 1>&2
    echo "" 1>&2
    echo "where <snapshot-dir> contains a 'neuprint' subdirectory containing CSV files" 1>&2
    echo " and scripts to use for neuprint ingestion." 1>&2
    echo "" 1>&2
    echo "If --debug-shell is given, then you'll be dropped into a bash shell within " 1>&2
    echo "the container instead of launching the ingestion script." 1>&2
    echo "" 1>&2
    echo "By default, this script uses /scratch for temporary storage."  1>&2
    echo "To override that, use WORKSPACE_DIR:" 1>&2
    echo "  WORKSPACE_DIR=/tmp/neo4j ingest-neuprint-snapshot-using-apptainer.sh <snapshot-dir>" 1>&2
    echo "" 1>&2
    exit 1
fi

SNAPSHOT_DIR=$1
DEBUG_SHELL=$2

if [[ ! -z "${DEBUG_SHELL}" && "${DEBUG_SHELL}" != "--debug-shell" ]]; then
    echo "Error: The only permitted value for the second argument is '--debug-shell', not '${DEBUG_SHELL}'" 1>&2
    exit 1
fi

if [[ ! -d "${SNAPSHOT_DIR}/neuprint" ]]; then
    echo "Error: The snapshot diretory you provided does not contain a 'neuprint' subdirectory:" 1>&2
    echo "${SNAPSHOT_DIR}" 1>&2
    exit 1
fi

# The directory in which this bash script resides.
# https://stackoverflow.com/questions/59895
SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# By default, we export the database to /scratch.
# Assuming I'm on a cluster node, this directory is available.
WORKSPACE_DIR=${WORKSPACE_DIR-/scratch/${USER}/$(basename ${SNAPSHOT_DIR})/neo4j}

if [[ -z "${DEBUG_SHELL}" ]]
then
    mv ${WORKSPACE_DIR} ${WORKSPACE_DIR}-old 2> /dev/null || true
fi

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

# I have no idea why, but neo4j balks if the log file
# doesn't already exist when the server is launched.
touch ${WORKSPACE_DIR}/logs/neo4j.log

# Note: The plugins still need to be installed into ${NEO4J_HOME}/plugins once the container is launched.
# cp /groups/flyem/data/neo4j-plugins/apoc-4.4.0.7-all.jar ${WORKSPACE_DIR}/plugins/
APOC_PLUGINS_URL=https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.7/apoc-4.4.0.7-all.jar
wget -q ${APOC_PLUGINS_URL} -P ${WORKSPACE_DIR}/plugins/

cp ${SCRIPTS_DIR}/* ${WORKSPACE_DIR}/scripts/
cp ${SNAPSHOT_DIR}/neuprint/create-indexes.cypher ${WORKSPACE_DIR}/scripts/
cp ${SCRIPTS_DIR}/neo4j.conf ${WORKSPACE_DIR}/conf/

# Note: By default, the container's networking is the same as the host,
# so there's no need to map ports explicitly unless we want to use
# different ports within the container and host.

# We use --writable-tmpfs since neo4j needs a writable filesystem.
# https://github.com/apptainer/singularity/issues/4546#issuecomment-537152617

if [[ ! -z "${DEBUG_SHELL}" ]]
then
    apptainer exec --writable-tmpfs docker://neo4j:4.4.16 /scripts/ingest-neuprint-snapshot-within-neo4j-container.sh --debug-shell
    exit $?
else
    apptainer exec --writable-tmpfs docker://neo4j:4.4.16 /scripts/ingest-neuprint-snapshot-within-neo4j-container.sh
    if [[ "$?" != "0" ]]
    then
        exit $?
    fi

    # In theory, the apptainer command above ought to have failed already if there was a failure in the log.
    # But I'm not sure if that works as it's supposed to, so here's an extra check
    if grep -i 'import failed' ${WORKSPACE_DIR}/logs/import.*.log > /dev/null
    then
        2>&1 echo "ERROR: Apptainer exited cleanly, but the neo4j log indicates that the import failed!"
        exit 1
    fi

    # Now copy the database files from /scratch to the snapshot directory
    echo "$(date '+%Y-%m-%d %H:%M:%S') Copying database to ${SNAPSHOT_DIR}"
    cp -R ${WORKSPACE_DIR} ${SNAPSHOT_DIR}/
    echo "$(date '+%Y-%m-%d %H:%M:%S') DONE"
fi
