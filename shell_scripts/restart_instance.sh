#!/usr/bin/bash
#
# restart_instance.sh — Swap in a new Neo4j database and restart neuprint services
#
# Usage:
#   restart_instance.sh [INSTANCE [SKIP_DB_SWAP]]
#
# Arguments:
#   INSTANCE      Name of the neuprint instance to restart (e.g. "production",
#                 "hemibrain").  If omitted, an interactive menu is presented
#                 listing every instance found under /opt/conf/neuprinthttp-*.
#   SKIP_DB_SWAP  Any second argument disables the database swap (see below).
#
# Normal operation (one argument, or interactive selection):
#   1. Validates that a config directory exists at
#        /opt/conf/neuprinthttp          (production)
#        /opt/conf/neuprinthttp-INSTANCE (all others)
#   2. Verifies that a today-dated database directory exists:
#        /data1/data15/app/neo4j/data/db/INSTANCE_YYYYMMDD
#        /data1/data15/app/neo4j/data/db/INSTANCE_YYYYMMDD/databases
#   3. Scales the Docker services to zero:
#        em_services_neuprinthttp[-SUFFIX]
#        em_services_neo4j[-SUFFIX]
#   4. Moves the live database to INSTANCE_YYYYMMDD.backup and promotes the
#      today-dated directory to be the new live database (INSTANCE_YYYYMMDD -> INSTANCE).
#   5. Scales neo4j back to 1 replica, waits 60 s, then scales neuprinthttp
#      back to 1 replica.
#
# Skip-swap mode (two arguments):
#   Steps 1 and 2 still run, but they check the existing INSTANCE directory
#   (not a today-dated one).  Steps 3 and 5 still execute; step 4 is skipped,
#   so the live database files are left untouched.  Useful for a plain service
#   restart without a database update.
#
# Requirements:
#   - Must be run with sufficient privileges to call `docker service scale`.
#   - Neo4j data lives under /data1/data15/app/neo4j/data/db/.
#       e.g. /data1/data15/app/neo4j/data/db/production
#            /data1/data15/app/neo4j/data/db/production_20260603
#            /data1/data15/app/neo4j/data/db/production_20260603.backup
#            /data1/data15/app/neo4j/data/db/hemibrain_20260603
#   - Instance configs live under /opt/conf/.
#       e.g. /opt/conf/neuprinthttp             (production)
#            /opt/conf/neuprinthttp-hemibrain

TODAY=$(date '+%Y%m%d')

echo "Running as `whoami`"
id

if [ "$#" -gt 0 ]
then
  export INSTANCE=$1
else
  declare -a servers=()
  IFS=$'\n' raw=($(ls -d /opt/conf/neuprinthttp-*))
  for srv in ${raw[@]}; do
    servers+=(`echo ${srv} | sed 's/.*-//'`)
  done
  echo "Select an instance to restart"
  PS3="Instance: "
  select opt in "${servers[@]}" "Cancel"
  do
    if [ "$opt" = 'Cancel' ]; then
      exit
    elif [ -n "$opt" ]; then
      break
    fi
  done
  export INSTANCE=${servers[$REPLY-1]}
fi

if [ "$INSTANCE" == "production" ]; then
  SUFFIX=""
else
  SUFFIX="-${INSTANCE}"
fi
CHECK="/opt/conf/neuprinthttp${SUFFIX}"
if [ ! -e "$CHECK" ]; then
  echo "There is no neuprinthttp directory at ${CHECK}"
  exit 1
fi

echo "Will restart ${INSTANCE}"

INSTANCEDIR="/data1/data15/app/neo4j/data/db/${INSTANCE}"
# To bypass the file check, just pass in a second parm (value unimportant)
if [ "$#" -ne 2 ]; then
  NEW_DIR="/data1/data15/app/neo4j/data/db/${INSTANCE}_${TODAY}"
  for test_dir in ${NEW_DIR} "${NEW_DIR}/databases"; do
    echo "Checking for ${test_dir}"
    if [ ! -e "$test_dir" ]; then
      echo "${test_dir} does not exist"
      exit 1
    fi
  done
else
  for test_dir in ${INSTANCEDIR} "${INSTANCEDIR}/databases"; do
    echo "Checking for ${test_dir}"
    if [ ! -e "$test_dir" ]; then
      echo "${test_dir} does not exist"
      exit 1
    fi
  done
fi

# Scale down the neuprint server and Neo4J database
docker service scale em_services_neuprinthttp${SUFFIX}=0
docker service scale em_services_neo4j${SUFFIX}=0
sleep 10

if [ "$#" -ne 2 ]; then
# Swap in the new database files
  echo "Contents of current database ${NEW_DIR}:"
  ls -l ${INSTANCEDIR}
  echo "Moving ${INSTANCEDIR} to ${INSTANCEDIR}_${TODAY}.backup"
  if ! mv ${INSTANCEDIR} ${INSTANCEDIR}_${TODAY}.backup; then
    echo "ERROR: failed to move ${INSTANCEDIR} to ${INSTANCEDIR}_${TODAY}.backup — aborting without touching ${NEW_DIR}"
    exit 1
  fi
  echo "Contents of new database ${NEW_DIR}:"
  ls -l ${NEW_DIR}
  echo "Moving ${NEW_DIR} to ${INSTANCEDIR}"
  if ! mv ${NEW_DIR} ${INSTANCEDIR}; then
    echo "ERROR: failed to move ${NEW_DIR} to ${INSTANCEDIR} — attempting to restore backup"
    if mv ${INSTANCEDIR}_${TODAY}.backup ${INSTANCEDIR}; then
      echo "Backup restored successfully to ${INSTANCEDIR} — no database change was made"
    else
      echo "ERROR: backup restore also failed — live database is missing; manual recovery required"
      echo "  Backup location: ${INSTANCEDIR}_${TODAY}.backup"
      echo "  New database:    ${NEW_DIR}"
      echo "To restore the database and restart services, run:"
      echo "  mv ${INSTANCEDIR}_${TODAY}.backup ${INSTANCEDIR}"
      echo "  docker service scale em_services_neo4j${SUFFIX}=1"
      echo "  sleep 60"
      echo "  docker service scale em_services_neuprinthttp${SUFFIX}=1"
    fi
    exit 1
  fi
  echo "Contents of ${INSTANCEDIR}:"
  ls -l ${INSTANCEDIR}
fi

# Scale up the neuprint server and Neo4J database
sleep 10
docker service scale em_services_neo4j${SUFFIX}=1
sleep 60
docker service scale em_services_neuprinthttp${SUFFIX}=1

# Send email to neuprint-admin-aaaabgimb576a7yacean6k2n2a@hhmi.org.slack.com
