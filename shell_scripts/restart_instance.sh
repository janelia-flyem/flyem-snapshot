#!/usr/bin/bash

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
    if [ $opt = 'Cancel' ]
    then
      exit
    else
      break
    fi
  done
  export INSTANCE=${servers[$REPLY-1]}
fi

if [ $INSTANCE == "production" ]; then
  SUFFIX=""
else
  SUFFIX="-${INSTANCE}"
fi
CHECK="/opt/conf/neuprinthttp${SUFFIX}"
if [ ! -e "$CHECK" ]; then
  echo "There is no neuprinthttp directory at ${CHECK}"
  exit
fi

echo "Will restart ${INSTANCE}"

INSTANCEDIR="/data15/app/neo4j/data/db/${INSTANCE}"
# To bypass the file check, just pass in a second parm (value unimportant)
if [ "$#" -ne 2 ]; then
  NEW_DIR="/data15/app/neo4j/data/db/${INSTANCE}_${TODAY}"
  for test_dir in ${NEW_DIR} "${NEW_DIR}/databases"; do
    echo "Checking for ${test_dir}"
    if [ ! -e "$test_dir" ]; then
      echo "${test_dir} does not exist"
      exit
    fi
  done
else
  for test_dir in ${INSTANCEDIR} "${INSTANCEDIR}/databases"; do
    echo "Checking for ${test_dir}"
    if [ ! -e "$test_dir" ]; then
      echo "${test_dir} does not exist"
      exit
    fi
  done
fi

# Scale down the neuprint server and Neo4J database
docker service scale em_services_neuprinthttp${SUFFIX}=0
docker service scale em_services_neo4j${SUFFIX}=0
sleep 10

if [ "$#" -ne 2 ]; then
# Swap in the new database files
  mv ${INSTANCEDIR} ${INSTANCEDIR}_${TODAY}.backup
  ls -l ${NEW_DIR}
  mv ${NEW_DIR} ${INSTANCEDIR}
  ls -l ${INSTANCEDIR}
fi

# Scale up the neuprint server and Neo4J database
sleep 10
docker service scale em_services_neo4j${SUFFIX}=1
sleep 60
docker service scale em_services_neuprinthttp${SUFFIX}=1

# Send email to neuprint-admin-aaaabgimb576a7yacean6k2n2a@hhmi.org.slack.com
