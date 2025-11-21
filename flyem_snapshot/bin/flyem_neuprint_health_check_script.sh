#!/bin/bash

source /groups/flyem/proj/cluster/miniforge/bin/activate flyem-312
cd /groups/flyem/data/snapshots

input_file="server_health.txt"
if [ ! -f "$input_file" ]; then
  echo "Error: File '$input_file' not found."
  exit 1
fi
echo "Reading ${input_file}"
readarray -t server_list < "$input_file"
for item in "${server_list[@]}"; do
  echo "$item"
done

for SERVER in "${server_list[@]}"
do
  if [[ $SERVER =~ ^#.* || $SERVER =~ ^\s*$ ]] ;
  then
    continue
  fi
  if [ $SERVER == "neuprint" ]; then
    SERVER=""
  fi
  if [ -z ${SERVER} ]; then
    INSTANCE="neuprint"
    python health_check.py
  else
    INSTANCE="neuprint-${SERVER}"
    python health_check.py --server ${SERVER}
  fi
  if [ $? -eq 0 ]; then
    echo "Server ${INSTANCE} is up"
  else
    echo "Server ${INSTANCE} is down"
    if [ -z ${SERVER} ]; then
      ssh flyem@emdata5 "/data15/app/neo4j/data/db/basic_restart_instance.sh"
    else
      ssh flyem@emdata5 "/data15/app/neo4j/data/db/basic_restart_instance.sh ${SERVER}"
    fi
    echo "Server: ${SERVER}"
    echo "Instance: ${INSTANCE}"
    if [ -z ${SERVER} ]; then
      SENDTO="neuprint-updates-aaaarfzyuewobt3wf3zhsqe6xm@hhmi.org.slack.com"
    elif [ ${SERVER} = "cns" ]; then
      SENDTO="neuprint-updates-cns-aaaardxlenzhljk3pvy4ivp4xm@hhmi.org.slack.com"
    elif [ ${SERVER} = "fish2" ]; then
      SENDTO="neuprint-updates-fish-aaaaqvjm2vabn3onhdnntzuory@hhmi.org.slack.com"
    elif [ ${SERVER} = "yakuba" ]; then
      SENDTO="neuprint-updates-yaku-aaaaqwi6eytoc76xuuazcemh2i@hhmi.org.slack.com"
    else
      SENDTO="neuprint-updates-aaaarfzyuewobt3wf3zhsqe6xm@hhmi.org.slack.com"
    fi
    echo "${INSTANCE} has been restarted" | mailx -s "${INSTANCE} restarted" ${SENDTO}
  fi
done
