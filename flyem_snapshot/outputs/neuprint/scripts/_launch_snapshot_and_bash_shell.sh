#!/bin/bash

##
## Intended to be executed WITHIN a neo4j container which was configured for neuprint ingestion.
## Launches neo4j and cypher-shell.
## Stops neo4j upon exit.
##

set -e

# If we used the normal docker entrypoint, then we could leave the config in /conf.
# But since we bypass the docker entrypoint, then /conf is ignored, apparently.
# We must overwrite the default config file.
cp /conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf

echo "Launching neo4j..."
trap "neo4j stop" EXIT
neo4j start --verbose

# Wait for neo4j to start. (Wait for the "Started." in the log file.)
grep -q 'Started\.' <(tail -n1 -f /logs/neo4j.log)

/bin/bash
