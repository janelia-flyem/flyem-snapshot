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

# We don't actually use plugins during ingestion,
# but it's convenient to have access to them when debugging.
# Install them by copying into NEO4J_HOME.
ls /plugins/* > /dev/null 2>&1 && cp /plugins/* ${NEO4J_HOME}/plugins/

# The neo4j.conf that ships alongside a snapshot is sized for a large cluster
# node (31G heap / 150G page cache). On a smaller machine the JVM cannot even
# reserve that much -- especially with -XX:+AlwaysPreTouch -- and neo4j dies
# with an unhelpful "Unexpected Neo4j server failure".
# Allow both to be overridden, e.g.
#   HEAP_SIZE=4G MAX_MEMORY=8G inspect-neuprint-snapshot <neo4j-export-dir>
# The defaults below match the committed neo4j.conf, so this is a no-op
# unless the caller overrides them.
HEAP_SIZE=${HEAP_SIZE:-31G}
MAX_MEMORY=${MAX_MEMORY:-150G}
sed -i \
    -e "s|^server\.memory\.heap\.initial_size=.*|server.memory.heap.initial_size=${HEAP_SIZE}|" \
    -e "s|^server\.memory\.heap\.max_size=.*|server.memory.heap.max_size=${HEAP_SIZE}|" \
    -e "s|^server\.memory\.pagecache\.size=.*|server.memory.pagecache.size=${MAX_MEMORY}|" \
    ${NEO4J_HOME}/conf/neo4j.conf

echo "Server memory settings in effect:"
grep -E '^server\.memory\.' ${NEO4J_HOME}/conf/neo4j.conf | sed 's/^/    /'

echo "Launching neo4j..."
trap "neo4j stop" EXIT
neo4j start --verbose

# Wait for neo4j to start. (Wait for the "Started." in the log file.)
grep -q 'Started\.' <(tail -n1 -f /logs/neo4j.log)

/bin/bash
