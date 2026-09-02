#!/bin/bash
##
## check-neuprint-db.sh <neo4j-export-dir>
##
## Launches a neuprint snapshot's neo4j database in a container and runs a
## series of validation checks against it, then shuts it down.
##
## Exits 0 if every check passed, 1 otherwise, so it is usable from CI or a
## post-ingestion hook.
##
## Usage:
##   pixi run bash check-neuprint-db.sh 2026-05-12-32c9ac/neo4j
##
## The dataset name is read from the :Meta node, so this works on any dataset
## (wasp, hemibrain, fish2, ...) without configuration.
##
## Environment overrides, all optional:
##
##   NEO4J_DB     Database to check. Default 'data', which is what this
##                pipeline builds under neo4j 5.x / CalVer. Use 'neo4j' to
##                check a pre-upgrade 4.4-era database, e.g. to capture a
##                baseline before swapping in a new one:
##                  NEO4J_DB=neo4j NEO4J_IMAGE=docker://neo4j:4.4.16 \
##                      check-neuprint-db.sh <dir>
##
##   NEO4J_IMAGE  Container image. Default docker://neo4j:2026.07.1. Must be
##                able to open the store you are pointing it at -- neo4j has
##                no downgrade path, so an older image cannot read a newer
##                store.
##
##   HEAP_SIZE    Override the database's own neo4j.conf memory sizing, which
##   MAX_MEMORY   is otherwise respected as-is. Needed only when checking a
##                cluster-sized snapshot on a smaller machine:
##                  HEAP_SIZE=4G MAX_MEMORY=8G check-neuprint-db.sh <dir>
##
## Version-dependent checks (Cypher default language, default database name)
## downgrade to informational when they do not apply, so the suite stays
## meaningful against an older database.
##

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: check-neuprint-db.sh <neo4j-export-dir>" 1>&2
    echo "  where <neo4j-export-dir> contains: conf data logs plugins" 1>&2
    exit 2
fi

if [[ ! -d "$1" ]]; then
    echo "Error: no such directory: $1" 1>&2
    exit 2
fi

NEO4J_DIR=$(cd -- "$1" && pwd)
NEO4J_IMAGE=${NEO4J_IMAGE:-docker://neo4j:2026.07.1}
NEO4J_DB=${NEO4J_DB:-data}

for d in conf data logs plugins; do
    if [[ ! -d "${NEO4J_DIR}/${d}" ]]; then
        echo "Error: ${NEO4J_DIR} does not contain a '${d}' subdirectory." 1>&2
        exit 2
    fi
done

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

##
## The in-container half. Quoted heredoc: nothing here is expanded by the host.
##
cat > "${WORK}/checks.sh" <<'CHECKS'
#!/bin/bash
set -uo pipefail

# Which database to open. The host always forwards this as APPTAINERENV_NEO4J_DB;
# defaulted here as well so this half is runnable standalone and so 'set -u'
# can never trip on it.
NEO4J_DB=${NEO4J_DB:-data}

cp /conf/neo4j.conf ${NEO4J_HOME}/conf/neo4j.conf
ls /plugins/* > /dev/null 2>&1 && cp /plugins/* ${NEO4J_HOME}/plugins/

# Respect the snapshot's own sizing unless explicitly overridden.
if [[ -n "${HEAP_SIZE:-}" ]]; then
    sed -i -e "s|^server\.memory\.heap\.initial_size=.*|server.memory.heap.initial_size=${HEAP_SIZE}|" \
           -e "s|^server\.memory\.heap\.max_size=.*|server.memory.heap.max_size=${HEAP_SIZE}|" \
           ${NEO4J_HOME}/conf/neo4j.conf
fi
if [[ -n "${MAX_MEMORY:-}" ]]; then
    sed -i -e "s|^server\.memory\.pagecache\.size=.*|server.memory.pagecache.size=${MAX_MEMORY}|" \
           ${NEO4J_HOME}/conf/neo4j.conf
fi

echo "Starting neo4j..."

# The log lives in the snapshot directory and PERSISTS between runs, so a
# stale 'Started.' from an earlier successful run would make the check below
# pass even when this start failed -- and then every query returns nothing,
# which looks like a corrupt database rather than a server that never came up.
# Truncate first so what we grep for can only have come from this run.
: > /logs/neo4j.log

START_OUT=$(neo4j start 2>&1)
for _ in $(seq 120); do
    grep -q 'Started\.' /logs/neo4j.log 2>/dev/null && break
    sleep 1
done
if ! grep -q 'Started\.' /logs/neo4j.log 2>/dev/null; then
    echo "ERROR: neo4j did not start." 1>&2
    echo "--- neo4j start output ---" 1>&2
    echo "${START_OUT}" | tail -30 1>&2
    echo "--- /logs/neo4j.log (tail) ---" 1>&2
    tail -30 /logs/neo4j.log 1>&2
    exit 1
fi
trap 'neo4j stop > /dev/null 2>&1' EXIT

# Constrain cypher-shell's thread pools.
#
# The JVM and Netty both size their pools from availableProcessors(), so on a
# many-core machine each cypher-shell invocation tries to spawn ~80 G1 GC
# threads plus a Netty event-loop group of 2x the core count. Observed on a
# 128-core node: every invocation died with
#
#   pthread_create failed (EAGAIN) ... epollEventLoopGroup-3-88
#   java.lang.OutOfMemoryError: unable to create native thread
#
# and this script makes dozens of invocations. ActiveProcessorCount is a single
# lever that shrinks everything derived from the CPU count at once. Nothing
# cypher-shell does here is CPU-bound -- it issues a query and prints a row.
#
# Set AFTER the server has started, so only the client is affected.
export JAVA_OPTS="${JAVA_OPTS:-} -XX:ActiveProcessorCount=4"

CS=/var/lib/neo4j/bin/cypher-shell

# Run a query, strip the header row and surrounding quotes.
#
# The grep is a safety net: if the JVM writes warnings or a stack trace to
# stdout, they would otherwise be treated as query results and embedded into
# the check messages, turning one failure into hundreds of lines of noise.
# Drop JVM unified-logging lines, stack-trace frames, and log4j errors.
q() {
    ${CS} -d "${NEO4J_DB}" --format plain "$1" 2>/dev/null \
        | grep -vE '^\[[0-9]+\.[0-9]+s\]\[|^[[:space:]]+at |^java\.lang\.|^[0-9]{4}-[0-9]{2}-[0-9]{2}T.*(ERROR|WARN)|Failed to (start|submit)' \
        | tail -n +2 | tr -d '"'
}

PASSES=0
FAILURES=0
ok()   { printf '  PASS  %s\n' "$1"; PASSES=$((PASSES+1)); }
bad()  { printf '  FAIL  %s\n' "$1"; FAILURES=$((FAILURES+1)); }
skip() { printf '  SKIP  %s\n' "$1"; }
info() { printf '  ....  %s\n' "$1"; }

# expect_eq <description> <actual> <expected>
expect_eq() {
    if [[ "$2" == "$3" ]]; then ok "$1 (= $2)"; else bad "$1 -- expected '$3', got '$2'"; fi
}

# expect_gt <description> <actual> <minimum>
expect_gt() {
    if [[ -n "$2" && "$2" =~ ^[0-9]+$ && "$2" -gt "$3" ]]; then
        ok "$1 (= $2)"
    else
        bad "$1 -- expected > $3, got '$2'"
    fi
}

echo
echo "=============================================================="
echo " Dataset / topology"
echo "=============================================================="

DS=$(q "MATCH (m:Meta) RETURN m.dataset;" | head -1)
if [[ -z "${DS}" ]]; then
    # Distinguish "the database is empty/odd" from "the server or database is
    # not actually usable", which otherwise present identically.
    echo "ERROR: could not read the dataset name from the :Meta node." 1>&2
    echo "--- SHOW DATABASES ---" 1>&2
    ${CS} -d system "SHOW DATABASES;" 2>&1 | tail -20 1>&2
    echo "--- raw response to the :Meta query ---" 1>&2
    ${CS} -d "${NEO4J_DB}" "MATCH (m:Meta) RETURN m.dataset;" 2>&1 | tail -20 1>&2
    exit 1
fi
info "dataset: ${DS}"
expect_eq "exactly one :Meta node" "$(q "MATCH (m:Meta) RETURN count(m);")" "1"

SEG_COUNT=$(q "MATCH (n:${DS}_Segment) RETURN count(n);")
expect_gt ":${DS}_Segment nodes"    "${SEG_COUNT}" 0
expect_gt ":${DS}_Neuron nodes"     "$(q "MATCH (n:${DS}_Neuron)     RETURN count(n);")" 0
expect_gt ":${DS}_Synapse nodes"    "$(q "MATCH (n:${DS}_Synapse)    RETURN count(n);")" 0
expect_gt ":${DS}_SynapseSet nodes" "$(q "MATCH (n:${DS}_SynapseSet) RETURN count(n);")" 0

# Every node should carry one of the labels we know about. Nodes with several
# labels (a :Neuron is also a :Segment) are counted once by the single MATCH,
# so this catches stray or mislabelled nodes without double-counting.
TOTAL_NODES=$(q "MATCH (n) RETURN count(n);")
ACCOUNTED=$(q "MATCH (n)
               WHERE n:${DS}_Segment OR n:${DS}_Synapse OR n:${DS}_SynapseSet
                  OR n:Meta OR n:${DS}_Element OR n:${DS}_ElementSet
               RETURN count(n);")
info "total nodes: ${TOTAL_NODES}"
expect_eq "every node accounted for by a known label" "${ACCOUNTED}" "${TOTAL_NODES}"

expect_gt "ConnectsTo relationships" "$(q "MATCH ()-[r:ConnectsTo]->() RETURN count(r);")" 0
expect_gt "SynapsesTo relationships" "$(q "MATCH ()-[r:SynapsesTo]->() RETURN count(r);")" 0
expect_gt "Contains relationships"   "$(q "MATCH ()-[r:Contains]->()   RETURN count(r);")" 0

expect_eq "every :Neuron is also a :Segment" \
    "$(q "MATCH (n:${DS}_Neuron) WHERE NOT n:${DS}_Segment RETURN count(n);")" "0"

echo
echo "=============================================================="
echo " Integrity"
echo "=============================================================="

expect_eq "no duplicate Segment bodyIds" \
    "$(q "MATCH (n:${DS}_Segment) WITH n.bodyId AS b, count(*) AS c WHERE c > 1 RETURN count(b);")" "0"

expect_eq "no Segment with a null bodyId" \
    "$(q "MATCH (n:${DS}_Segment) WHERE n.bodyId IS NULL RETURN count(n);")" "0"

expect_gt "uniqueness constraints present" \
    "$(q "SHOW CONSTRAINTS YIELD name RETURN count(name);")" 1

echo
echo "=============================================================="
echo " Indexes"
echo "=============================================================="

expect_eq "no index in a non-ONLINE state" \
    "$(q "SHOW INDEXES YIELD state WHERE state <> 'ONLINE' RETURN count(state);")" "0"

expect_eq "no index below 100% populated" \
    "$(q "SHOW INDEXES YIELD populationPercent WHERE populationPercent < 100.0 RETURN count(populationPercent);")" "0"

info "total indexes: $(q "SHOW INDEXES YIELD name RETURN count(name);")"

# Which property names actually exist on Segment nodes?
# (ROI booleans are sparse, so this must span all Segments, not a sample.
# That means a full scan -- slow on a large dataset, hence the notice.)
printf '  ....  scanning property keys across all %s Segments...\n' "${SEG_COUNT}"
SEG_PROPS=$(q "MATCH (n:${DS}_Segment) UNWIND keys(n) AS k RETURN DISTINCT k;" | sort -u)

# Which property names do the Segment RANGE indexes claim to index?
IDX_PROPS=$(q "SHOW INDEXES YIELD labelsOrTypes, properties, entityType, type
               WHERE entityType = 'NODE' AND type = 'RANGE' AND '${DS}_Segment' IN labelsOrTypes
               RETURN properties[0];" | sort -u)

INERT=$(comm -23 <(echo "${IDX_PROPS}") <(echo "${SEG_PROPS}") | sed '/^$/d')
if [[ -z "${INERT}" ]]; then
    ok "every Segment index refers to a property that exists"
else
    bad "indexes referring to non-existent properties: $(echo "${INERT}" | wc -l | tr -d ' ')"
    echo "${INERT}" | head -10 | sed 's/^/          /'
fi

echo
echo "=============================================================="
echo " ROI naming consistency"
echo "=============================================================="

# The Meta node's roiInfo is the contract clients rely on. Every ROI it
# advertises should exist as a property on at least one Segment. This is the
# regression test for the sanitize_roi_name mismatch.
META_ROIS=$(q "MATCH (m:${DS}_Meta)
               UNWIND keys(apoc.convert.fromJsonMap(m.roiInfo)) AS r
               RETURN DISTINCT r;" | sort -u | sed '/^$/d')

if [[ -z "${META_ROIS}" ]]; then
    skip "could not read Meta.roiInfo (is the APOC plugin present?)"
else
    info "ROIs advertised by Meta.roiInfo: $(echo "${META_ROIS}" | wc -l | tr -d ' ')"
    MISSING=$(comm -23 <(echo "${META_ROIS}") <(echo "${SEG_PROPS}") | sed '/^$/d')
    if [[ -z "${MISSING}" ]]; then
        ok "every ROI in Meta.roiInfo exists as a Segment property"
    else
        bad "ROIs in Meta.roiInfo with no matching Segment property: $(echo "${MISSING}" | wc -l | tr -d ' ')"
        echo "${MISSING}" | head -10 | sed 's/^/          /'
    fi

    # Every ROI the Meta node advertises should have a Segment index backing it.
    # This is deterministic, unlike asking whether the planner *chooses* that
    # index for a given query -- that is a cost decision based on selectivity,
    # and a label scan is legitimately cheaper for an ROI covering a large
    # fraction of Segments.
    MISSING_IDX=$(comm -23 <(echo "${META_ROIS}") <(echo "${IDX_PROPS}") | sed '/^$/d')
    if [[ -z "${MISSING_IDX}" ]]; then
        ok "every ROI in Meta.roiInfo has a Segment index"
    else
        bad "ROIs in Meta.roiInfo with no Segment index: $(echo "${MISSING_IDX}" | wc -l | tr -d ' ')"
        echo "${MISSING_IDX}" | head -10 | sed 's/^/          /'
    fi

    # Informational only: report whether the planner picks an index seek for a
    # ROI whose name contains characters that naive sanitizing would rewrite.
    # A scan here is not a failure -- see above.
    TRICKY=$(echo "${META_ROIS}" | grep -F '(' | head -1 || true)
    if [[ -n "${TRICKY}" ]]; then
        TRICKY_N=$(q "MATCH (n:${DS}_Segment) WHERE n.\`${TRICKY}\` = true RETURN count(n);")
        PLAN=$(${CS} -d "${NEO4J_DB}" "EXPLAIN MATCH (n:${DS}_Segment) WHERE n.\`${TRICKY}\` = true RETURN count(n);" 2>&1)
        if grep -q 'NodeIndexSeek' <<<"${PLAN}"; then
            info "planner uses NodeIndexSeek for ROI '${TRICKY}' (${TRICKY_N} segments)"
        else
            info "planner uses a scan for ROI '${TRICKY}' (${TRICKY_N} of ${SEG_COUNT} segments)"
        fi
    fi
fi

echo
echo "=============================================================="
echo " Index usability"
echo "=============================================================="

# Whether the planner *chooses* an index is a cost decision we must not assert
# on -- a scan is legitimately cheaper for an ROI covering many Segments.
# Whether an index *can* be used is a different question, and it is testable:
# an index hint either resolves or neo4j raises an error. So force every ROI
# index with a hint; a missing, broken or inapplicable index then fails loudly.
#
# Total index reads are snapshotted around that workload as direct evidence the
# indexes served real reads rather than merely existing.

READS_BEFORE=$(q "SHOW INDEXES YIELD readCount WHERE readCount IS NOT NULL RETURN sum(readCount);")

HINT_TESTED=0
HINT_FAILED=0
HINT_FAILED_NAMES=""

if [[ -n "${META_ROIS}" ]]; then
    HINT_FILE=$(mktemp)
    while IFS= read -r roi; do
        [[ -z "${roi}" ]] && continue
        printf 'MATCH (n:%s_Segment) USING INDEX n:%s_Segment(`%s`) WHERE n.`%s` = true RETURN count(n);\n' \
            "${DS}" "${DS}" "${roi}" "${roi}" >> "${HINT_FILE}"
        HINT_TESTED=$((HINT_TESTED+1))
    done <<< "${META_ROIS}"

    # Fast path: run all of them through a single cypher-shell. Starting a JVM
    # per query would cost a couple of seconds each, which is minutes over ~100
    # ROIs. --fail-at-end runs every statement rather than stopping at the first
    # error, so a clean exit means all hints resolved.
    printf '  ....  exercising %s ROI indexes via index hints...\n' "${HINT_TESTED}"
    if ${CS} -d "${NEO4J_DB}" --fail-at-end -f "${HINT_FILE}" > /dev/null 2>&1; then
        HINT_FAILED=0
    else
        # Something failed (or --fail-at-end is unsupported on this server).
        # Fall back to per-ROI probing to name the offenders. Slow, but only
        # ever runs when there is something to report.
        printf '  ....  a hint failed; probing individually to identify which...\n'
        n=0
        while IFS= read -r roi; do
            [[ -z "${roi}" ]] && continue
            n=$((n+1))
            printf '\r  ....  probing %s/%s' "${n}" "${HINT_TESTED}"
            if ! ${CS} -d "${NEO4J_DB}" \
                    "MATCH (n:${DS}_Segment)
                     USING INDEX n:${DS}_Segment(\`${roi}\`)
                     WHERE n.\`${roi}\` = true
                     RETURN count(n);" > /dev/null 2>&1; then
                HINT_FAILED=$((HINT_FAILED+1))
                HINT_FAILED_NAMES="${HINT_FAILED_NAMES}${roi}"$'\n'
            fi
        done <<< "${META_ROIS}"
        printf '\r                                        \r'
    fi
    rm -f "${HINT_FILE}"
fi

if [[ "${HINT_TESTED}" -eq 0 ]]; then
    skip "no ROI indexes to exercise"
elif [[ "${HINT_FAILED}" -eq 0 ]]; then
    ok "all ${HINT_TESTED} ROI indexes usable when forced via index hint"
else
    bad "${HINT_FAILED} of ${HINT_TESTED} ROI indexes could not be used"
    echo "${HINT_FAILED_NAMES}" | sed '/^$/d' | head -10 | sed 's/^/          /'
fi

# readCount is sampled telemetry, not a synchronous counter: neo4j flushes
# index usage statistics periodically, so a fast workload can finish before the
# numbers move. Poll briefly to give them a chance to catch up.
READS_AFTER="${READS_BEFORE}"
for _ in $(seq 12); do
    READS_AFTER=$(q "SHOW INDEXES YIELD readCount WHERE readCount IS NOT NULL RETURN sum(readCount);")
    [[ -n "${READS_AFTER}" && "${READS_AFTER}" =~ ^[0-9]+$ && "${READS_AFTER}" -gt "${READS_BEFORE}" ]] && break
    sleep 5
done

info "total index reads: ${READS_BEFORE} before workload, ${READS_AFTER} after"

if [[ "${HINT_TESTED}" -gt 0 ]]; then
    if [[ "${READS_AFTER}" =~ ^[0-9]+$ && "${READS_AFTER}" -gt "${READS_BEFORE}" ]]; then
        ok "indexes served reads during the workload (= ${READS_AFTER})"
    else
        # Not a failure. The index-hint check above already proved every index
        # is usable, and those queries necessarily read them. An unmoved
        # counter here means the statistics had not been flushed yet, which we
        # cannot distinguish from genuine non-use -- so report, don't fail.
        info "readCount did not move within the polling window (statistics flush lag)"
    fi
fi

echo
echo "=============================================================="
echo " Server configuration"
echo "=============================================================="

expect_eq "'${NEO4J_DB}' database is online" \
    "$(q "SHOW DATABASES YIELD name, currentStatus WHERE name = '${NEO4J_DB}' RETURN currentStatus;")" "online"

expect_eq "'${NEO4J_DB}' database is the default" \
    "$(q "SHOW DATABASES YIELD name, default WHERE default = true RETURN name;")" "${NEO4J_DB}"

# Cypher was decoupled from the server in neo4j 2025.06, so this setting does
# not exist on older servers. Absent setting => report it, don't fail: this
# suite is meant to be usable against a pre-upgrade database too.
CYPHER_LANG=$(q "SHOW SETTINGS YIELD name, value WHERE name = 'db.query.default_language' RETURN value;")
if [[ -z "${CYPHER_LANG}" ]]; then
    info "db.query.default_language not present (server predates neo4j 2025.06)"
else
    expect_eq "Cypher default language pinned" "${CYPHER_LANG}" "CYPHER_5"
fi

# dbms.components() can return more than one row; take only the first.
info "neo4j server version: $(q "CALL dbms.components() YIELD name, versions WHERE name = 'Neo4j Kernel' RETURN versions[0];" | head -1)"

echo
echo "=============================================================="
echo " Summary"
echo "=============================================================="
echo "  passed: ${PASSES}"
echo "  failed: ${FAILURES}"
echo

if [[ "${FAILURES}" -gt 0 ]]; then
    echo "RESULT: FAILED"
    exit 1
fi
echo "RESULT: OK"
exit 0
CHECKS

chmod +x "${WORK}/checks.sh"

export APPTAINER_BIND="${NEO4J_DIR}/conf:/conf,${NEO4J_DIR}/data:/data,${NEO4J_DIR}/logs:/logs,${NEO4J_DIR}/plugins:/plugins,${WORK}:/checks"

# Forward the (optional) memory overrides into the container.
for v in HEAP_SIZE MAX_MEMORY; do
    if [[ -n "${!v:-}" ]]; then
        export APPTAINERENV_${v}="${!v}"
    fi
done

# Always forwarded -- the in-container half needs to know which database to
# open, and it has its own matching default if this were ever missing.
export APPTAINERENV_NEO4J_DB="${NEO4J_DB}"

echo "Checking database in: ${NEO4J_DIR}"
echo "Database:             ${NEO4J_DB}"
echo "Using image:          ${NEO4J_IMAGE}"

apptainer exec --writable-tmpfs "${NEO4J_IMAGE}" /checks/checks.sh
