#!/bin/bash
##
## check-neuprint-snapshot.sh <neo4j-export-dir>
##
## Launches a neuprint snapshot's neo4j database in a container and runs a
## series of validation checks against it, then shuts it down.
##
## Exits 0 if every check passed, 1 otherwise, so it is usable from CI or a
## post-ingestion hook.
##
## Usage:
##   pixi run bash check-neuprint-snapshot.sh 2026-05-12-32c9ac/neo4j
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
##                      check-neuprint-snapshot.sh <dir>
##
##   NEO4J_IMAGE  Container image. Default docker://neo4j:2026.07.1. Must be
##                able to open the store you are pointing it at -- neo4j has
##                no downgrade path, so an older image cannot read a newer
##                store.
##
##   CHECK_CSV_COUNTS
##                Set to 0 to skip reconciling label counts against the
##                exported CSV row counts. On by default; it is the strongest
##                check here, but reads every CSV, which is slow on a large
##                dataset over network storage.
##
##   MAX_QUERY_MS Turn the complex-query timing into an assertion instead of
##                an informational line. Milliseconds, because the whole
##                plausible range at snapshot scale sits under one second.
##                Unset by default: a threshold picked without baselines
##                fails for environmental reasons more often than for real
##                regressions.
##
##   QUERY_SEARCH_TERM
##                Search term for the complex query. This, not the bodyId,
##                determines how long that query takes. Defaults to the
##                commonest letter in the dataset's type names.
##
##   QUERY_SEARCH_TERMS
##                Comma-separated terms to time in one run ('a,in,LC'), for
##                finding a term heavy enough to assert on. neo4j is booted
##                once and each term timed in turn.
##
##   QUERY_BODY_ID
##                bodyId for the complex query. Defaults to the lowest one
##                carrying a type; has almost no effect on the timing.
##
##   SHOW_QUERY   Set to 1 to print the generated Cypher. Printed anyway
##                when the query errors or exceeds MAX_QUERY_MS.
##
##   HEAP_SIZE    Override the database's own neo4j.conf memory sizing, which
##   MAX_MEMORY   is otherwise respected as-is. Needed only when checking a
##                cluster-sized snapshot on a smaller machine:
##                  HEAP_SIZE=4G MAX_MEMORY=8G check-neuprint-snapshot.sh <dir>
##
## Version-dependent checks (Cypher default language, default database name)
## downgrade to informational when they do not apply, so the suite stays
## meaningful against an older database.
##

set -euo pipefail

# Print the header block above as help. Keeps one copy of the documentation
# rather than a second one that drifts from it.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    awk 'NR==1 {next} /^##/ {sub(/^## ?/, ""); print; next} {exit}' "$0"
    exit 0
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: check-neuprint-snapshot.sh <neo4j-export-dir>" 1>&2
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

SEG_COUNT=$(q "MATCH (n:\`${DS}_Segment\`) RETURN count(n);")
expect_gt ":${DS}_Segment nodes"    "${SEG_COUNT}" 0
expect_gt ":${DS}_Neuron nodes"     "$(q "MATCH (n:\`${DS}_Neuron\`)     RETURN count(n);")" 0
SYN_COUNT=$(q "MATCH (n:\`${DS}_Synapse\`)    RETURN count(n);")
SS_COUNT=$( q "MATCH (n:\`${DS}_SynapseSet\`) RETURN count(n);")
expect_gt ":${DS}_Synapse nodes"    "${SYN_COUNT}" 0
expect_gt ":${DS}_SynapseSet nodes" "${SS_COUNT}" 0

# :Synapse is a specialization of :Element, and :SynapseSet of :ElementSet --
# the same nesting as :Neuron within :Segment. Every dataset measured so far
# labels every synapse as an element whether or not the config declares element
# tables (wasp, yakuba and fish2 all do), so the raw Element count merely
# restates the Synapse count and asserting it is > 0 proves nothing.
#
# The informative number is the remainder: elements that are not synapses,
# such as somas. That is the population 'non-synaptic-bodies: element-presence'
# selects on in a report config -- fish2 uses it and has 190,774; wasp and
# yakuba have none. Were it to reach zero on a dataset that expects it,
# element-presence would degrade to 'none' and every report's body ranking
# would change with nothing failing here.
#
# Reported rather than asserted, since zero is correct for most datasets.
# Note these two are label scans, not count-store lookups, so they cost more
# than the counts beside them.
ELM_COUNT=$(q "MATCH (n:\`${DS}_Element\`)    RETURN count(n);")
ELMSET_COUNT=$(q "MATCH (n:\`${DS}_ElementSet\`) RETURN count(n);")
NONSYN_ELM=$(q "MATCH (n:\`${DS}_Element\`)    WHERE NOT n:\`${DS}_Synapse\`    RETURN count(n);")
NONSYN_ELMSET=$(q "MATCH (n:\`${DS}_ElementSet\`) WHERE NOT n:\`${DS}_SynapseSet\` RETURN count(n);")

info ":${DS}_Element nodes: ${ELM_COUNT} (${NONSYN_ELM} non-synaptic)"
info ":${DS}_ElementSet nodes: ${ELMSET_COUNT} (${NONSYN_ELMSET} non-synaptic)"

# The nesting itself, asserted for free from the numbers already collected.
# |Element AND Synapse| = Element - non-synaptic. If every synapse carries
# :Element then that equals the Synapse count, i.e. Element - Synapse ==
# non-synaptic. A synapse missing its :Element label breaks the identity.
expect_eq "every :${DS}_Synapse is also an :${DS}_Element" \
    "$(( ELM_COUNT - SYN_COUNT ))" "${NONSYN_ELM}"
expect_eq "every :${DS}_SynapseSet is also an :${DS}_ElementSet" \
    "$(( ELMSET_COUNT - SS_COUNT ))" "${NONSYN_ELMSET}"

# Every node should carry one of the labels we know about. Nodes with several
# labels (a :Neuron is also a :Segment) are counted once by the single MATCH,
# so this catches stray or mislabelled nodes without double-counting.
TOTAL_NODES=$(q "MATCH (n) RETURN count(n);")
ACCOUNTED=$(q "MATCH (n)
               WHERE n:\`${DS}_Segment\` OR n:\`${DS}_Synapse\` OR n:\`${DS}_SynapseSet\`
                  OR n:Meta OR n:\`${DS}_Element\` OR n:\`${DS}_ElementSet\`
               RETURN count(n);")
info "total nodes: ${TOTAL_NODES}"
expect_eq "every node accounted for by a known label" "${ACCOUNTED}" "${TOTAL_NODES}"

expect_gt "ConnectsTo relationships" "$(q "MATCH ()-[r:ConnectsTo]->() RETURN count(r);")" 0
expect_gt "SynapsesTo relationships" "$(q "MATCH ()-[r:SynapsesTo]->() RETURN count(r);")" 0
expect_gt "Contains relationships"   "$(q "MATCH ()-[r:Contains]->()   RETURN count(r);")" 0

expect_eq "every :Neuron is also a :Segment" \
    "$(q "MATCH (n:\`${DS}_Neuron\`) WHERE NOT n:\`${DS}_Segment\` RETURN count(n);")" "0"

echo
echo "=============================================================="
echo " Reconciliation against the import"
echo "=============================================================="

# The count checks above only assert "> 0", which would pass a truncated
# import as long as it was internally consistent. Reconcile against what the
# importer itself reported instead. import.out.log is persisted next to the
# database, so this costs nothing.
#
#   Imported:
#     5278783 nodes
#     10702773 relationships
#     26561538 properties
IMPORT_LOG=/logs/import.out.log
if [[ ! -r "${IMPORT_LOG}" ]]; then
    skip "no import.out.log next to the database -- cannot reconcile against the import"
else
    IMP_NODES=$(grep -A3 '^Imported:' "${IMPORT_LOG}" | grep -oE '[0-9]+ nodes'         | tail -1 | grep -oE '[0-9]+')
    IMP_RELS=$( grep -A3 '^Imported:' "${IMPORT_LOG}" | grep -oE '[0-9]+ relationships' | tail -1 | grep -oE '[0-9]+')

    if [[ -z "${IMP_NODES}" || -z "${IMP_RELS}" ]]; then
        skip "could not parse counts out of ${IMPORT_LOG}"
    else
        expect_eq "node count matches the import report" "${TOTAL_NODES}" "${IMP_NODES}"
        TOTAL_RELS=$(q "MATCH ()-[r]->() RETURN count(r);")
        info "total relationships: ${TOTAL_RELS}"
        expect_eq "relationship count matches the import report" "${TOTAL_RELS}" "${IMP_RELS}"
    fi

    # A tolerated bad record is silent data loss: --bad-tolerance permits some
    # number of malformed rows to be skipped and merely logged. Match on the
    # phrase rather than trying to parse a count out of it -- neo4j writes
    # "There were bad entries which were skipped and logged into <file>",
    # so a pattern expecting a leading number silently reports success.
    if grep -qi 'bad entries' "${IMPORT_LOG}"; then
        bad "the import skipped bad entries -- rows were silently dropped"
        grep -i 'bad entries' "${IMPORT_LOG}" | head -3 | sed 's/^/          /'
    else
        ok "the import reported no bad entries"
    fi
fi

# Opt-in: compare label counts against the exported CSV row counts, forwarded
# in by the host when CHECK_CSV_COUNTS is set. Stronger than the import log,
# because it notices rows the importer dropped as bad entries.
if [[ -n "${CSV_SEGMENTS:-}" ]]; then
    expect_eq "Segment count matches Neuprint_Neurons CSV rows" \
        "${SEG_COUNT}" "${CSV_SEGMENTS}"
    expect_eq "Synapse count matches Neuprint_Synapses CSV rows" \
        "${SYN_COUNT}" "${CSV_SYNAPSES}"
    expect_eq "SynapseSet count matches Neuprint_SynapseSet.csv rows" \
        "${SS_COUNT}" "${CSV_SYNAPSESETS}"
fi

echo
echo "=============================================================="
echo " Integrity"
echo "=============================================================="

expect_eq "no duplicate Segment bodyIds" \
    "$(q "MATCH (n:\`${DS}_Segment\`) WITH n.bodyId AS b, count(*) AS c WHERE c > 1 RETURN count(b);")" "0"

expect_eq "no Segment with a null bodyId" \
    "$(q "MATCH (n:\`${DS}_Segment\`) WHERE n.bodyId IS NULL RETURN count(n);")" "0"

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
SEG_PROPS=$(q "MATCH (n:\`${DS}_Segment\`) UNWIND keys(n) AS k RETURN DISTINCT k;" | sort -u)

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

# roiInfo is read with apoc.convert.fromJsonMap, so APOC is load-bearing for
# this section and for the index-usability section below. The plugin is loaded
# from the snapshot's own plugins/ dir and its jar is version-coupled to the
# server, so a snapshot built for one neo4j release can silently fail to load
# it under another.
#
# Check it directly, because the failure was invisible. q() sends stderr to
# /dev/null, so an APOC error is indistinguishable from an empty result: the
# roiInfo read below returned nothing, took a skip, and skip does not count as
# a failure. Both the ROI checks and the whole index-usability section would
# disappear while the run still reported OK -- and those are the regression
# test for the ROI naming bug this branch exists to fix.
#
# Probe the function this script actually calls rather than only counting
# what is installed: a partial or mismatched plugin can advertise names it
# cannot execute.
APOC_FUNCS=$(q "SHOW FUNCTIONS YIELD name WHERE name STARTS WITH 'apoc.' RETURN count(*);")
APOC_OK=0
if [[ "$(q "RETURN apoc.convert.fromJsonMap('{}') IS NOT NULL;")" == "true" ]]; then
    APOC_OK=1
    ok "APOC usable (${APOC_FUNCS:-?} apoc functions)"
else
    bad "APOC is not usable -- apoc.convert.fromJsonMap did not execute"
    info "the apoc jar in the snapshot's plugins/ dir must match the server version"
fi

# The Meta node's roiInfo is the contract clients rely on. Every ROI it
# advertises should exist as a property on at least one Segment. This is the
# regression test for the sanitize_roi_name mismatch.
META_ROIS=$(q "MATCH (m:\`${DS}_Meta\`)
               UNWIND keys(apoc.convert.fromJsonMap(m.roiInfo)) AS r
               RETURN DISTINCT r;" | sort -u | sed '/^$/d')

if [[ -z "${META_ROIS}" ]]; then
    if [[ "${APOC_OK}" -eq 0 ]]; then
        # Already reported above; do not fail twice for one cause.
        skip "Meta.roiInfo unreadable because APOC is unusable -- ROI and index-usability checks cannot run"
    else
        bad "Meta.roiInfo is empty or unreadable even though APOC is usable"
    fi
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
        TRICKY_N=$(q "MATCH (n:\`${DS}_Segment\`) WHERE n.\`${TRICKY}\` = true RETURN count(n);")
        PLAN=$(${CS} -d "${NEO4J_DB}" "EXPLAIN MATCH (n:\`${DS}_Segment\`) WHERE n.\`${TRICKY}\` = true RETURN count(n);" 2>&1)
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
        printf 'MATCH (n:`%s_Segment`) USING INDEX n:`%s_Segment`(`%s`) WHERE n.`%s` = true RETURN count(n);\n' \
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
                    "MATCH (n:\`${DS}_Segment\`)
                     USING INDEX n:\`${DS}_Segment\`(\`${roi}\`)
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
echo " Complex query"
echo "=============================================================="

# neuPrintExplorer's search query: a full :Neuron scan with eleven
# toLower(...) CONTAINS predicates, so no index can serve it, followed by an
# unbounded ORDER BY over everything matched. It is the worst case a user can
# trigger from the search box, which makes it the right thing to time.
#
# Two purposes:
#  1. Smoke test -- it exercises the annotation properties the frontend expects
#     (type, instance, hemibrainType, class, entryNerve, ...). Properties a
#     given dataset lacks come back null, and 'null CONTAINS q' is null rather
#     than an error, so this degrades to fewer matches instead of failing. An
#     actual error means a real schema problem.
#  2. Timing -- reported by default. Set MAX_QUERY_MS to turn it into an
#     assertion once you have baselines; a threshold picked without them fails
#     for environmental reasons more often than for real regressions.
#
# Be clear about what the timing measures. On wasp (50,564 neurons) the range
# across search terms was 251 ms for a term matching 702 rows to 635 ms for
# one matching 44,933 -- only 2.5x for 64x the rows. The floor is the label
# scan itself, and the term adds comparatively little. So this tracks scan and
# sort throughput far more than it tracks search selectivity, and no term will
# make it slow on a dataset this size.
#
# Note this deliberately uses the unprefixed :Neuron label, exactly as the
# frontend does, rather than ${DS}_Neuron. Each snapshot database holds a
# single dataset, so the two are equivalent here.

# Timing is measured as wall clock, minus the cost of a trivial query.
#
# cypher-shell prints its own figures ("ready to start consuming query after
# 548 ms, ...") but ONLY when stdout is a terminal; captured through $(...)
# those lines are absent, so they cannot be parsed from a script. And a raw
# wall-clock measurement is dominated by ~1-2s of JVM startup per invocation.
#
# So: time a trivial query to establish the per-invocation overhead, and
# subtract it. What remains is attributable to the query itself.
now_ms() {
    local t
    t=$(date +%s%3N 2>/dev/null)
    if [[ "${t}" =~ ^[0-9]+$ ]]; then
        echo "${t}"
    else
        # date without %N support (not expected on the debian-based image)
        echo $(( $(date +%s) * 1000 ))
    fi
}

# Both inputs can be pinned from the environment for a reproducible
# measurement. Otherwise they are derived deterministically from the data --
# no rand() -- so two runs against the same snapshot are comparable.
#
# The bodyId is nearly free: it contributes one equality test on an indexed
# property plus a branch in the priority CASE. The SEARCH TERM is what costs.
# Every neuron is scanned and tested against eleven toLower(...) CONTAINS
# predicates, and every row that matched is then sorted, so the runtime
# tracks how large a fraction of the label the term matches.
#
# The default is therefore the most expensive realistic term available: the
# two-character type prefix shared by the largest number of neurons. Set
# QUERY_SEARCH_TERM shorter or more common to make it heavier, longer to make
# it lighter. Anything outside [A-Za-z0-9] is stripped, since the term is
# interpolated into the query text rather than passed as a parameter.
# QUERY_SEARCH_TERMS (plural, comma-separated) times several terms in one
# container start, which is the cheap way to hunt for a term heavy enough to
# be worth asserting on. Booting the container and neo4j per term instead
# costs minutes each and swamps what you are trying to measure.
if [[ -n "${QUERY_SEARCH_TERMS:-}" ]]; then
    IFS=',' read -ra RAW_TERMS <<< "${QUERY_SEARCH_TERMS}"
    TERM_SRC="from QUERY_SEARCH_TERMS"
elif [[ -n "${QUERY_SEARCH_TERM:-}" ]]; then
    RAW_TERMS=("${QUERY_SEARCH_TERM}")
    TERM_SRC="from QUERY_SEARCH_TERM"
else
    # The single letter appearing in the most type names. Measured on wasp, a
    # two-character prefix matched ~1% of the label while the commonest single
    # letter matched 89%, so this is materially closer to the worst case a
    # user can trigger -- and typing one letter in the search box is exactly
    # what a user does.
    RAW_TERMS=("$(q "MATCH (n:\`${DS}_Neuron\`) WHERE n.type IS NOT NULL
                     UNWIND range(0, size(n.type) - 1) AS i
                     WITH n, toLower(substring(n.type, i, 1)) AS ch
                     WHERE ch >= 'a' AND ch <= 'z'
                     WITH ch, count(DISTINCT n) AS c
                     RETURN ch ORDER BY c DESC, ch LIMIT 1;" | head -1)")
    TERM_SRC="commonest letter in type names"
fi

# Anything outside [A-Za-z0-9] is dropped, since terms are interpolated into
# the query text rather than passed as parameters.
TERMS=()
for _t in "${RAW_TERMS[@]}"; do
    _t=$(printf '%s' "${_t}" | tr -cd 'A-Za-z0-9')
    [[ -n "${_t}" ]] && TERMS+=("${_t}")
done

if [[ -n "${QUERY_BODY_ID:-}" ]]; then
    SAMPLE_BODY=$(printf '%s' "${QUERY_BODY_ID}" | tr -cd '0-9')
    BODY_SRC="from QUERY_BODY_ID"
    if [[ "${SAMPLE_BODY}" != "${QUERY_BODY_ID}" ]]; then
        bad "QUERY_BODY_ID='${QUERY_BODY_ID}' is not a plain integer"
        SAMPLE_BODY=""
    fi
else
    SAMPLE_BODY=$(q "MATCH (n:\`${DS}_Neuron\`) WHERE n.type IS NOT NULL
                     RETURN min(n.bodyId);" | head -1 | tr -cd '0-9')
    BODY_SRC="lowest bodyId carrying a type"
fi

# Label count store lookup, so this is cheap. Reported alongside the row
# count to show what fraction of the label the term actually matched -- the
# number to look at when a timing comes back suspiciously fast.
NEURON_TOTAL=$(q "MATCH (n:\`${DS}_Neuron\`) RETURN count(n);" | head -1)

if [[ -z "${SAMPLE_BODY}" || "${#TERMS[@]}" -eq 0 ]]; then
    skip "no usable bodyId/search term -- cannot build the query"
else
    info "bodyId ${SAMPLE_BODY} (${BODY_SRC}), ${#TERMS[@]} search term(s) ${TERM_SRC}"

    # Built per term, so a sweep can vary the term without re-booting.
    build_query() {
        local q="$1"
        cat <<QRY
WITH toLower('${q}') as q, ${SAMPLE_BODY} as user_body, '(' + toLower('${q}') as parenQ
MATCH (n:Neuron)
WHERE n.bodyId = user_body
   OR any(prop IN [
       n.type, n.instance, n.hemibrainType, n.flywireType,
       n.systematicType, n.itoleeHl, n.trumanHl, n.synonyms,
       n.class, n.entryNerve, n.exitNerve
   ] WHERE toLower(prop) CONTAINS q)
WITH n, q, parenQ, user_body,
     [toLower(n.type), toLower(n.instance), toLower(n.hemibrainType),
      toLower(n.flywireType), toLower(n.systematicType), toLower(n.itoleeHl),
      toLower(n.trumanHl), toLower(n.synonyms), toLower(n.class),
      toLower(n.entryNerve), toLower(n.exitNerve)] as props
WITH n, q, parenQ, props, user_body,
     CASE
         WHEN n.bodyId = user_body AND user_body <> 0 THEN 0
         WHEN any(p IN props WHERE p = q) THEN 1
         WHEN any(p IN props WHERE p STARTS WITH q) THEN 2
         WHEN any(p IN props WHERE p STARTS WITH parenQ) THEN 3
         WHEN any(p IN props WHERE p CONTAINS q) THEN 4
         ELSE 5
     END as priority,
     CASE
         WHEN toLower(n.type) STARTS WITH q THEN 0
         WHEN toLower(n.type) CONTAINS q THEN 1
         ELSE 2
     END as type_priority
RETURN
    toString(n.bodyId) as bodyId, n.type as type, n.instance as instance,
    n.hemibrainType as hemibrainType, n.flywireType as flywireType,
    n.systematicType as systematicType, n.itoleeHl as itoLeeHl,
    n.trumanHl as trumanHl, n.synonyms as synonyms, n.class as class,
    n.entryNerve as entryNerve, n.exitNerve as exitNerve,
    priority, type_priority
ORDER BY priority, type_priority, n.type, n.instance
QRY
    }

    # Per-invocation overhead: JVM startup plus connect, measured once with a
    # query that does no work, then subtracted from every timing below. A
    # single cypher-shell invocation spends 1-2s here, which would otherwise
    # swamp the query itself.
    T0=$(now_ms); ${CS} -d "${NEO4J_DB}" --format plain "RETURN 1;" > /dev/null 2>&1; T1=$(now_ms)
    OVERHEAD_MS=$(( T1 - T0 ))
    info "per-invocation overhead (JVM startup + connect): ${OVERHEAD_MS} ms"

    QUERY_FAILED=0
    SLOWEST_MS=0
    SLOWEST_TERM=""

    # The query runs to ~35 lines, and this suite is otherwise one line per
    # check, so it is not printed on every run. It is printed when something
    # went wrong -- which is when you want to paste it into cypher-shell --
    # or on request via SHOW_QUERY=1. Printed at most once, since a sweep
    # would otherwise repeat it per term.
    SHOWN=0
    show_query() {
        echo "          ----- Cypher, term '$1' -----"
        build_query "$1" | sed 's/^/          /'
        echo "          ----- end -----"
    }

    for TERM in "${TERMS[@]}"; do
        SLOW_QUERY=$(build_query "${TERM}")

        # The first run doubles as the smoke test -- its exit status says
        # whether the query executes, so there is no need to run it again just
        # to find out. It also pays the page-cache misses; the second run is
        # the number worth comparing between runs. Neither holds the result
        # set in a variable, since a short term can match a large fraction of
        # the label. Here stdout is discarded and only stderr kept, which is
        # where an error would appear ('2>&1 >/dev/null' -- order matters:
        # stderr to the pipe, then stdout away).
        T0=$(now_ms)
        COLD_OUT=$(${CS} -d "${NEO4J_DB}" --format plain "${SLOW_QUERY}" 2>&1 >/dev/null)
        COLD_RC=$?
        T1=$(now_ms)
        COLD_MS=$(( T1 - T0 - OVERHEAD_MS )); (( COLD_MS < 0 )) && COLD_MS=0

        if [[ "${COLD_RC}" -ne 0 ]]; then
            bad "search query failed to execute for term '${TERM}'"
            grep -viE '^[[:space:]]*$' <<<"${COLD_OUT}" | tail -5 | sed 's/^/          /'
            if [[ "${SHOWN}" -eq 0 ]]; then show_query "${TERM}"; SHOWN=1; fi
            QUERY_FAILED=1
            continue
        fi

        # Rows counted as they stream past rather than buffered. The timing
        # still covers fetching the whole result set, which is the cost a user
        # actually waits on. --format plain emits a header plus one row each.
        T0=$(now_ms)
        ROWS=$(${CS} -d "${NEO4J_DB}" --format plain "${SLOW_QUERY}" 2>/dev/null | grep -c .)
        T1=$(now_ms)
        WARM_MS=$(( T1 - T0 - OVERHEAD_MS )); (( WARM_MS < 0 )) && WARM_MS=0
        (( ROWS > 0 )) && ROWS=$(( ROWS - 1 ))

        printf "  ....  term %-10s rows %8s / %-10s cold %7s ms   warm %7s ms\n" \
            "'${TERM}'" "${ROWS}" "${NEURON_TOTAL}" "${COLD_MS}" "${WARM_MS}"

        # First successful term always takes the slot, so the slowest term is
        # reported even when every timing clamps to zero.
        if [[ -z "${SLOWEST_TERM}" ]] || (( WARM_MS > SLOWEST_MS )); then
            SLOWEST_MS="${WARM_MS}"
            SLOWEST_TERM="${TERM}"
        fi
    done

    if [[ "${QUERY_FAILED}" -eq 0 ]]; then
        ok "the neuPrintExplorer search query executes (${#TERMS[@]} term(s))"
    fi

    if [[ -z "${SLOWEST_TERM}" ]]; then
        : # every term failed; already reported above
    elif [[ -n "${MAX_QUERY_MS:-}" ]]; then
        # Milliseconds, not seconds: the whole plausible range on a
        # snapshot-sized dataset sits under one second, so an integer-second
        # threshold cannot express anything useful.
        if [[ "${SLOWEST_MS}" -le "${MAX_QUERY_MS}" ]]; then
            ok "slowest term '${SLOWEST_TERM}' within ${MAX_QUERY_MS} ms (${SLOWEST_MS} ms)"
        else
            bad "term '${SLOWEST_TERM}' took ${SLOWEST_MS} ms, over the ${MAX_QUERY_MS} ms threshold"
            if [[ "${SHOWN}" -eq 0 ]]; then show_query "${SLOWEST_TERM}"; SHOWN=1; fi
        fi
    else
        info "slowest: ${SLOWEST_MS} ms (term '${SLOWEST_TERM}'); set MAX_QUERY_MS to assert"
    fi

    if [[ -n "${SHOW_QUERY:-}" && "${SHOWN}" -eq 0 && -n "${SLOWEST_TERM}" ]]; then
        show_query "${SLOWEST_TERM}"
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

# The verdict is written to the bind-mounted work dir as well as returned as
# an exit status. apptainer's own exit code can be perturbed by teardown
# problems that have nothing to do with the checks (fuse-overlayfs cleanup
# failures are common on cluster nodes), and this is used as a pipeline gate,
# so the host reads the verdict from here rather than trusting that code.
if [[ "${FAILURES}" -gt 0 ]]; then
    echo 1 > /checks/verdict
    echo "RESULT: FAILED"
    exit 1
fi
echo 0 > /checks/verdict
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

# Derive expected counts from the exported CSVs. This is the strongest check
# in the suite: it catches rows the importer skipped as bad entries, which the
# import-log reconciliation cannot see. neo4j-admin creates exactly one node
# per row of a --nodes file, so a node file's row count is the expected label
# count (verified exactly against both wasp and yakuba).
#
# On by default. It reads every exported CSV, which is tens of millions of
# lines on a large dataset, so opt out when you want a quick check:
#
#   CHECK_CSV_COUNTS=0 check-neuprint-snapshot <dir>
#
# Skipped automatically, with a notice, if the snapshot's neuprint/ directory
# is not alongside the neo4j/ directory -- e.g. a database deployed on its own.
CHECK_CSV_COUNTS=${CHECK_CSV_COUNTS:-1}
if [[ "${CHECK_CSV_COUNTS}" != "0" ]]; then
    CSV_DIR="$(dirname "${NEO4J_DIR}")/neuprint"
    if [[ ! -d "${CSV_DIR}" ]]; then
        echo "No CSVs at ${CSV_DIR} -- skipping CSV reconciliation" 1>&2
    else
        # Sum (lines - 1) per file to discount each file's header row.
        csv_rows() {
            local total=0 f l
            for f in "$@"; do
                [[ -f "${f}" ]] || continue
                l=$(wc -l < "${f}")
                total=$(( total + l - 1 ))
            done
            echo "${total}"
        }
        echo "Counting CSV rows (reads every exported CSV; CHECK_CSV_COUNTS=0 to skip)..."
        export APPTAINERENV_CSV_SEGMENTS=$(csv_rows "${CSV_DIR}"/Neuprint_Neurons/*.csv)
        export APPTAINERENV_CSV_SYNAPSES=$(csv_rows "${CSV_DIR}"/Neuprint_Synapses/*.csv)
        export APPTAINERENV_CSV_SYNAPSESETS=$(csv_rows "${CSV_DIR}"/Neuprint_SynapseSet.csv)
        echo "  segments=${APPTAINERENV_CSV_SEGMENTS} synapses=${APPTAINERENV_CSV_SYNAPSES} synapsesets=${APPTAINERENV_CSV_SYNAPSESETS}"
    fi
fi

echo "Checking database in: ${NEO4J_DIR}"
echo "Database:             ${NEO4J_DB}"
echo "Using image:          ${NEO4J_IMAGE}"

apptainer exec --writable-tmpfs "${NEO4J_IMAGE}" /checks/checks.sh
APPTAINER_RC=$?

# Prefer the verdict the checks themselves recorded. A missing file means the
# container never got as far as the summary (image pull failed, neo4j never
# started, OOM kill), which is a failure regardless of what apptainer returned.
if [[ -f "${WORK}/verdict" ]]; then
    VERDICT=$(tr -cd '0-9' < "${WORK}/verdict")
    if [[ "${VERDICT}" != "${APPTAINER_RC}" ]]; then
        echo "NOTE: checks reported ${VERDICT}, apptainer exited ${APPTAINER_RC}" \
             "(container teardown noise); using the checks' own verdict."
    fi
    exit "${VERDICT:-1}"
fi

echo "ERROR: the checks did not run to completion (no verdict recorded)."
exit 1
