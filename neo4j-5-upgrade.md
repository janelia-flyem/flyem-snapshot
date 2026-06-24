# Neo4j Upgrade: 4.4.16 → 5.26.27

## Background

The `flyem-snapshot` pipeline uses Neo4j as the backing database for neuPrint.
It does **not** run Neo4j directly — instead it:

1. **Phase 1** (`flyem-snapshot --config ...`) — pulls data from DVID and writes
   CSV files under a `neuprint/` subdirectory.
2. **Phase 2** (`ingest-neuprint-snapshot-using-apptainer`) — launches a Neo4j
   Docker image via Apptainer, bulk-imports all CSVs with `neo4j-admin`, then
   creates indexes via `cypher-shell`.

The Neo4j version only affects Phase 2.  The CSV files produced in Phase 1 are
not expected to change.

---

## What Changed Between 4.4.16 and 5.x

### neo4j-admin import command

| 4.4 | 5.x |
|-----|-----|
| `neo4j-admin import [flags] @args-file` | `neo4j-admin database import full <dbname> [flags] @args-file` |
| `--database=<name>` flag | Database name is now a **positional argument** |
| `--max-memory=<size>` | `--max-off-heap-memory=<size>` |

In 5.x the subcommand is `database import full` and the target database name
moves from a `--database` flag to a positional argument immediately after `full`.

### Configuration (`neo4j.conf`)

Nearly all settings were renamed in 5.x.  The top-level namespaces changed:

| Old prefix (4.4) | New prefix (5.x) | Scope |
|---|---|---|
| `dbms.memory.*` | `server.memory.*` | heap, pagecache |
| `dbms.connector.*` | `server.bolt.*` / `server.http.*` / `server.https.*` | network |
| `dbms.directories.*` | `server.directories.*` | plugin/log paths |
| `dbms.logs.*` | `server.logs.*` | log levels |
| `dbms.jvm.additional` | `server.jvm.additional` | JVM flags |
| `dbms.default_database` | `initial.dbms.default_database` | default DB |
| `dbms.db.timezone` | `db.temporal.timezone` | timestamps |
| `dbms.transaction.timeout` | `db.transaction.timeout` | query timeout |
| `dbms.tx_state.memory_allocation` | `db.tx_state.memory_allocation` | tx memory |
| `dbms.tx_log.rotation.*` | `db.tx_log.rotation.*` | WAL retention |

Authentication changed from a single toggle to two separate settings:

| 4.4 | 5.x |
|-----|-----|
| `dbms.security.auth_enabled=false` | `dbms.security.authentication.enabled=false` |
|  | `dbms.security.authorization.enabled=false` |

### Cypher index/constraint syntax

The old pre-4.x "legacy" syntax was fully removed in 5.x:

| 4.4 (legacy syntax, still accepted) | 5.x (only accepted syntax) |
|---|---|
| `CREATE CONSTRAINT ON (n:Label) ASSERT n.prop IS UNIQUE` | `CREATE CONSTRAINT name FOR (n:Label) REQUIRE n.prop IS UNIQUE` |
| `CREATE INDEX ON :Label(prop)` | `CREATE INDEX FOR (n:Label) ON (n.prop)` |

Note: constraints now **require a name**.

### APOC plugin

In 4.x, APOC was maintained separately at `neo4j-contrib/neo4j-apoc-procedures`
and distributed as a single `apoc-X.X.X-all.jar`.

In 5.x, APOC was split into two packages maintained at `neo4j/apoc`:
- **APOC Core** — bundled inside the Neo4j Docker image at
  `$NEO4J_HOME/labs/apoc-*.jar`; copy to `plugins/` to enable.
- **APOC Extended** — separate download for less-common procedures.

We only use APOC for convenience during debugging (not during ingestion itself),
so `apoc-core` is sufficient.

### Other notable 5.x changes

- **Incremental import** (Enterprise only): 5.x adds `database import incremental`
  for updating an existing database without a full rebuild.  We continue to use
  full import (Community-compatible).
- **Cypher version**: The default Cypher version is 5; some deprecated Cypher 4
  constructs may warn or fail.
- **`db.awaitIndexes`**: Still available in 5.x but deprecated; it waits for all
  background index population to complete.  No change needed now, but a future
  upgrade may require polling `SHOW INDEXES` instead.

---

## What Was Changed in This Branch

### `flyem_snapshot/outputs/neuprint/scripts/ingest-neuprint-snapshot-using-apptainer.sh`

- Docker image bumped: `neo4j:4.4.16` → `neo4j:5.26.27`
- APOC download URL updated to the 5.x core jar:
  `neo4j/apoc` `5.26.27-core.jar` (replaces `neo4j-contrib` `4.4.0.7-all.jar`)

### `flyem_snapshot/outputs/neuprint/scripts/ingest-neuprint-snapshot-within-neo4j-container.sh`

- `neo4j-admin import` → `neo4j-admin database import full data`
  (database name `data` is now a positional argument, not a `--database` flag)
- `--database=data` removed from `ingestion-args.txt`
- `--max-memory` → `--max-off-heap-memory`
- Stale version comments updated

### `flyem_snapshot/outputs/neuprint/scripts/inspect-neuprint-snapshot.sh`

- Docker image bumped: `neo4j:4.4.16` → `neo4j:5.26.27`

### `flyem_snapshot/outputs/neuprint/scripts/neo4j.conf`

- Replaced the entire file with a clean 5.x configuration.
- All `dbms.*` settings renamed per the table above.
- `dbms.security.auth_enabled=false` split into the two 5.x equivalents.
- Removed the 4.x commented-out boilerplate (it was misleading in a 5.x context).

### `flyem_snapshot/outputs/neuprint/templates/create-indexes.cypher`

- All `CREATE CONSTRAINT ON … ASSERT … IS UNIQUE` updated to
  `CREATE CONSTRAINT <name> FOR … REQUIRE … IS UNIQUE`
- All `CREATE INDEX ON :Label(prop)` updated to
  `CREATE INDEX FOR (n:Label) ON (n.prop)`
- The `CREATE POINT INDEX` for `Element.location` was already 5.x-compatible
  and was left unchanged.
- Already-commented-out legacy constraints were left as-is (they are not executed).

---

## Using an Existing Snapshot

The CSV data files produced by Phase 1 are compatible with both Neo4j 4.x and
5.x — **they do not need to be regenerated**.

However, the ingest script copies `create-indexes.cypher` directly from the
snapshot directory, not from the repo:

```bash
cp ${SNAPSHOT_DIR}/neuprint/create-indexes.cypher ${WORKSPACE_DIR}/scripts/
```

Any snapshot generated before this upgrade will contain a copy of
`create-indexes.cypher` with the old 4.x Cypher syntax, which Neo4j 5.x will
reject.  You have two options:

### Option 1 — Re-run Phase 1 (clean, recommended if time allows)

```bash
flyem-snapshot --config /path/to/your-snapshot.yaml
```

This regenerates the entire `neuprint/` subdirectory, including a fresh
`create-indexes.cypher` rendered from the updated 5.x template.

### Option 2 — Overwrite just the Cypher file (faster)

Render a fresh copy of `create-indexes.cypher` from the updated template
and drop it into the existing snapshot directory:

```bash
# From within the flyem-snapshot repo, render the file by running a minimal
# flyem-snapshot invocation, or copy the already-rendered file from a newer
# snapshot that was generated with this branch.
cp /path/to/new-snapshot/neuprint/create-indexes.cypher \
   /path/to/existing-snapshot/neuprint/create-indexes.cypher
```

Use this option when re-running Phase 1 is impractical (e.g. the DVID UUID is
no longer accessible, or Phase 1 takes many hours).

---

## How to Test

### Smoke test (local, no cluster needed)

Run a small snapshot through the full Phase 2 pipeline using the `--debug-shell`
flag to enter the container and run the import manually:

```bash
ingest-neuprint-snapshot-using-apptainer <snapshot-dir> --debug-shell
```

Inside the shell, verify the import command runs:

```bash
/var/lib/neo4j/bin/neo4j-admin database import full data --help
```

Then run the actual ingestion and check the logs:

```bash
# exit the shell and let the script run normally
ingest-neuprint-snapshot-using-apptainer <snapshot-dir>
tail -f /scratch/$USER/<snapshot-name>/neo4j/logs/import.out.log
```

### Verify the database starts and indexes are created

After ingestion completes, use `inspect-neuprint-snapshot` to launch neo4j
against the exported database and run a few Cypher queries:

```bash
inspect-neuprint-snapshot <snapshot-dir>/neo4j

# Inside the container:
/var/lib/neo4j/bin/cypher-shell -d data "SHOW INDEXES YIELD name, state, type WHERE state <> 'ONLINE' RETURN name, state, type;"
# Expect zero rows (all indexes online).

/var/lib/neo4j/bin/cypher-shell -d data "MATCH (n:Meta) RETURN n LIMIT 1;"
/var/lib/neo4j/bin/cypher-shell -d data "MATCH (n:Neuron) RETURN count(n);"
```

### Full end-to-end test

Use the wasp v0.8 snapshot (the smallest available release config) as a
realistic but manageable test case:

```bash
# Phase 1 (if CSVs not already available)
flyem-snapshot --config /groups/flyem/data/snapshots/snapshot-configs/wasp/v0.8/wasp-release-snapshot.yaml

# Phase 2
ingest-neuprint-snapshot-using-apptainer 2026-05-12-32c9ac
```

Check that neuprinthttp can connect to the resulting database and serve queries.

### Things to watch for

- **Import failure**: check `/scratch/$USER/.../neo4j/logs/import.err.log` for
  errors. The script already checks for `import failed` in the logs.
- **Config rejection**: Neo4j 5.x will refuse to start if it encounters unknown
  or renamed configuration keys, and will log the offending key to
  `logs/neo4j.log`. If startup fails, check that log first.
- **Index failures**: check `logs/create-indexes.out.log` and
  `logs/create-indexes.err.log`. The script checks for `database is unavailable`
  and an empty output log.
- **APOC jar**: if APOC procedures are needed (e.g. during a debug session),
  verify the jar is present at `$NEO4J_HOME/plugins/apoc-5.26.27-core.jar` inside
  the container.

---

## Useful References

- [Neo4j 5 upgrade/migration guide](https://neo4j.com/docs/upgrade-migration-guide/current/version-5/)
- [neo4j-admin database import full (5.x docs)](https://neo4j.com/docs/operations-manual/current/tools/neo4j-admin/neo4j-admin-import/)
- [Neo4j 5 configuration settings reference](https://neo4j.com/docs/operations-manual/current/reference/configuration-settings/)
- [Neo4j 5 Cypher constraint syntax](https://neo4j.com/docs/cypher-manual/current/constraints/)
- [APOC 5.x releases](https://github.com/neo4j/apoc/releases)
- [Neo4j Docker Hub](https://hub.docker.com/_/neo4j)
