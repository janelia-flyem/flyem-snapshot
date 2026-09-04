# Neo4j Upgrade: 4.4.16 → 2026.07.1

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

## What Additionally Changed for the CalVer Target

The original target for this branch was `5.26.27`; it was retargeted to
`2026.06.0`, and then to **`2026.07.1`** once that shipped.  Neo4j switched from
SemVer to calendar versioning (`YYYY.MM.Patch`) with the `2025.01` release, which
is the first release after the `5.26` LTS checkpoint — so all of the 5.x-era
changes above still apply, plus the following.

### Support lifecycle (important)

CalVer releases are **not** LTS.  Each monthly release is supported only until
the next one ships, whereas `5.26 LTS` remains supported until **June 2028**.
Targeting CalVer therefore implies a recurring upgrade cadence — this branch has
already moved once for exactly that reason, `2026.06.0` having been superseded
within weeks.

Bumping between monthly releases is cheap by design: breaking changes land only
in the release immediately following an LTS (`2025.01`), so a bump is normally a
handful of version strings plus a re-validation run with
`check-neuprint-snapshot`.

### Avoid `2026.07.0`

`2026.07.0` introduced a UTF-8 encoding bug in the **block format** that affected
the Cypher `trim()` function, causing query failures and, in some cases,
unreadable stored string data.  It was reverted in `2026.07.1`, which is why that
patch release exists.  Doubly moot here — block format is Enterprise-only and we
run Community, whose databases report store format `standard` — but do not pin
`2026.07.0`.

### Java 21 required

Neo4j `2025.01`+ requires **Java 21** (Java 17 is no longer supported); Java 25 is
also supported from `2025.10` onward.  This is handled automatically for us
because we run the official `neo4j` Docker image, which bundles its own JDK —
but it matters for any non-container deployment.

Consequence for our config: `-XX:-UseBiasedLocking` was **removed** from
`neo4j.conf`.  Biased locking no longer exists in Java 21, so the flag is at
best ignored with an "obsolete option" warning and at worst rejected as an
unrecognized VM option, which would stop the JVM from starting.

### Removed configuration settings

- **`db.tx_state.memory_allocation`** — removed without replacement in
  `2025.01`.  Deleted from `neo4j.conf`.
- **`server.memory.off_heap.*`** — also removed in `2025.01`.  We never set
  these in `neo4j.conf`.  Note this is unrelated to the
  `--max-off-heap-memory` flag on `neo4j-admin database import`, which is
  still supported.

Note that we set `server.config.strict_validation.enabled=false`, which means
removed or misspelled settings are **silently ignored** rather than causing a
startup failure.  That makes stale config easy to miss — worth auditing rather
than relying on the server to complain.

### Cypher language versioning

As of `2025.06` the Cypher language is versioned independently of the server, and
**Cypher 25** exists alongside Cypher 5.  From `2026.02` the *distributed*
`neo4j.conf` explicitly sets `db.query.default_language=CYPHER_25`.

Because we replace `neo4j.conf` wholesale, we would otherwise silently inherit
the built-in default (`CYPHER_5`).  We now pin `db.query.default_language=CYPHER_5`
explicitly, so the language version can't shift underneath us on a future server
bump.  `create-indexes.cypher` is Cypher 5 syntax.

### `neo4j-admin database import` default change

`2025.12` changed the default `--bad-tolerance` from `1000` to `-1` (unlimited).
That means a malformed CSV row would be **skipped and logged instead of failing
the import** — silent row loss in a connectome export.  The script now passes
`--bad-tolerance` explicitly (default `1000`, matching pre-CalVer behaviour,
overridable via the `BAD_TOLERANCE` environment variable; set `0` to fail on the
first bad record).

All other import options we rely on (`--overwrite-destination`,
`--normalize-types`, `--high-parallel-io`, `--max-off-heap-memory`, `--threads`,
`--multiline-fields`) are unchanged.

### No store-format migration

Irrelevant for this pipeline in any case: every run does a full fresh import
from CSV into a brand-new database, so there is no existing store to migrate.

### Downstream components

The original component review concluded that neuprint-python, neuPrintExplorer
and neuPrintHTTP needed no code changes. That was assessed against a `5.26`
server and **does not fully hold for a CalVer server** — neuPrintHTTP needs a
Bolt driver bump. See *Related component findings* under Current Status below
for the details and the deployment sequencing this implies.

---

## What Was Changed in This Branch

### `flyem_snapshot/outputs/neuprint/scripts/ingest-neuprint-snapshot-using-apptainer.sh`

- Docker image bumped: `neo4j:4.4.16` → `neo4j:2026.07.1`
- APOC download URL updated to the CalVer core jar:
  `neo4j/apoc` `2026.07.1-core.jar` (replaces `neo4j-contrib` `4.4.0.7-all.jar`)

### `flyem_snapshot/outputs/neuprint/scripts/ingest-neuprint-snapshot-within-neo4j-container.sh`

- `neo4j-admin import` → `neo4j-admin database import full data`
  (database name `data` is now a positional argument, not a `--database` flag)
- `--database=data` removed from `ingestion-args.txt`
- `--max-memory` → `--max-off-heap-memory`
- **All CSV paths are now absolute**, anchored on `SNAPSHOT_DIR=/snapshot`.
  Seven arguments were bare filenames, and `neo4j-admin` rejects a path with no
  directory component: `Neuprint_Meta.csv: Unable to find the parent of the
  path`. Arguments generated by `find` were unaffected because they carry a
  directory component; the `find` calls were switched to absolute paths too, for
  consistency.
- **`python` replaced with `nproc`** for the `--threads` count. The `neo4j:5+`
  images do not ship Python, so the old
  `python -c 'multiprocessing.cpu_count()//2'` died with
  `python: command not found`. Clamped to a minimum of 1, since `--threads=0` is
  invalid. `LSB_MAX_NUM_PROCESSORS` still takes precedence inside an LSF job.
- **`--bad-tolerance` is now passed explicitly** (see the CalVer section above).
- **`HEAP_SIZE` / `MAX_MEMORY` are overridable**, and are written into *both*
  copies of `neo4j.conf` — the in-container one the server reads, and the
  bind-mounted one that is persisted next to the database. Writing only the
  former left the shipped conf claiming 31G/150G, which then broke
  `inspect-neuprint-snapshot` on any machine smaller than a cluster node.
  Defaults remain 31G/150G, so cluster behaviour is unchanged.

### `flyem_snapshot/outputs/neuprint/scripts/inspect-neuprint-snapshot.sh` and `_launch_snapshot_and_bash_shell.sh`

- Docker image bumped: `neo4j:4.4.16` → `neo4j:2026.07.1`
- `HEAP_SIZE` / `MAX_MEMORY` forwarded into the container as `APPTAINERENV_*`
- **The inspect path has no memory defaults.** The snapshot's own `neo4j.conf`
  is authoritative — the ingestion records the sizing it actually used — so an
  unset variable means "respect the conf" rather than "force the cluster
  defaults". Defaulting to 31G/150G here would overwrite a snapshot correctly
  recorded as 4G/8G, and the JVM would then fail to reserve the heap under
  `-XX:+AlwaysPreTouch`, dying with an unhelpful
  `Unexpected Neo4j server failure`.

  Note the asymmetry with the ingest path, which *does* keep its defaults: it
  needs a concrete value to hand `neo4j-admin`, and it is the step that
  establishes the sizing in the first place.

### `flyem_snapshot/outputs/neuprint/scripts/neo4j.conf`

- Replaced the entire file with a clean 5.x/CalVer configuration.
- All `dbms.*` settings renamed per the table above.
- Removed the 4.x commented-out boilerplate (it was misleading in a 5.x context).
- **Authentication is disabled with `dbms.security.auth_enabled=false`.**
  An earlier revision of this file used `dbms.security.authentication.enabled`
  and `dbms.security.authorization.enabled`, **neither of which is a real Neo4j
  setting**. Because `server.config.strict_validation.enabled=false` is set,
  they were silently ignored — logged only as `Unrecognized setting` warnings —
  so authentication stayed *enabled* and the index-creation step hung forever on
  a `cypher-shell` `username:` prompt.

  This is the cautionary example for `strict_validation`: it turns a typo into a
  warning nobody reads. Audit this file by hand rather than trusting the server
  to complain.

### `flyem_snapshot/outputs/neuprint/templates/create-indexes.cypher`

- All `CREATE CONSTRAINT ON … ASSERT … IS UNIQUE` updated to
  `CREATE CONSTRAINT <name> FOR … REQUIRE … IS UNIQUE`
- All `CREATE INDEX ON :Label(prop)` updated to
  `CREATE INDEX FOR (n:Label) ON (n.prop)`
- The `CREATE POINT INDEX` for `Element.location` was already 5.x-compatible
  and was left unchanged.
- Already-commented-out legacy constraints were left as-is (they are not executed).

---

## ROI property names are NOT sanitized

An earlier commit on this branch (`2c88ad9`) introduced `sanitize_roi_name()`,
which rewrote ROI names when building CSV headers — `BU(R)` became `BU_R_` —
on the premise that Neo4j rejects parenthesised property names in CSV headers
and collapses similar names into duplicates.

**That premise is wrong, and the rewrite caused a real bug.** It was reverted in
`76fbadf` and replaced with `check_roi_name()`, which validates and returns the
name unchanged.

Why it was wrong — a minimal import against `2026.06.0`:

```
header: bodyId:ID,BU(R):boolean,VLNP(-AOTU)(R):boolean,VLNP(R):boolean
stored: ["BU(R)", "VLNP(-AOTU)(R)", "VLNP(R)", "bodyId"]
```

`neo4j-admin` accepts the names and stores them verbatim, and `VLNP(-AOTU)(R)`
and `VLNP(R)` remain **distinct** — the exact collision the sanitizer claimed to
prevent.

Why it mattered — only the CSV headers were rewritten. The Meta node's
`roiInfo`, the indexes emitted by `create-indexes.cypher`, and therefore every
client query all refer to ROIs by their real names, so the node properties
became the odd one out:

```
MATCH (n:wasp3_Segment)
RETURN count(n.`BU(R)`) AS unsanitized, count(n.BU_R_) AS sanitized;
-> unsanitized: 0, sanitized: 388        (before the fix)
-> unsanitized: 388, sanitized: 0        (after)
```

Roughly 194 of the 232 indexes referenced properties that did not exist, so they
were inert and any ROI filter matched nothing.

`check_roi_name()` still rejects `,` `:` `"` CR and LF — the characters that
genuinely break the `name:type` header format — and raises rather than silently
rewriting, so a new ROI name from DVID cannot quietly corrupt the export.

> **Checking ROI headers correctly:** the boolean ROI columns are *sparse* —
> each `Neuprint_Neurons/*.csv` batch carries only the ROIs its bodies touch.
> Sampling one file proves nothing (`0000.csv` happens to contain only `AGNG`
> and `GNG`, neither of which has special characters). Always aggregate:
>
> ```bash
> head -qn1 Neuprint_Neurons/*.csv | tr ',' '\n' | grep ':boolean' | sort -u
> ```

---

## Using an Existing Snapshot

**Snapshots generated before `76fbadf` must have Phase 1 re-run.** Their CSV
headers contain sanitized ROI names (`BU_R_`), which will not match the ROI
names in `create-indexes.cypher` or in the Meta node, producing a database whose
ROI indexes are inert. This supersedes the earlier guidance that Phase 1 output
did not need regenerating.

Note that a re-run will *not* regenerate the CSVs on its own: `neuprint-export`
is cached behind a sentinel file, and a cached run completes in under two
minutes having done nothing. Clear it first:

```bash
rm -f <snapshot>/cache/neuprint-export-*.sentinel
```

or move the whole `cache/` and `neuprint/` directories aside for a genuinely
clean rebuild.

Separately, the ingest script copies `create-indexes.cypher` directly from the
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
# Phase 1
# Note: the wasp and wasp2 config trees are byte-identical; this is the path
# the production wrapper script (run_wasp_snapshot.sh) uses.
pixi run flyem-snapshot --config /groups/flyem/data/snapshots/snapshot-configs/wasp/v0.8/wasp-release-snapshot.yaml

# Phase 2
pixi run ingest-neuprint-snapshot-using-apptainer 2026-05-12-32c9ac

# Validate the result
pixi run check-neuprint-snapshot 2026-05-12-32c9ac/neo4j
```

On a machine smaller than a cluster node, size the ingest explicitly — its
defaults are 31G heap / 150G page cache:

```bash
HEAP_SIZE=4G MAX_MEMORY=8G pixi run ingest-neuprint-snapshot-using-apptainer \
    2026-05-12-32c9ac --scratch-dir /tmp
```

Note that a re-run will silently skip CSV regeneration if the export sentinel is
still present — see *Using an Existing Snapshot* above.

Check that neuprinthttp can connect to the resulting database and serve queries
(with the driver bump and `"database": "data"` config change described below).

### Things to watch for

- **Import failure**: check `/scratch/$USER/.../neo4j/logs/import.err.log` for
  errors. The script already checks for `import failed` in the logs.
- **Config problems are NOT fatal here.** Neo4j would normally refuse to start
  on an unknown or renamed setting, but we set
  `server.config.strict_validation.enabled=false`, so unrecognised keys are
  downgraded to `Unrecognized setting` **warnings** and otherwise ignored. Read
  the validation block that `neo4j start` prints — `N issues found` with a list
  — rather than assuming a clean startup means a correct config. This is exactly
  how the bogus `dbms.security.authentication.enabled` setting went unnoticed
  and left authentication enabled.
- **Server won't start / "Unexpected Neo4j server failure"**: usually the memory
  sizing. Check the `-Xms`/`-Xmx` values in the `Executing command line:` output;
  a 31 GB heap cannot be reserved on a smaller machine, especially under
  `-XX:+AlwaysPreTouch`. Override with `HEAP_SIZE` / `MAX_MEMORY`.
- **Index failures**: check `logs/create-indexes.out.log` and
  `logs/create-indexes.err.log`. The script checks for `database is unavailable`
  and an empty output log.
- **APOC jar**: if APOC procedures are needed (e.g. during a debug session),
  verify the jar is present at `$NEO4J_HOME/plugins/apoc-2026.07.1-core.jar` inside
  the container.

---

## Current Status (as of 2026-09-04)

**The upgrade runs end-to-end and is validated on two machines.** The neuclease
blocker recorded in earlier revisions of this document is resolved.

### Environment

`pixi.toml` now pins `neuclease >= 0.81.post0.dev106`, which fixed the
`IncompleteLabelNamesError` / `DEFAULT_BODY_STATUS_CATEGORIES` problems. Four
further dependency changes were needed:

- **`pandas` pinned to `>=2.2,<3`.** The neuclease bump left pandas unpinned and
  pixi resolved 3.0.3, where PDEP-14 makes a new `StringDtype` the default and
  Copy-on-Write becomes mandatory. That breaks `series_checksum` and
  `neo4j_type_suffix`. Migrating to pandas 3 is separate work — and the risk is
  not merely crashes: patching the dtype inference could silently emit *wrong*
  neo4j type suffixes in CSV headers, corrupting property types with no error.
- **`pip` added.** Without it, `pixi run pip install -e .` falls through `PATH`
  to the system pip, which builds against an old setuptools/versioneer and dies
  looking for a nonexistent `setup.cfg`.
- **`neuroglancer`, `tensorstore`, `ngsidekick` added.** neuclease's
  `misc.neuroglancer` submodule imports them without declaring them.
- **`linux-64` platform, the `apptainer` target dependency and the
  `update-service` environment restored** after a local edit had dropped them.
  The lock file remains **format v6** — older pixi on the cluster nodes cannot
  read v7.

Note that `pixi install` does *not* install `flyem-snapshot` itself; the CLI
entry points require `pixi run pip install -e . --no-deps` afterwards.

### Verified results

The wasp figures below were produced against `2026.06.0`, before the version
bump; yakuba and fish2 were validated against `2026.07.1` directly.

Full pipeline, wasp v0.8, against `neo4j:2026.06.0`:

| | |
|---|---|
| Import | 5,278,783 nodes / 10,702,773 relationships / 26,561,538 properties |
| Bad entries | none |
| Indexes | 232, all `ONLINE` at 100% |
| Database | `data`, online, default |

Run independently on `vm7181` (a 23 GB VM) and on LSF cluster node `h07u06`,
producing **identical counts in every measured dimension** — 640,170 Segment /
50,564 Neuron / 2,336,820 Synapse / 2,301,792 SynapseSet nodes, 2,301,792
ConnectsTo / 2,103,064 SynapsesTo / 6,297,917 Contains relationships, 232
indexes, 97 ROIs. The pipeline is reproducible across machines.

Since then the pipeline has been validated end-to-end on two further datasets
against `neo4j:2026.07.1`, each passing with zero failures:

| dataset | nodes | relationships | Neuron | indexes | ROIs |
|---|---|---|---|---|---|
| wasp v0.8 | 5,278,783 | 10,702,773 | 50,564 | 232 | 97 |
| yakuba-vnc | 137,632,747 | 261,841,354 | 87,627 | 132 | 25 |
| fish2 | 77,830,526 | 128,491,329 | 224,391 | 697 | 210 |

All three pass **38 of 38** at default settings. Note that the total depends on
the flags: `CHECK_CSV_COUNTS=0` drops three checks and `MAX_QUERY_MS` adds one,
so totals are only comparable between runs invoked the same way.

Every reconciliation was exact on all three: node and relationship totals match
the importer's own report, no bad entries were skipped, and both the node
counts (`Segment` / `Synapse` / `SynapseSet`) and the relationship counts
(`ConnectsTo` / `SynapsesTo` / `Contains` / `CloseTo`) match the exported CSV
row counts.

The relationship grouping is taken from the `--relationships=<TYPE>=<file>`
arguments in `ingest-neuprint-snapshot-within-neo4j-container.sh`, because it
is not one file per type — `ConnectsTo` is built from two CSVs and `Contains`
from four, two of them globs over element tables. **The checker is coupled to
that script**: adding a `--relationships=` line there without updating the
checker makes `Contains count matches CSV rows` fail.

Working through that mapping also exposed a fourth relationship type,
`CloseTo`, which the suite had never counted. No dataset measured has any, so
the omission was invisible — the three counted types happened to sum to the
reported total. There is now a relationship-type accounting check mirroring the
node-label one, so an uncounted type inflates the total loudly instead.

#### Element labels are nested inside Synapse labels

**`:Synapse` is a specialization of `:Element`, and `:SynapseSet` of
`:ElementSet`** — the same nesting as `:Neuron` within `:Segment`.

Every dataset measured carries `:Element` on every synapse, whether or not its
config declares element tables. wasp and yakuba both report `Element` counts
exactly equal to their `Synapse` counts. So a dataset does not divide into
"has elements" and "has none"; what varies is how many elements are *not*
synapses — somas and the like. fish2 is the first dataset here with a non-zero
count of those:

| dataset | Element | Synapse | non-synaptic |
|---|---|---|---|
| wasp | 2,336,820 | 2,336,820 | 0 |
| yakuba | 51,824,503 | 51,824,503 | 0 |
| fish2 | 41,916,590 | 41,725,816 | 190,774 |

The labels are not disjoint, so the element counts are dominated by synapses:

```
Element    41,916,590  -  Synapse     41,725,816  =  190,774 non-synaptic
ElementSet 32,382,125  -  SynapseSet  32,192,910  =  189,215 non-synaptic
                                            total =  379,989
```

Only that 379,989 remainder is unaccounted for by `Segment + Synapse +
SynapseSet + Meta`, and `Segment + Element + ElementSet + Meta` sums to
77,830,526 exactly — the reported total.

The non-synaptic remainder is the population that `non-synaptic-bodies:
element-presence` selects on in a report config, which fish2 uses. If it
silently went to zero, `element-presence` would degrade to `none` and every
report's body ranking would change, with nothing failing. The checker therefore
reports it explicitly rather than leaving it to be inferred by subtraction.

The nesting itself is asserted, at no extra query cost. Since
`|Element ∩ Synapse| = Element − non-synaptic`, every synapse carrying
`:Element` is equivalent to `Element − Synapse == non-synaptic`, so the
identity is checked from figures already collected. A synapse missing its
`:Element` label breaks it. Asserting the raw `Element` count is `> 0` would
prove nothing, since it merely restates the `Synapse` count.

> The commit message on `8e573c9` describes fish2 as having "379,989 Element and
> ElementSet nodes". That is wrong — it has 74.3M, of which 379,989 are
> non-synaptic. The check itself is correct; only the message is misleading.

#### Complex-query timing drifts downward across repeated runs

Warm timings for the neuPrintExplorer search query:

| dataset | neurons | warm | vs `MAX_QUERY_MS=1500` |
|---|---|---|---|
| wasp | 50,564 | 502-611 ms | ~146% headroom |
| yakuba | 87,627 | 766-808 ms | ~86% headroom |
| fish2 | 224,391 | 1000-1347 ms | **~11% headroom** |

The fish2 spread is not random. Three consecutive runs on the same idle node
were **monotonically decreasing in both columns**:

| run | cold | warm |
|---|---|---|
| 1 | 1749 ms | 1347 ms |
| 2 | 1562 ms | 1079 ms |
| 3 | 1347 ms | 1000 ms |

26% faster on warm, 23% on cold, strictly decreasing — not the signature of
noise. The likely cause is the **host** page cache. Each run starts a fresh
container, but the store files stay cached on the node in between, so the
container's own "cold" run is progressively less cold on each repeat. The
cold/warm distinction the checker reports only covers Neo4j's page cache, not
the host's.

Two consequences for calibration:

1. **Calibrate on the first run on a node, not a repeat.** A number derived
   from run 3 would be tuned to the most-cached case and would fail on exactly
   the runs that matter — a freshly ingested database on a node that has never
   touched those files. Expect the true worst case to be at or above run 1.
2. **Set `MAX_QUERY_MS` per dataset**, in the wrapper scripts. A single global
   value cannot serve both ends: 1500 ms leaves fish2 only 11% headroom over
   its slowest observed run, while being too loose to catch a 2x regression on
   wasp.

Suggested starting points: **1500 ms for wasp and yakuba, 2000-2500 ms for
fish2** (48% and 86% headroom respectively over fish2's slowest observed run).
Treat these as provisional until a threshold has survived a few first-runs on
cold nodes.

### Running on the cluster (LSF)

- `LSF_UNIT_FOR_LIMITS=MB`, and it lives in `lsf.conf` — **not** `lsb.params`,
  so `bparams` will not show it.
- **Phase 1 peaks at 9.4 GB** (`MAX MEM` from `bjobs -l`), averaging 1.4 GB.
- Phase 1 is largely serial; reserve wide slot counts for Phase 2, where
  `neo4j-admin` actually uses `--threads`. The ingest script honours
  `LSB_MAX_NUM_PROCESSORS` inside an LSF job.
- Use `bsub … -Is /bin/bash`. Without `-Is` the job runs and exits while your
  prompt stays on the **submit host**, where a small per-user memory limit will
  produce a `MemoryError` that looks like a pipeline bug.
- Cluster nodes have a real `/scratch`, so `--scratch-dir` can be omitted.

### Validating a built database

`check-neuprint-snapshot` launches a snapshot's database in a container, runs
38 checks and shuts it down, exiting 0/1 so it can gate a pipeline run. It
covers
node/relationship counts, node-label accounting, `bodyId` integrity, index state
and population, that every index refers to a property that exists, that every
ROI in `Meta.roiInfo` has both a matching property and an index, that every ROI
index is usable when forced via an index hint, and the server's database name
and Cypher language pin.

A third lesson: **`skip` does not count as a failure**, so anything that
degrades to a skip can hide a whole section. `Meta.roiInfo` is parsed with
`apoc.convert.fromJsonMap`, and `q()` sends stderr to `/dev/null`, so an APOC
that failed to load looked exactly like an empty result — skipping every ROI
assertion and the entire index-usability section while the run still printed
`RESULT: OK`. Since the apoc jar ships in the snapshot's own `plugins/` dir and
is version-coupled to the server, that is a live risk for this upgrade, not a
hypothetical one. That path now fails. The check is derived from the real
`roiInfo` read rather than a synthetic probe — an earlier attempt to probe APOC
separately failed on a database where APOC plainly worked.

Two further lessons are baked into it. **Do not assert that a query produces a
`NodeIndexSeek`** — plan choice is a planner cost decision, not a correctness
property, and a scan is legitimately cheaper for an ROI covering many Segments.
Force the index with a hint instead: that either resolves or errors. And
**`readCount` is sampled telemetry**, flushed periodically and cumulative, so it
is reported rather than asserted on.

It also reconciles the node and relationship totals against what the importer
reported, fails if the import skipped bad entries, and reconciles the
`Segment` / `Synapse` / `SynapseSet` counts against the exported CSV row counts
(on by default; `CHECK_CSV_COUNTS=0` skips it). Finally it runs the query
neuPrintExplorer issues from its search box and reports the timing;
`MAX_QUERY_MS` turns that into an assertion. Run `check-neuprint-snapshot.sh
--help` for the full list of environment overrides.

Be clear about what that timing measures. On wasp the spread across search
terms was 251 ms for a term matching 702 rows to 635 ms for one matching
44,933 — only 2.5x for 64x the rows. The floor is the label scan, so it tracks
scan and sort throughput far more than search selectivity, and no search term
makes it slow at snapshot scale. It is a regression detector, not a model of
user-visible latency.

### Remaining work

1. **Packaging gap:** `pyproject.toml` has no `MANIFEST.in` or `package-data`,
   so the `.sh` / `.conf` / `.cypher` files are absent from a built wheel
   (verified — a built wheel contains zero non-`.py` files). This works today
   only because the install is editable, and it blocks deploying this branch as
   a built package.
2. Consider deriving the ingest memory sizing from the cgroup / LSF limit rather
   than defaulting to 31G/150G, so neither path needs environment variables.
3. The `.sh` header and the Python wrapper's docstring are two hand-maintained
   copies of the same documentation, which is how they drifted apart once
   already. Have the wrapper shell out to `--help` instead.

### Merged `origin/master` mid-course (2026-09-04)

The branch forked from master on 2026-06-09 and had fallen 9 commits behind by
the time fish2 was attempted. That surfaced as a config-validation failure
rather than anything Neo4j-related:

```
jsonschema.exceptions.ValidationError: Additional properties are not allowed
    ('non-synaptic-bodies' was unexpected)
```

The snapshot configs under `snapshot-configs/` are maintained against master,
not against this branch. The fish2 config uses the `non-synaptic-bodies` report
setting added upstream in `5417ef9`, which this branch did not have. Note that
config validation stops at the *first* unknown key, so a cherry-pick of that one
commit would only have revealed the next gap; merging closes them all at once.

**Merged `origin/master` into the branch** (merge commit `7940dcc`). Two
conflicts, both resolved in favour of this branch:

- **`pixi.toml` — keep ours.** Upstream `7e37c92` ("Update pixi (use arm on mac,
  not x64)") narrowed `platforms` to `["osx-arm64"]` and dropped the
  `[target.linux-64.dependencies] apptainer` block with it. Taking that would
  break every cluster run. Ours is otherwise a superset: every dependency master
  declares is present, including `ngsidekick`, plus the `pandas <3` pin and the
  `pip` / webdriver / `google-cloud-storage` additions. **This branch therefore
  restores `linux-64` to master when it merges** — worth calling out in review,
  since it reverts part of an upstream commit.
- **`pixi.lock` — keep ours.** It matches our `pixi.toml`, and a lock file
  cannot be meaningfully three-way merged. Ours is lock format v7; master's is
  v6, so merging this branch also moves master to v7 (pixi >= 0.50 required).

`flyem_snapshot/outputs/neuprint/segment.py` auto-merged cleanly: master added
per-compartment (axon/dendrite) synapse breakdowns, this branch added the
`check_roi_name` call. Both survive in the result.

`.gitattributes` and `.gitignore` had uncommitted local edits adding the same
pixi boilerplate master had already committed; master's `.gitattributes` is the
better version, as it also marks `pixi.lock` as `-diff`.

After pulling this on a machine, re-sync the environment before running:

```bash
cd /groups/flyem/home/flyem/flyem-snapshot
git pull && pixi install && pixi run pip install -e . --no-deps
```

**fish2 has not yet been run to completion** — the merge unblocks config
validation, but Phase 1 and Phase 2 for fish2 remain unverified. wasp and
yakuba are both validated end-to-end.

### Related component findings

- **neuprint-python** — no changes needed (HTTP only, no direct Neo4j/Bolt connection)
- **neuPrintExplorer** — no changes needed (React frontend, HTTP only)
- **neuPrintHTTP** — re-reviewed against a CalVer server. The earlier conclusion
  that no code change was needed held for `5.26` but **does not hold here**:

  1. **Driver bump required.** It pins `neo4j-go-driver/v5 v5.27.0`; Neo4j
     documents `5.28` as the forward-compatibility floor for 2025.x/2026.x.
     Driver `v5.28.0` added Bolt Handshake Manifest v1 (ADR 30), which is how a
     driver negotiates the newer Bolt versions a CalVer server offers — `5.27`
     still performs the legacy fixed handshake. `v5.28.4` is a drop-in upgrade.
  2. **Config change**, as before: `"database": "neo4j"` → `"data"`. This fails
     *silently* if wrong — the service connects but finds no data. The field
     must be present and explicit on every store entry; omitting it falls back
     to the server's home database.
  3. **Keep the `neuPrint-bolt` engine.** The `neuPrint-neo4j` backend speaks
     the legacy HTTP transactional API removed in Neo4j 5.

  **Deployment sequencing:** the production Neo4j server must be upgraded to
  `2026.07.1` *before* a database built by this branch is swapped in. A store
  written by the 2026.07.1 importer cannot be read by an older server, and Neo4j
  has no downgrade path.

---

## Useful References

- [Changes from Neo4j 5.26 LTS to 2025.01 and later](https://neo4j.com/docs/upgrade-migration-guide/current/version-2025-2026/upgrade/)
- [Changes, deprecations and removals in the 2025–2026 series](https://neo4j.com/docs/operations-manual/current/changes-2025-2026/)
- [Configure the Cypher default version](https://neo4j.com/docs/operations-manual/current/configuration/cypher-version-configuration/)
- [Select Cypher version](https://neo4j.com/docs/cypher-manual/current/queries/select-version/)
- [Neo4j 5 upgrade/migration guide](https://neo4j.com/docs/upgrade-migration-guide/current/version-5/) (still relevant — the 4.4 → 5.x changes all apply)
- [neo4j-admin database import full](https://neo4j.com/docs/operations-manual/current/tools/neo4j-admin/neo4j-admin-import/)
- [Neo4j configuration settings reference](https://neo4j.com/docs/operations-manual/current/reference/configuration-settings/)
- [Neo4j Cypher constraint syntax](https://neo4j.com/docs/cypher-manual/current/constraints/)
- [APOC releases](https://github.com/neo4j/apoc/releases)
- [Neo4j version support / EOL dates](https://endoflife.date/neo4j)
- [Neo4j Docker Hub](https://hub.docker.com/_/neo4j)
