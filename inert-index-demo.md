# Why an inert index is invisible

A worked demonstration of the `fish2_:Soma` defect: 217 Neo4j indexes that
covered zero nodes while reporting themselves perfectly healthy.

The demo is deliberately synthetic rather than run against fish2. The defect is
fixed, so reproducing it on a real snapshot would mean deliberately
reintroducing it; a standalone container makes the same point in two minutes and
can be re-run by anyone.

---

## The defect

Element labels are written in snapshot configs with a leading colon — the
schema's own example is `':Mito'`. A colon is Cypher *punctuation*, not part of
a label, so it has to be stripped before the name is used.

`flyem_snapshot/outputs/neuprint/element.py` stripped it, so the exported nodes
carried the labels `Soma` and `fish2_Soma`:

```python
specific_label = cfg['element-labels'].get(config_name, '').lstrip(':')
```

`flyem_snapshot/outputs/neuprint/indexes.py` did not. So
`create-indexes.cypher` rendered `{{dataset}}_{{label}}` as `fish2_:Soma`:

```cypher
CREATE INDEX FOR (n:`fish2_:Soma`) ON (n.`Forebrain`);
```

Two code paths disagreeing about one name. Every element ROI index — **217 of
them, 31% of fish2's 707 indexes** — was built on a label no node carried.

Both spellings are *legal* Neo4j labels. Inside backticks a label may contain
almost anything, colons included. Nothing was malformed; the index simply
pointed at nothing.

## Why nothing noticed

This is the part worth internalising. Neo4j's health metrics cannot distinguish
an index that works from one that covers nothing:

```
label          state     populationPercent
"fish2_:Soma"  "ONLINE"  100.0
```

An index over a label no node carries is **trivially fully populated** — there
is nothing left to populate. So `state: ONLINE` and `populationPercent: 100.0`
are both entirely accurate and entirely useless. Anyone auditing index health
sees a clean bill across all 217.

The checker's pre-existing assertions could not catch it either:

- *no index in a non-ONLINE state* — passes, it is ONLINE
- *no index below 100% populated* — passes, it is at 100%
- *every Segment index refers to a property that exists* — passes, it was scoped
  to `Segment` indexes and checked property names, not labels

It was found by adding an assertion that every indexed **label** is carried by
at least one node, which failed on its first run against fish2 — the only
dataset here with element tables, so wasp and yakuba could never have caught it.

## The demonstration

`demo-inert-index.sh` (standalone, needs podman or `CLI=docker`). It starts a
`neo4j:2026.08.1` container, creates 190,774 `:fish2_Soma` nodes — fish2's
actual non-synaptic element count — each with a boolean ROI property, builds the
**broken** index, and profiles the query a client would really run:

```cypher
MATCH (n:`fish2_Soma`) WHERE n.`Forebrain` = true RETURN count(n)
```

Then it adds the **correct** index and profiles the identical query again.

## Results

| index present | query plan | DbHits | Time |
|---|---|---|---|
| only the broken `fish2_:Soma` | `NodeByLabelScan` + `Filter` | **381,549** | 269 ms |
| the correct `fish2_Soma` | `NodeIndexSeek` | **191** | 58 ms |

**About 2000x more reads.** The DbHits decompose exactly:

- `381,549` ≈ 2 × 190,774 — every node read, plus its property. A full label
  scan.
- `191` = the 190 matching nodes, plus one. Only what the query needs.

Both queries return `190`. **The results are correct either way** — this costs
performance and produces misleading diagnostics, but never wrong data.

### A methodological trap in measuring this

The first attempt at this comparison was misleading, and the mistake is easy to
repeat. With *both* indexes present, querying the broken label directly:

```cypher
MATCH (n:`fish2_:Soma`) WHERE n.`Forebrain` = true RETURN count(n)
```

reports a `NodeIndexSeek` — it seeks its own empty index and instantly returns
nothing, which looks like the index working. The honest comparison is the query
a *client* actually issues, against the label the nodes carry (`fish2_Soma`),
with only the broken index present. That is what the table above measures.

## How bad was it, really

Worth keeping the severity calibrated. Three bugs of this shape were found on
the `neo4j-5-upgrade` branch, and they are not equally serious:

| defect | index pointed at | nodes had | consequence | kind |
|---|---|---|---|---|
| ROI property names | `BU(R)` | `BU_R_` | ~194 inert indexes | performance |
| fulltext property list | 3 of 11 searched | — | **42% of search results dropped on yakuba** | **correctness** |
| element labels | `fish2_:Soma` | `fish2_Soma` | 217 inert indexes | performance |

Only the fulltext gap returned wrong answers to users. The other two are "fix
it, it is cheap and it is wrong" rather than "this was shipping broken
results" — but all three came from the same root cause, and that root cause has
produced a correctness bug once already.

## What now guards it

Two complementary checks, because neither alone is sufficient:

**`check-neuprint-snapshot`, after an ingest** — asserts that every indexed
label is carried by at least one node. This catches *any* label mismatch,
including a well-formed but wrong one such as a bad dataset prefix
(`wasp_Segment` instead of `wasp3_Segment`), which contains no suspicious
characters at all.

**`check_element_label()`, at Phase 1** — rejects a label containing a backtick,
an embedded colon, or a line break before the index script is even written. This
catches malformed labels in seconds instead of after an ingest. The backtick
case matters most: it would end the backtick quoting early, producing invalid
Cypher and killing the ingest during index creation, hours into a run.

A character check alone would miss the wrong-prefix case; a label-existence
check alone would only report the problem after a full rebuild. Hence both.
