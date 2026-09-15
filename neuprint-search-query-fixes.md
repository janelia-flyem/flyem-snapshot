# FindNeurons search: one change still needed

Found while adding neuPrintExplorer's two FindNeurons search queries to
`check-neuprint-snapshot`, validating against wasp, yakuba and fish2 —
originally on Neo4j 2026.07.1, and since re-validated on **2026.08.1**, the
version this branch now targets. Measurements and reasoning are in
[`neo4j-upgrade.md`](neo4j-upgrade.md) under *What the fulltext "fast" query
actually saves*.

Three problems were found. **Two are now fixed in this repository** and are
kept below for the record; the remaining one is in neuPrintExplorer and is
still outstanding.

| # | where | status |
|---|---|---|
| 1 | `neuPrintExplorer` — `buildFastQuery` does not compile on Neo4j 5+ | **outstanding** |
| 2 | `flyem-snapshot` — fulltext index covered 3 of 11 searched properties | fixed, `644158a` |
| 3 | `flyem-snapshot` — `itoLeeHl` misspelling | fixed with 2 |

Change 1 is independent of the Neo4j upgrade and can be made at any time.

---

## 1. `buildFastQuery` does not compile on Neo4j 5.x or later

| | |
|---|---|
| **Repo** | `connectome-neuprint/neuPrintExplorer` |
| **File** | `src/js/plugins/query/shared/NeuronInputField.jsx` |
| **Line** | 41 (inside `buildFastQuery`, which begins at line 28) |

**Current:**

```cypher
OPTIONAL MATCH (b:Neuron) WHERE user_body <> 0 AND b.bodyId = user_body
WITH textMatches + collect(b) as allMatches, q, user_body
UNWIND allMatches as n
```

**Proposed:**

```cypher
OPTIONAL MATCH (b:Neuron) WHERE user_body <> 0 AND b.bodyId = user_body
WITH textMatches, q, user_body, collect(b) as bodyMatches
WITH textMatches + bodyMatches as allMatches, q, user_body
UNWIND allMatches as n
```

**Why.** The query fails outright from Neo4j 5 onward. On 2026.07.1 and
identically on 2026.08.1:

```
42I18: syntax error or access rule violation - reference to non-grouping
sub-expression. The expression contains a non-grouping sub-expression
`textMatches`. In an aggregating context only grouping sub-expressions and
constants are allowed.
```

`collect(b)` makes that `WITH` an aggregating clause, so its grouping keys are
`q` and `user_body`. `textMatches` then appears inside the aggregating
expression without being a grouping key, which Cypher rejects. The fix makes
`textMatches` a grouping key of the aggregating `WITH` and moves the
concatenation into a separate non-aggregating one. Semantically identical.

**This is a Cypher 4.4 to 5 breaking change, not a 2026-specific one.**
Verified by running `EXPLAIN` on both forms against stock containers:

| server | as written | with the fix |
|---|---|---|
| 4.4.48 | compiles | compiles |
| 5.26.30 | **fails** — "Aggregation column contains implicit grouping expressions ... Illegal expression(s): textMatches" | compiles |
| 2026.07.1 | **fails** — `42I18` | compiles |
| 2026.08.1 | **fails** — `42I18` | compiles |

Each row is from running `EXPLAIN` on both forms against that version in a
stock container, not from reading release notes.

Cypher 4.4 accepted implicit grouping expressions; Cypher 5 rejects them, so
this breaks as soon as you leave 4.4 rather than only on CalVer. 5.26's error
message prescribes the same fix: "It may be possible to rewrite the query by
extracting these grouping/aggregation expressions into a preceding WITH
clause."

**So the fix can be made now, independently of the server upgrade.** It
compiles on 4.4 as well as 5.x and 2026.x, so it needs no coordination with
the rollout and cannot break the current deployment.

The query is selected by a `useFastQuery` toggle (line 213). Since it works on
4.4, it may well have been exercised there; what is certain is that it cannot
work on any 5.x or later server. It arrived in `a2edf26` and was extended by
`5e1d77c`, so this is current work rather than legacy.

**Verify.** Any snapshot built by the `neo4j-5-upgrade` branch will exercise
it — `check-neuprint-snapshot` runs the query and fails if it does not execute.
The checker currently carries the corrected form so that it can measure the
fast path; that copy is a stand-in, not the fix.

---

## 2. The fulltext index did not cover the properties the query searches — FIXED

| | |
|---|---|
| **Repo** | `janelia-flyem/flyem-snapshot`, plus the per-dataset snapshot configs |
| **File** | `flyem_snapshot/outputs/neuprint/indexes.py`, lines 55–68 (schema default), and/or each dataset's `*-snapshot.yaml` |
| **Setting** | `indexes.find-neurons-fulltext-index-properties` |

**Current default:**

```python
"default": [
    "type",
    "instance",
    "synonyms",
]
```

**Why.** `buildFastQuery` finds candidates through
`find_neurons_fulltext_properties_index`, then ranks them using all **eleven**
annotation properties (`type`, `instance`, `hemibrainType`, `flywireType`,
`systematicType`, `itoleeHl`, `trumanHl`, `synonyms`, `class`, `entryNerve`,
`exitNerve`). A neuron whose only match is in an unindexed property is never
returned as a candidate, so the fast query silently yields fewer rows than the
slow one. `a2edf26` expanded the searched fields without expanding the index.

This makes equivalence **dataset-dependent**, which is the trap — a dataset
that happens to populate only indexed properties gives a false all-clear:

| dataset | populated & indexed | populated & NOT indexed | equivalent? |
|---|---|---|---|
| wasp | `type` 6%, `instance` 98% | none | yes |
| yakuba | `type` 14% | `class` 24%, `entryNerve` 4%, `systematicType` <1%, `exitNerve` <1% | **no** |

yakuba's best-populated searchable property, `class` at 24%, is not indexed.
On both datasets the index also spends a slot on `synonyms`, which is null
throughout.

**Measured on yakuba, term `n`** (on 2026.07.1, before both the index fix and
the version bump):

| | rows | warm |
|---|---|---|
| slow query (label scan) | 21,158 | 803 ms |
| fast query (fulltext) | 12,228 | 387 ms |
| **missing** | **8,930 — 42% of the correct result** | |

So this is a correctness defect, not a tuning issue: the fast search returns
under two-thirds of what a user should see.

Note the trap in those timings. They look like a 2.07x speedup, but the fast
query is partly faster *because* it returns 42% less data. Adjusting for the
rows it never produces, at the ~19 us/row fitted on wasp, gives roughly 557 ms
— about 1.44x. That constant is wasp's rather than yakuba's, so treat the
figure as indicative; the direction is not in doubt. **Fixing the index
coverage will shrink the apparent speedup**, and anyone benchmarking the fast
query against a dataset with incomplete coverage will overstate its benefit.

**Options.** Either set the index to all eleven properties, or set it
per-dataset to whatever that dataset populates. All eleven is simpler and
self-maintaining as annotation coverage changes; per-dataset keeps each index
smaller. Note that indexing an absent property is harmless — it simply
contributes nothing.

**Fixed in `644158a`**, which sets the default to all eleven. Confirmed by
rebuilding all three datasets, latterly on 2026.08.1: yakuba's slow and fast
queries now return **identical row counts** where they previously differed by
8,930, and all three pass the coverage check.

That also corrected the headline speedup. yakuba's apparent 2.07x was inflated
by the 42% of rows the fast query was dropping; like-for-like it measures
**1.53x** (960 ms against 627 ms on 2026.08.1), and has held between 1.47x and
1.57x across rebuilds.

Note the before and after row counts are not directly comparable — 21,158 then,
21,167 now. yakuba is under active annotation, so its totals drift between
builds. What matters is that the two queries *disagreed by 8,930* before and
agree exactly after.

---

## 3. `itoLeeHl` was misspelled in the eleven-property list — FIXED

| | |
|---|---|
| **Repo** | `janelia-flyem/flyem-snapshot` |
| **File** | `flyem_snapshot/outputs/neuprint/indexes.py`, line 66 |

**Current (commented out):**

```python
# "default": [
#     "type", "instance", "hemibrainType", "flywireType", "systematicType",
#     "itoLeeHl", "trumanHl", "synonyms", "class", "entryNerve", "exitNerve"
# ]
```

**Proposed.** Change `"itoLeeHl"` to `"itoleeHl"` — lowercase `l`.

**Why.** Both FindNeurons queries read the property as `n.itoleeHl` and alias
it for display as `itoLeeHl`. The commented list uses the display alias rather
than the property name, so enabling it as written would index a property that
does not exist and quietly contribute nothing. This only matters if change 2
is done by uncommenting this list.

**Caveat on the evidence.** The spelling is inferred from neuPrintExplorer's
queries, which are authoritative about the schema they read. It could not be
confirmed against our data: none of wasp, yakuba or fish2 populates
`itoleeHl`, so a coverage count of zero is consistent with either an
unpopulated property or a wrong name. Worth confirming against a dataset that
does populate it (hemibrain) before relying on it.

---

## Related, and probably more impactful than any of the three

**Neither query has a `LIMIT`.** Both return every matched row.

What the index removes is the label scan plus `CONTAINS` filtering. Everything
after the match — the eleven `toLower()` calls, both `CASE` ladders, the
`DISTINCT`, the `ORDER BY`, serialising fourteen columns — is shared, so the
per-row cost is identical and grows with the result set either way.

Two things follow, and they pull in opposite directions.

**Within one dataset, the saving is a fixed amount that a large result set
swamps.** Four term lengths on wasp:

| term | rows | slow | fast | speedup |
|---|---|---|---|---|
| `l` | 25,278 | 572 ms | 562 ms | **1.02x** |
| `lc` | 184 | 214 ms | 102 ms | 2.10x |
| `lc10` | 7 | 193 ms | 103 ms | 1.87x |
| `SNxx07` | 1 | 178 ms | 81 ms | 2.20x |

**Across datasets, that fixed amount scales with the scan being avoided** —
label size times populated properties — so it is far from negligible on a large
label:

| dataset | neurons | slow | fast | speedup | saving |
|---|---|---|---|---|---|
| wasp | 50,564 | 498 ms | 504 ms | 0.99x | ~0 |
| fish2 | 224,391 | 1247 ms | 428 ms | **2.91x** | **819 ms** |

(fish2's row counts differ by 1, so that is very nearly like-for-like.)

So the index is worth having, and more so as datasets grow. But an autocomplete
field issues a short, common term on every keystroke, which maximises the
result set — the part the index cannot help with. If the reported slowness is
that case, the index will improve it by the fixed amount and no more, while the
dominant cost remains returning and serialising tens of thousands of rows a
dropdown cannot display. **A `LIMIT` addresses what the index cannot**, and the
two are complementary rather than alternatives.

This is a frontend design decision, not a defect, which is why it is listed
separately.
