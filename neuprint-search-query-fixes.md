# FindNeurons search: two items still outstanding

Found while adding neuPrintExplorer's two FindNeurons search queries to
`check-neuprint-snapshot`, validating against wasp, yakuba and fish2 —
originally on Neo4j 2026.07.1, and since re-validated on **2026.08.1**, the
version this branch now targets. Measurements and reasoning are in
[`neo4j-upgrade.md`](neo4j-upgrade.md) under *What the fulltext "fast" query
actually saves*.

Four problems were found. **Two are now fixed in this repository** and are kept
below for the record; the other two are in neuPrintExplorer and remain
outstanding.

| # | where | status |
|---|---|---|
| 1 | `neuPrintExplorer` — `buildFastQuery` does not compile on Neo4j 5+ | **outstanding**; fix verified equivalent on 4.4 |
| 2 | `flyem-snapshot` — fulltext index covered 3 of 11 searched properties | fixed, `644158a` |
| 3 | `flyem-snapshot` — `itoLeeHl` misspelling | fixed with 2 |
| 4 | `neuPrintExplorer` — fast query cannot run on hemibrain at all | **outstanding**, and not fixed by 1 |

Change 1 is independent of the Neo4j upgrade and can be made at any time.
Change 4 is a deployment constraint rather than a code defect, and it is not
resolved by change 1.

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
| **4.4.16** — the production version | compiles | compiles |
| 4.4.48 — latest 4.4 patch | compiles | compiles |
| 5.26.30 | **fails** — "Aggregation column contains implicit grouping expressions ... Illegal expression(s): textMatches" | compiles |
| 2026.07.1 | **fails** — `42I18` | compiles |
| 2026.08.1 | **fails** — `42I18` | compiles |

Each row is from running `EXPLAIN` on both forms against that version in a
stock container, not from reading release notes.

Both 4.4 patches are listed because 4.4.48 was tested first by accident — the
`neo4j:4.4` tag resolves to the latest patch, not to the version actually
deployed. 4.4.16 was then tested explicitly. The production version is also
covered far more strongly by the real-data comparisons below: every one of
those 16 runs executed **both** forms successfully against live 4.4.16 servers,
which is direct evidence of compilation rather than an `EXPLAIN` on a stand-in.

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

### The fix returns identical results

Compiling is not the same as being equivalent, so the two forms were run and
their full outputs compared. Any such comparison is only possible on **4.4** —
the original does not compile on 5.x or later, so the version where the
original still works is the only one where both can be run *and* compared.

This was done twice: first on a synthetic fixture, then against real
production datasets. Both are below, and the distinction matters — the
synthetic tests were designed to hit specific edge cases, the real ones to
catch data shapes nobody would think to invent.

#### First, on a synthetic fixture

**Not one of the Janelia databases.** Seven `:Neuron` nodes hand-written into a
throwaway `neo4j:4.4.16` container, with invented values (`LC10`, `SNxx`,
`MBON01`) and a `find_neurons_fulltext_properties_index` over all eleven
searched properties. The property *names* are real; the data is not.

Eight cases, chosen for where the two forms could plausibly diverge:

| term / bodyId | rows | what it exercises |
|---|---|---|
| `lc` / 0 | 4 | no bodyId — `OPTIONAL MATCH` yields null, `collect` must give `[]` |
| `lc` / 300 | 5 | bodyId **not** among the text matches → appended, gets `priority 0` |
| `lc` / 100 | 4 | bodyId **is** among the text matches → `DISTINCT` must dedupe, not 5 rows |
| `lc` / 999 | 4 | nonexistent bodyId → no phantom row |
| `zzz` / 0 | 0 | no text matches, no bodyId |
| `zzz` / 300 | 1 | no text matches *but* a bodyId — the empty-`textMatches` case the source comment is about |
| `l` / 0 | 5 | single-character term |
| `mbon` / 400 | 1 | bodyId identical to the sole text match |

**All eight produced byte-identical output**, compared as whole result sets so
that row content and ordering counted too — including the `priority` and
`type_priority` columns the frontend sorts on.

The `lc` / 100 case is the one worth singling out: body 100 appears in
`textMatches` *and* in `collect(b)`, so the concatenated list contains a
duplicate and both forms depend on `WITH DISTINCT` to collapse it. Both return
4 rows, with body 100 promoted to `priority 0`:

```
bodyId, type, priority, type_priority
"100", "LC10", 0, 0
"200", "LC10b", 2, 0
"500", "(LC)paren", 3, 1
"600", NULL, 4, 2
```

Mechanically the agreement is expected. `collect()` skips nulls, so an
unmatched `OPTIONAL MATCH` yields `[]` in both forms; and `textMatches` is a
single value from the `CALL` subquery, so making it a grouping key produces
exactly one group — the same grouping the original had implicitly.

**A second synthetic test, for tie-breaking at scale.** Seven distinct nodes
cannot tie, and the query ends `ORDER BY priority, type_priority, n.type,
n.instance` — where tied rows have *unspecified* order. At realistic scale ties
are near-certain, since many neurons share a null `type` and null `instance`, so
the two forms could in principle return the same rows in different orders.

3,000 nodes were created that tie on **every** sort key — `class` matching the
term, `type` and `instance` both null. Output was byte-identical, with and
without a bodyId, and with a bodyId both inside and outside the tied set; the
promoted body came first in both forms. Tie order is stable because both build
`allMatches` in the same order (`textMatches`, then `collect(b)`) and the sort
preserves it.

That was the one plausible volume-dependent failure, which is why a still
larger synthetic population would add little — the difference between the forms
is a single clause and does not otherwise interact with row count.

#### Then, on real production data

The synthetic fixture tests semantics; real data tests shapes nobody invented —
apostrophes and unicode in type names, very long strings, realistic null
patterns across the eleven properties. Both forms were run against four live
datasets via `/api/custom/custom`, with
their full result sets compared — two search terms each, with and without a
sampled bodyId, so the `collect(b)` path is genuinely exercised:

| dataset | rows compared | result |
|---|---|---|
| `male-cns:v1.0` | 5,138 / 5,139 / 67,449 / 67,449 | identical |
| `wasp3:v0.8` | 178 / 179 / 51,883 / 51,884 | identical |
| `yakuba-vnc` | 0 / 1 / 5,929 / 5,929 | identical |
| `fish2` | 235 / 236 / 4,133 / 4,134 | identical |

**16 comparisons, all identical in content *and* row order**, the largest at
67,449 rows. Note the pairs where adding a bodyId does not change the count —
`yakuba-vnc` at 5,929 and `male-cns` at 67,449 — those are cases where the
sampled body was already among the text matches, so `DISTINCT` collapsed the
duplicate. That is the behaviour most at risk from the change, and it held.

So the evidence stands at: 8 semantic cases and 3,000 fully-tied rows on a
synthetic fixture, plus 16 comparisons on four production datasets up to 67,449
rows — none of the 24 showing any difference.

**What this does not establish.** Every comparison is on a 4.4 server, because
the original does not compile on 5.x or later and so cannot be compared against
there. Equivalence on the versions that actually matter rests on these results
together with the mechanical argument above — that `collect()` skips nulls, and
that `textMatches` is a single value so grouping on it yields one group.

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

## 4. The fast query cannot run on hemibrain at all

| | |
|---|---|
| **Where** | `https://neuprint.janelia.org`, dataset `hemibrain:v1.2.1` |
| **Cause** | its backing Neo4j is **3.5.3**, which has no `CALL {}` subqueries |

`buildFastQuery` wraps its fulltext search in a `CALL { ... }` subquery.
Subquery support arrived in the 4.x line, so on 3.5 the parser rejects it
outright — **in both the original and the fixed form**:

```
Invalid input '{': expected whitespace, comment, namespace of a procedure
or a procedure name
```

This is not the aggregation defect and the fix does not address it. The
query's structure is unsupported.

**The fleet, measured:**

| dataset | Neo4j | fast query |
|---|---|---|
| `hemibrain:v1.2.1` | **3.5.3** † | cannot run |
| `male-cns:v1.0` | 4.4.16 † | runs |
| `wasp3:v0.8` | 4.4.16 | runs |
| `yakuba-vnc` | 4.4.16 | runs |
| `fish2` | 4.4.16 | runs |

† measured with `CALL dbms.components()`. The other three are inferred from the
deployment config and from queries succeeding, not directly measured.

Uniformly 4.4.16 with hemibrain as the sole outlier — and note two datasets on
the *same hostname* sit on different backends, since neuPrintHTTP's `MasterDB`
fronts several stores and `/api/custom/custom` routes per dataset.

**Consequence: `useFastQuery` cannot be enabled globally.** Turning it on for
all datasets gives hemibrain a syntax error, not a slow result. It needs to be
gated per dataset, or hemibrain's server upgraded first — and hemibrain is the
least likely candidate for upgrading, being a published frozen dataset that
this pipeline does not re-ingest. Per-dataset gating is probably the only
practical answer.

---

## Related, and probably more impactful than any of the four

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
