"""
Validate a neuprint export BEFORE ingesting it.

Checks the CSV/feather artifacts that Phase 1 produced, without starting
neo4j and without needing a built database.  This is a gate to run before
spending an hour on Phase 2, not a replacement for check-neuprint-snapshot --
it can tell you the artifacts are internally consistent, but only the
post-ingestion check can tell you the import did what it was told.

Usage:

    check-neuprint-export <snapshot>/neuprint

... where that directory contains Neuprint_*.csv and create-indexes.cypher,
as produced by flyem-snapshot.  (check-neuprint-snapshot takes the sibling
<snapshot>/neo4j instead.)

What it checks:

    - every index in create-indexes.cypher refers to a label that some CSV
      actually produces, and to a property that appears in that CSV's header.
      This is the inert-index regression test, moved before ingestion: an
      index over a label no node carries reports ONLINE at 100% populated,
      because an index over nothing is trivially fully populated.  Both
      values are accurate and both are useless.  217 such indexes reached
      production on fish2 -- 30% of its node indexes -- from a colon in an
      element label (`fish2_:Soma` where nodes carry `fish2_Soma`).
    - every ROI in Meta.roiInfo has a matching boolean column in the Neuron
      CSVs, and vice versa.  This is the ROI-naming regression test: a
      mismatch means client queries filtering on real ROI names match
      nothing, which is invisible until someone searches.
    - CSV headers are well formed -- each field parses as `name:type`, no
      duplicate column names, and no name contains the characters that break
      the format (comma, colon, double-quote, CR, LF).
    - the two tables exported as both feather and CSV (Neuprint_Neurons and
      Neuprint_Neuron_Connections) have matching row counts.  Both are
      written from one DataFrame, so this checks the CSV batching rather
      than the data -- an error upstream of that DataFrame appears in both
      and passes.
    - Meta.totalPreCount/totalPostCount do not exceed the synapses that
      exist.  They may legitimately be smaller, when the config restricts
      the dataset totals to in-bounds ROIs.
    - every relationship CSV names an ID space that some node CSV defines,
      so a :START_ID(Body-ID) cannot point at an ID space nothing provides.

Exits 0 if every check passed, 1 if any failed, 2 on a bad invocation.

Note on cost: the label inventory is read from the :LABEL column of every
node CSV.  By default only the first --label-rows rows of each file are
scanned, which is ample when labels are assigned by rule (they always are
here) and keeps this fast on a 22-million-row Neuron export.  A file that
was truncated says so.  Use --full to scan every row; on yakuba that is
minutes rather than seconds, and the only thing it buys is certainty about
a label applied to very few bodies.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Characters that break the `name:type` CSV header format. Matches
# check_roi_name() in outputs/neuprint/util.py -- keep them in step.
FORBIDDEN_IN_NAME = set(',:"\r\n')

DEFAULT_LABEL_ROWS = 200_000


class Checker:
    """Collects pass/fail lines the way check-neuprint-snapshot does, so the
    two tools read alike."""

    def __init__(self, use_color=None):
        if use_color is None:
            use_color = (os.environ.get('NO_COLOR') is None
                         and sys.stdout.isatty())
        self.color = use_color
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.failures = []

    def _c(self, code, text):
        return f'\033[{code}m{text}\033[0m' if self.color else text

    def ok(self, msg):
        self.passed += 1
        print(f"  {self._c('32', 'PASS')}  {msg}")

    def bad(self, msg, detail=None):
        self.failed += 1
        self.failures.append(msg)
        print(f"  {self._c('31;1', 'FAIL')}  {msg}")
        for line in (detail or []):
            print(f'            {line}')

    def warn(self, msg):
        self.warnings += 1
        print(f"  {self._c('33;1', 'WARN')}  {msg}")

    def info(self, msg):
        print(f"  {self._c('2', '....')}  {msg}")

    def head(self, msg):
        print()
        print(self._c('36', '=' * 66))
        print(self._c('36', f' {msg}'))
        print(self._c('36', '=' * 66))

    def expect_eq(self, msg, got, want):
        if str(got) == str(want):
            self.ok(f'{msg} ({got})')
        else:
            self.bad(msg, [f'expected {want}, got {got}'])


# --------------------------------------------------------------------------
# reading the export
# --------------------------------------------------------------------------

def header_of(path):
    """The header fields of a CSV, as written."""
    with open(path, newline='') as f:
        for row in csv.reader(f):
            return row
    return []


def column_names(header):
    """Property names from a header, dropping neo4j's type suffixes and the
    structural columns. 'bodyId:long' -> 'bodyId'; ':LABEL' -> skipped."""
    names = []
    for field in header:
        if field.startswith(':'):            # :LABEL, :ID(..), :START_ID(..)
            continue
        names.append(field.split(':', 1)[0])
    return names


def label_field(header):
    """The label column's name as written, or None.

    The exporter is inconsistent on purpose-by-accident: segment.py writes
    ':LABEL', while synapse.py, synapseset.py, element.py, elementset.py and
    meta.py write ':Label'. neo4j-admin treats header field types
    case-insensitively, so both import fine -- and a checker that matched only
    one spelling silently ignored every Synapse, SynapseSet, Element,
    ElementSet and Meta CSV, which is most of the export.
    """
    for field in header:
        if field.lower() == ':label':
            return field
    return None


def id_space(field):
    """':ID(Body-ID)' -> 'Body-ID'; ':ID' -> ''; not an id column -> None."""
    m = re.match(r':(?:START_|END_)?ID(?:\(([^)]*)\))?$', field, re.I)
    if not m:
        return None
    return m.group(1) or ''


def node_csvs(export):
    """Every CSV that neo4j-admin will import as nodes, i.e. that has a
    :LABEL column. Relationship CSVs have :START_ID/:END_ID instead."""
    # Glob every CSV, not Neuprint_*.csv: the batched tables are written as
    # Neuprint_Neurons/0000.csv, so matching on the FILE name finds only the
    # unbatched ones (Meta, Neuron_Connections) and silently ignores every
    # Neuron and Synapse batch -- which is most of the export.
    nodes, rels = [], []
    for path in sorted(export.rglob('*.csv')):
        header = header_of(path)
        if label_field(header):
            nodes.append((path, header))
        elif any(f.upper().startswith((':START_ID', ':END_ID'))
                 for f in header):
            rels.append((path, header))
    return nodes, rels


def scan_labels(path, header, limit):
    """The distinct label sets in a node CSV's :LABEL column.

    Returns (labels, rows_read, truncated).
    """
    idx = header.index(label_field(header))
    labels = set()
    rows = 0
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)                   # header
        for row in reader:
            rows += 1
            if idx < len(row):
                labels.update(x for x in row[idx].split(';') if x)
            if limit and rows >= limit:
                return labels, rows, True
    return labels, rows, False


def csv_rows(paths):
    """Total data rows across CSVs, one header discounted per file."""
    total = 0
    for p in paths:
        with open(p, 'rb') as f:
            total += sum(1 for _ in f) - 1
    return total


# --------------------------------------------------------------------------
# parsing create-indexes.cypher
# --------------------------------------------------------------------------

def parse_indexes(text):
    """(label, property) pairs the rendered Cypher will create.

    Covers the forms indexes.py renders: plain and POINT indexes, the
    uniqueness constraints, and the FULLTEXT index's ON EACH list.
    """
    def unquote(s):
        return s.strip().strip('`')

    pairs = []

    # CREATE [POINT] INDEX [`name`] FOR (n:`Label`) ON (n.`prop`)
    for label, prop in re.findall(
            r'CREATE\s+(?:POINT\s+)?INDEX\s+(?:`?[\w-]+`?\s+)?FOR\s*\(\s*\w+\s*:\s*(`[^`]+`|\w+)\s*\)'
            r'\s*ON\s*\(\s*\w+\.\s*(`[^`]+`|\w+)\s*\)', text, re.I):
        pairs.append((unquote(label), unquote(prop), 'index'))

    # CREATE CONSTRAINT `name` FOR ( `var`:`Label` ) REQUIRE `var`.prop IS UNIQUE
    for label, prop in re.findall(
            r'CREATE\s+CONSTRAINT\s+`?[\w-]+`?\s+FOR\s*\(\s*`?[\w-]+`?\s*:\s*(`[^`]+`|\w+)\s*\)'
            r'\s*REQUIRE\s+`?[\w-]+`?\.\s*(`[^`]+`|\w+)', text, re.I):
        pairs.append((unquote(label), unquote(prop), 'constraint'))

    # CREATE FULLTEXT INDEX name FOR (n:`Label`) ON EACH [ n.`a`, n.`b` ]
    for m in re.finditer(
            r'CREATE\s+FULLTEXT\s+INDEX\s+[\w-]+\s+FOR\s*\(\s*\w+\s*:\s*(`[^`]+`|\w+)\s*\)'
            r'\s*ON\s+EACH\s*\[(.*?)\]', text, re.I | re.S):
        label = unquote(m.group(1))
        for prop in re.findall(r'\w+\.\s*(`[^`]+`|\w+)', m.group(2)):
            pairs.append((label, unquote(prop), 'fulltext'))

    return pairs


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def check_headers(chk, node_csvs_, rel_csvs):
    chk.head('CSV headers')
    bad_names, dupes = [], []
    for path, header in node_csvs_ + rel_csvs:
        seen = set()
        for field in header:
            name = field.split(':', 1)[0] if not field.startswith(':') else ''
            if name:
                if FORBIDDEN_IN_NAME & set(name):
                    offending = ''.join(sorted(FORBIDDEN_IN_NAME & set(name)))
                    bad_names.append(f'{path.name}: {name!r} contains {offending!r}')
                if name in seen:
                    dupes.append(f'{path.name}: duplicate column {name!r}')
                seen.add(name)
    if bad_names:
        chk.bad('no column name contains a character that breaks name:type',
                bad_names[:10])
    else:
        chk.ok('no column name contains a character that breaks name:type')
    if dupes:
        chk.bad('no duplicate column names within a file', dupes[:10])
    else:
        chk.ok('no duplicate column names within a file')


def check_id_spaces(chk, node_csvs_, rel_csvs):
    chk.head('ID spaces')
    defined = set()
    for _, header in node_csvs_:
        for field in header:
            sp = id_space(field)
            if sp is not None:
                defined.add(sp)
    chk.info(f'node CSVs define: {", ".join(sorted(defined)) or "(none)"}')
    missing = []
    for path, header in rel_csvs:
        for field in header:
            sp = id_space(field)
            if sp is not None and sp not in defined:
                missing.append(f'{path.name}: {field} -- no node CSV defines {sp!r}')
    if missing:
        chk.bad('every relationship ID space is defined by some node CSV',
                missing[:10])
    else:
        chk.ok('every relationship ID space is defined by some node CSV')


def check_indexes(chk, export, node_csvs_, label_rows):
    chk.head('create-indexes.cypher vs the CSV headers')
    cypher_path = export / 'create-indexes.cypher'
    if not cypher_path.is_file():
        chk.bad(f'{cypher_path.name} is present')
        return
    pairs = parse_indexes(cypher_path.read_text())
    if not pairs:
        chk.bad('create-indexes.cypher declares at least one index',
                ['parsed zero (label, property) pairs -- has the template '
                 'changed shape?'])
        return
    chk.info(f'{len(pairs)} index/constraint target(s) declared')

    # What labels will exist, and which properties each label's CSVs carry.
    label_props = defaultdict(set)
    all_labels = set()
    truncated = []
    for path, header in node_csvs_:
        labels, rows, cut = scan_labels(path, header, label_rows)
        if cut:
            truncated.append(path.name)
        all_labels |= labels
        props = set(column_names(header))
        for lbl in labels:
            label_props[lbl] |= props
    if truncated:
        chk.info(f'label scan stopped at {label_rows:,} rows in '
                 f'{len(truncated)} file(s); use --full to scan every row')
    chk.info(f'{len(all_labels)} distinct label(s) across the node CSVs')

    dead_label, dead_prop, dead_fulltext = [], [], []
    for label, prop, kind in pairs:
        if label not in all_labels:
            dead_label.append(f'{kind} on :{label}({prop}) -- no CSV produces '
                              f'that label')
        elif prop not in label_props[label]:
            # A FULLTEXT index naming an absent property is not the same
            # defect. The index still serves the properties that do exist;
            # that one simply contributes nothing. check-neuprint-snapshot
            # treats fulltext coverage gaps as warnings for the same reason.
            # A plain index over an absent property IS inert, and fails.
            (dead_fulltext if kind == 'fulltext' else dead_prop).append(
                f'{kind} on :{label}({prop}) -- that label\'s CSVs have no '
                f'{prop!r} column')

    if dead_label:
        chk.bad('every indexed label is produced by some CSV', dead_label[:12])
    else:
        chk.ok('every indexed label is produced by some CSV')
    if dead_prop:
        chk.bad('every indexed property exists in that label\'s CSVs',
                dead_prop[:12])
    else:
        chk.ok('every indexed property exists in that label\'s CSVs')
    if dead_fulltext:
        chk.warn(f'{len(dead_fulltext)} fulltext propert(ies) are indexed but '
                 f'not exported -- the index works, those contribute nothing')
        for line in dead_fulltext[:12]:
            print(f'            {line}')
        chk.info('this usually means the config\'s '
                 'find-neurons-fulltext-index-properties lists more than this '
                 'dataset populates')


def check_rois(chk, export, node_csvs_):
    chk.head('ROI names: Meta.roiInfo vs the Neuron CSV columns')
    meta = export / 'Neuprint_Meta.csv'
    if not meta.is_file():
        chk.bad('Neuprint_Meta.csv is present')
        return
    header = header_of(meta)
    try:
        with open(meta, newline='') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            chk.bad('Neuprint_Meta.csv has a data row')
            return
        record = dict(zip(header, rows[1]))
    except Exception as e:                                  # noqa: BLE001
        chk.bad('Neuprint_Meta.csv parses', [str(e)])
        return

    roi_info_field = next((k for k in record if k.split(':', 1)[0] == 'roiInfo'),
                          None)
    if not roi_info_field:
        chk.warn('Neuprint_Meta.csv has no roiInfo column -- ROI check skipped')
        return
    import json
    try:
        roi_info = json.loads(record[roi_info_field])
    except Exception as e:                                  # noqa: BLE001
        chk.bad('Meta.roiInfo parses as JSON', [str(e)])
        return
    meta_rois = set(roi_info)
    chk.info(f'{len(meta_rois)} ROI(s) in Meta.roiInfo')

    # Boolean ROI columns, aggregated across every Neuron CSV batch. They are
    # SPARSE -- each batch only carries the ROIs its bodies touch -- so
    # sampling one file proves nothing.
    csv_rois = set()
    for path, hdr in node_csvs_:
        if 'Neuprint_Neurons' not in str(path) and path.name != 'Neuprint_Neurons.csv':
            continue
        for field in hdr:
            if field.endswith(':boolean'):
                csv_rois.add(field.split(':', 1)[0])
    chk.info(f'{len(csv_rois)} boolean ROI column(s) across the Neuron CSVs')

    only_meta = sorted(meta_rois - csv_rois)
    only_csv = sorted(csv_rois - meta_rois)
    if only_meta:
        chk.bad('every Meta.roiInfo ROI has a boolean column',
                [f'{len(only_meta)} missing: {", ".join(only_meta[:8])}'
                 + (' ...' if len(only_meta) > 8 else '')])
    else:
        chk.ok('every Meta.roiInfo ROI has a boolean column')
    if only_csv:
        chk.warn(f'{len(only_csv)} boolean column(s) not in Meta.roiInfo: '
                 + ', '.join(only_csv[:8]) + (' ...' if len(only_csv) > 8 else ''))


def check_feather_parity(chk, export):
    chk.head('feather vs CSV row counts')
    try:
        import pyarrow.feather as feather
    except ImportError:
        chk.warn('pyarrow is not importable -- feather checks skipped. '
                 'Run via `pixi run check-neuprint-export`.')
        return
    pairs = [
        ('Neuprint_Neurons.feather', sorted((export / 'Neuprint_Neurons').glob('*.csv'))
         or [export / 'Neuprint_Neurons.csv']),
        ('Neuprint_Neuron_Connections.feather',
         [export / 'Neuprint_Neuron_Connections.csv']),
    ]
    for fname, csvs in pairs:
        fpath = export / fname
        csvs = [p for p in csvs if p.is_file()]
        if not fpath.is_file() or not csvs:
            chk.info(f'{fname}: not exported, skipped')
            continue
        n_feather = feather.read_table(fpath).num_rows
        chk.expect_eq(f'{fname} rows match its CSV(s)',
                      csv_rows(csvs), n_feather)


def check_meta_totals(chk, export):
    chk.head('Meta synapse totals')
    meta = export / 'Neuprint_Meta.csv'
    if not meta.is_file():
        return
    header = header_of(meta)
    with open(meta, newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return
    record = {k.split(':', 1)[0]: v for k, v in zip(header, rows[1])}
    syn_csvs = sorted((export / 'Neuprint_Synapses').glob('*.csv'))
    if not syn_csvs:
        chk.info('no Synapse CSVs, skipped')
        return
    n_syn = csv_rows(syn_csvs)
    chk.info(f'{n_syn:,} Synapse rows exported')
    for key in ('totalPreCount', 'totalPostCount'):
        raw = record.get(key, '')
        if not raw:
            continue
        try:
            val = int(float(raw))
        except ValueError:
            chk.warn(f'Meta.{key} is not a number: {raw!r}')
            continue
        if val <= n_syn:
            chk.ok(f'Meta.{key} ({val:,}) does not exceed the exported '
                   f'synapses')
        else:
            chk.bad(f'Meta.{key} does not exceed the exported synapses',
                    [f'{key}={val:,} but only {n_syn:,} Synapse rows exist'])


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('export_dir',
                        help='the <snapshot>/neuprint directory, containing '
                             'Neuprint_*.csv and create-indexes.cypher')
    parser.add_argument('--label-rows', type=int, default=DEFAULT_LABEL_ROWS,
                        metavar='N',
                        help=f'rows per node CSV to scan for labels '
                             f'(default {DEFAULT_LABEL_ROWS:,})')
    parser.add_argument('--full', action='store_true',
                        help='scan every row for labels, not just the first '
                             '--label-rows. Minutes rather than seconds on a '
                             'large export.')
    parser.add_argument('--color', choices=['auto', 'always', 'never'],
                        default='auto')
    args = parser.parse_args()

    export = Path(args.export_dir).resolve()
    if not export.is_dir():
        print(f'ERROR: {export} is not a directory.', file=sys.stderr)
        return 2
    if not any(export.glob('Neuprint_*.csv')):
        print(f'ERROR: {export} contains no Neuprint_*.csv.\n'
              f'       Point this at <snapshot>/neuprint, not <snapshot> or '
              f'<snapshot>/neo4j.', file=sys.stderr)
        return 2

    use_color = {'always': True, 'never': False}.get(args.color)
    chk = Checker(use_color)
    print(f'export:  {export}')

    nodes, rels = node_csvs(export)
    print(f'node CSVs: {len(nodes)}    relationship CSVs: {len(rels)}')
    if not nodes:
        print('ERROR: no node CSVs (none has a :LABEL column).',
              file=sys.stderr)
        return 2

    label_rows = 0 if args.full else args.label_rows
    check_headers(chk, nodes, rels)
    check_id_spaces(chk, nodes, rels)
    check_indexes(chk, export, nodes, label_rows)
    check_rois(chk, export, nodes)
    check_feather_parity(chk, export)
    check_meta_totals(chk, export)

    chk.head('RESULT')
    print(f'  passed: {chk.passed}    failed: {chk.failed}    '
          f'warnings: {chk.warnings}')
    if chk.failed:
        print()
        for msg in chk.failures:
            print(f"  {chk._c('31;1', 'FAILED')}  {msg}")
        print()
        print('  The export has a problem that ingestion will not fix. '
              'Phase 2 would')
        print('  succeed and produce a database with this defect baked in.')
        return 1
    print()
    print('  RESULT: OK -- the export is internally consistent.')
    print('  This does not verify the import itself; run '
          'check-neuprint-snapshot after Phase 2.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
