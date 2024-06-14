"""
Business logic for translating arbitrary body annotations (e.g. from DVID/Clio)
into the format neuprint needs (column names, status values, etc.)
"""
import re
import logging

from neuclease.util import snakecase_to_camelcase

logger = logging.getLogger(__name__)

# For most fields, we formulaically convert from snake_case to camelCase,
# but for some fields the terminology isn't translated by that formula.
# This list provides explicit translations for the special cases.
# Also, to exclude a DVID/clio body annotation field from neuprint entirely,
# list it here and map it to "".
CLIO_TO_NEUPRINT_PROPERTIES = {
    'bodyid': 'bodyId',
    'hemibrain_bodyid': 'hemibrainBodyId',

    # Note:
    #   The DVID 'status' is named 'statusLabel' in neuprint.
    #   Then neuprint creates a new 'status' property which contains
    #   a "simplified" translation of the statusLabel values.
    #   (See mapping below.)
    'status': 'statusLabel',

    # Make sure these never appear in neuprint.
    'last_modified_by': '',
    'old_bodyids': '',
    'old_type': '',
    'reviewer': '',
    'to_review': '',
    'typing_notes': '',
    'user': '',
    'notes': '',
    'halfbrain_body': '',
    'halfbrainBody': '',
    'group_old': '',
    'confidence': '',

    # Hand-edited positions.
    # Should these be discarded?
    'position': 'location',
    'position_type': 'locationType',

    # These generally won't be sourced from Clio anyway;
    # they should be sourced from the appropriate DVID annotation instance.
    'soma_position': 'somaLocation',
    'tosoma_position': 'tosomaLocation',
    'root_position': 'rootLocation',

    # hemibrain stuff
    'primary neurite': 'primaryNeurite',
    'cell body fiber': 'cellBodyFiber',
    'synonym': 'synonyms',  # Wasn't in the original hemibrain release; now we add it for compatibility with MANC

    # hemibrain stuff to exclude
    'body ID': '',
    'comment': '',
    'naming user': '',
    '0.5 status': '',
    '0.2 status': '',
    'assigned': '',
    'tips sparsely traced status': '',
    'property': '',
    'location': '',
    'cross midline': '',
    'major input': '',
    'major output': '',
}

# Note any 'statusLabel' (DVID status) that isn't
# listed here will appear in neuprint unchanged.
NEUPRINT_STATUSLABEL_TO_STATUS = {
    'Unimportant':              'Unimportant',  # noqa
    'Glia':                     'Glia',         # noqa
    'Hard to trace':            'Orphan',       # noqa
    'Orphan-artifact':          'Orphan',       # noqa
    'Orphan':                   'Orphan',       # noqa
    'Orphan hotknife':          'Orphan',       # noqa

    'Putative Leaves':          '',             # noqa
    'Out of scope':             '',             # noqa
    'Not examined':             '',             # noqa
    '':                         '',             # noqa

    '0.5assign':                '0.5assign',    # noqa

    'Anchor':                   'Anchor',       # noqa
    'Cleaved Anchor':           'Anchor',       # noqa
    'Sensory Anchor':           'Anchor',       # noqa
    'Cervical Anchor':          'Anchor',       # noqa
    'Soma Anchor':              'Anchor',       # noqa
    'Primary Anchor':           'Anchor',       # noqa
    'Partially traced':         'Anchor',       # noqa

    'Leaves':                   'Traced',       # noqa
    'PRT Orphan':               'Traced',       # noqa
    'Prelim Roughly traced':    'Traced',       # noqa
    'RT Orphan':                'Traced',       # noqa
    'Roughly traced':           'Traced',       # noqa
    'Traced in ROI':            'Traced',       # noqa
    'Traced':                   'Traced',       # noqa
    'Finalized':                'Traced',       # noqa
}


def neuprint_segment_annotations(cfg, ann):
    """
    Translate input body annotations (e.g. clio annotations)
    to neuprint terminology and values.

    - Rename columns to match neuprint conventions
    - Drop columns which are meant to be excluded.
    - Translate status values to the reduced neuprint set.
    - Drop body 0 (if present)
    - Drop empty strings (replace with null)
    - Translate 'location' and 'position' [x,y,z] lists with neo4j spatial points.
    """
    ann = ann.query('body != 0')

    renames = {c: snakecase_to_camelcase(c.replace(' ', '_'), False) for c in ann.columns}
    renames.update({
        c: c.replace('Position', 'Location')
        for c in renames.keys()
        if 'Position' in c
    })
    renames.update(CLIO_TO_NEUPRINT_PROPERTIES)
    renames.update(cfg['annotation-property-names'])

    # Drop the columns that map to "", and rename the rest.
    renames = {k:v for k,v in renames.items() if (k in ann) and v}

    # Due to source data problems, it's possible that
    # renaming columns introduces DUPLICATE columns.
    # We will consolidate them before deleting the 'duplicates'.
    for k, v in list(renames.items()):
        if (k != v) and (k in ann) and (v in ann):
            ann[k].update(ann[v])
            del ann[v]
            del renames[v]

    ann = ann[[*renames.keys()]]
    ann = ann.rename(columns=renames)
    logger.info(f"Annotation columns after renaming: {ann.columns.tolist()}")

    # Erase any values which are just "".
    # Better to leave them null.
    ann = ann.replace('', None)

    # If any columns are completely empty, remove them.
    allnull = ann.isnull().all(axis=0)
    empty_cols = allnull.loc[allnull].index
    if len(empty_cols) > 0:
        logger.info(f"Deleting empty annotation columns: {empty_cols.tolist()}")
        ann = ann.drop(columns=empty_cols)

    # Neuprint uses 'simplified' status choices,
    # referring to the original (dvid) status as 'statusLabel'.
    ann['status'] = ann['statusLabel'].replace(NEUPRINT_STATUSLABEL_TO_STATUS)

    # Points must be converted to neo4j spatial points.
    # FIXME: What about point-annotations which DON'T contain 'location' or 'position' in the name?
    for col in ann.columns:
        if not re.search('position|location', col.lower()):
            continue

        # FIXME: Is there a better way to catch positionType instead of hard-coding this?
        if col in ('positionType', 'locationType'):
            continue

        ispoint = ann[col].map(lambda x: hasattr(x, '__len__') and len(x) == 3)
        if (ann[col].notnull() & ~ispoint).any():
            ann[col] = ann[col].map(_convert_point)
            ispoint = ann[col].map(lambda x: hasattr(x, '__len__') and len(x) == 3)
            if (ann[col].notnull() & ~ispoint).any():
                # Even after conversion, there are _still_ bad entries!
                logger.warning(f"Annotation column {col} has non-null values that aren't points. Ignoring those items.")

        valid = ann[col].notnull() & ispoint
        ann.loc[~valid, col] = None
        ann.loc[valid, col] = [
            f"{{x:{x}, y:{y}, z:{z}}}"
            for (x,y,z) in ann.loc[valid, col].values
        ]

    return ann


def _convert_point(x):
    # Try converting strings like '123, 456, 789' to lists
    if not isinstance(x, str) or ',' not in x:
        return x
    try:
        # In clio, points might be strings, such as:
        # - "123, 456, 789"
        # - "[123, 456, 789]"
        p = eval(x)
        if len(p) == 3:
            return list(p)
        return x
    except Exception:
        return x
