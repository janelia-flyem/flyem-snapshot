"""
Business logic for translating arbitrary body annotations (e.g. from DVID/Clio)
into the format neuprint needs (column names, status values, etc.)
"""
import re
import logging

import pandas as pd

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
}

# Note:
#   Any 'statusLabel' (DVID status) that isn't
#   listed here will appear in neuprint unchanged.
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

    '0.5assign':                'Assign',       # noqa

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
    'RT Hard to trace':         'Traced',       # noqa
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
    rn = pd.Series(renames)
    for _v, s in rn.groupby(rn.values):
        if len(s) == 1:
            continue
        ann[s.index[0]] = ann[s.index].bfill(axis=1).iloc[:, 0]
        ann = ann.drop(columns=s.index[1:])
        for k in s.index[1:]:
            del renames[k]

    ann = ann[[*renames.keys()]]
    ann = ann.rename(columns=renames)
    logger.info(f"Annotation columns after renaming: {ann.columns.tolist()}")

    # Drop categorical dtype for this column before using replace()
    ann['statusLabel'] = ann['statusLabel'].astype('string')

    # Erase any values which are just "".
    # Better to leave them null.
    ann = ann.replace(["", pd.NA], [None, None])

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

        count = ann[col].notnull().sum()
        ann[col] = ann[col].map(_convert_point)
        newcount = ann[col].notnull().sum()
        if newcount != count:
            logger.warning(
                f"Point annotation column {col} has {count - newcount}"
                " values which could not be processed as points."
            )

        valid = ann[col].notnull()
        ann.loc[~valid, col] = None
        ann.loc[valid, col] = [
            f"{{x:{x}, y:{y}, z:{z}}}"
            for (x,y,z) in ann.loc[valid, col].values
        ]

    return ann


def _convert_point(p):
    """
    Convert the given entity from various possible forms of point data into to a standard form.

    The input may be one of the following:

    - A string like '123, 456, 789'
    - A string like '[123, 456, 789]'
    - A string like '{x: 123, y: 456, z: 789}'
    - A list or array of three ints or floats
    - a neo4j spatial coordinate like this:
        {
            'coordinates': [24481, 36044, 67070],
            'crs': {'name': 'cartesian-3d',
            'properties': {'href': 'http://spatialreference.org/ref/sr-org/9157/ogcwkt/',
            'type': 'ogcwkt'},
            'srid': 9157,
            'type': 'link'},
            'type': 'Point'
        }

    The output form is just a list: [x, y, z].
    If the input is None or cannot be interpreted as a point, then None is returned.
    """
    match p:
        case None:
            return None

        case {'coordinates': coords} if len(coords) == 3:
            return coords

        case {'x': x, 'y': y, 'z': z}:
            return [x, y, z]

        case [x, y, z] if all(isinstance(v, int) for v in (x, y, z)):
            return [x, y, z]

        case [x, y, z]:
            try:
                return [float(x), float(y), float(z)]
            except (ValueError, TypeError):
                return None

        case str() as s:
            s = s.strip('[]{} ')

            try:
                pattern = (
                    r'(?:x:\s*)?(-?\d+\.?\d*)\s*,\s*'
                    r'(?:y:\s*)?(-?\d+\.?\d*)\s*,\s*'
                    r'(?:z:\s*)?(-?\d+\.?\d*)\s*$'
                )
                if match := re.match(pattern, s):
                    return [eval(x) for x in match.groups()]
            except (ValueError, TypeError):
                return None

        case _:
            return None
