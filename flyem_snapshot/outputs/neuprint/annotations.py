"""
Business logic for translating arbitrary body annotations (e.g. from DVID/Clio)
into the format neuprint needs (column names, status values, etc.)
"""
import re
import logging

import pandas as pd

from neuclease.util import snakecase_to_camelcase
from neuclease.dvid.keyvalue import DEFAULT_BODY_STATUS_CATEGORIES

from .util import convert_point_cols_to_neo4j_spatial

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
    'Will be merged':           'Anchor',       # noqa
    'Partially traced':         'Anchor',       # noqa
    'Sensory Anchor':           'Anchor',       # noqa
    'Cervical Anchor':          'Anchor',       # noqa
    'Soma Anchor':              'Anchor',       # noqa
    'Examined Soma Anchor':     'Anchor',       # noqa
    'Primary Anchor':           'Anchor',       # noqa

    'Leaves':                   'Traced',       # noqa
    'PRT Orphan':               'Traced',       # noqa
    'Reviewed':                 'Traced',       # noqa
    'Prelim Roughly traced':    'Traced',       # noqa
    'RT Hard to trace':         'Traced',       # noqa
    'RT Orphan':                'Traced',       # noqa
    'Roughly traced':           'Traced',       # noqa
    'Traced in ROI':            'Traced',       # noqa
    'Traced':                   'Traced',       # noqa
    'Finalized':                'Traced',       # noqa
}

# We need to make sure we're handling all status labels that we use in DVID,
# which is generally tracked in the neuclease DEFAULT_BODY_STATUS_CATEGORIES.
assert list(NEUPRINT_STATUSLABEL_TO_STATUS.keys()) == DEFAULT_BODY_STATUS_CATEGORIES


def neuprint_segment_annotations(cfg, ann, convert_points_to_neo4j_spatial=True):
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
    if ann.index.name != 'body':
        if 'body' in ann.columns:
            ann = ann.set_index('body')
        else:
            raise ValueError("Body annotations table must have a 'body' column or index.")

    ann = ann.query('body != 0')

    # Note that dvid/clio neuronjson annotations come with the bodyid column,
    # but annotations loaded from CSV or from dvid point annotations don't
    # have it until we set it here.
    ann['bodyid'] = ann.index

    renames = {c: snakecase_to_camelcase(c.replace(' ', '_'), False) for c in ann.columns}
    renames.update({
        c: c.replace('Position', 'Location')
        for c in renames.keys()
        if 'Position' in c
    })
    renames.update(CLIO_TO_NEUPRINT_PROPERTIES)
    renames.update(cfg.get('annotation-property-names', {}))

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
    statusLabel_dtype = ann['statusLabel'].dtype
    ann['statusLabel'] = ann['statusLabel'].astype('string')

    # Neuprint uses 'simplified' status choices,
    # referring to the original (dvid) status as 'statusLabel'.
    ann['status'] = ann['statusLabel'].replace(NEUPRINT_STATUSLABEL_TO_STATUS)

    # Erase any values which are just "".
    # Better to leave them null.
    ann = ann.replace(["", pd.NA], [None, None])
    ann['statusLabel'] = ann['statusLabel'].astype(statusLabel_dtype)

    # If any columns are completely empty (other than statusLabel), remove them.
    allnull = ann.drop(columns=['statusLabel']).isnull().all(axis=0)
    empty_cols = allnull.loc[allnull].index
    if len(empty_cols) > 0:
        logger.info(f"Deleting empty annotation columns: {empty_cols.tolist()}")
        ann = ann.drop(columns=empty_cols)

    if convert_points_to_neo4j_spatial:
        # Convert '.*location.*' and '.*position.*' columns to neo4j spatial points.
        # FIXME: What about point-annotations which DON'T contain 'location' or 'position' in the name?
        convert_point_cols_to_neo4j_spatial(ann)

    return ann
