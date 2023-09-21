"""
Business logic for translating arbitrary body annotations (e.g. from DVID/Clio)
into the format neuprint needs (column names, status values, etc.)
"""
from neuclease.util import snakecase_to_camelcase

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
    'reviewer': '',
    'to_review': '',
    'typing_notes': '',
    'user': '',
    'notes': '',
    'halfbrainBody': '',

    # These generally won't be sourced from Clio anyway;
    # they should be sourced from the appropriate DVID annotation instance.
    'soma_position': 'somaLocation',
    'tosoma_position': 'tosomaLocation',
    'root_position': 'rootLocation',
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
    renames.update({c: c.replace('Position', 'Location') for c in renames})
    renames.update(CLIO_TO_NEUPRINT_PROPERTIES)
    renames.update(cfg['annotation-property-names'])

    # Drop the columns that map to "", and rename the rest.
    renames = {k:v for k,v in renames.items() if (k in ann) and v}
    ann = ann[[*renames.keys()]]
    ann = ann.rename(columns=renames)

    # Erase any values which are just "".
    # Better to leave them null.
    ann = ann.replace('', None)

    # Neuprint uses 'simplified' status choices,
    # referring to the original (dvid) status as 'statusLabel'.
    ann['status'] = ann['statusLabel'].replace(NEUPRINT_STATUSLABEL_TO_STATUS)

    # Points must be converted to neo4j spatial points.
    # FIXME: What about point-annotations which DON'T contain 'location' or 'position' in the name?
    for col in ann.columns:
        if 'location' in col.lower() or 'position' in col.lower():
            valid = ann[col].notnull()
            ann.loc[valid, col] = [
                f"{{x:{x}, y:{y}, z:{z}}}"
                for (x,y,z) in ann.loc[valid, col].values
            ]

    return ann
