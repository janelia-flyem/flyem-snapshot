import logging
import numpy as np
import pandas as pd

# For certain types, we need to ensure that 'long' is used in neo4j, not int.
# Also, in some cases (e.g. 'group'), the column in our pandas DataFrame is
# stored with float dtype because pandas uses NaN for missing values.
# We want to emit actual ints in the CSV.
NEUPRINT_TYPE_OVERRIDES = {
    'bodyId': 'long',
    'group': 'long',
    'hemibrainBodyid': 'long',
    'size': 'long',
    'pre': 'int',
    'post': 'int',
    'upstream': 'int',
    'downstream': 'int',
    'synweight': 'int',
    # 'halfbrainBody': 'long',
    'voxelSize': 'float[]',
    'primaryRois': 'string[]',
    'superLevelRois': 'string[]',
    'nonHierarchicalROIs': 'string[]',
}


def append_neo4j_type_suffixes(df, exclude=()):
    """
    Return a renamed DataFrame wholes columns now have
    type suffixes such as ':float'.
    Only rename columns which DON'T ALREADY HAVE a
    colon (:) in the name.
    """
    typed_renames = neo4j_column_names(df, exclude)
    return df.rename(columns=typed_renames)


def neo4j_column_names(df, exclude=()):
    """
    Determine type suffixes such as ':float' for columns
    which DON'T ALREADY HAVE a colon (:) in the name.
    Returns a dict of columns to rename.
    """
    typed_renames = {}
    for col, series in df.items():
        if col in exclude or ':' in col:
            continue
        suffix = neo4j_type_suffix(series)
        typed_renames[col] = f'{col}:{suffix}'
    return typed_renames


def neo4j_type_suffix(series):
    # In some cases, the name determines the dtype suffix
    # We check these first because these names override the dtype-based rules.
    if 'location' in series.name.lower() or 'position' in series.name.lower():
        # The weird srid:9157 means 'cartesian-3d' according to the neo4j docs.
        # https://neo4j.com/docs/cypher-manual/current/values-and-types/spatial/#spatial-values-crs-cartesian
        return 'point{srid:9157}'
    if series.name in NEUPRINT_TYPE_OVERRIDES:
        return NEUPRINT_TYPE_OVERRIDES[series.name]

    # In most cases, the dtype determines the type suffix
    if isinstance(series.dtype, pd.CategoricalDtype):
        return 'string'
    if series.dtype == bool:
        return 'boolean'
    if np.issubdtype(series.dtype, np.integer):
        return 'int'
    if np.issubdtype(series.dtype, np.floating):
        return 'float'

    assert series.dtype == object, \
        f"Unsupported column: {series.name}, {series.dtype}"

    # If dtype is 'object', we have to distinguish between a few cases:
    # - string
    # - list-of-strings
    # - list-of-float

    valid = series.dropna()
    if len(valid) == 0:
        logger = logging.getLogger(__name__)
        logger.warning(f"No data to infer correct neo4j type for column: {series.name}")
        return ':IGNORE'

    if valid.map(lambda s: isinstance(s, str)).all():
        return 'string'

    if not valid.map(lambda x: isinstance(x, (list, tuple))).all():
        raise RuntimeError(
            f"Column {series.name} has object "
            "dtype but contains non-string, non-list values."
        )

    if len(valid.iloc[0]) == 0:
        return 'string[]'

    if isinstance(valid.iloc[0][0], str):
        return 'string[]'
    if np.issubdtype(type(valid.iloc[0][0]), np.floating):
        return 'float[]'
    if np.issubdtype(type(valid.iloc[0][0]), np.integer):
        return 'int[]'

    raise RuntimeError(
        f"Column {series.name} contains lists with unsupported "
        f"list entries (e.g. {valid.iloc[0][0]})"
    )