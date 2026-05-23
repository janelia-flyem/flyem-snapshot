import re
import logging

import numpy as np
import pandas as pd


# For certain types, we need to ensure that 'long' is used in neo4j, not int.
# Also, in some cases (e.g. 'group'), the column in our pandas DataFrame is
# stored with float dtype because pandas uses NaN for missing values.
# We want to emit actual ints in the CSV.
NEUPRINT_TYPE_OVERRIDES = {
    'bodyId': 'long',
    #'sv': 'long',
    'group': 'long',
    'hemibrainBodyid': 'long',
    'mitoSegment': 'long',
    'size': 'long',
    'pre': 'int',
    'post': 'int',
    'upstream': 'int',
    'downstream': 'int',
    'synweight': 'int',
    # 'halfbrainBody': 'long',
    'positionType': 'string',
    'locationType': 'string',

    # fish2 :Soma properties
    'somaId': 'long',
    'zapbenchId': 'long',
}


def append_neo4j_type_suffixes(df, exclude=(), drop_empty=True):
    """
    Return a renamed DataFrame wholes columns now have
    type suffixes such as ':float'.
    Only rename columns which DON'T ALREADY HAVE a
    colon (:) in the name.
    """
    typed_renames = neo4j_column_names(df, exclude)
    cols = df.columns.tolist()
    if drop_empty:
        cols = [c for c in cols if typed_renames.get(c, None) != ':IGNORE']
    return df[cols].rename(columns=typed_renames)


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
    if series.name in NEUPRINT_TYPE_OVERRIDES:
        return NEUPRINT_TYPE_OVERRIDES[series.name]
    if re.search('position|location', series.name.lower()):
        # The weird srid:9157 means 'cartesian-3d' according to the neo4j docs.
        # https://neo4j.com/docs/cypher-manual/current/values-and-types/spatial/#spatial-values-crs-cartesian
        return 'point{srid:9157}'

    # In most cases, the dtype determines the type suffix
    if isinstance(series.dtype, pd.CategoricalDtype):
        return 'string'

    if series.dtype == 'string[python]':
        return 'string'

    if series.dtype == bool:
        # https://neo4j.com/docs/operations-manual/4.4/tools/neo4j-admin/neo4j-admin-import/#import-tool-header-format-properties
        msg = (
            f"Problem with column '{series.name}':\n"
            "You shouldn't export columns as bool dtype because pandas will not export "
            "lowercase 'true', and neo4j will silently fail to interpret your data correctly.\n"
            "You should convert your data to string, make sure it's lowercase, "
            "and give it an explicit foo:boolean header."
        )
        raise RuntimeError(msg)

    if np.issubdtype(series.dtype, np.integer):
        if series.abs().max() < 2**31:
            return 'int'
        else:
            return 'long'

    if np.issubdtype(series.dtype, np.floating):
        return 'float'

    assert series.dtype == object, \
        f"Unsupported column: {series.name}, {series.dtype}"

    # If dtype is 'object', we have to distinguish between a few cases:
    # - int (with empty rows)
    # - float (with empty rows)
    # - string
    # - list-of-strings
    # - list-of-float

    valid = series.dropna()
    if len(valid) == 0:
        logger = logging.getLogger(__name__)
        logger.warning(f"No data to infer correct neo4j type for column: {series.name}")
        return 'IGNORE'

    if valid.map(lambda v: np.issubdtype(type(v), np.integer)).all():
        return 'long'

    if valid.map(lambda v: np.issubdtype(type(v), np.floating)).all():
        return 'float'

    if valid.map(lambda s: isinstance(s, str)).all():
        return 'string'

    if (valid.map(type) == bool).all():
        return 'boolean'

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


def prepare_int_cols_for_export(df):
    """
    Works in-place.

    Some columns might have the wrong dtype due to the way
    pandas expresses missing values with NaN.
    To ensure that the non-missing values will be written
    correctly in the CSV, we first convert to the correct dtype
    (if the dataframe contains no missing entries) or to 'object'
    dtype (if necessary) and cast the available values to the
    correct type.

    Args:
        DataFrame in which the column names already have neo4j
        type suffixes (e.g. 'foo:long').
        Any columns without a suffix are left unchanged.
    """
    neo4j_to_numpy = {
        'long': np.int64,
        'int': np.int32
    }

    for c, pandas_dtype in df.dtypes.items():
        if ':' not in c:
            continue

        neo4j_type = c.split(':')[1]
        export_type = neo4j_to_numpy.get(neo4j_type, None)
        if not export_type or export_type == pandas_dtype:
            continue

        # For columns which should be exported as numerics,
        # coerce incompatible values to null (but warn about them).
        if np.issubdtype(export_type, np.number):
            nullcount = df[c].isnull().sum()
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if df[c].isnull().sum() > nullcount:
                # (The current function is running in a subprocess,
                # so we can't use the global logger variable.)
                logging.getLogger(__name__).warning(
                    f"Segment annotation column {c} has values which cannot be "
                    "converted to numeric types. Setting those values to null."
                )

        if df[c].notnull().all():
            df[c] = df[c].astype(export_type)
        else:
            # Convert column to dtype 'object' (instead of float)
            # so that we can replace floats with ints while
            # allowing for missing values.
            df[c] = df[c].astype(object)
            nn = df[c].notnull()
            df.loc[nn, c] = df.loc[nn, c].astype(export_type)


def convert_point_cols_to_neo4j_spatial(df):
    """
    Convert any columns whose name contains 'location' or 'position'
    to the string format required for neo4j spatial points in CSV files,

    e.g. "{x: 123, y: 456, z: 789}"

    Works in-place.
    """
    logger = logging.getLogger(__name__)

    # Points must be converted to neo4j spatial points.
    # FIXME: What about point-annotations which DON'T contain 'location' or 'position' in the name?
    for col in df.columns:
        if not re.search('position|location', col.lower()):
            continue

        # FIXME: Is there a better way to catch positionType instead of hard-coding this?
        if col in ('positionType', 'locationType'):
            continue

        df[col] = df[col].astype(object).where(df[col].notnull(), None)
        count = df[col].notnull().sum()
        df[col] = df[col].map(_convert_point)
        newcount = df[col].notnull().sum()
        if newcount != count:
            logger.warning(
                f"Point annotation column {col} has {count - newcount}"
                " values which could not be processed as points."
            )

        valid = df[col].notnull()
        df.loc[~valid, col] = None
        df.loc[valid, col] = [
            f"{{x:{x}, y:{y}, z:{z}}}"
            for (x,y,z) in df.loc[valid, col].values
        ]


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
