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
}


def append_neo4j_type_suffixes(cfg, df, exclude=()):
    """
    Return a renamed DataFrame wholes columns now have
    type suffixes such as ':float'.
    Only rename columns which DON'T ALREADY HAVE a
    colon (:) in the name.
    """
    typed_renames = neo4j_column_names(cfg, df, exclude)
    return df.rename(columns=typed_renames)


def neo4j_column_names(cfg, df, exclude=()):
    """
    Determine type suffixes such as ':float' for columns
    which DON'T ALREADY HAVE a colon (:) in the name.
    Returns a dict of columns to rename.
    """
    if not cfg:
        point_properties = []
    else:
        point_properties = [
            pa['property-name']
            for pa in cfg['neuprint']['point-annotations']
        ]

    typed_renames = {}
    for col, col_dtype in df.dtypes.items():
        if col in exclude or ':' in col:
            continue
        if 'location' in col.lower() or 'position' in col.lower() or col in point_properties:
            # The weird srid:9157 means 'cartesian-3d' according to the neo4j docs.
            # https://neo4j.com/docs/cypher-manual/current/values-and-types/spatial/#spatial-values-crs-cartesian
            typed_renames[col] = col + ':point{srid:9157}'
        elif col in NEUPRINT_TYPE_OVERRIDES:
            typed_renames[col] = col + ':' + NEUPRINT_TYPE_OVERRIDES[col]
        elif isinstance(col_dtype, pd.CategoricalDtype):
            typed_renames[col] = col + ':string'
        elif col_dtype == bool:
            typed_renames[col] = col + ':boolean'
        elif np.issubdtype(col_dtype, np.integer):
            typed_renames[col] = col + ':int'
        elif np.issubdtype(col_dtype, np.floating):
            typed_renames[col] = col + ':float'
        elif col_dtype == object:
            # Assume that 'object' means str
            if df[col].dropna().map(lambda s: isinstance(s, str)).all():
                typed_renames[col] = col + ':string'
            else:
                msg = f"Column {col} has object dtype but contains non-string values."
                raise RuntimeError(msg)
    return typed_renames
