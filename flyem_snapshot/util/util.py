from google.cloud import storage
import numpy as np
import pandas as pd


def replace_object_nan_with_none(df):
    """
    For columns in the given DataFrame with dtype 'object',
    ensure that None is used as the null value rather than np.nan.
    This ensures perfect round-trip feather serialization, for instance.

    Works in-place.
    """
    for col, dtype in df.dtypes.items():
        if dtype != object:
            continue
        df[col] = df[col].replace([np.nan], [None])


def restrict_synapses_to_roi(roiset, roi, point_df, partner_df):
    """
    Drop synapses from point_df and partner_df if they don't fall
    within the given roi, as found in the given roiset column.
    Partners are dropped iff the 'post' side doesn't fall in the roi,
    and then points that end up with no partners at all are dropped.

    Args:
        roiset:
            Name of a roiset column.  If roiset is None, then
            the input tables are returned unmodified.
        roi:
            Name of a roi that can be found in that roiset column.
            If roi is None, then we drop points in the <unspecified> and keep all others.
        point_df:
            Synapse point table (indexed by point_id)
        partner_df:
            Synapse partner table  (pre_id, post_id)

    Returns:
        point_df, partner_df
    """
    if not roiset:
        return point_df, partner_df

    # We keep *connections* that are in-bounds.
    # In neuprint, this is defined by the 'post' side.
    # On the edge, there can be 'pre' points that are out-of-bounds but
    # preserved here because they are partnered to an in-bounds 'post' point.
    if roi:
        inbounds_partners = (partner_df[roiset] == roi)
    else:
        inbounds_partners = (partner_df[roiset] != "<unspecified>")
    partner_df = partner_df.loc[inbounds_partners]

    # Keep the points which are still referenced in partner_df
    valid_ids = pd.concat(
        (
            partner_df['pre_id'].drop_duplicates().rename('point_id'),
            partner_df['post_id'].drop_duplicates().rename('point_id')
        ),
        ignore_index=True
    )
    point_df = point_df.loc[point_df.index.isin(valid_ids)]

    return point_df, partner_df


def upload_file_to_gcs(bucket_name, source, destination):
    """ Upload a file to Google Cloud Storage
        Keyword arguments:
          bucket_name: name of the GCS bucket
          source: local path to the file to upload
          destination: GCS destination
        Returns:
            None
    """
    if not bucket_name:
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_filename(source)
