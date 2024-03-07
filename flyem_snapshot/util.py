import os
import time
import platform

from zlib import adler32
from datetime import datetime
from subprocess import check_output
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from bokeh.plotting import output_file, save as bokeh_save
from bokeh.io import export_png


def rm_f(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def cache_dataframe(df, path):
    """
    A wrapper around feather.write_feather with an extra check to ensure
    that NaN data will not change after a round-trip of write/read.
    Raises an error if the dataframe doesn't conform.
    """
    # Standardize on None as the null value (instead of NaN or "").
    # https://stackoverflow.com/questions/46283312/how-to-proceed-with-none-value-in-pandas-fillna
    FAKENULL = '__cache_dataframe_nullval__'
    bad_columns = []
    for col, dtype in df.dtypes.items():
        if dtype != object:
            continue

        if (df[col].replace([np.nan], [FAKENULL]) != df[col].replace([None], FAKENULL)).any():
            bad_columns.append(col)
    if bad_columns:
        msg = (
            "DataFrame cannot be cached because it contains column(s) of 'object' dtype "
            "that contain np.nan. write_feather() will convert them to None, giving them "
            "a different checksum.  Convert them to None yourself using Series.replace([np.nan], [None])) "
            "before returning the dataframe, so it can be cached.\n"
            f"The nan-containing columns are: [{bad_columns}]"
        )
        raise RuntimeError(msg)
    feather.write_feather(df, path)


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
        df[col].replace([np.nan, None], inplace=True)


def dataframe_checksum(df):
    checksums = []
    checksums.append(series_checksum(df.index))
    for c in df.columns:
        checksums.append(series_checksum(df[c]))
    return adler32(np.array(checksums))


def series_checksum(s):
    if s.dtype == 'category':
        return adler32(s.cat.codes.values)
    if s.dtype == 'object':
        return adler32(s.astype(str).values.astype(str))
    else:
        return adler32(s.values)


def checksum(data):
    if data is None:
        return 999999999
    if isinstance(data, str):
        return adler32(data.encode('utf-8'))
    if isinstance(data, (int, float)) or np.issubdtype(type(data), np.number):
        return adler32(str(data).encode('utf-8'))
    if isinstance(data, np.ndarray):
        return adler32(data)
    if isinstance(data, pd.DataFrame):
        return dataframe_checksum(data)
    if isinstance(data, pd.Series):
        return dataframe_checksum(data.to_frame())
    if isinstance(data, Sequence):
        if all(isinstance(d, (int, float)) for d in data):
            return adler32(np.array(data))
        return adler32(np.array([checksum(e) for e in data]))
    if isinstance(data, Mapping):
        csums = []
        for k,v in sorted(data.items()):
            csums.append(checksum(k))
            csums.append(checksum(v))
        return adler32(np.array(csums))

    # Doesn't match any supported types, but we can see if the object
    # can be checksummed directly (i.e. supports buffer protocol)
    return adler32(data)


def export_bokeh(p, filename, title):
    path = os.path.splitext(filename)[0]
    png_path = f"png/{path}.png"
    rm_f(png_path)
    export_png(p, filename=png_path)

    html_path = f"html/{path}.html"
    rm_f(html_path)
    output_file(filename=html_path, title=title)
    bokeh_save(p)


def log_lsf_details(logger):
    logger.info(f"Running on {platform.uname().node}")
    job_id = os.environ.get("LSB_JOBID", None)
    if not job_id:
        return
    rtm_url = construct_rtm_url(job_id)
    logger.info(f"LSB_JOBID is {job_id}")
    logger.info(f"RTM graphs: {rtm_url}")


def construct_rtm_url(job_id=None, tab='jobgraph'):
    """
    Construct a URL that can be used to browse a job's host
    graphs on Janelia's RTM web server.
    """
    assert tab in ('hostgraph', 'jobgraph')
    job_id = job_id or os.environ.get("LSB_JOBID", None)
    if not job_id:
        return None

    submit_time = get_job_submit_time(job_id)
    submit_timestamp = int(submit_time.timestamp())
    rtm_url = (
        "http://lsf-rtm.int.janelia.org/cacti/plugins/grid/grid_bjobs.php"
        "?action=viewjob"
        f"&tab={tab}"
        "&clusterid=1"
        "&indexid=0"
        f"&jobid={job_id}"
        f"&submit_time={submit_timestamp}"
    )
    return rtm_url


def get_job_submit_time(job_id=None):
    """
    Return the job's submit_time as a datetime object.
    """
    job_id = job_id or os.environ["LSB_JOBID"]
    bjobs_output = check_output(f'bjobs -X -noheader -o SUBMIT_TIME {job_id}', shell=True).strip().decode()
    # Example:
    # Sep  6 13:10:09 2017
    submit_time = datetime.strptime(f"{bjobs_output} {time.localtime().tm_zone}", "%b %d %H:%M:%S %Y %Z")
    return submit_time


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
