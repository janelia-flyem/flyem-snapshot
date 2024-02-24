import os
import hashlib
import platform
import time
from datetime import datetime
from subprocess import check_output

import pandas as pd
from bokeh.plotting import output_file, save as bokeh_save
from bokeh.io import export_png


def rm_f(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def det_hash(s, nbytes=4):
    """
    Deterministic hash of a string, using sha1 but
    only taking the last N bytes to create an int.
    Note:
        Python's builtin hash() is NOT deterministic across
        interpreter startup or from one process to the next
        (e.g. with multiprocessing).
    """
    return int.from_bytes(hashlib.sha1(s.encode('utf-8')).digest()[-nbytes:], 'little')


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
