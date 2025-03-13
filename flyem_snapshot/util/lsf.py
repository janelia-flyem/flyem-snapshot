"""
Utility functions for logging details related to an LSF job.
"""
import os
import time
import platform
from datetime import datetime
from subprocess import check_output


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

