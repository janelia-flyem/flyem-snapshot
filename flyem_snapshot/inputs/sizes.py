import os
import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer
from neuclease.dvid.labelmap import fetch_sizes, fetch_mutations, compute_affected_bodies

from google.cloud import storage

from ..util.util import upload_file_to_gcs

logger = logging.getLogger(__name__)

BodySizesSchema = {
    "type": "object",
    "description":
        "An on-disk cache for body sizes (voxel counts), obtained from a prior uuid.\n"
        "We'll have to fetch sizes only for those bodies which have been\n"
        "modified since this cache was created.\n",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "load-sizes": {
            "description": "If false, don't load sizes at all.",
            "type": "boolean",
            "default": False
        },
        "cache-file": {
            "description": "Feather file with columns 'body' and 'size'",
            "type": "string",
            "default": "",
        },
        "cache-uuid": {
            "description":
                "The locked uuid from which these sizes were obtained.\n"
                "If you genereted these sizes from an unlocked node, then provide the UUID of the PARENT node.\n",
            "type": "string",
            "default": "",
        },
        "processes": {
            "description":
                "How many processes should be used to load body sizes?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
        "gc-project": {
            "description":
                "Google Cloud project.\n",
            "type": "string",
            "default": "FlyEM-Private"
        },
        "gcs-bucket": {
            "description":
                "Google Cloud Storage bucket to export files to.\n",
            "type": "string",
            "default": "flyem-snapshots"
        },
    }
}


@PrefixFilter.with_context('body-sizes')
def load_body_sizes(cfg, pointlabeler, body_lists, snapshot_tag):
    """
    Load/export the sizes (voxel count) of the given bodies (after deduplication).
    It's up to the caller to decide which set of bodies to process.
    But for connectivity snapshots, we typically fetch sizes
    ONLY for bodies which have at least one synapse.
    """
    if not cfg['load-sizes']:
        logger.info("Body sizes will not be emitted due to load-sizes: false")
        return None

    # Initialize Google cloud client and select project
    client = storage.Client()
    if cfg['gc-project'] and cfg['gcs-bucket']:
        client.project = cfg['gc-project']
    else:
        cfg['gcs-bucket'] = ""

    dvidseg = pointlabeler and pointlabeler.dvidseg
    cache_file = cfg['cache-file']
    cache_uuid = cfg['cache-uuid']

    if not cache_file and not dvidseg:
        logger.info("No body size info provided. Body sizes will not be emitted.")
        return None

    if not cache_file:
        return _fetch_all_body_sizes(dvidseg, body_lists, snapshot_tag, cfg['processes'], cfg)

    cached_sizes = feather.read_feather(cache_file).astype({'body': np.int64}).set_index('body')['size']
    if bool(dvidseg) != bool(cache_uuid):
        logger.error("body sizes cache is not specified properly; disregarding it.")
        cache_file = cache_uuid = ""

    if not dvidseg and not cache_uuid:
        logger.info("Using cached body sizes without updating from DVID")
        return cached_sizes

    outofdate_bodies = _determine_out_of_date_bodies(dvidseg, cache_uuid)
    if len(outofdate_bodies) == 0:
        return cached_sizes

    with Timer("Fetching non-cached neuron sizes", logger):
        new_sizes = fetch_sizes(*dvidseg, outofdate_bodies, processes=cfg['processes'])
        new_sizes.index = new_sizes.index.astype(np.int64)

    combined_sizes = (
        new_sizes.to_frame()
        .combine_first(cached_sizes.to_frame())['size']
        .astype(np.int64)
    )

    os.makedirs('tables', exist_ok=True)
    fname = f'tables/body-size-cache-{snapshot_tag}.feather'
    feather.write_feather(
        combined_sizes.reset_index(),
        fname)
    upload_file_to_gcs(cfg['gcs-bucket'], fname, f"{snapshot_tag}/{fname}")

    return combined_sizes


def _fetch_all_body_sizes(dvidseg, body_lists, snapshot_tag, processes, cfg):
    # No cache: Gotta fetch them all from DVID
    # (Takes ~1 hour for the full CNS -- would be worse if we had to also fetch sizes of NON-synaptic bodies.)
    logger.info("No body sizes cache provided.")
    bodies = pd.unique(np.concatenate(body_lists))
    with Timer("Loading all neuron sizes from DVID", logger):
        sizes = fetch_sizes(
            *dvidseg,
            bodies,
            processes=processes
        )

    os.makedirs('tables', exist_ok=True)
    fname = f'tables/body-size-cache-{snapshot_tag}.feather'
    feather.write_feather(
        sizes.reset_index(),
        fname)
    upload_file_to_gcs(cfg['gcs-bucket'], fname, f"{snapshot_tag}/{fname}")

    return sizes


def _determine_out_of_date_bodies(dvidseg, cache_uuid):
    # Note: Using 1 parenthesis and 1 bracket to indicate
    #       exclusive/inclusive mutation range: (a,b]
    server, snapshot_uuid, instance = dvidseg
    delta_range = f"({cache_uuid}, {snapshot_uuid}]"
    muts = fetch_mutations(server, delta_range, instance)
    effects = compute_affected_bodies(muts)
    outofdate_bodies = np.concatenate((effects.changed_bodies, effects.new_bodies))
    outofdate_bodies = pd.unique(outofdate_bodies)
    return outofdate_bodies
