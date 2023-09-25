import logging

import numpy as np
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer
from neuclease.dvid.labelmap import fetch_sizes, fetch_mutations, compute_affected_bodies

logger = logging.getLogger(__name__)

BodySizesSchema = {
    "type": "object",
    "description":
        "An on-disk cache for body sizes, obtained from a prior uuid.\n"
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
            "description": "The locked uuid from which these sizes were obtained.\n",
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
    }
}


@PrefixFilter.with_context('body-sizes')
def load_body_sizes(cfg, dvid_seg, df, snapshot_tag):
    """
    Load/export the sizes of all bodies listed in df
    (in either the index or the 'body' column).
    It's up to the caller to decide which set of bodies to process.
    But for connectivity snapshots, we typically fetch sizes
    ONLY for bodies which have at least one synapse.
    """
    if not cfg['load-sizes']:
        logger.info("Body sizes will not be emitted due to load-sizes: false")
        return None

    cache_file = cfg['cache-file']
    cache_uuid = cfg['cache-uuid']
    if not cache_file and not dvid_seg:
        logger.info("No body size info provided. Body sizes will not be emitted.")
        return None

    if df.index.name == 'body':
        bodies = df.index.drop_duplicates().values
    elif 'body' in df.columns:
        bodies = df['body'].unique()

    if not cache_file:
        # No cache: Gotta fetch them all from DVID
        # (Takes ~1 hour for the full CNS -- would be worse if we had to also fetch sizes of NON-synaptic bodies.)
        logger.info("No body sizes cache provided.")
        with Timer("Loading all neuron sizes from DVID", logger):
            sizes = fetch_sizes(
                *dvid_seg,
                bodies,
                processes=cfg['processes']
            )

        feather.write_feather(
            sizes.reset_index(),
            f'tables/body-size-cache-{snapshot_tag}.feather')
        return sizes

    cached_sizes = feather.read_feather(cache_file).set_index('body')['size']
    if bool(dvid_seg) != bool(cache_uuid):
        logger.error("body sizes cache is not specified properly; disregarding it.")
        cache_file = cache_uuid = ""

    if not dvid_seg and not cache_uuid:
        logger.info("Using cached body sizes without updating from DVID")
        return cached_sizes

    # Note: Using 1 parenthesis and 1 bracket to indicate
    #       exclusive/inclusive mutation range: (a,b]
    server, snapshot_uuid, instance = dvid_seg
    delta_range = f"({cache_uuid}, {snapshot_uuid}]"
    muts = fetch_mutations(server, delta_range, instance)
    effects = compute_affected_bodies(muts)
    outofdate_bodies = np.concatenate((effects.changed_bodies, effects.new_bodies))
    if len(outofdate_bodies) == 0:
        return cached_sizes

    with Timer("Fetching non-cached neuron sizes", logger):
        new_sizes = fetch_sizes(*dvid_seg, outofdate_bodies, processes=cfg['processes'])

    combined_sizes = (
        new_sizes.to_frame()
        .combine_first(cached_sizes.to_frame())['size']
        .astype(np.int64)
    )
    return combined_sizes
