"""
Export skeletons from a DVID server. Files are written to:
    skeletons/skeletons-swc/
    skeletons/skeletons-precomputed/
The skeletons-swc/ directory contains the SWC files, and the skeletons-precomputed/
directory contains the Neuroglancer precomputed files.
"""
import logging
import os
import re

import pandas as pd

from neuclease import PrefixFilter
from neuclease.dvid.keyvalue import fetch_keys, fetch_keyvalues
from neuclease.util import (
    compute_parallel, skeleton_to_neuroglancer, swc_to_dataframe, tqdm_proxy
)

logger = logging.getLogger(__name__)

# TODO:
#   - Need to provide resolution (in nanometers) in the config.
#   - Allow parallel process count to be configured?
#   - Allow user to narrow the set of skeletons to export by including or excluding body statuses?
#   - Auto-set the skeleton instance from the dvid segmentation instance: f"{seg_instance}_skeletons"

SkeletonSchema = {
    "description": "Settings for skeleton export.",
    "type": "object",
    "default": {},
    "properties": {
        "export-skeletons": {
            "description": "If true, export the skeletons.",
            "type": "boolean",
            "default": True,
        },
        "dvid": {
            "description": "DVID server/UUID and instance to export skeletons from.",
            "type": "object",
            "additionalProperties": False,
            "default": {},
            "properties": {
                "server": {
                    "type": "string",
                    "default": ""
                },
                "uuid": {
                    "type": "string",
                    "default": ""
                },
                "instance": {
                    "type": "string",
                    "default": ""
                }
            }
        }
    }
}


def _process_single_skeleton(server, uuid, instance, key):
    """
    Fetch a single skeleton from the DVID server and write two files:
        - an SWC file
        - a Neuroglancer "precomputed" skeleton file

    Returns:
        (key, success) tuple, where success is True if the skeleton
        exists on the DVID server and was written successfully.
    """
    assert key.endswith('_swc')
    body = key[:-4]

    try:
        swc_bytes = fetch_key(server, uuid, instance, key)
    except HTTPError as e:
        return key, False

    if not swc_bytes:
        return key, False

    fname = f"skeletons/skeletons-swc/{body}.swc"
    with open(fname, "wb") as f:
        f.write(swc_bytes)

    df = swc_to_dataframe(swc_bytes)

    # FIXME: Need to provide resolution here, using a value from the config
    #        that is auto-filled using the DVID segmentation by default.
    skeleton_to_neuroglancer(df, output_path=f"skeletons/skeletons-precomputed/{body}")
    return key, True


@PrefixFilter.with_context('skeletons')
def export_skeletons(cfg, ann=None):
    """
    Export skeletons in both SWC and Neuroglancer precomputed format.
    The set of skeletons to export is taken from the body annotations table.
    """
    skeleton_src = (
        cfg['dvid']['server'],
        cfg['dvid']['uuid'],
        cfg['dvid']['instance'],
    )
    if not (cfg['export-skeletons'] and all(skeleton_src)):
        return
    
    if ann is None:
        logger.info(f"Fetching all keys from {'/'.join(skeleton_src)}")
        keys = fetch_keys(*skeleton_src)
        keys = [k for k in keys if re.match(r"\d+_swc$", k)]
        if not keys:
            logger.warning(
                "Not exporting skeletons: No skeleton keys found in the DVID instance "
                f"({'/'.join(skeleton_src)})"
            )
            return
    elif len(ann) == 0:
        logger.warning("Not exporting skeletons: No body IDs in the annotations table.")
        return
    else:
        assert ann.index.name == 'body'
        keys = ann.index.astype(str) + "_swc"

    os.makedirs('skeletons/skeletons-swc', exist_ok=True)
    os.makedirs('skeletons/skeletons-precomputed', exist_ok=True)
    with open('skeletons/skeletons-swc/info', 'w', encoding='utf-8') as f:
        f.write('{"@type": "neuroglancer_skeletons"}\n')

    logger.info(f"Processing skeletons for {len(keys)} body IDs")

    results = compute_parallel(
        partial(_process_single_skeleton, *skeleton_src),
        keys,
        processes=8,
        show_progress=False,
    )
    results = pd.DataFrame(results, columns=['key', 'success'])

    if not results['success'].all():
        missing_keys = results.loc[~results['success'], 'key']
        missing_keys.to_csv('skeletons/missing-skeletons.csv', index=False, header=True)
        logger.warning(
            f"Did not find skeletons for {len(missing_keys)} body IDs from the annotations. "
            "See skeletons/missing-skeletons.csv."
        )

# Command-line entry point for testing with hemibrain (no annotations)
if __name__ == "__main__":
    test_config = {
        "export-skeletons": True,
        "dvid": {
            "server": "https://hemibrain-dvid.janelia.org",
            "uuid": "15aee239",
            "instance": "segmentation_skeletons",
        },
    }

    export_skeletons(test_config)
