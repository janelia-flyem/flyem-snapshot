"""
Export skeletons from a DVID server. Files are written to:
    skeletons/skeletons-swc/
    skeletons/skeletons-precomputed/
The skeletons-swc/ directory contains the SWC files, and the skeletons-precomputed/
directory contains the Neuroglancer precomputed files. If skeletons are missing,
they are written to skeletons/missing-skeletons.csv.
"""
from functools import partial
import logging
import os
import re
import copy

import pandas as pd
import numpy as np
import requests.exceptions

from neuclease import PrefixFilter
from neuclease.dvid.keyvalue import fetch_key, fetch_keys
from neuclease.util import (
    compute_parallel, skeleton_to_neuroglancer, swc_to_dataframe
)
from ..util.checksum import checksum
from ..caches import cached, SerializerBase

logger = logging.getLogger(__name__)

# TODO:
#   - Need to provide resolution (in nanometers) in the config.
#   - Allow parallel process count to be configured?
#   - Allow user to narrow the set of skeletons to export by including or excluding body statuses?

SkeletonSchema = {
    "description": "Settings for skeleton export.",
    "type": "object",
    "default": {},
    "properties": {
        "export-skeletons": {
            "description": "If true, export the skeletons.",
            "type": "boolean",
            "default": False,
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
        },
        "processes": {
            "description":
                "How many processes should be used to export skeletons?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    }
}

class SkeletonSerializer(SerializerBase):
    """
    Serializer that just stores the table of exported skeletons vs. missing skeletons.
    Avoids downloading skeletons repeatedly for a release snapshot whose segmentation hasn't changed,
    but attempts to re-export skeletons that weren't successfully exported in the last run.
    """

    def get_cache_key(self, cfg, snapshot_tag, ann=None, pointlabeler=None, subset_bodies=None):
        self.cfg = cfg
        self.snapshot_tag = snapshot_tag
        self.ann = ann
        self.pointlabeler = pointlabeler

        cfg = copy.copy(cfg)
        cfg['processes'] = 0

        cfg_hash = hex(checksum(cfg))

        if ann is not None:
            ann_body_hash = hex(checksum(np.sort(ann.index.values)))
        else:
            ann_body_hash = '0x0'

        if pointlabeler is None:
            mutid = 0
        else:
            mutid = pointlabeler.last_mutation["mutid"]

        return f'{snapshot_tag}-seg-{mutid}-ann-{ann_body_hash}-skeletons-{cfg_hash}.csv'

    def save_to_file(self, results, path):
        results.to_csv(path, index=True, header=True)
        if not results['success'].all():
            num_failed = (~results['success']).sum()
            logger.warning(
                f"Failed to export skeletons for {num_failed} body IDs. See {path}"
            )

    def load_from_file(self, path):
        results = pd.read_csv(path)
        if not results['success'].all():
            failed_bodies = results.loc[~results['success']].index.str[:-4].map(int).unique()
            new_results = export_skeletons(self.cfg, self.snapshot_tag, self.ann, self.pointlabeler, failed_bodies)
            results.loc[new_results.index, 'success'] = new_results['success']
        return results


@PrefixFilter.with_context('skeletons')
@cached(SkeletonSerializer('skeletons'))
def export_skeletons(cfg, snapshot_tag, ann=None, pointlabeler=None, subset_bodies=None):
    """
    Export skeletons in both SWC and Neuroglancer precomputed format.
    The set of skeletons to export is taken from the body annotations table.
    """
    del snapshot_tag
    del pointlabeler

    skeleton_src = (
        cfg['dvid']['server'],
        cfg['dvid']['uuid'],
        cfg['dvid']['instance'],
    )
    if not (cfg['export-skeletons'] and all(skeleton_src)):
        return

    if subset_bodies is not None:
        logger.info(f"Exporting skeletons for a subset of {len(subset_bodies)} bodies")
        keys = [f'{body}_swc' for body in subset_bodies]
    elif ann is None:
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
    with open('skeletons/skeletons-precomputed/info', 'w', encoding='utf-8') as f:
        f.write('{"@type": "neuroglancer_skeletons"}\n')

    logger.info(f"Processing skeletons for {len(keys)} body IDs")

    results = compute_parallel(
        partial(_process_single_skeleton, *skeleton_src),
        keys,
        processes=cfg['processes'],
    )
    results = pd.DataFrame(results, columns=['key', 'success']).set_index('key')
    return results


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
    except requests.exceptions.HTTPError:
        return key, False
    except Exception as e:
        logger.warning(f"An exception of type {type(e).__name__} occurred. Arguments:\n{e.args}")
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
