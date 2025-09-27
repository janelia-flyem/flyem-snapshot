"""
Export skeletons from a DVID server. Files are written to:

    skeletons/skeletons-swc/
    skeletons/skeletons-precomputed/

The skeletons-swc/ directory contains the SWC files, and the skeletons-precomputed/
directory contains the Neuroglancer precomputed skeleton files.
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
from neuclease.dvid.keyvalue import fetch_instance_info, fetch_key, fetch_keys
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

SingleInstanceSkeletonSchema = {
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
        "coordinate-resolution-nm": {
            "description": "Units of the resolution (in nanometers).",
            "type": "array",
            "items": {
                "type": "number",
            },
            "minItems": 3,
            "maxItems": 3,
            "default": [0, 0, 0],
        }
    }
}

SkeletonsSchema = {
    "description": "Settings for exporting one or more skeleton instances from DVID.",
    "type": "object",
    "default": {},
    "properties": {
        "processes": {
            "description":
                "How many processes should be used to export skeletons?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    },
    "additionalProperties": SingleInstanceSkeletonSchema,
}


def export_skeletons(cfg, snapshot_tag, ann=None, pointlabeler=None):
    cfg = copy.copy(cfg)
    processes = cfg['processes']
    del cfg['processes']
    for name, instance_cfg in cfg.items():
        export_skeleton_instance(name, instance_cfg, snapshot_tag, ann, pointlabeler, processes=processes)


class SkeletonSerializer(SerializerBase):
    """
    Serializer that just stores the table of exported skeletons vs. missing skeletons.
    Avoids downloading skeletons repeatedly for a release snapshot whose segmentation hasn't changed,
    but attempts to re-export skeletons that weren't successfully exported in the last run.
    """

    def get_cache_key(self, cfg_name, cfg, snapshot_tag, ann=None, pointlabeler=None, subset_bodies=None, processes=0):
        self.cfg_name = cfg_name
        self.cfg = cfg
        self.snapshot_tag = snapshot_tag
        self.ann = ann
        self.pointlabeler = pointlabeler
        self.processes = processes
        cfg_hash = hex(checksum(cfg))

        if ann is not None:
            ann_body_hash = hex(checksum(np.sort(ann.index.values)))
        else:
            ann_body_hash = '0x0'

        if pointlabeler is None:
            mutid = 0
        else:
            mutid = pointlabeler.last_mutation["mutid"]

        return f'{snapshot_tag}-seg-{mutid}-ann-{ann_body_hash}-{self.name}-{cfg_hash}.csv'

    def save_to_file(self, results, path):
        if results is None and os.path.exists(path):
            os.remove(path)
            return

        results.to_csv(path, index=True, header=True)
        if not results['success'].all():
            num_failed = (~results['success']).sum()
            logger.warning(
                f"Failed to export skeletons for {num_failed} body IDs. See {os.path.abspath(path)}"
            )

    def load_from_file(self, path):
        results = pd.read_csv(path)
        if not results['success'].all():
            failed_bodies = results.loc[~results['success']].index.str[:len('_swc')].map(int).unique()
            new_results = export_skeletons(
                self.cfg_name,
                self.cfg,
                self.snapshot_tag,
                self.ann,
                self.pointlabeler,
                failed_bodies,
                processes=self.processes
            )
            results.loc[new_results.index, 'success'] = new_results['success']
        return results


@PrefixFilter.with_context('skeletons-{cfg_name}')
@cached(SkeletonSerializer('skeletons-{cfg_name}'))
def export_skeleton_instance(cfg_name, cfg, snapshot_tag, ann=None, pointlabeler=None, subset_bodies=None, processes=0):
    """
    Export skeletons in both SWC and Neuroglancer precomputed format.
    The set of skeletons to export is taken from the body annotations table.

    The SkeletonSerializer ensures that we don't rerun the full export if it's already been
    completed, but it does call this function if there are missing skeletons from the last run.
    """
    if (np.array(cfg['coordinate-resolution-nm']) == 0).any():
        if pointlabeler is None:
            raise RuntimeError("Skeleton coordinate resolution must be set in the config.")
        info = fetch_instance_info(*pointlabeler.dvidseg)
        cfg['coordinate-resolution-nm'] = info['Extended']['VoxelSize']

    del snapshot_tag
    del pointlabeler

    skeleton_src = (
        cfg['dvid']['server'],
        cfg['dvid']['uuid'],
        cfg['dvid']['instance'],
    )
    if not (cfg['export-skeletons'] and all(skeleton_src)):
        return None

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

    dirname = f'skeletons-{cfg_name}'
    os.makedirs(f'{dirname}/skeletons-swc', exist_ok=True)
    os.makedirs(f'{dirname}/skeletons-precomputed', exist_ok=True)
    with open(f'{dirname}/skeletons-precomputed/info', 'w', encoding='utf-8') as f:
        f.write('{"@type": "neuroglancer_skeletons"}\n')

    logger.info(f"Processing skeletons for {len(keys)} body IDs")

    results = compute_parallel(
        partial(_process_single_skeleton, *skeleton_src, cfg['coordinate-resolution-nm'], dirname),
        keys,
        processes=processes,
    )
    results = pd.DataFrame(results, columns=['key', 'success']).set_index('key')
    return results


def _process_single_skeleton(server, uuid, instance, resolution, dirname, key):
    """
    Fetch a single skeleton from the DVID server and write two files:
        - an SWC file
        - a Neuroglancer "precomputed" skeleton file

    Returns:
        (key, success) tuple, where success is True if the skeleton
        exists on the DVID server and was written successfully.
    """
    assert key.endswith('_swc')
    body = key[:-len('_swc')]

    try:
        swc_bytes = fetch_key(server, uuid, instance, key)
    except requests.exceptions.HTTPError:
        return key, False
    except Exception as e:
        logger.warning(f"An exception of type {type(e).__name__} occurred. Arguments:\n{e.args}")
        return key, False

    if not swc_bytes:
        return key, False

    fname = f"{dirname}/skeletons-swc/{body}.swc"
    with open(fname, "wb") as f:
        f.write(swc_bytes)

    df = swc_to_dataframe(swc_bytes)

    skeleton_to_neuroglancer(df, orig_resolution_nm=resolution, output_path=f"{dirname}/skeletons-precomputed/{body}")
    return key, True
