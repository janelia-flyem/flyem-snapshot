import os
import json
import logging

import numpy as np
import pyarrow.feather as feather

from neuclease.util import Timer, extract_labels_from_volume, dump_json
from neuclease.dvid.node import fetch_instance_info
from neuclease.dvid.voxels import fetch_volume_box
from neuclease.dvid.labelmap import fetch_labelmap_voxels_chunkwise
from neuclease.dvid.roi import fetch_combined_roi_volume

logger = logging.getLogger(__name__)

RoiSetSchema = {
    "description": "Settings to describe a set of disjoint ROIs",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["rois"],
    "properties": {
        "rois": {
            "description": "Either a list of ROI names or a mapping of ROI names to integers.\n",
            "oneOf": [
                {
                    # Optionally provide a path to a JSON file from which the ROI set will be read
                    "type": "string",
                },
                {
                    "type": "array",
                    "items": {"type": "string"},
                    # No default
                },
                {
                    "type": "object",
                    "additionalProperties": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    # No default
                }
            ]
        },
        "labelmap": {
            "description":
                "Optional. If provided, ROI segments will be loaded from the given\n"
                "labelmap in DVID instead of from individual 'roi' instances in DVID.\n"
                "The mapping between segment IDs and ROI names is determined from the 'rois' mapping.",
            "type": "string",
            "default": ""
        }
    }
}

RoisSchema = {
    "description": "Settings to describe a set of disjoint ROIs",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["roi-sets"],
    "properties": {
        "roi-sets": {
            "type": "object",
            "additionalProperties": RoiSetSchema,
            "default": {},
        },
        "processes": {
            "description":
                "How many processes should be used to fetch ROI masks from DVID?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    }
}


def load_rois(cfg, point_df, partner_df):
    """
    For each named ROI set in the config,
    add a column to point_df for the ROI of the PostSyn side.
    """
    os.makedirs("volumes", exist_ok=True)

    snapshot_tag = cfg['snapshot-tag']
    point_df = point_df.copy()
    for roiset_name, roiset_cfg in cfg['neuprint']['roi-sets'].items():
        roi_ids = roiset_cfg['rois']
        if isinstance(roi_ids, str):
            roi_ids = json.load(open(roi_ids, 'r'))
        if isinstance(roi_ids, list):
            roi_ids = dict(zip(roi_ids, range(1, len(roi_ids)+1)))
        assert isinstance(roi_ids, dict)

        roi_vol, roi_box = _load_roi_vol(roiset_name, roi_ids, roiset_cfg['labelmap'], cfg['dvid-seg'], cfg['processes'])
        extract_labels_from_volume(point_df, roi_vol, roi_box, 5, roi_ids, roiset_name, skip_index_check=True)

    # Merge those extracted ROI values onto the partner_df, too.
    # Note that we use the 'post' side when defining the location of a synapse connection.
    # The post side is used consistently for definining aggregate per-ROI synapse strengths
    # such as 'weight', 'synweight', 'upstream', 'downstream', 'weight'.
    with Timer("Adding roi columns to partner table", logger):
        roiset_names = [*cfg['neuprint']['roi-sets'].keys()]
        partner_df = partner_df.drop(columns=roiset_names, errors='ignore')
        partner_df = partner_df.merge(point_df[roiset_names].rename_axis('post_id'), 'left', on='post_id')

    # Overwrite the point_df we saved earlier now that we've updated the ROIs.
    with Timer("Exporting filtered/updated synapse tables", logger):
        feather.write_feather(
            point_df.reset_index(),
            f'tables/point_df-{snapshot_tag}.feather'
        )
        feather.write_feather(
            partner_df,
            f'tables/partner_df-{snapshot_tag}.feather'
        )
    return point_df, partner_df


def _load_roi_vol(roiset_name, roi_ids, roi_labelmap_name, snapshot_seg, processes):
    """
    Load an ROI volume, either from a list of DVID 'roi' instances,
    or from a single low-res DVID labelmap instance.

    If possible, load it from a locally cached copy.

    Args:
        roiset_name:
            The name of an ROI set from the config
        roi_ids:
            The dict of {roi: id} for the ROIs in the set.
        roi_labelmap_name:
            The name of a low-res (256nm) segmentation that has the ROI segments.
        snapshot_seg:
            The main neuron segmentation's (server, uuid, instance).

    Returns:
        roi_vol, roi_box (both in scale-5 resolution)
    """
    vol_cache_name = f'volumes/{roiset_name}-vol.npy'
    box_cache_name = f'volumes/{roiset_name}-box.json'
    ids_cache_name = f'volumes/{roiset_name}-ids.json'
    if all(os.path.exists(p) for p in (vol_cache_name, box_cache_name, ids_cache_name)):
        cached_ids = json.load(open(ids_cache_name, 'r'))
        if cached_ids != roi_ids:
            logger.warning(f"Can't use cached volume for {roiset_name}: cached roi_ids don't match!")
        else:
            with Timer(f"Loading {box_cache_name}", logger):
                roi_vol = np.load(vol_cache_name)
                roi_box = np.array(json.load(open(box_cache_name, 'r')))
                return roi_vol, roi_box

    if roi_labelmap_name:
        roi_seg = (*snapshot_seg[:2], roi_labelmap_name)
        neuron_res = fetch_instance_info(*snapshot_seg)['Extended']['VoxelSize']
        roi_res = fetch_instance_info(*roi_seg)['Extended']['VoxelSize']
        if (np.array(roi_res) / neuron_res != 2**5).any():
            msg = f"{roi_labelmap_name}: ROI volumes must be 32x lower resolution than the full-res segmentation"
            raise RuntimeError(msg)

        roi_box = np.array(fetch_volume_box(*roi_seg))
        with Timer(f"Fetching ROI volume: {roi_labelmap_name}"):
            roi_vol = fetch_labelmap_voxels_chunkwise(*roi_seg, roi_box, progress=False)
    else:
        roi_vol, roi_box, _ = fetch_combined_roi_volume(
            *snapshot_seg[:2],
            roi_ids,
            processes=processes
        )

    with Timer(f"Writing {vol_cache_name}", logger):
        np.save(vol_cache_name, roi_vol)
        dump_json(roi_box, box_cache_name, unsplit_int_lists=True)
        dump_json(roi_ids, ids_cache_name)
    return roi_vol, roi_box
