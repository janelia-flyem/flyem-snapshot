import os
import json
import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, extract_labels_from_volume, dump_json, narrowest_dtype
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
                    # Optionally provide a path to a '.json' from which the ROI set will be read,
                    # OR a python expression which produces an ROI name from an ROI segment ID.
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
        "source": {
            "description":
                "Whether to load the ROIs from DVID or to use a column in the synapse point table input,\n"
                "in which case your input synapses must already have the appropriate column.\n",
            "type": "string",
            "enum": ["dvid-rois", "dvid-labelmap", "synapse-point-table"],
            "default": "dvid-rois"
        },
        "labelmap": {
            "description":
                "Optional. If provided, ROI segments will be loaded from the given\n"
                "labelmap in DVID instead of from individual 'roi' instances in DVID.\n"
                "The mapping between segment IDs and ROI names is determined from the 'rois' mapping.\n"
                "Note: The ROI volume MUST have exactly 32x lower resolution than the full neuron resolution.",
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
        "dvid": {
            "description":
                "Which DVID server to use for reading ROIs.\n"
                "If not specified here, the settings from the 'synapses.update-to' setting will be used.\n",
            "default": {},
            "additionalProperties": False,
            "properties": {
                "server": {
                    "type": "string",
                    "default": ""
                },
                "uuid": {
                    "type": "string",
                    "default": ""
                }
            }
        },
        "roi-sets": {
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


@PrefixFilter.with_context('rois')
def load_rois(cfg, snapshot_tag, point_df, partner_df):
    """
    For each named ROI set in the config,
    add a column to point_df for the ROI of the PostSyn side.
    """
    os.makedirs("volumes", exist_ok=True)

    point_df = point_df.copy()
    for roiset_name, roiset_cfg in cfg['roi-sets'].items():
        roi_ids = roiset_cfg['rois']
        if isinstance(roi_ids, str) and roi_ids.endswith('.json'):
            roi_ids = json.load(open(roi_ids, 'r'))
        elif isinstance(roi_ids, list):
            roi_ids = dict(zip(roi_ids, range(1, len(roi_ids)+1)))

        if isinstance(roi_ids, str):
            # If roi_ids is still a string, it must be a valid python expression,
            # and it shouldn't give the same result for ID 0 as ID 1
            assert eval(roi_ids.format(x=1)) != eval(roi_ids.format(x=0))  # pylint: disable=eval-used
        else:
            assert isinstance(roi_ids, dict)

        if roiset_cfg['source'] == "synapse-point-table":
            _load_roi_col(roiset_name, roi_ids, point_df)
        else:
            if bool(roiset_cfg['labelmap']) != (roiset_cfg['source'] == 'dvid-labelmap'):
                raise RuntimeError("Please supply a labelmap for your ROIs IFF you selected source: dvid-labelmap")

            roi_vol, roi_box = _load_roi_vol(roiset_name, roi_ids, roiset_cfg['labelmap'], cfg['dvid'], cfg['processes'])
            if isinstance(roi_ids, str):
                # Now we can compute the mapping from name to label
                unique_ids = pd.unique(roi_vol.reshape(-1))
                roi_ids = {
                    eval(roi_ids.format(x=x)): x  # pylint: disable=eval-used
                    for x in unique_ids if x != 0
                }
            extract_labels_from_volume(point_df, roi_vol, roi_box, 5, roi_ids, roiset_name, skip_index_check=True)

    # Merge those extracted ROI values onto the partner_df, too.
    # Note that we use the 'post' side when defining the location of a synapse connection.
    # The post side is used consistently for definining aggregate per-ROI synapse strengths
    # such as 'weight', 'synweight', 'upstream', 'downstream', 'weight'.
    with Timer("Adding roi columns to partner table", logger):
        roiset_names = [*cfg['roi-sets'].keys()]
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


def _load_roi_col(roiset_name, roi_ids, point_df):
    """
    For a given roiset, ensure that point_df contains a column for
    the ROI names (a categorical) AND the ROI integer IDs (the 'label').

    For example:
        - primary_roi (categorical strings)
        - primary_roi_label (integers)

    If it contains only one or the other, then we calculate the missing one using roi_ids.
    """
    if isinstance(roi_ids, str):
        if f'{roiset_name}_label' not in point_df.columns:
            raise RuntimeError(
                f"If you are specifying {roiset_name} roi_ids via a python expression, "
                f"then you must supply the {roiset_name}_label column.")
        unique_ids = point_df[f'{roiset_name}'].unique()
        roi_ids = {
            eval(roi_ids.format(x=x)): x  # pylint: disable=eval-used
            for x in unique_ids if x != 0
        }

    assert isinstance(roi_ids, dict)
    expected_cols = {roiset_name, f'{roiset_name}_label'}
    if not (expected_cols & {*point_df.columns}):
        msg = (
            f"Since your config specifies that the ROI column '{roiset_name}' "
            "will be supplied by your synapse table, you must supply at least one "
            "of the following columns in the synapse point table:\n"
            f"- {roiset_name} (a column of strings or categories)"
            f"{roiset_name}_label (a column of integer IDs)"
        )
        raise RuntimeError(msg)

    # Ensure that the string column is categorical
    if not isinstance(point_df[roiset_name].dtype, pd.CategoricalDtype):
        point_df[roiset_name] = point_df[roiset_name].astype('category')

    if expected_cols <= {*point_df.columns}:
        # Necessary columns are already present
        return

    if f'{roiset_name}_label' not in point_df.columns:
        # Produce integers from names according to roi_ids
        dtype = narrowest_dtype(max(roi_ids.values()), signed=False)
        point_df[f'{roiset_name}_label'] = point_df[roiset_name].map(roi_ids).astype(dtype)

    if roiset_name not in point_df.columns:
        # Produce names from integers according to (reversed) roi_ids
        reverse_ids = {v:k for k,v in roi_ids.items()}
        if len(roi_ids) != len(reverse_ids):
            logger.warning("ROI IDs are not unique; precise mapping of ROI labels to ROI names is undefined.")
        point_df[roiset_name] = point_df[f'{roiset_name}_label'].map(reverse_ids)


def _load_roi_vol(roiset_name, roi_ids, roi_labelmap_name, dvid_cfg, processes):
    """
    Load an ROI volume, either from a list of DVID 'roi' instances,
    or from a single low-res DVID labelmap instance.

    If possible, load it from a locally cached copy.

    Args:
        roiset_name:
            The name of an ROI set from the config
        roi_ids:
            The dict of {roi: id} for the ROIs in the set,
            or a python expression which can produce a string for each unique ID in a labelmap volume.
        roi_labelmap_name:
            The name of a low-res (256nm) segmentation that has the ROI segments.
        snapshot_seg:
            The main neuron segmentation's (server, uuid, instance).

    Returns:
        roi_vol, roi_box (both in scale-5 resolution)
    """
    dvid_node = (dvid_cfg['server'], dvid_cfg['uuid'])
    if not all(dvid_node):
        raise RuntimeError(
            f"Can't read {roiset_name} ROIs from DVID. "
            "You didn't supply DVID server/uuid to read from.")

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
        roi_seg = (*dvid_node, roi_labelmap_name)

        # It would be nice to add this check back in, but currently
        # the snapshot_seg is not passed to this function.
        # neuron_res = fetch_instance_info(*snapshot_seg)['Extended']['VoxelSize']
        # roi_res = fetch_instance_info(*roi_seg)['Extended']['VoxelSize']
        # if (np.array(roi_res) / neuron_res != 2**5).any():
        #     msg = f"{roi_labelmap_name}: ROI volumes must be 32x lower resolution than the full-res segmentation"
        #     raise RuntimeError(msg)

        roi_box = np.array(fetch_volume_box(*roi_seg))
        with Timer(f"Fetching ROI volume: {roi_labelmap_name}"):
            roi_vol = fetch_labelmap_voxels_chunkwise(*roi_seg, roi_box, progress=False)
    else:
        if isinstance(roi_ids, str):
            raise RuntimeError(
                f"If you didn't supply a labelmap volume from which to fetch the {roiset_name} "
                "ROIs, then you must supply a list of ROI instance names to fetch from DVID."
            )
        roi_vol, roi_box, _ = fetch_combined_roi_volume(*dvid_node, roi_ids, processes=processes)

    with Timer(f"Writing {vol_cache_name}", logger):
        np.save(vol_cache_name, roi_vol)
        dump_json(roi_box, box_cache_name, unsplit_int_lists=True)
        dump_json(roi_ids, ids_cache_name)

    return roi_vol, roi_box
