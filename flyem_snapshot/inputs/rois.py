import os
import json
import logging
from collections import namedtuple

import numpy as np
import pandas as pd

from neuclease import PrefixFilter
from neuclease.util import Timer, extract_labels_from_volume, dump_json, narrowest_dtype
from neuclease.dvid.voxels import fetch_volume_box
from neuclease.dvid.labelmap import fetch_labelmap_voxels_chunkwise
from neuclease.dvid.roi import fetch_combined_roi_volume

logger = logging.getLogger(__name__)

# FIXME: A better name than 'roiset' would be 'roilayer'
RoiSetSchema = {
    "description": "Settings to describe a set of disjoint ROIs",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["rois"],
    "properties": {
        "rois": {
            "description":
                "One of the following:\n"
                "  - list of ROI names: ['FOO(R)', 'FOO(L)', ...]\n"
                "  - mapping to segment IDs: {'FOO(R)': 1, 'FOO(L)': 2}\n"
                "  - path to a .json file with either of the above: /path/to/my-roi-ids.json\n"
                "  - a format string to compute each ROI name from its ROI segment ID 'x',\n"
                "    such as: 'ME_R_col_{x // 100}_{x % 100}'\n",
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
                "Whether to load the ROIs from DVID or to use a column in the synapse (or landmark) point table input,\n"
                "in which case your input synapses (or landmarks) must already have the appropriate column.\n",
            "type": "string",
            "enum": ["dvid-rois", "dvid-labelmap", "point-table"],
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
            "default": {
                "example_roiset": {}
            },
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
def load_point_rois(cfg, point_df, roiset_names):
    """
    For each named ROI set in the config,
    add a column to point_df for the ROI of the PostSyn side.
    """
    if point_df is None:
        return None, {}

    os.makedirs("volumes", exist_ok=True)

    point_df = point_df.copy()
    roisets = {}
    for roiset_name in roiset_names:
        roiset_cfg = cfg['roi-sets'][roiset_name]
        roi_ids = _load_columns_for_roiset(roiset_name, roiset_cfg, point_df, cfg['dvid'], cfg['processes'])
        assert isinstance(roi_ids, dict)
        roisets[roiset_name] = roi_ids

    return point_df, roisets


@PrefixFilter.with_context('rois')
def merge_partner_rois(cfg, point_df, partner_df):
    # Merge extracted ROI values in point_df onto the partner_df, too.
    # Note that we use the 'post' side when defining the location of a synapse connection.
    # The post side is used consistently for definining aggregate per-ROI synapse strengths
    # such as 'weight', 'synweight', 'upstream', 'downstream', 'weight'.
    with Timer("Adding roi columns to partner table", logger):
        roiset_names = [*cfg['roi-sets'].keys()]
        partner_df = partner_df.drop(columns=roiset_names, errors='ignore')
        partner_df = partner_df.merge(point_df[roiset_names].rename_axis('post_id'), 'left', on='post_id')

    return partner_df


def _load_columns_for_roiset(roiset_name, roiset_cfg, point_df, dvid_cfg, processes):
    """
    Ensure that point_df contains name and label columns for the given roiset,
    and that the name column is of Categorical dtype.
    Modifies point_df IN-PLACE.

    Returns the dict of roi_ids which was either loaded directly from the
    user's config or extracted from the ROI data itself.

    There are different possible sources of the ROI column data,
    depending on the 'source' specified in the user's config:

        - If the user chose to pre-populate the synapse table (or landmark table)
          with the ROI column, then we just load the ROI values and assign
          roi_ids if necessary.
        - If the user chose to provide an ROI volume (from DVID),
          then we load the volume and then extract the ROI labels from the
          appropriate locations in the volume.

    Note:
        We emit two columns per ROI set: One for ROI name and another for the
        corresponding integer segment ID.
        If we could guarantee that ROI IDs are always consecutive integers from 1..N,
        then we could forego the segment ID column and just rely on pd.Categorical codes.
        But that guarantee doesn't hold for all ROI sets (e.g. optic lobe column ROI
        segments, which encode their hex position).
    """
    roi_ids = roiset_cfg['rois']
    if isinstance(roi_ids, str) and roi_ids.endswith('.json'):
        roi_ids = json.load(open(roi_ids, 'r'))
    elif isinstance(roi_ids, list):
        roi_ids = dict(zip(roi_ids, range(1, len(roi_ids)+1)))

    assert isinstance(roi_ids, (dict, str))
    if isinstance(roi_ids, str):
        # If roi_ids is still a string, it must be a valid format string,
        # and it shouldn't give the same result for ID 0 as ID 1
        eval_0 = eval(f'f"{roi_ids}"', None, {'x': 0})  # pylint: disable=eval-used
        eval_1 = eval(f'f"{roi_ids}"', None, {'x': 1})  # pylint: disable=eval-used
        if eval_0 == eval_1:
            raise RuntimeError("ROI ID expression should not produce the same output for ID 0 and ID 1.")

    if roiset_cfg['source'] == "point-table":
        roi_ids = _load_roi_col(roiset_name, roi_ids, point_df)
        return roi_ids

    if bool(roiset_cfg['labelmap']) != (roiset_cfg['source'] == 'dvid-labelmap'):
        raise RuntimeError("Please supply a labelmap for your ROIs IFF you selected source: dvid-labelmap")

    roi_vol, roi_box, roi_ids = _load_roi_vol(roiset_name, roi_ids, roiset_cfg['labelmap'], dvid_cfg, processes)
    extract_labels_from_volume(point_df, roi_vol, roi_box, 5, roi_ids, roiset_name, skip_index_check=True)

    return roi_ids


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
                f"If you are specifying {roiset_name} roi_ids via a format string, "
                f"then you must supply the {roiset_name}_label column.")
        unique_ids = point_df[f'{roiset_name}'].unique()
        roi_ids = {
            eval(f'f"{roi_ids}"', None, {'x': x})  # pylint: disable=eval-used
            for x in unique_ids if x != 0
        }

    assert isinstance(roi_ids, dict)
    expected_cols = {roiset_name, f'{roiset_name}_label'}
    if not (expected_cols & {*point_df.columns}):
        msg = (
            f"Since your config specifies that the ROI column '{roiset_name}' "
            "will be supplied by your synapse (or landmark) table, you must supply "
            "at least one of the following columns in the point table:\n"
            f"- {roiset_name} (a column of strings or categories)"
            f"{roiset_name}_label (a column of integer IDs)"
        )
        raise RuntimeError(msg)

    # Ensure categorical
    if not isinstance(point_df[roiset_name].dtype, pd.CategoricalDtype):
        point_df[roiset_name] = point_df[roiset_name].astype('category')

    if expected_cols <= {*point_df.columns}:
        # Necessary columns are already present
        return roi_ids

    if f'{roiset_name}_label' not in point_df.columns:
        # Produce integers from names according to roi_ids
        dtype = narrowest_dtype(max(roi_ids.values()), signed=None)
        point_df[f'{roiset_name}_label'] = point_df[roiset_name].map(roi_ids).astype(dtype)

    if roiset_name not in point_df.columns:
        # Produce names from integers according to (reversed) roi_ids
        reverse_ids = {v:k for k,v in roi_ids.items()}
        if len(roi_ids) != len(reverse_ids):
            logger.warning("ROI IDs are not unique; precise mapping of ROI labels to ROI names is undefined.")
        point_df[roiset_name] = point_df[f'{roiset_name}_label'].map(reverse_ids)

    return roi_ids


def _load_roi_vol(roiset_name, roi_ids, roi_labelmap_name, dvid_cfg, processes):
    """
    Load an ROI volume, either from a list of DVID 'roi' instances,
    or from a single low-res DVID labelmap instance.

    If possible, load it from a previously cached copy without loading from DVID.
    If not, load from DVID and cache the volume on disk before returning.

    Args:
        roiset_name:
            The name of an ROI set from the config
        roi_ids:
            The dict of {roi: id} for the ROIs in the set,
            or a python expression which can produce a string for each unique ID in a labelmap volume.
        roi_labelmap_name:
            The name of a low-res (256nm) segmentation that has the ROI segments.
        dvid_cfg:
            The 'dvid' portion of the roi volume config,
            which contains the 'server', and 'uuid'.
        processes:
            How many processes to use when reading from DVID.

    Returns:
        roi_vol, roi_box (both in scale-5 resolution)
    """
    cache = RoiVolCache(roiset_name, roi_ids)
    roi_vol, roi_box = cache.load()
    if roi_vol is None:
        roi_vol, roi_box = _load_roi_vol_from_dvid(roiset_name, roi_ids, roi_labelmap_name, dvid_cfg, processes)
        cache.save(roi_vol, roi_box)

    if isinstance(roi_ids, str):
        # Now we can compute the mapping from name to label
        unique_ids = pd.unique(roi_vol.reshape(-1))
        roi_ids = {
            eval(f'f"{roi_ids}"', None, {'x': x}): x  # pylint: disable=eval-used
            for x in unique_ids if x != 0
        }

    return roi_vol, roi_box, roi_ids


def _load_roi_vol_from_dvid(roiset_name, roi_ids, roi_labelmap_name, dvid_cfg, processes):
    dvid_node = (dvid_cfg['server'], dvid_cfg['uuid'])
    if not all(dvid_node):
        raise RuntimeError(
            f"Can't read {roiset_name} ROIs from DVID. "
            "You didn't supply DVID server/uuid to read from.")

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
            dtype = narrowest_dtype(roi_vol.max(), signed=None)
            roi_vol = roi_vol.astype(dtype)
            return roi_vol, roi_box

    if isinstance(roi_ids, str):
        raise RuntimeError(
            f"If you didn't supply a labelmap volume from which to fetch the {roiset_name} "
            "ROIs, then you must supply a list of ROI instance names to fetch from DVID."
        )
    roi_vol, roi_box, _ = fetch_combined_roi_volume(*dvid_node, roi_ids, processes=processes)
    return roi_vol, roi_box


class RoiVolCache:
    """
    Utility class for loading/saving a cached ROI volume and its box on disk.
    The segment IDs are stored and comapared against the requested IDs
    to ensure that the cached volume is valid to be used.
    """

    CacheFiles = namedtuple('CacheFiles', 'vol box ids')

    def __init__(self, roiset_name, roi_ids):
        self.roiset_name = roiset_name
        self.roi_ids = roi_ids
        self.files = RoiVolCache.CacheFiles(
            f'volumes/{roiset_name}-vol.npy',
            f'volumes/{roiset_name}-box.json',
            f'volumes/{roiset_name}-ids.json'
        )

    def load(self):
        """
        Load the cached ROI volume from disk.
        If the ROI segment IDs stored on disk don't match the IDs
        this cache was initialized with, then the stored files
        aren't valid and can't be used.

        If the stored file doesn't exist or isn't valid for the given IDs,
        then None is returned.
        """
        files = self.files
        if not all(os.path.exists(p) for p in files):
            return None, None

        stored_ids = json.load(open(files.ids, 'r'))
        if stored_ids != self.roi_ids:
            logger.warning(f"Can't use cached volume for {self.roiset_name}: cached roi_ids don't match!")
            return None, None

        with Timer(f"Loading cached volume: {files.box}", logger):
            roi_vol = np.load(files.vol)
            roi_box = np.array(json.load(open(files.box, 'r')))
            return roi_vol, roi_box

    def save(self, roi_vol, roi_box):
        files = self.files
        with Timer(f"Writing {files.vol}", logger):
            np.save(files.vol, roi_vol)
            dump_json(roi_box, files.box, unsplit_int_lists=True)
            dump_json(self.roi_ids, files.ids)
