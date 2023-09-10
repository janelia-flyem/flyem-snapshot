"""
Export a connectivity snapshot from a DVID segmentation,
along with other denormalizations.
"""
import os
import sys
import json
import shutil
import logging
import warnings
from functools import cache, partial
from argparse import ArgumentParser

import requests
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import holoviews as hv
import hvplot.pandas
from bokeh.plotting import output_file, save as bokeh_save
from bokeh.io import export_png
from confiddler import load_config, dump_config, dump_default_config


from neuclease import configure_default_logging, PrefixFilter
from neuclease.util import (
    switch_cwd, Timer, timed, encode_coords_to_uint64, decode_coords_from_uint64,
    extract_labels_from_volume, compute_parallel, tqdm_proxy,
    snakecase_to_camelcase, dump_json, iter_batches
)
from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.node import fetch_instance_info
from neuclease.dvid.voxels import fetch_volume_box
from neuclease.dvid.roi import fetch_combined_roi_volume
from neuclease.dvid.keyvalue import DEFAULT_BODY_STATUS_CATEGORIES, fetch_body_annotations
from neuclease.dvid.annotation import fetch_all_elements
from neuclease.dvid.labelmap import (
    resolve_snapshot_tag, fetch_mutations, fetch_complete_mappings,
    fetch_bodies_for_many_points, fetch_labelmap_voxels_chunkwise,
    fetch_labels_batched, compute_affected_bodies, fetch_sizes
)
from neuclease.misc.neuroglancer import format_nglink
from neuclease.misc.completeness import (
    completeness_forecast,
    plot_categorized_connectivity_forecast,
    variable_width_hbar,
)

MALECNS_VNC_CUTOFF_Z = 45056

# As luck would have it, we it's easy to filter by Z values directly from the
# point_id since Z occupies the most-significant position in the encoded point_id.
MALECNS_VNC_CUTOFF_POINT_ID = encode_coords_to_uint64(np.array([[MALECNS_VNC_CUTOFF_Z, 0, 0]], np.uint64))[0]


LabelmapSchema = {
    "description": "dvid labelmap location",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["server", "uuid", "instance"],
    "properties": {
        "server": {
            "description": "DVID server",
            "type": "string",
            # No default
        },
        "uuid": {
            "description": "Use ':master' for the current HEAD node, or ':master~1' for it's parent node.",
            "type": "string",
            "default": ":master~1"
        },
        "instance": {
            "description": "Name of the labelmap instance",
            "type": "string",
            "default": "segmentation"
        }
    }
}

ConfigSchema = {
    "properties": {
       "snapshot": LabelmapSchema,
       "syndir": {
            "description":
                "Optional. This value can be used within format strings in the other synapse settings.",
            "type": "string",
            "default": ""
        },
        "synapse-points": {
            "description":
                "A feather file containing the synapse points, with columns for sv and roi",
            "type": "string",
            # NO DEFAULT
        },
        "synapse-partners": {
            "description":
                "A feather file containing the synapse partners, with columns for pre_id, post_id",
            "type": "string",
            # NO DEFAULT
        },
        "zone": {
            "description":
                "Specifically for the male CNS. Whether to process the brain only, vnc only, or whole cns",
            "type": "string",
            "enum": ["brain", "vnc", ""],
            "default": "",
        },
        "min-confidence": {
            "description":
                "Before generating any results, exclude synapse predictions which fall below this confidence level.\n"
                "Note: This pre-filters the synapse table, so it is effectively a minimum bound on the confidence threshold for neuprint exports.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.0,
        },
    }
}

@PrefixFilter.with_context('Loading Synapses')
def _load_synapses(cfg):
    snapshot_tag = cfg['snapshot-tag']
    output_dir = cfg['output-dir']

    # Do the files already exist for this snapshot?
    # If so, just load those and return.
    # FIXME: I'm specifying these file names in two different places,
    #        and they must agree.
    if os.path.exists(f'tables/partner_df-{snapshot_tag}.feather'):
        logger.info(f"Loading previously-written synapse files from {output_dir}")
        partner_df = feather.read_feather(f'tables/partner_df-{snapshot_tag}.feather')
        point_df = feather.read_feather(f'tables/point_df-{snapshot_tag}.feather').set_index('point_id')
        return point_df, partner_df

    point_df, partner_df = _load_raw_synapses(cfg)
    mutations, mapping = _fetch_and_export_complete_mappings(cfg)
    point_df, partner_df = _update_synapses(cfg, point_df, partner_df, mutations, mapping)
    return point_df, partner_df


def _load_raw_synapses(cfg):
    points_path = cfg['synapse-points'].format(syndir=cfg['syndir'])
    partners_path = cfg['synapse-partners'].format(syndir=cfg['syndir'])

    with Timer("Loading synapses from disk", logger):
        point_df = feather.read_feather(points_path).set_index('point_id')
        partner_df = feather.read_feather(partners_path)
    logger.info(f"Loaded {len(point_df)} points and {len(partner_df)} partners")

    with Timer("Dropping unecessary columns and narrowing dtypes", logger):
        if (point_df[['sv', 'body']] <= 2**32).all().all():
            label_dtype = np.uint32
        else:
            label_dtype = np.uint64

        point_df = point_df.drop(
            errors='ignore',
            columns=[
                'roi_label',
                'user',
                'tbar_region',
            ]
        )

        point_df = point_df.astype({
            'conf': np.float32,
            'sv': label_dtype,
            'body': label_dtype,
            'kind': 'category',
            'roi': 'category',
        })

        partner_df = partner_df.astype({
            'conf_pre': np.float32,
            'conf_post': np.float32,
        })

    zone = cfg['zone']
    with Timer(f"Filtering for zone {zone}", logger):
        if zone == 'brain':
            point_df = point_df.loc[point_df['z'] < MALECNS_VNC_CUTOFF_Z].copy()
            partner_df = partner_df.loc[
                (partner_df['pre_id'] < MALECNS_VNC_CUTOFF_POINT_ID) &  # noqa
                (partner_df['post_id'] < MALECNS_VNC_CUTOFF_POINT_ID)].copy()
        elif zone == 'vnc':
            point_df = point_df.loc[point_df['z'] >= MALECNS_VNC_CUTOFF_Z].copy()
            partner_df = partner_df.loc[
                (partner_df['pre_id'] >= MALECNS_VNC_CUTOFF_POINT_ID) &  # noqa
                (partner_df['post_id'] >= MALECNS_VNC_CUTOFF_POINT_ID)].copy()

    if 'conf_pre' not in partner_df.columns:
        logger.info("Merging conf_pre column onto partner_df")
        partner_df = partner_df.merge(
            point_df['conf'].rename('conf_pre').rename_axis('pre_id'),
            'left',
            on='pre_id'
        )
    if 'conf_post' not in partner_df.columns:
        logger.info("Merging conf_post column onto partner_df")
        partner_df = partner_df.merge(
            point_df['conf'].rename('conf_post').rename_axis('post_id'),
            'left',
            on='post_id'
        )

    min_conf = cfg['min-confidence']
    with Timer(f"Filtering for min-confidence: {min_conf}", logger):
        point_df = point_df.loc[point_df['conf'] >= min_conf].copy()
        partner_df = partner_df.loc[
            (partner_df['conf_pre'] >= min_conf) &  # noqa
            (partner_df['conf_post'] >= min_conf)].copy()

    logger.info(f"Kept {len(point_df)} points and {len(partner_df)} partners")
    return point_df, partner_df


def _fetch_and_export_complete_mappings(cfg):
    dvid_seg = cfg['dvid-seg']
    snapshot_tag = cfg['snapshot-tag']
    mutations = fetch_mutations(*cfg['dvid-seg'])

    # Fetching the 'complete' mappings takes 2x as long as the minimal mappings,
    # but it's more useful because it
    mapping = fetch_complete_mappings(*dvid_seg, mutations)

    feather.write_feather(
        mapping.reset_index(),
        f"tables/complete-nonsingleton-mapping-{snapshot_tag}.feather"
    )

    return mutations, mapping


def _update_synapses(cfg, point_df, partner_df, mutations, mapping):
    dvid_seg = cfg['dvid-seg']
    snapshot_tag = cfg['snapshot-tag']

    with Timer(f"Updating supervoxels/bodies for UUID {dvid_seg[1][:6]}", logger):
        fetch_bodies_for_many_points(*dvid_seg, point_df, mutations, mapping, processes=cfg['processes'])

    with Timer("Adding columns to partner table for body_pre, body_post", logger):
        partner_df = partner_df.drop(columns=['body_pre', 'body_post'], errors='ignore')
        partner_df = partner_df.merge(point_df['body'], 'left', left_on='pre_id', right_index=True)
        partner_df = partner_df.merge(point_df['body'], 'left', left_on='post_id', right_index=True, suffixes=['_pre', '_post'])

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

