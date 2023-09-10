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

_ = hvplot.pandas  # linting

logger = logging.getLogger(__name__)



# In most cases, we formulaically convert from snake_case to camelCase,
# but in some cases the terminology doesn't follow that formula.
# This list provides the override set.
# To exclude a column from neuprint, list it here and map it to "".
CLIO_TO_NEUPRINT_PROPERTIES = {
    'bodyid': 'bodyId',
    'status': 'statusLabel',
    'hemibrain_bodyid': 'hemibrainBodyId',

    # Make sure these never appear in neuprint.
    'last_modified_by': '',
    'old_bodyids': '',
    'reviewer': '',
    'to_review': '',
    'typing_notes': '',
    'user': '',
    'notes': '',
    'halfbrainBody': '',

    # These generally won't be sourced from Clio anyway;
    # they should be sourced from the appropriate DVID annotation instance.
    'soma_position': 'somaLocation',
    'tosoma_position': 'tosomaLocation',
    'root_position': 'rootLocation',
}

# Note any 'statusLabel' (DVID status) that isn't
# listed here will appear in neuprint unchanged.
NEUPRINT_STATUSLABEL_TO_STATUS = {
    'Unimportant':              'Unimportant',  # noqa
    'Glia':                     'Glia',         # noqa
    'Hard to trace':            'Orphan',       # noqa
    'Orphan-artifact':          'Orphan',       # noqa
    'Orphan':                   'Orphan',       # noqa
    'Orphan hotknife':          'Orphan',       # noqa

    'Out of scope':             '',             # noqa
    'Not examined':             '',             # noqa
    '':                         '',             # noqa

    '0.5assign':                '0.5assign',    # noqa

    'Anchor':                   'Anchor',       # noqa
    'Cleaved Anchor':           'Anchor',       # noqa
    'Sensory Anchor':           'Anchor',       # noqa
    'Cervical Anchor':          'Anchor',       # noqa
    'Soma Anchor':              'Anchor',       # noqa
    'Primary Anchor':           'Anchor',       # noqa
    'Partially traced':         'Anchor',       # noqa

    'Leaves':                   'Traced',       # noqa
    'PRT Orphan':               'Traced',       # noqa
    'Prelim Roughly traced':    'Traced',       # noqa
    'RT Orphan':                'Traced',       # noqa
    'Roughly traced':           'Traced',       # noqa
    'Traced in ROI':            'Traced',       # noqa
    'Traced':                   'Traced',       # noqa
    'Finalized':                'Traced',       # noqa
}

# For certain types, we need to ensure that 'long' is used in neo4j, not int.
# Also, in some cases (e.g. 'group'), the column in our pandas DataFrame is
# stored with float dtype because pandas uses NaN for missing values.
# We want to emit actual ints in the CSV.
NEUPRINT_TYPE_OVERRIDES = {
    'bodyId': 'long',
    'group': 'long',
    'hemibrainBodyid': 'long',
    'size': 'long',
    'pre': 'int',
    'post': 'int',
    'upstream': 'int',
    'downstream': 'int',
    'synweight': 'int',
    # 'halfbrainBody': 'long',
}

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
        },
        "primary": {
            "description": "Set to true if this is the 'primary' roi set.",
            "type": "boolean",
            "default": False
        },
        "synapse-properties": {
            "type": "object",
            "default": {},
            "description":
                "In some cases, the values of the ROI segment IDs are semantically meaningful.\n"
                "If that's the case, you may wish to add related properties to the synapse nodes.\n"
                "Here, we allow you to specify such properties, calculated via formula from the segment IDs.\n",
            "additionalProperties": {
                "type": "string",
                "description":
                    "A Python expression in which the roi segment ID 'x'\n"
                    "is used to produce a property value for each synapse.\n",

                # By default, the synapse property is assigned the value of the ROI segment ID itself.
                "default": "x"
            }
        }
    }
}

PointAnnotationSchema = {
    "description": "Settings to describe a source of point annotations in DVID which should be loaded into neuprint",
    "type": "object",
    "default": {},
    "required": ['instance', 'property-name'],
    "properties": {
        "instance": {
            "description":
                "Name of the DVID annotation instance which contains the points.\n"
                "Note: If more than one annotation point falls on the same body,\n"
                "      only one will be used, and the others simply dropped.\n"
                "Example: 'nuclei-centroids'\n",
            "type": "string",
            # no default
        },
        "property-name": {
            "description": "The name of the property as it will appear on :Segment/:Neuron nodes",
            "type": "string",
            # no default
        }
    }
}

NeuprintSchema = {
    "description": "settings for constructing a neuprint snapshot database",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["dataset"],
    "properties": {
        "export-neuprint-snapshot": {
            "description": "If true, export CSV files needed to construct a neo4j neuprint database.",
            "type": "boolean",
            "default": True,
        },
        "dataset": {
            "description":
                "The name of the dataset in neuprint, e.g. 'cns'\n"
                "Do NOT include the version here (such as cns:v1.0)\n",
            "type": "string",
            "default": ""
        },
        "roi-sets": {
            "type": "object",
            "additionalProperties": RoiSetSchema,
            "default": {},
        },
        "point-annotations": {
            "type":"array",
            "items": PointAnnotationSchema,
            "default": []
        },
        "neuron-label-criteria": {
            "type": "object",
            "default": {},
            "properties": {
                "synweight": {
                    "description": "Segments with this synweight (or more) will also be labeled as Neurons.\n",
                    "type": "number",
                    "minimum": 0,
                    "default": 100,
                },
                "properties": {
                    "description": "Segments with non-empty values for any properties listed here will be labeled as Neurons\n",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["rootLocation", "somaLocation", "class", "type", "instance", "group", "somaSide", "synonym"],
                },
                "status": {
                    "description": "Segments with one of these neuprint statuses will be labeled Neurons\n",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["Traced", "Anchor"],
                }
            }
        },
        # TODO Move this into a larger sub-structure for all :Meta info
        "postHPThreshold": {
            "description": "Which confidence threshold to use when calculating each connection's weightHP\n",
            "type": "number",
            "default": 0.0
            # Note: On both hemibrain and MANC, we used 0.7
        },

        # TODO Move this into a larger sub-structure for all :Meta info
        "postHighAccuracyThreshold": {
            "description": "Which confidence threshold to use when calculating each connection's standard 'weight'\n",
            "type": "number",
            "default": 0.0
            # Note: On both hemibrain we used 0.5, on MANC we used 0.4
        },

        # TODO: roi hierarchy
        # TODO: weightHP, and get rid of weightHR (a failed experiment, I think.)
        #       If I don't get rid if it, then I need to avoid pre-filtering by min-confidence,
        #       and apply that filter before neuron/weight export (but not synapse export).
    }
}

@PrefixFilter.with_context('Neuprint')
def _export_neuprint(cfg, point_df, partner_df, ann):
    """
    Export CSV files for each of the following:

    Nodes:
        - Synapse
        - Neuron
        - SynapseSet
        - Meta

    Edges:
        - Neuron -[:Contains]-> SynapseSet
        - SynapseSet -[:Contains]-> Synapses
        - Neuron -[:ConnectsTo]-> Neuron
        - SynapseSet -[:ConnectsTo]-> SynapseSet
        - Synapse -[:SynapsesTo]-> Synapse
    """
    if not cfg['export-neuprint-snapshot']:
        logger.info("Not generating neuprint snapshot.")
        return

    if not cfg['neuprint']['dataset']:
        raise RuntimeError("Please define a dataset name in neuprint:dataset")

    # Drop body 0 entirely.
    point_df = point_df.loc[point_df['body'] != 0]
    partner_df = partner_df.loc[(partner_df['body_pre'] != 0) & (partner_df['body_post'] != 0)]

    # with Timer("Decoding zyx coordinates from post_id"):
    #     partner_df[[*'zyx']] = decode_coords_from_uint64(partner_df['post_id'].values)

    point_df, partner_df = _determine_neuprint_synapse_rois(cfg, point_df, partner_df)
    _export_neuprint_synapses(cfg, point_df)
    _export_neuprint_neurons(cfg, point_df, partner_df, ann)
    connectome = _export_neuprint_neuron_connections(cfg, partner_df)
    _export_neuprint_synapse_connections(partner_df)
    _export_synapsesets(cfg, partner_df, connectome)







