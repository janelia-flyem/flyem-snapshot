"""
Export function for the neuprint :Meta node.

In neuprint, global constants such as the dataset name and ROI hierarchy
are stored in a special node in the neo4j database. We chose the label :Meta
for this node.  Like all other data we ingest into neo4j, we create a CSV
file with the data for this node.  Since there is only a single :Meta node,
the generated CSV file will have only one row.
"""
import re
import copy
import json
import logging
from textwrap import dedent
from collections.abc import Sequence, Mapping

import pandas as pd

from confiddler import load_config
from neuclease.util import dump_json

from .util import append_neo4j_type_suffixes

# These :Meta properties will be exported for neo4j with special dtypes
# in the column names and written as semicolon-delimited lists.
NEUPRINT_META_LIST_PROPERTIES = {
    'voxelSize': 'float[]',
    'primaryRois': 'string[]',
    'superLevelRois': 'string[]',
    'nonHierarchicalROIs': 'string[]',
    'overviewRois': 'string[]',
}

logger = logging.getLogger(__name__)

_ = StatusDefinitionsSchema = {
    # This is what the hemibrain/MANC neuprint repos had stored:
    # These are not 'status' values, but rather 'statusLabel', and this list is incomplete.
    # I therefore infer that no one is actually using this field, so let's just stop storing it.
    # "default": {
    #   "Roughly traced": "neuron high-level shape correct and validated by biological expert",
    #   "Prelim Roughly traced": "neuron high-level shape most likely correct or nearly complete, not yet validated by biological expert",
    #   "Anchor": "Big segment that has not been roughly traced",
    #   "0.5assign": "Segment fragment that is within the set required for a 0.5 connectome"
    # }

    "description":
        "Deprecated.  Do not bother with this.\n"
        "Mapping of status names to textual descriptions of what those names are supposed to represent.\n"
        "In existing neuprint databases, we apparently stored an incomplete list of _statusLabel_ definitions, not status definitions.\n",
    "additionalProperties": {
        "type": "string"
    },
    "default": {},
}

RoiHierarchyDefinition = {
    "oneOf": [
        # no children
        {
            "type": "null"
        },

        # leaf children, expressed as a simple list
        # instead of a dict where all values are null.
        {
            "type": "array",
            "items": {"type": "string"}
        },

        # A string containing one of the following:
        # - a path to a json/yaml file OR
        # - a reference to the name of one of the roi-sets named in the top-level config,
        #   using the syntax 'roiset://my-roiset'.  In that case, it's effectively similar
        #   to the above alternative of inserting a list of children,
        #   i.e. every ROI in the roiset becomes a 'leaf' ROI.
        {
            "type": "string",
            "pattern": r"(.*\.(json|yaml))|(roiset://.*)$"
        },

        # Recursive definition
        {
            "type": "object",
            "additionalProperties": {
                "$ref": "#/definitions/rh-def-recursive"
            }
        }
    ],
}

# This defines a recursive schema for the ROI hierarchy in the config.
# There are multiple ways to list bottom-level (leaf) ROIs:
# - Leaf ROIs have no children, as indicated by EITHER an empty dict or a null.
#
#
# Example config:
# CNS:
#   MB(R):
#     aL(R): {}            # <-- leaf ROI
#     bL(R): null          # <-- also a leaf ROI
#   MB(L): [al(L), bL(L)]  # <-- two leaf ROIs
#
#   ME(R):
#     ME(R)-columns: [ME_col_1, ME_col_2, ...] # <-- leaf ROIs
#   ME(L):
#     ME(L)-columns: /path/to/ME(L)-columns.json  # <-- Arbitrary JSON will be loaded and swapped in
#
RoiHierarchySchema = {
    "description":
        "The hierarchy of ROI parent-child relationships,\n"
        "expressed here as dict-of-dict, with a few optional tweaks.\n"
        "This structure will be translated into the roiHierarchy JSON\n"
        "structure that neuprint actually uses.",

    "additionalProperties": {
        "$ref": "#/definitions/rh-def-recursive"
    }
}

NeuronColumnSchema = {
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "name": {
            "description": "human-readable name",
            "type": "string",
            # By default, use the id
            "default": ""
        },
        "id": {
            "description": "Neuron property name",
            "type": "string",
            # no default
        },
        # The 'visible' property is inserted into the meta,
        # but we don't ask for it in this part of the config.
        # Instead, the config has a separate list for neuronColumnsVisible,
        # which is used to auto-populate the 'visible' flag for each column.
        # "visible": {
        #     "type": "boolean",
        #     "default": False
        # },
        "choices": {
            "description":
                "Auto-complete choices for this property.\n"
                "If no auto-complete should be loaded, then use null.\n"
                "Write 'auto' to have the list auto-populated from the possible property values.\n",
            "default": None,
            "oneOf": [
                {
                    "type": "null"
                },
                {
                    "type": "array",
                    "items": {"type": "string"},
                },
                {
                    "type": "string",
                    "enum": ["auto"]
                }
            ]
        }
    }
}

NeuronColumnsSchema = {
    "description":
        "The list of Neuron properties which should appear in NeuprintExplorer's\n"
        "FindNeurons filter widget, along with settings for each.\n",
    "type": "array",
    "items": NeuronColumnSchema,
    "default": [
        {
            'id': 'bodyId',
            'name': 'body ID',
            'choices': None,
        },
        {
            'id': 'type',
            'choices': None,
        },
        {
            'id': 'status',
            'choices': 'auto',
        }
    ]
}

NeuprintMetaSchema = {
    "default": {},
    "definitions": {
        "rh-def-recursive": RoiHierarchyDefinition
    },
    "additionalProperties": False,
    "properties": {
        "dataset": {
            "description":
                "The name of the dataset in neuprint, e.g. 'cns'\n"
                "Do NOT include the version here (such as cns:v1.0)\n",
            "type": "string",
            # no default
        },
        "tag": {
            "description": "The version tag (if any), e.g. 'v1.0'",
            "type": "string",
            "default": ""
        },
        "hideDataSet": {
            "description": "Should this dataset be hidden from the neuprintExplorer drop-down menu?\n",
            "type": "boolean",
            "default": False
        },
        "voxelSize": {
            "type": "array",
            "items": {"type": "number"},
            "default": [8.0, 8.0, 8.0]
        },
        "voxelUnits": {
            "type": "string",
            "default": "nanometers"
        },
        "info": {
            "type": "string",
            # "default": "https://www.janelia.org/project-team/flyem"
            "default": "https://www.janelia.org/project-team/flyem"
        },
        "logo": {
            "description":
                "URL to the dataset logo.  Can be a relative path to a file in the neuprintHTTP server.\n"
                "(Default URL is the flyem logo, but you should pick something better.)\n",
            "type": "string",

            # "default": "https://www.janelia.org/sites/default/files/Project%20Teams/Fly%20EM/hemibrain_logo.png",
            # "default": "public/VNC_all_neuropils.png",

            # This is just the generic FlyEM logo.
            "default": "https://www.janelia.org/sites/default/files/styles/epsa_580x580/public/flyEM-logo_580x580.jpg",
        },
        "postHighAccuracyThreshold": {
            "description":
                "Which confidence threshold to use when calculating each connection's standard 'weight'.\n"
                "This is determined by finding a 'balanced' point on our synapse precision/recall curve.\n"
                "For hemibrain we used 0.5; for MANC we used 0.4\n",
            "type": "number",
            "default": 0.0
        },
        "postHPThreshold": {
            "description": "Which confidence threshold to use when calculating each connection's weightHP\n"
                "This is determined via analysis of our on our synapse precision/recall curve to select\n"
                "a point that favors higher precision at the expense of lower recall.\n"
                "For both hemibrain and MANC, we used 0.7\n",
            "type": "number",
            "default": 0.0
        },
        "preHPThreshold": {
            "description": "Deprecated. Do not set.",
            "type": "number",
            "default": 0.0
        },
        "meshHost": {
            "description":
                "URL of a DVID server from which ROI meshes (and maybe neuron skeletons??) can be downloaded.\n"
                "Example values:\n"
                "  https://hemibrain-dvid.janelia.org\n"
                "  https://manc-dvid.janelia.org\n",
            "type": "string",
            # no default
        },
        "uuid": {
            "description":
                "The DVID UUID from which the neuprint snapshot was generated.\n"
                "Leave this unset to have it automatically set during the snapshot export.\n"
                "Explicitly configure this only if you are building a neuprint database from flat\n"
                "synapse tables (without DVID). Otherwise, let it be configured automatically.\n",
            "type": "string",
            "default": ""

        },
        "latestMutationId": {
            "description":
                "The mutation ID of the main segmentation in DVID for the snapshot used.\n"
                "Leave this unset to have it automatically set during the snapshot export.\n"
                "Configure this only if you are building a neuprint database from flat synapse tables (without DVID).\n",
            "type": "integer",
            "default": 0
        },
        "lastDatabaseEdit": {
            "description":
                "The timestamp of the most recent edit in the main segmentation in DVID for the snapshot used.\n"
                "Leave this unset to have it automatically set during the snapshot export.\n"
                "Configure this only if you are building a neuprint database from flat synapse tables (without DVID).\n",
            "type": "string",
            "default": ""
        },

        "neuronColumns": NeuronColumnsSchema,

        "neuronColumnsVisible": {
            # Discussion here:
            # https://github.com/janelia-flyem/flyem-recon/issues/110
            "description": "List of neuron columns to make visible by default in neuprint neuron searches.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": ["bodyId", "type", "status"]
        },

        "primaryRois": {
            "description": dedent("""\
                A typical neuprint dataset includes ROIs which may overlap with one another.
                For example, most datasets define an ROI hierarchy, in which 'parent' ROIs
                encompass child ROIs.  (Obviously, parents and their children overlap.)
                Another example is the CNS optic lobe, which defines intersecting 'layer' and 'column' ROIs.

                Since ROIs can overlap, then certain queries will return duplicate synapses and/or inflate
                the apparent synaptic weight of a connection or neuron, if the user were to naively
                aggregate synapse weights across all ROIs.

                One way to avoid such double-counting is to restrict a query to 'primary' ROIs only.
                The 'primary' ROIs are a specific subset of ROIs -- not necessarily at the bottom of the
                hierarchy, which are guaranteed not to overlap with each other and guaranteed to cover all
                other ROIs.  Every synapse is either included in exactly one primary ROI or not included in
                _any_ ROI at all (primary or otherwise).
                """),
            "type": "array",
            "items": {"type": "string"},
            # no default
        },
        "nonHierarchicalROIs":{
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"},
                },
                {
                    "type": "null"
                }
            ],
            "default": None
        },

        "overviewRois": {
            "description":
                "Which ROIs to show in the neuprint explorer overview heatmap.\n"
                "By default, neuprint explorer shows the primary ROIs, except for those we list in 'excludeFromOverview'.\n",
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"},
                },
                {
                    "type": "null"
                }
            ],
            "default": None
        },
        "overviewOrder": {
            "description":
                "Whether neuprint explorer should auto-choose the order of rows and columns in the overview heatmap (clustered)\n"
                "or whether it should use the order you specified in overviewRois (explicit)\n",
            "type": "string",
            "enum": ["clustered", "explicit"],
            "default": "clustered",
        },

        # The roiHierarchy is written into this config in a different
        # JSON format than the JSON will be written into neuprint.
        "roiHierarchy": RoiHierarchySchema,

        # This is deprecated; we don't use it any more.
        # (See comments above.)
        # "statusDefinitions": StatusDefinitionsSchema,

        # These will be automatically set from the synapse tables.
        # "totalPreCount",
        # "totalPostCount",

        # This has been renamed 'primaryRois', but we keep this key for backwards compatibility.
        # It will be populated using the primaryRois list.
        # "superLevelRois",

        # "roiInfo",
        # We'll populate "roiInfo" by weaving together a few different lists.
        # Ultimately, the dict will have an entry for every ROI,
        # looking something like this:
        #
        # "Ov(L)": {
        #   "description": "Ovoid (left)",
        #   "excludeFromOverview": false,
        #   "isNerve": false,
        #   "isPrimary": true,
        #   "parent": "ventral nerve core",
        #   "post": 1352587,
        #   "pre": 193036,
        # }

        "roiDescriptions": {
            "description": "dict of {roi: description}",
            "additionalProperties": {
                "type": "string"
            }
        },

        "nerveRois": {
            "description":
                "The list of ROIs which are nerves, rather than neuropils.\n"
                "Nerves are excluded from the completeness table.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },

        "excludeFromOverview": {
            "description":
                "The list of ROIs which should not be included in\n"
                "Neuprint Explorer's front-page ROI-to-ROI connection summary.\n"
                "Usually nerves should be excluded, at least.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },

        # This will be automatically populated using the properties we wrote into :Segment nodes.
        # "neuronProperties",
    }
}


def load_roi_hierarchy(rhcfg, roisets):
    if rhcfg is None:
        return None

    if isinstance(rhcfg, str):
        if rhcfg.startswith('roiset://'):
            roiset_name = rhcfg[len('roiset://'):]
            rois = roisets[roiset_name].keys()
            return {k: None for k in rois}
        elif re.match('.*(yaml|json)$', rhcfg):
            rhcfg = load_config(rhcfg, RoiHierarchySchema)
            return load_roi_hierarchy(rhcfg, roisets)
        raise RuntimeError(f"bad string entry in roi-hierarchy: '{rhcfg}'")

    if isinstance(rhcfg, Mapping):
        if rhcfg:
            return {k: load_roi_hierarchy(v, roisets) for k,v in rhcfg.items()}

        # We translate {} to None, though either works.
        return None

    if isinstance(rhcfg, Sequence):
        return {k: None for k in rhcfg}

    raise AssertionError("Shouldn't reach this line")


def construct_neuprint_roi_hierarchy(rh):
    """
    Convert from the simple dict hierarchy we loaded from the
    config into the asinine format used in neuprint's :Meta node.

    Args:
        rh:
            An roi hierarchy from the config,
            standardizsed via load_roi_hierarchy()
            so it consists solely of nested dicts
            whose values are either dict or None.

    Returns:
        A data structure of nested alternating dicts and lists,
        with explicit 'name' and 'children' keys instead of just
        letting the key represent the name and the values represent
        the children.

        Example:

            >>> rh = {
            ...    'A': {
            ...         'B': None,
            ...         'C': {
            ...             'D': None,
            ...             'E': None
            ...         }
            ...     }
            ... }

            >>> construct_neuprint_roi_hierarchy(rh)
            {'name': 'A',
             'children': [
                {'name': 'B'},
                {'name': 'C',
                 'children': [
                    {'name': 'D'},
                    {'name': 'E'}
                 ]
                }
             ]
            }
    """
    def _convert(d):
        rois = []
        for k, v in d.items():
            roi = {'name': k}
            if v:
                roi['children'] = _convert(v)
            rois.append(roi)
        return rois

    assert len(rh) == 1, \
        "Expected only 1 top-level ROI at the root of the hierarchy."
    return _convert(rh)[0]


# This is the complete list of :Meta properties.
# See assert statement below.
META_PROPERTIES = [
    'dataset', 'tag', 'hideDataSet',
    'voxelSize', 'voxelUnits',
    'info', 'logo', 'meshHost',
    'postHighAccuracyThreshold', 'preHPThreshold', 'postHPThreshold',
    'totalPreCount', 'totalPostCount',
    'superLevelRois',
    'primaryRois', 'nonHierarchicalROIs',
    'overviewRois', 'overviewOrder',
    'roiInfo', 'roiHierarchy',
    'neuronProperties',
    'neuronColumns',
    'neuronColumnsVisible',

    'uuid', 'latestMutationId', 'lastDatabaseEdit',

    # 'nerveRois'  # <-- used in the config above, but not copied verbatim to meta.
]


def export_neuprint_meta(cfg, last_mutation, neuron_df, dataset_totals, roi_totals, roisets):
    """
    """
    metacfg = cfg['meta']

    # This will become the final metadata (before conversion to CSV)
    meta = {}

    # Copy this subset directly from the config
    verbatim_keys = (
        'dataset', 'tag', 'hideDataSet',
        'voxelSize', 'voxelUnits',
        'info', 'logo', 'meshHost',
        'postHighAccuracyThreshold', 'preHPThreshold', 'postHPThreshold',
        # 'totalPreCount', 'totalPostCount',
        # 'superLevelRois',
        'primaryRois', 'nonHierarchicalROIs',
        'overviewRois', 'overviewOrder',
        # 'roiInfo', 'roiHierarchy',
        # 'neuronProperties',

        # 'neuronColumns'
        'neuronColumnsVisible',

        # These will be used if the user gave them, but overwritten otherwise.
        'uuid', 'latestMutationId', 'lastDatabaseEdit',

        # 'nerveRois',
    )
    for k in verbatim_keys:
        meta[k] = metacfg[k]

    # These are listed under two names.
    # (superLevelRois is the legacy name.)
    meta['superLevelRois'] = meta['primaryRois']

    # If our snapshot came from a DVID UUID, we can fill in these
    # values automatically (if the user's config didn't override).
    if last_mutation:
        if not metacfg['uuid']:
            meta['uuid'] = last_mutation['uuid']
        if not metacfg['latestMutationId']:
            meta['latestMutationId'] = last_mutation['mutid']
        if not metacfg['lastDatabaseEdit']:
            meta['lastDatabaseEdit'] = str(last_mutation['timestamp'])

    # These were determined during the snapshot export.
    meta['totalPreCount'] = dataset_totals['pre']
    meta['totalPostCount'] = dataset_totals['post']

    neuron_prop_splits = [name.split(':', 1) for name in neuron_df.columns]
    neuron_prop_splits = filter(lambda s: len(s) > 1 and s[0], neuron_prop_splits)
    neuron_property_types = dict(neuron_prop_splits)
    meta['neuronProperties'] = neuron_property_types

    # These are created with info from the config (and annotations).
    rh = load_roi_hierarchy(metacfg['roiHierarchy'], roisets)
    meta['roiHierarchy'] = construct_neuprint_roi_hierarchy(rh)
    meta['roiInfo'] = _load_roi_info(metacfg, roi_totals, rh)
    meta['neuronColumns'] = _load_neuron_columns(metacfg, neuron_df)

    # Just for debug
    dump_json(meta, 'neuprint/Neuprint_Meta_debug.json')

    # Export for neo4j!
    _export_meta_as_csv(meta)


def _load_roi_info(metacfg, roi_totals, roi_hierarchy):
    def _parents(d, root):
        parents = {}
        for k, v in d.items():
            parents[k] = root
            if v:
                parents.update(_parents(v, k))
        return parents

    # Construct a flat dict of {roi: parent}
    # from the roi hierarchy.
    roi_parents = _parents(roi_hierarchy, None)

    # Use sets for quick lookups
    primary_rois = set(metacfg['primaryRois'])
    nerve_rois = set(metacfg['nerveRois'])
    exclude_from_overview = set(metacfg['excludeFromOverview'])

    # roi_totals contains the full set of ROIs,
    # even if some of them weren't listed in the config.
    roi_info = {}
    for row in roi_totals.reset_index().itertuples():
        roi = row.roi
        roi_info[roi] = entry = {}
        entry['description'] = metacfg['roiDescriptions'].get(roi, '')

        entry['excludeFromOverview'] = (roi in exclude_from_overview)
        entry['isNerve'] = (roi in nerve_rois)
        entry['isPrimary'] = (roi in primary_rois)

        if roi not in roi_parents:
            logger.warning(f"ROI {roi} is not found in the roiHierarchy")
        entry['parent'] = roi_parents.get(roi, None)

        entry['post'] = row.post
        entry['pre'] = row.pre

    return roi_info


def _load_neuron_columns(metacfg, neuron_df):
    neuron_df = neuron_df.rename(columns={c: c.split(':')[0] for c in neuron_df.columns})
    neuron_columns = copy.deepcopy(metacfg['neuronColumns'])
    for item in neuron_columns:
        col = item['id']
        item['visible'] = (col in metacfg['neuronColumnsVisible'])
        item['name'] = (item['name'] or col)
        if col not in neuron_df.columns:
            logger.error(f"Meta config lists column '{col}' which is not present in the data.")
            continue
        if item['choices'] == 'auto':
            # Note:
            #   Empty strings were already removed when the annotations
            #   were prepped for neuprint, so none of these choices will be "".
            item['choices'] = sorted(neuron_df[col].dropna().unique())
    return neuron_columns


def _export_meta_as_csv(meta):
    meta = copy.deepcopy(meta)
    assert set(meta.keys()) == set(META_PROPERTIES)
    dataset = meta['dataset']
    meta[':Label'] = f"Meta;{dataset}_Meta"

    # For columns with neo4j types like float[] or string[],
    # convert them to a string using ';' as list delimiter.
    for key in list(meta.keys()):
        if NEUPRINT_META_LIST_PROPERTIES.get(key, '').endswith('[]'):
            meta[key] = meta[key] or []
            meta[key] = ';'.join(map(str, meta[key]))

    # For everything else, convert to JSON if it's a dict.
    for key in list(meta.keys()):
        if isinstance(meta[key], dict):
            meta[key] = json.dumps(meta[key])

    # And these are JSON strings, too.
    meta['neuronColumns'] = json.dumps(meta['neuronColumns'])
    meta['neuronColumnsVisible'] = json.dumps(meta['neuronColumnsVisible'])

    # We can't export as bool since neo4j requires 'true' not 'True'.
    # Convert to string.
    # https://neo4j.com/docs/operations-manual/4.4/tools/neo4j-admin/neo4j-admin-import/#import-tool-header-format-properties
    meta['hideDataSet:boolean'] = str(meta['hideDataSet']).lower()
    del meta['hideDataSet']

    # One column per key, with exactly one row.
    meta_df = pd.DataFrame({k: [v] for k,v in meta.items()})
    meta_df = meta_df.rename(columns={
        prop: f'{prop}:{dtype}'
        for prop,dtype in NEUPRINT_META_LIST_PROPERTIES.items()
    })
    meta_df = append_neo4j_type_suffixes(meta_df)
    meta_df.to_csv('neuprint/Neuprint_Meta.csv', index=False, header=True)
    logger.info("Wrote Neuprint_Meta.csv")
