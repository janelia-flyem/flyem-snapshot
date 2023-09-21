"""
Export function for the neuprint :Meta node.

In neuprint, global constants such as the dataset name and ROI hierarchy
are stored in a special node in the neo4j database. We chose the label :Meta
for this node.  Like all other data we ingest into neo4j, we create a CSV
file with the data for this node.  Since there is only a single :Meta node,
the generated CSV file will have only one row.
"""
import copy
import json
import logging
from collections.abc import Sequence, Mapping

import pandas as pd

from confiddler import load_config
from neuclease.util import dump_json

from .util import append_neo4j_type_suffixes, NEUPRINT_TYPE_OVERRIDES

logger = logging.getLogger(__name__)

NeuroglancerInfoLayerSchema = {
    "default": {},
    "properties": {
        "host": {
            "description": "DVID server",
            "type": "string",
            "default": ""
        },
        "uuid": {
            "description": "",
            "type": "string",
            "default": ""
        },
        "dataType": {
            "description":
                "I have no idea what this is for.\n"
                "It does not directly correspond to anything in the neuroglancer JSON.\n"
                "Choices: 'segmentation' or 'grayscalejpeg'",
            "type": "string",
            "enum": ["segmentation", "grayscalejpeg"]
            # no default
        },
    }
}

NeuroglancerInfoSchema = {
    # I am not clear on why some neuroglancer settings are
    # in 'neuroglancerInfo' and others are in 'neuroglancerMeta'.
    # Neither one of them has a 1-to-1 correspondence with
    # neuroglancer's own JSON structure.
    "default": {},
    "description": "neuroglancer settings for segmentation and grayscale layers\n",
    "properties": {
        "segmentation": NeuroglancerInfoLayerSchema,
        "grayscalejpeg": NeuroglancerInfoLayerSchema,
    }
}

NeuroglancerMetaLayerSchema = {
    # It's not clear which of these properties is always necessary,
    # aside from 'name', 'source' and (perhaps) 'dataType'.
    # It's also not clear why we don't just use precisely the same
    # keys and values that are used within neuroglancer's own JSON structure.
    "default": {},
    "properties": {
        "name": {
            "type": "string",
            # no default
        },
        "source": {
            "type": "string",
            # no default
        },
        "dataType": {
            "description": "In the neuroglancer layer JSON, this is just called 'type'.\n",
            "type": "string",
            "enum": ["image", "segmentation", "annotation"],
            # no default
        },
        "host": {
            "description": "DVID server",
            "type": "string",
            "default": ""
        },
        "dataInstance": {
            "type": "string",
            "default": ""
        },
        # FIXME: In the MANC neuprint :Meta, this is a boolean,
        # but that makes no sense from neuroglancer's point of view.
        "linkedSegmentationLayer": {
            "description":
                "Normally, this neuroglancer setting refers to which\n"
                "segmentation layer is associated with an annotation layer.\n"
                "It's not clear why we have a boolean here.\n",
            "type": "boolean",
            # no default
        },
    }
}

NeuroglancerMetaSchema = {
    "type": "array",
    "items": NeuroglancerMetaLayerSchema,
    "default": []
}

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

        # Path to a json/yaml file
        {
            "type": "string",
            "pattern": r".*\.(json|yaml)$"
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
        # "visible": {
        #     "description": "This is redundant with the 'neuronColumnsVisible' config setting.",
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
        "neuroglancerInfo": NeuroglancerInfoSchema,
        "neuroglancerMeta": NeuroglancerMetaSchema,

        "neuronColumns": NeuronColumnsSchema,

        "neuronColumnsVisible": {
            # Discussion here:
            # https://github.com/janelia-flyem/flyem-recon/issues/110
            "description": "List of neuron columns to make visible by default in neuprint neuron searches.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": ["bodyId", "type", "status"]
        },

        # # I think this isn't actually used,
        # # so I'm omitting it for now...
        # # See https://github.com/janelia-flyem/flyem-recon/issues/110
        # "neuronFilter": {
        #     "description": {
        #         "List of neuron properties to include in the neuron filters UI.\n",
        #     "type": "array",
        #     "items": {"type": "string"},
        #     "default": ["bodyId", "type", "status"]
        #     }
        # },

        "primaryRois": {
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
        #   "hasROI": true,
        #   "isNerve": false,
        #   "isPrimary": true,
        #   "parent": "ventral nerve core",
        #   "post": 1352587,
        #   "pre": 193036,
        #   "showHierarchy": true
        # }
        #
        # Note:
        #   I can find no references to hasROI or showHierarchy
        #   in neuPrintHTTP or neuPrintExplorer(Plugins),
        #   so I'm going to INGORE them.

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

        # #   I have no idea what this field is used for.
        # #   If it's not used, let's drop it.
        # "neuropilRois": {
        #     "description": "The list of ROIs which are neuropil compartments.\n",
        #     "type": "array",
        #     "items": {"type": "string"},
        #     "default": []
        # },

        "excludeFromOverview": {
            "description":
                "The list of ROIs which should not be included in\n"
                "Neuprint Explorer's front-page ROI-to-ROI connection summary.\n"
                "Usually nerves should be excluded, at least.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": []
        },

        # I can't find any references to either hasROI or showHierarchy
        # anywhere in neuprintExplorer or neuprintHTTP
        # I'm going to ignore these for now.
        # "showHierarchy": {},
        # "hasROI": {},

        # This will be automatically populated using the properties we wrote into :Segment nodes.
        # "neuronProperties",
    }
}


def load_roi_hierarchy(rhcfg):
    if rhcfg is None:
        return None

    if isinstance(rhcfg, str):
        rhcfg = load_config(rhcfg, RoiHierarchySchema)
        return load_roi_hierarchy(rhcfg)

    if isinstance(rhcfg, Mapping):
        if rhcfg:
            return {k: load_roi_hierarchy(v) for k,v in rhcfg.items()}
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
    'dataset', 'tag',
    'voxelSize', 'voxelUnits',
    'info', 'logo', 'meshHost',
    'neuroglancerInfo', 'neuroglancerMeta',
    'postHighAccuracyThreshold', 'preHPThreshold', 'postHPThreshold',
    'totalPreCount', 'totalPostCount',
    'superLevelRois',
    'primaryRois', 'nonHierarchicalROIs',
    'roiInfo', 'roiHierarchy',
    'neuronProperties',
    'neuronColumns',
    'neuronColumnsVisible',

    'uuid', 'latestMutationId', 'lastDatabaseEdit',

    # 'objectProperties', -- needed?

    # obsolete, I think:
    # 'nerveRois', 'neuropilRois',
    # 'statusDefinitions',
    # 'neuronFilter',
]


def export_neuprint_meta(cfg, last_mutation, neuron_prop_names, dataset_totals, roi_totals, neuprint_ann):
    """
    TODO: objectProperties?
    """
    metacfg = cfg['meta']

    # This will become the final metadata (before conversion to CSV)
    meta = {}

    # Copy this subset directly from the config
    verbatim_keys = (
        'dataset', 'tag',
        'voxelSize', 'voxelUnits',
        'info', 'logo', 'meshHost',
        'neuroglancerInfo', 'neuroglancerMeta',
        'postHighAccuracyThreshold', 'preHPThreshold', 'postHPThreshold',
        # 'totalPreCount', 'totalPostCount',
        # 'superLevelRois',
        'primaryRois', 'nonHierarchicalROIs',
        # 'roiInfo', 'roiHierarchy',
        # 'neuronProperties',

        # 'neuronColumns'
        'neuronColumnsVisible',

        # These will be used if the user gave them, but overwritten otherwise.
        'uuid', 'latestMutationId', 'lastDatabaseEdit',

        # obsolete, I think:
        # 'nerveRois',  # <-- used by config, but not copied verbatim to meta
        # 'neuropilRois',
        # 'statusDefinitions',
        # 'neuronFilter',
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
    meta['neuronProperties'] = neuron_prop_names

    # These are created with info from the config (and annotations).
    rh = load_roi_hierarchy(metacfg['roiHierarchy'])
    meta['roiHierarchy'] = construct_neuprint_roi_hierarchy(rh)
    meta['roiInfo'] = _load_roi_info(metacfg, roi_totals, rh)
    meta['neuronColumns'] = _load_neuron_columns(metacfg, neuprint_ann)

    # Just for debug
    dump_json(meta, 'neuprint/meta.json')

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

        # Note:
        #   I can find no references to 'hasROI' or 'showHierarchy'
        #   in neuPrintHTTP or neuPrintExplorer(Plugins),
        #   so I'm going to ignore them.
        #
        # entry['hasROI'] = ???
        # entry['showHierarchy'] = ???

    return roi_info


def _load_neuron_columns(metacfg, neuprint_ann):
    neuron_columns = copy.deepcopy(metacfg['neuronColumns'])
    for item in neuron_columns:
        col = item['id']
        item['visible'] = (col in metacfg['neuronColumnsVisible'])
        item['name'] = (item['name'] or col)
        if item['choices'] == 'auto':
            item['choices'] = sorted(filter(lambda choice: choice != '', neuprint_ann[col].unique()))
    return neuron_columns


def _export_meta_as_csv(meta):
    assert set(meta.keys()) == set(META_PROPERTIES)

    # For columns with neo4j types like float[] or string[],
    # convert them to a string using ';' as list delimiter.
    for key in list(meta.keys()):
        if NEUPRINT_TYPE_OVERRIDES.get(key, '').endswith('[]'):
            meta[key] = meta[key] or []
            meta[key] = ';'.join(map(str, meta[key]))

    # For everything else, convert to JSON if it's a dict.
    for key in list(meta.keys()):
        if isinstance(meta[key], dict):
            meta[key] = json.dumps(meta[key])

    # And these are JSON strings, too.
    meta['neuroglancerMeta'] = json.dumps(meta['neuroglancerMeta'])
    meta['neuronColumns'] = json.dumps(meta['neuronColumns'])
    meta['neuronColumnsVisible'] = json.dumps(meta['neuronColumnsVisible'])

    # One column per key, with exactly one row.
    meta_df = pd.DataFrame({k: [v] for k,v in meta.items()})
    meta_df = append_neo4j_type_suffixes(meta_df)
    meta_df.to_csv('neuprint/Neuprint_Meta.csv', index=False, header=True)
    logger.info("Wrote Neuprint_Meta.csv")
