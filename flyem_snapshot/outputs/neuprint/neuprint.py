"""
Export a connectivity snapshot in the form of CSV files that can be converted to a neo4j neuprint database.
"""
import os
import logging

from neuclease import PrefixFilter

from .annotations import neuprint_segment_annotations
from .meta import export_neuprint_meta
from .neuroglancer import NeuroglancerSettingsSchema, export_neuroglancer_json_state
from .indexes import IndexesSettingsSchema, export_neuprint_indexes_script
from .segment import export_neuprint_segments, export_neuprint_segment_connections
from .synapse import export_neuprint_synapses, export_neuprint_synapse_connections
from .synapseset import export_synapsesets

logger = logging.getLogger(__name__)

RoiSynapsePropertiesSchema = {
    "description": "Synapse properties derived from the roi values in a single roi-set (roi column).",
    "default": {},
    "additionalProperties": {
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

            # By default, the synapse property is assigned
            # the value of the ROI segment ID itself.
            "default": "x"
        }
    }
}

NeuprintSchema = {
    "description": "settings for constructing a neuprint snapshot database",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": [],
    "properties": {
        "export-neuprint-snapshot": {
            "description": "If true, export CSV files needed to construct a neo4j neuprint database.",
            "type": "boolean",
            "default": False,
        },
        "meta": {
            "description":
                "File path to a separate config file describing the Neuprint :Meta info.\n"
                "See the command-line argument: flyem-snapshot -M\n",
            "type": "string",
            "default": ""
        },
        "neuroglancer": NeuroglancerSettingsSchema,
        "roi-set-names": {
            "description":
                "The set of ROI sets (ROI column names) from the input synapse\n"
                "table to actually copy into neuprint as ROIs.\n",
            "default": None,
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"}
                },
                {
                    "type": "null"
                }
            ]
        },
        "roi-synapse-properties": {
            "description":
                "Synapse properties derived from the roi values in a single roi-set (roi column).\n"
                "The canonical example of this is the CNS optic lobe columns, which define special\n"
                "properties for hex1,hex2 based on the ROI segment ID values.\n",
            "additionalProperties": RoiSynapsePropertiesSchema,
            "default": {},
        },
        "neuron-label-criteria": {
            "description":
                "In neuprint, all synaptic bodies are :Segment nodes, but only 'important' ones also become :Neuron nodes.\n"
                "The following settings determine which criteria will promote a :Segment to a :Neuron\n",
            "default": {},
            "additionalProperties": False,
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
        "annotation-property-names": {
            "description":
                "Mapping from annotation columns to neuprint property names.\n"
                "Any columns NOT mentioned here will be translated to property names according to predefined rules.\n"
                "Note: Currently, all spatial (point) properties MUST contain the word 'Location' in the name.",
            "additionalProperties": {
                "type": "string"
            },
            "default": {},
        },
        "indexes": IndexesSettingsSchema,
        "processes": {
            "description":
                "For steps specifically in the neuprint build process which could\n"
                "benefit from multiprocessing, how many processes should be used?\n"
                "By default, use the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },

        # TODO: roi hierarchy
        # TODO: weightHP, and get rid of weightHR (a failed experiment, I think.)
        #       If I don't get rid if it, then I need to avoid pre-filtering by min-confidence,
        #       and apply that filter before neuron/weight export (but not synapse export).

        # TODO: Optionally update ROI meshes in DVID...
    }
}


@PrefixFilter.with_context('neuprint')
def export_neuprint(cfg, point_df, partner_df, ann, body_sizes, last_mutation):
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

    TODO: Mito
    """
    if not cfg['export-neuprint-snapshot']:
        logger.info("Not generating neuprint snapshot.")
        return

    os.makedirs('neuprint', exist_ok=True)

    # Drop body 0 entirely.
    point_df = point_df.loc[point_df['body'] != 0]
    partner_df = partner_df.loc[(partner_df['body_pre'] != 0) & (partner_df['body_post'] != 0)]

    neuprint_ann = neuprint_segment_annotations(cfg, ann)
    neuron_prop_names, dataset_totals, roi_totals = export_neuprint_segments(cfg, point_df, partner_df, neuprint_ann, body_sizes)
    export_neuprint_meta(cfg, last_mutation, neuron_prop_names, dataset_totals, roi_totals, neuprint_ann)
    export_neuroglancer_json_state(cfg, last_mutation)
    export_neuprint_indexes_script(cfg, neuron_prop_names, roi_totals.index)

    connectome = export_neuprint_segment_connections(cfg, partner_df)
    export_synapsesets(cfg, partner_df, connectome)

    export_neuprint_synapses(cfg, point_df)
    export_neuprint_synapse_connections(partner_df)