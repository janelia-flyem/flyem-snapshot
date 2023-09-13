"""
Export a connectivity snapshot from a DVID segmentation,
along with other denormalizations.
"""
import logging
from neuclease import PrefixFilter

from .synapse import export_neuprint_synapses, export_neuprint_synapse_connections
from .segment import export_neuprint_segments, export_neuprint_segment_connections
from .synapseset import export_synapsesets

logger = logging.getLogger(__name__)

RoiSetMetaFieldsSchema = {
    "description":
        "Special neuprint settings related to the roi-sets (roi columns)\n"
        "that were loaded for the currently processing snapshot.\n",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
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
        "roi-set-meta": {
            "type": "object",
            "additionalProperties": RoiSetMetaFieldsSchema,
            "default": {},
        },
        "neuron-label-criteria": {
            "description":
                "In neuprint, all synaptic bodies are :Segment nodes, but only 'important' ones also become :Neuron nodes.\n"
                "The following settings determine which criteria will promote a :Segment to a :Neuron\n",
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
        "postHighAccuracyThreshold": {
            "description": "Which confidence threshold to use when calculating each connection's standard 'weight'\n",
            "type": "number",
            "default": 0.0
            # Note: On both hemibrain we used 0.5, on MANC we used 0.4
        },
        # TODO Move this into a larger sub-structure for all :Meta info
        "postHPThreshold": {
            "description": "Which confidence threshold to use when calculating each connection's weightHP\n",
            "type": "number",
            "default": 0.0
            # Note: On both hemibrain and MANC, we used 0.7
        },
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
    }
}


@PrefixFilter.with_context('Neuprint')
def export_neuprint(cfg, point_df, partner_df, ann, body_sizes):
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

    if not cfg['dataset']:
        raise RuntimeError("Please define a dataset name in neuprint:dataset")

    # Drop body 0 entirely.
    point_df = point_df.loc[point_df['body'] != 0]
    partner_df = partner_df.loc[(partner_df['body_pre'] != 0) & (partner_df['body_post'] != 0)]

    # with Timer("Decoding zyx coordinates from post_id"):
    #     partner_df[[*'zyx']] = decode_coords_from_uint64(partner_df['post_id'].values)

    export_neuprint_synapses(cfg, point_df)
    export_neuprint_segments(cfg, point_df, partner_df, ann, body_sizes)
    connectome = export_neuprint_segment_connections(cfg, partner_df)
    export_synapsesets(cfg, partner_df, connectome)
    export_neuprint_synapse_connections(partner_df)
