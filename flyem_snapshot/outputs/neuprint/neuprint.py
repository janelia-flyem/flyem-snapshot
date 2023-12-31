"""
Export a connectivity snapshot in the form of CSV files that can be converted to a neo4j neuprint database.
"""
import os
import logging

import pandas as pd

from neuclease import PrefixFilter
from neuclease.util import Timer

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
                "table to actually copy into neuprint as ROIs.\n"
                "If nothing is listed here, all ROI sets are used.",
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
        "restrict-synapses-to-roiset": {
            "description":
                "Entirely discard synapses that fall outside the given roiset.\n"
                "Synapses which do not fall on a non-zero ROI in the given roiset\n"
                "will not be used to calculate Neuron roiInfo totals, compartment totals,\n"
                "connection weights, or anything else. It's as if they don't exist.",
            "type": "string",
            "default": ""
        },
        "restrict-connectivity-to-roiset": {
            "description":
                "Discard synapse *connections* that fall outside the given roiset.\n"
                "The discarded synapses won't be included in the database individually,\n"
                "nor will they contribute to Neuron :ConnectsTo weights.\n"
                "However, they WILL contribute to Neuron roiInfo totals, indicating which compartments\n"
                "a Neuron innervates and what portion of the Neuron's total synapses reside in each compartment.\n",
            "type": "string",
            "default": ""
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
                "By default, annotation columns are automatically translated to neuprint properties,\n"
                "with names translated according to standard rules.\n"
                "But you can customize the translations here (or exclude some properties by translating them to ''.\n"
                "Note: Currently, all spatial (point) properties MUST contain the word 'Location' in the name.",
            "additionalProperties": {
                "type": "string"
            },
            "default": {},
        },
        "indexes": IndexesSettingsSchema,
        "max-segment-files": {
            "description":
                "The :Segment nodes will be exported as a series of CSV files.\n"
                "This setting controls the number of CSV files to produce.\n",
            "type": "integer",
            "default": 30_000,
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

        # TODO: Optionally update ROI meshes in DVID...
    }
}


@PrefixFilter.with_context('neuprint')
def export_neuprint(cfg, point_df, partner_df, ann, body_sizes, tbar_nt, body_nt, roisets, last_mutation):
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

    point_df, partner_df = restrict_synapses_to_roiset(
        cfg, 'restrict-synapses-to-roiset', point_df, partner_df)

    point_df, partner_df, inbounds_bodies, inbounds_rois = drop_out_of_bounds_bodies(
        cfg, point_df, partner_df)

    neuron_df, dataset_totals, roi_totals = export_neuprint_segments(
        cfg, point_df, partner_df, neuprint_ann, body_sizes, body_nt, inbounds_bodies, inbounds_rois)

    export_neuprint_meta(
        cfg,
        last_mutation,
        neuron_df,
        dataset_totals,
        roi_totals,
        roisets
    )

    export_neuroglancer_json_state(cfg, last_mutation)

    export_neuprint_indexes_script(
        cfg, neuron_df.columns, roi_totals.index, roisets)

    point_df, partner_df = restrict_synapses_to_roiset(
        cfg, 'restrict-connectivity-to-roiset', point_df, partner_df)

    connectome = export_neuprint_segment_connections(cfg, partner_df)
    export_synapsesets(cfg, partner_df, connectome)

    export_neuprint_synapses(cfg, point_df, tbar_nt)
    export_neuprint_synapse_connections(partner_df)


def drop_out_of_bounds_bodies(cfg, point_df, partner_df):
    """
    Determine which bodies intersect the 'in-bounds' ROIs,
    and preserve all of their synaptic connections (even if the partner bod is entirely out-of-bounds.)
    Discard synaptic connections in which neither side is an in-bounds body.
    """
    setting = 'restrict-connectivity-to-roiset'
    roiset = cfg[setting]
    if not roiset:
        return point_df, partner_df, None, None

    with Timer(f"Filtering out bodies according to {setting}: '{roiset}'"):
        inbounds_rois = {*partner_df[roiset].unique()} - {"<unspecified>"}
        inbounds_partners = (partner_df[roiset] != "<unspecified>")
        inbounds_body_pairs = partner_df.loc[inbounds_partners, ['body_pre', 'body_post']]
        inbounds_bodies = pd.unique(inbounds_body_pairs.values.reshape(-1))

        # Preserve partners as long as at least one side belongs to an in-bounds body.
        # (This will include out-of-bounds synapses, but no synapses where neither
        # body touches the in-bounds region.)
        keep_pre = partner_df['body_pre'].isin(inbounds_bodies)
        keep_post = partner_df['body_post'].isin(inbounds_bodies)
        partner_df = partner_df.loc[keep_pre | keep_post]

        # Keep the points which are still referenced in partner_df
        valid_ids = pd.concat(
            (
                partner_df['pre_id'].drop_duplicates().rename('point_id'),
                partner_df['post_id'].drop_duplicates().rename('point_id')
            ),
            ignore_index=True
        )
        point_df = point_df.loc[point_df.index.isin(valid_ids)]
        return point_df, partner_df, inbounds_bodies, inbounds_rois


def restrict_synapses_to_roiset(cfg, setting, point_df, partner_df):
    roiset = cfg[setting]
    if not roiset:
        return point_df, partner_df

    with Timer(f"Filtering out synapses according to {setting}: '{roiset}'"):
        # We keep *connections* that are in-bounds.
        # In neuprint, this is defined by the 'post' side.
        # On the edge, there can be 'pre' points that are out-of-bounds but
        # preserved here because they are partnered to an in-bounds 'post' point.
        inbounds_partners = (partner_df[roiset] != "<unspecified>")
        partner_df = partner_df.loc[inbounds_partners]
        # Keep the points which are still referenced in partner_df
        valid_ids = pd.concat(
            (
                partner_df['pre_id'].drop_duplicates().rename('point_id'),
                partner_df['post_id'].drop_duplicates().rename('point_id')
            ),
            ignore_index=True
        )
        point_df = point_df.loc[point_df.index.isin(valid_ids)]

    return point_df, partner_df
