"""
Export a connectivity snapshot in the form of CSV files that can be converted to a neo4j neuprint database.
"""
import os
import copy
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
from .element import export_neuprint_elements, export_neuprint_elements_closeto
from .elementset import export_neuprint_elementsets
from ...util.util import restrict_synapses_to_roi
from ...util.checksum import checksum
from ...caches import cached, SentinelSerializer

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
                "The config settings which define the info that will be loaded into the Neuprint :Meta node.\n"
                "This should be a path to a separate config file containing the meta settings.\n"
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
        # This setting is applied BEFORE either of the two that follow.
        # If a point is dropped by this setting, it is definitely excluded,
        # but the two following settings can filter the set even further.
        "restrict-info-totals-to-roiset": {
            "description":
                "Entirely discard synapses and elements that fall outside the given roiset.\n"
                "Synapses/elements which do not fall on a non-zero ROI in the given roiset\n"
                "will not be used to calculate Neuron roiInfo totals, compartment totals, connection\n"
                "weights, or anything else. For all intents and purposes, they don't exist at all,\n"
                "but this settings saves you from having to pre-filter them from your input tables.\n",
            "type": "string",
            "default": ""
        },
        "restrict-connectivity-to-roiset": {
            "description":
                "Discard synapse *connections* that fall outside the given roiset.\n"
                "The discarded synapses won't be included in the database individually,\n"
                "nor will they contribute to Neuron :ConnectsTo weights.\n"
                "However, they may still contribute to Neuron roiInfo totals, indicating which compartments\n"
                "a Neuron innervates and what portion of the Neuron's total synapses reside in each compartment.\n",
            "type": "string",
            "default": ""
        },
        "restrict-elements-to-roiset": {
            "description":
                "Discard non-Synapse :Element points that fall outside the given roiset.\n"
                "The discarded Elements won't be included in the database individually\n"
                "However, they may still contribute to Neuron roiInfo totals, indicating which\n"
                "compartments within a Neuron contain various non-Synapse Elements (and their unfiltered totals).\n",
            "type": "string",
            "default": ""
        },
        # FIXME: These apply to all Elements, not just Synapses, maybe this should be renamed.
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
                "pre": {
                    "description": "Segments with this many presynapses (or more) will also be labeled as Neurons.\n",
                    "type": "number",
                    "minimum": 0,
                    "default": 100,
                },
                "post": {
                    "description": "Segments with this many postsynapses (or more) will also be labeled as Neurons.\n",
                    "type": "number",
                    "minimum": 0,
                    "default": 100,
                },
                "properties": {
                    "description": "Segments with non-empty values for any properties listed here will be labeled as Neurons\n",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["rootLocation", "somaLocation", "class", "type", "instance", "group", "somaSide", "synonyms"],
                },
                "status": {
                    "description":
                    "Segments with one of these neuprint statuses will be labeled Neurons.\n"
                    "Note: These should be neuprint 'status' values, not 'statusLabel' values.",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["Traced", "Anchor"],
                },
                "excluded-status": {
                    "description":
                        "Segments with one of these neuprint statuses will NOT be labeled as Neurons, regardless of their other properties.\n"
                        "Note: These should be neuprint 'status' values, not 'statusLabel' values.",
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["Unimportant", "Glia"],
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
        "keep-non-synaptic-segments": {
            "description":
                "If bodies without synapses (or other elements) are present in the body annotations table,\n"
                "should we create :Segment nodes (and possibly :Neuron) for them, or exclude them from\n"
                "the neuprint export?",
            "type": "boolean",
            "default": False
        },
        "element-labels": {
            "description":
                "For element sets listed in the inputs.elements config, specify the neuprint "
                "label that should be used when exporting them, e.g. ':Mito' or ':ColumnPin'",
            "type": "object",
            "default": {},
            "additionalProperties": {
                "type": "string"
            }
        },
        "element-totals": {
            "description":
                "Specify which non-synapse Element sets should be included as totals \n"
                "in Segment properties and Segment roiInfo values.\n",
            "type": "object",
            "default": {},
            "additionalProperties": {
                "type": "object",
                "default": {},
                "properties": {
                    "compute-segment-totals-with-name": {
                        "type": "string",
                        "default": ""
                    },
                    "compute-roi-totals-with-name": {
                        "type": "string",
                        "default": ""
                    },
                    "compute-kind-totals-for-column": {
                        "description":
                            "Which column (if any) to on which run value_counts() for inclusion among the overall counts.\n"
                            "Example: mitoType\n",
                        "type": "string",
                        "default": ""
                    }
                }
            }
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

        # TODO: weightHP, and get rid of weightHR (a failed experiment, I think.)
        #       If I don't get rid if it, then I need to avoid pre-filtering by min-confidence,
        #       and apply that filter before neuron/weight export (but not synapse export).

        # TODO: Optionally update ROI meshes in DVID...
    }
}


class NeuprintSentinelSerializer(SentinelSerializer):

    def get_cache_key(self, cfg, point_df, partner_df, element_tables,
                      ann, body_sizes, tbar_nt, body_nt,
                      syn_roisets, element_roisets, pointlabeler):

        cfg = copy.copy(cfg)
        cfg['processes'] = 0

        with Timer("Computing data checksum", logger, log_start=False):
            csums = [
                checksum(data) for data in (
                    cfg, point_df, partner_df, element_tables,
                    ann, body_sizes, tbar_nt, body_nt,
                    syn_roisets, element_roisets
                )
            ]
            logger.debug(str([hex(c) for c in csums]))
            key = hex(checksum(csums))

        if pointlabeler is not None:
            key = f'{key}-seg-{pointlabeler.last_mutation["mutid"]}'

        return f'{self.name}-{key}.sentinel'


@PrefixFilter.with_context('neuprint')
@cached(NeuprintSentinelSerializer('neuprint-export', True))
def export_neuprint(cfg, point_df, partner_df, element_tables, ann, body_sizes, tbar_nt, body_nt,
                    syn_roisets, element_roisets, pointlabeler):
    """
    Export CSV files for each of the following:

    Nodes:
        - Meta
        - Neuron
        - SynapseSet
        - ElementSet
        - Synapse
        - Element

    Edges:
        - Neuron -[:ConnectsTo]-> Neuron
        - SynapseSet -[:ConnectsTo]-> SynapseSet
        - Synapse -[:SynapsesTo]-> Synapse
        - Element -[:SynapsesTo]-> Element

        - Neuron -[:Contains]-> SynapseSet
        - Neuron -[:Contains]-> ElementSet
        - SynapseSet -[:Contains]-> Synapse
        - ElementSet -[:Contains]-> Element
    """
    if not cfg['export-neuprint-snapshot']:
        logger.info("Per config, not generating neuprint snapshot.")
        return

    os.makedirs('neuprint', exist_ok=True)

    last_mutation = pointlabeler and pointlabeler.last_mutation

    # Drop body 0 entirely.
    point_df = point_df.loc[point_df['body'] != 0]
    partner_df = partner_df.loc[(partner_df['body_pre'] != 0) & (partner_df['body_post'] != 0)]

    # We don't store neuprint properties for the "<unspecified>" ROI.
    syn_roisets = copy.deepcopy(syn_roisets)
    for roi_ids in syn_roisets.values():
        roi_ids.pop('<unspecified>', None)

    element_roisets = copy.deepcopy(element_roisets)
    for roi_ids in element_roisets.values():
        roi_ids.pop('<unspecified>', None)

    neuprint_ann = neuprint_segment_annotations(cfg, ann)

    point_df, partner_df = restrict_synapses_for_setting(
        cfg, 'restrict-info-totals-to-roiset', point_df, partner_df)

    element_tables = restrict_elements_to_roiset(
        cfg, 'restrict-info-totals-to-roiset', element_tables, point_df)

    point_df, partner_df, inbounds_bodies, inbounds_rois = drop_out_of_bounds_bodies(
        cfg, point_df, partner_df)

    neuron_df, dataset_totals, roi_totals = export_neuprint_segments(
        cfg, point_df, partner_df, element_tables, neuprint_ann, body_sizes, body_nt, inbounds_bodies, inbounds_rois)

    export_neuprint_meta(
        cfg,
        last_mutation,
        neuron_df,
        dataset_totals,
        roi_totals,
        syn_roisets
    )

    export_neuroglancer_json_state(cfg, last_mutation)

    export_neuprint_indexes_script(
        cfg, neuron_df.columns, roi_totals.index, syn_roisets, element_roisets)

    point_df, partner_df = restrict_synapses_for_setting(
        cfg, 'restrict-connectivity-to-roiset', point_df, partner_df)

    element_tables = restrict_elements_to_roiset(
        cfg, 'restrict-elements-to-roiset', element_tables, point_df)

    connectome = export_neuprint_segment_connections(cfg, partner_df)

    # TODO: It would be good to verify that there are no duplicated Element IDs (including Synapses)
    export_neuprint_elementsets(cfg, element_tables, connectome)
    export_neuprint_elements(cfg, element_tables, element_roisets)
    export_neuprint_elements_closeto(element_tables)

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


def restrict_synapses_for_setting(cfg, setting, point_df, partner_df):
    """
    Drop synapses from point_df and partner_df which fall outside a config-defined roiset.
    (No individual ROI in the roiset is used -- only points which the
    roiset lists as <unspecified> will be dropped.)

    Args:
        cfg:
            neuprint section of the snapshot config
        setting:
            The name of the config setting which holds the roiset name
            (Depending on WHEN this function is called in the overall pipeline,
            this function may exclude synapses from all calculations or may exclude
            synapses only from the connectivity export, after aggregate counts have
            already been computed.)
        point_df:
            Synapse points dataframe
        partner_df:
            Synapse partner dataframe

    Returns:
        Filtered versions of point_df and partner_df.
    """
    roiset = cfg[setting]
    if not roiset:
        return point_df, partner_df

    with Timer(f"Filtering out synapses according to {setting}: '{roiset}'"):
        return restrict_synapses_to_roi(roiset, None, point_df, partner_df)


def restrict_elements_to_roiset(cfg, setting, element_tables, syn_point_df):
    """
    From all element point tables, drop elements which are "<unspecified>"
    in the given roiset.

    Then, from all element distance tables, drop relationships if either of the
    points no longer exists in either the synapse point set or the generic element
    point set.

    Note:
        Unlike synapses, ordinary Elements are permitted to exist
        even if they have no relationships in the dataset.
        (Synapses, on the other hand must always have at least one partner.)
    """
    roiset = cfg[setting]
    if not roiset:
        return element_tables

    # Filter out points that fall outside the roiset
    with Timer(f"Filtering out Elements according to {setting}: '{roiset}'"):
        for name in list(element_tables.keys()):
            el_points, el_distances = element_tables[name]
            el_points = el_points.loc[el_points[roiset] != "<unspecified>"]
            element_tables[name] = (el_points, el_distances)

        # Filter out relationships (distances) if either point no longer exists after filtering.
        # We count synapses as valid elements (they were pre-filtered in a different function).
        valid_el_ids = (points[[]] for points, _ in element_tables.values())
        valid_ids = pd.concat((syn_point_df[[]], *valid_el_ids)).index
        for name in list(element_tables.keys()):
            el_points, el_distances = element_tables[name]
            valid_sources = el_distances['source_id'].isin(valid_ids)
            valid_targets = el_distances['target_id'].isin(valid_ids)
            el_distances = el_distances.loc[valid_sources & valid_targets]
            element_tables[name] = (el_points, el_distances)

    return element_tables
