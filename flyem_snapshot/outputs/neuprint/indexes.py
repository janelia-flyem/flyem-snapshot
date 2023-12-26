import logging
from itertools import chain
from jinja2 import Environment, PackageLoader

logger = logging.getLogger(__name__)

IndexesSettingsSchema = {
    "default": {},
    "additionalProperties": False,
    "properties": {
        "exclude-properties": {
            "description":
                "These segment properties won't be indexed. (All others will be.)\n"
                "Individual ROIs can be listed here, but usually ROI exclusions will be listed in exclude-roisets, below.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": ['roiInfo', 'notes']
        },
        "exclude-roisets": {
            "description": "ROI properties from these roi-sets (columns) won't be indexed. (All others will be.)",
            "type": "array",
            "items": {"type": "string"},
            "default": []
        }
    }
}


def export_neuprint_indexes_script(cfg, neuron_columns, roi_names, synapse_roisets, landmark_roisets):
    """
    Using the jinja template stored in create-indexes.cypher,
    export a script of cypher commands that will create indexes for all
    :Neuron/:Segment properties, including ROI properties, except for
    those excluded via the config.

    TODO: Actually index the landmark rois.

    Args:
        cfg:
            The 'neuprint' subsection of the main config.
        neuron_prop_names:
            The names of all :Segment properties except for ROI properties
        roi_names:
            The names of all ROIs actually found in the data.
            (If the config lists ROIs that didn't end up being used, we won't try to index them.)
        synapse_roisets:
            The mapping of {roiset_name: {roi_name: roi_id, roi_name: roi_id, ...}}.
            Used to create indexes for :Synapse and :Segment roi properties.
        landmark_roisets:
            The mapping of {roiset_name: {roi_name: roi_id, roi_name: roi_id, ...}}.
            Used to create indexes for :Landmark roi properties.
    """
    exclude_props = set(cfg['indexes']['exclude-properties'])
    exclude_roisets = cfg['indexes']['exclude-roisets']
    exclude_rois = set(chain(*[synapse_roisets[roiset].keys() for roiset in exclude_roisets]))

    # Note:
    #   We don't need to explicitly create an index for bodyId because
    #   our script adds a uniqueness constraint, which automatically
    #   creates an index, too. (We'll get an error from neo4j if we
    #   attempt to create an additional index.)
    neuron_prop_splits = [name.split(':', 1) for name in neuron_columns if ':' in name]
    neuron_prop_names = [name for (name, dtype) in neuron_prop_splits if name and dtype != "IGNORE"]
    neuron_prop_names = sorted(set(neuron_prop_names) - exclude_props - {'bodyId'})
    roi_names = sorted(set(roi_names) - exclude_props - exclude_rois)

    env = Environment(loader=PackageLoader('flyem_snapshot.outputs.neuprint'))
    template = env.get_template('create-indexes.cypher')
    rendered = template.render({
        'segment_properties': neuron_prop_names,
        'rois': roi_names,
        'dataset': cfg['meta']['dataset']
    })

    logger.info("Writing neuprint/create-indexes.cypher")
    with open('neuprint/create-indexes.cypher', 'w') as f:
        f.write(rendered)
