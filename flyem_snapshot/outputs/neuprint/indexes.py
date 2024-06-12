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
        },
        "element-roisets-to-index": {
            "description":
                "Ordinarily, we do not create indexes for Element ROI properties,\n"
                "but you can explicitly create selected ones with this setting.\n"
                "Presumably this is more useful when you are working with relatively\n"
                "sparse Elements or small ROIs (or both).\n",
            "type": "array",
            "default": [],
            "items": {
                "type": "object",
                "properties": {
                    "neuprint-label": {
                        "description":
                            "The neo4j node label (e.g. ':Mito') for the elements to index.\n"
                            "Must match the neuprint-label setting of the element input config.\n"
                            "Do not use the generic ':Element' label, since that would also apply to synapses, which we do not index by ROI.\n",
                        "type": "string",
                        "default": ""
                    },
                    "roisets": {
                        "description":
                            "A list of roiset names (as listed in the 'roi' config).\n"
                            "All of the constituent ROIs in these roisets will be indexed.\n",
                        "type": "array",
                        "default": [],
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }
}


def export_neuprint_indexes_script(cfg, neuron_columns, roi_names, synapse_roisets, element_roisets):
    """
    Using the jinja template stored in create-indexes.cypher,
    export a script of cypher commands that will create indexes for all
    :Neuron/:Segment properties, including ROI properties, except for
    those excluded via the config.

    TODO: Actually index the other element rois.

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
        element_roisets:
            Used to create indexes for :Element roi properties.
            Structure:
                {
                    element_config_name: {
                        roiset_name: {
                            roi_name: roi_id,
                            roi_name: roi_id,
                            ...
                        }
                    }
                }
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

    indexed_label_roisets = {item['neuprint-label']: item['roisets']
                             for item in cfg['indexes']['element-roisets-to-index']}

    invalid_labels = {*indexed_label_roisets.keys()} - {*cfg['element-labels'].values()}
    if invalid_labels:
        raise RuntimeError(f"Some requested Element indexes refer to non-existent neuprint labels: {invalid_labels}")

    element_rois_to_index = {}
    for config_name, d in element_roisets.items():
        label = cfg['element-labels'].get(config_name)
        if label not in indexed_label_roisets:
            continue
        rois = element_rois_to_index.get(label, set())
        rois |= set(chain(*(v.keys() for k,v in d.items() if k in indexed_label_roisets[label])))
        element_rois_to_index[label] = rois

    element_rois_to_index = {k: sorted(v) for k,v in element_rois_to_index.items()}

    env = Environment(loader=PackageLoader('flyem_snapshot.outputs.neuprint'))
    template = env.get_template('create-indexes.cypher')
    rendered = template.render({
        'segment_properties': neuron_prop_names,
        'rois': roi_names,
        'dataset': cfg['meta']['dataset'],
        'element_rois_to_index': element_rois_to_index,
    })

    logger.info("Writing neuprint/create-indexes.cypher")
    with open('neuprint/create-indexes.cypher', 'w') as f:
        f.write(rendered)
