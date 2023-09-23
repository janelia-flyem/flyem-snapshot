import logging
from jinja2 import Environment, PackageLoader

logger = logging.getLogger(__name__)

IndexesSettingsSchema = {
    "default": {},
    "additionalProperties": False,
    "properties": {
        "exclude": {
            "description": "These segment properties won't be indexed. (All others will be.)",
            "type": "array",
            "items": {"type": "string"},
            "default": ['roiInfo', 'notes']
        }
    }
}


def export_neuprint_indexes_script(cfg, neuron_prop_names, roi_names):
    exclude = set(cfg['indexes']['exclude'])

    # Note:
    #   We don't need to explicitly create an index for bodyId because
    #   our script adds a uniqueness constraint, which automatically
    #   creates an index, too. (We'll get an error from neo4j if we
    #   attempt to create an additional index.)
    neuron_prop_names = sorted(set(neuron_prop_names) - exclude - {'bodyId'})
    roi_names = sorted(set(roi_names) - exclude)

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
