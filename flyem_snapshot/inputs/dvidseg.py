import pyarrow.feather as feather

from neuclease.util import dump_json
from neuclease.dvid.labelmap import fetch_complete_mappings
from neuclease.dvid.labelmap.pointlabeler import PointLabeler, DvidSeg

DvidSegSchema = {
    "description": "dvid segmentation (labelmap) location",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "required": ["server", "uuid", "instance"],
    "properties": {
        "server": {
            "description": "DVID server",
            "type": "string",
            "default": "",
        },
        "uuid": {
            "description": "Use ':master' for the current HEAD node, or ':master~1' for it's parent node.",
            "type": "string",
            "default": ":master~1"
        },
        "instance": {
            "description": "Name of the labelmap instance",
            "type": "string",
            "default": "segmentation"
        },
        "export-mapping": {
            "type": "boolean",
            "default": False
        }
    }
}


def load_dvidseg(cfg, snapshot_tag):
    if not cfg['server']:
        return None, None, None
    dvidseg = DvidSeg(cfg['server'], cfg['uuid'], cfg['instance'])

    mapping = None
    if cfg['export-mapping']:
        # We export the complete mapping (rather than the minimal mapping),
        # since that can be more convenient for certain analyses.
        mapping = fetch_complete_mappings(*dvidseg)
        feather.write_feather(
            mapping.reset_index(),
            f"tables/complete-nonsingleton-mapping-{snapshot_tag}.feather"
        )

    pointlabeler = PointLabeler(*dvidseg, mapping=mapping)
    dump_json(pointlabeler.last_mutation, 'tables/last-mutation.json')
    return pointlabeler
