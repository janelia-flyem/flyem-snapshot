import os
import pyarrow.feather as feather

from neuclease import PrefixFilter
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
            "description": "Use ':master' for the current HEAD node, or ':master~1' for its parent node.",
            "type": "string",
            "default": ":master~1"
        },
        "instance": {
            "description": "Name of the labelmap instance",
            "type": "string",
            "default": "segmentation"
        },
        "export-mapping": {
            "description": "Set to true if you'd like to export DVID's in-memory mapping (supervoxel: body) for your own needs.",
            "type": "boolean",
            "default": False
        }
    }
}


@PrefixFilter.with_context('dvidseg')
def load_dvidseg(cfg, snapshot_tag):
    """
    This function doesn't actually read segmentation voxels.
    It initializes a PointLabeler object, which can then be used
    to extract body IDs from arbitrary coordinates in a DVID segmentation
    (as specified via the uuid/instance in the config).

    The main purpose of the PointLabeler is to determine the body IDs
    under synapse locations or other point annotations (such as soma
    locations, etc.).
    """
    if not cfg['server']:
        return None
    dvidseg = DvidSeg(cfg['server'], cfg['uuid'], cfg['instance'])

    mapping = None
    if cfg['export-mapping']:
        os.makedirs('tables', exist_ok=True)

        # We export the complete mapping (rather than the minimal mapping),
        # since that can be more convenient for certain analyses.
        mapping = fetch_complete_mappings(*dvidseg)
        feather.write_feather(
            mapping.reset_index(),
            f"tables/complete-nonsingleton-mapping-{snapshot_tag}.feather"
        )

    pointlabeler = PointLabeler(*dvidseg, mapping=mapping)
    dump_json(pointlabeler.last_mutation, 'last-mutation.json')
    return pointlabeler
