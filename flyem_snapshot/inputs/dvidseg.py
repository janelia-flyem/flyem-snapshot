import os
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import dump_json
from neuclease.dvid.repo import is_locked
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
    pointlabeler = PointLabeler(*dvidseg)
    dump_json(pointlabeler.last_mutation, 'last-mutation.json')

    if not cfg['export-mapping']:
        return pointlabeler

    os.makedirs('tables', exist_ok=True)
    if is_locked(dvidseg.server, dvidseg.uuid):
        mapping_path = f"tables/complete-nonsingleton-mapping-{snapshot_tag}.feather"
    else:
        mutid = pointlabeler.last_mutation['mutid']
        mapping_path = f"tables/complete-nonsingleton-mapping-{snapshot_tag}-mutid-{mutid}.feather"

    # Note:
    #   We could consider loading the mapping into the pointlabeler from the cached
    #   file, but in most cases, the fact that the mapping cache is still valid is
    #   a sign that the synapses and elements won't need to be relabeled, either.
    #   Pre-emptively loading the mapping into RAM would incur a potentially large RAM
    #   overhead (~8GB for the MaleCNS) in such cases even though it wouldn't be used.
    #   In cases where it *is* needed, it will be automatically fetched (albeit slowly)
    #   by the pointlabeler at the time it is needed.
    if not os.path.exists(mapping_path):
        # We export the complete mapping (rather than the minimal mapping),
        # since that can be more convenient for certain analyses.
        # NOTE:
        #   For unlocked UUIDs which may be undergoing edits,
        #   there's a race condition here: the mapping may change
        #   while we're downloading it.
        #   See https://github.com/janelia-flyem/flyem-snapshot/issues/6
        mapping = fetch_complete_mappings(*dvidseg)
        feather.write_feather(mapping.reset_index(), mapping_path)

        # Spare the pointlabeler from re-fetching the mapping.
        pointlabeler = PointLabeler(*dvidseg, mapping=mapping)

    return pointlabeler
