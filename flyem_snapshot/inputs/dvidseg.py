from collections import namedtuple
import pyarrow.feather as feather

from neuclease.util import dump_json
from neuclease.dvid.repo import fetch_branch_nodes
from neuclease.dvid.labelmap import fetch_complete_mappings, fetch_mutations
from neuclease.dvid.labelmap.pointlabeler import PointLabeler

DvidSeg = namedtuple('DvidSeg', 'server uuid instance')

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

    # Look for the last mutation, searching backwards in the DAG until a non-empty UUID is found.
    last_mutation = {}
    branch_nodes = fetch_branch_nodes(*dvidseg)
    for uuid in branch_nodes[::-1]:
        muts = fetch_mutations(dvidseg.server, uuid, dvidseg.instance, dag_filter='leaf-only')
        if len(muts):
            last_mutation = muts.iloc[-1].to_dict()
            break

    dump_json(last_mutation, 'tables/last-mutation.json')
    pointlabeler = PointLabeler(*dvidseg, mapping=mapping)
    return dvidseg, last_mutation, pointlabeler
