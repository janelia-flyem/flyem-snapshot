"""
Export skeletons from a DVID server. Files are written to:
    skeletons/skeletons-swc/
    skeletons/skeletons-precomputed/
The skeletons-swc/ directory contains the SWC files, and the skeletons-precomputed/
directory contains the Neuroglancer precomputed files.
"""
import logging
import os
import re

from neuclease.dvid.keyvalue import fetch_keys, fetch_keyvalues
from neuclease.util import compute_parallel, iter_batches, skeleton_to_neuroglancer, \
                           swc_to_dataframe


logger = logging.getLogger(__name__)

TestConfig = {"export-skeletons": True,
              "server": "https://hemibrain-dvid.janelia.org",
              "uuid": "15aee239",
              "instance": "segmentation_skeletons",
}
SkeletonSchema = {
    "properties": {
        "export-skeletons": {
            "description": "If true, export the skeletons.",
            "type": "boolean",
            "default": True,
        },
        "server": {
            "description": "DVID server",
            "type": "string",
            "default": ""
        },
        "uuid": {
            "description": "UUID",
            "type": "string",
            "default": ""
        },
        "instance": {
            "description": "Instance",
            "type": "string",
            "default": "segmentation_skeletons"
        }
    }
}

def _write_single_skeleton(swcdict):
    """
    Write a single skeleton to an SWC file and a Neuroglancer file.
    """
    key = list(swcdict.keys())[0]
    val = swcdict[key]
    fname = f"skeletons/skeletons-swc/{key}".replace('_', '.')
    with open(fname, "wb") as f:
        f.write(val)
        df = swc_to_dataframe(val)
        skeleton_to_neuroglancer(df, output_path="skeletons/skeletons-precomputed/" \
                                                 + key.replace('_swc', ''))


def export_skeletons(cfg, ann=None):
    """
    Export skeletons.
    """
    if not (cfg['export-skeletons'] and cfg['server'] and cfg['uuid'] and cfg['instance']):
        return
    if ann is None:
        # Get body IDs
        logger.info(f"Fetching {cfg['instance']} for {cfg['server']} {cfg['uuid']}")
        ret = fetch_keys(cfg['server'], cfg['uuid'], cfg['instance'])
        bid = []
        for key in ret:
            if not re.match(r"\d+_swc$", key):
                continue
            bid.append(key)
    else:
        if ann.empty:
            return
        bid = [f"{str(x)}_swc" for x in ann['bodyid'].tolist()]

    # Get SWCs
    # Split the body IDs into batches of 1000 and then fetch and write each batch
    os.makedirs('skeletons/skeletons-swc', exist_ok=True)
    os.makedirs('skeletons/skeletons-precomputed', exist_ok=True)
    with open('skeletons/skeletons-swc/info', 'w', encoding='ascii') as f:
        f.write('{"@type": "neuroglancer_skeletons"}\n')
    id_batches = iter_batches(bid, batch_size=5000)
    logger.info(f"Found {len(id_batches):,} batches containing {len(bid):,} body IDs")
    cnt = 0
    bnum = 0
    for btch in id_batches:
        bnum += 1
        ret = fetch_keyvalues(cfg['server'], cfg['uuid'], cfg['instance'], btch, batch_size=1000)
        swclist = []
        for key, val in ret.items():
            if val:
                swclist.append(({key: val}))
        if not swclist:
            continue
        # Write files
        cnt += len(swclist)
        compute_parallel(_write_single_skeleton, swclist, processes=8)
    logger.info(f"Wrote {cnt:,} skeleton{'' if cnt == 1 else 's'}")


if __name__ == "__main__":
    test_cfg = TestConfig
    export_skeletons(test_cfg)
