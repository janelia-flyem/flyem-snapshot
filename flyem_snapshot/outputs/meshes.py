"""
Export meshes from a DVID server. Files are written to meshes-{cfg_name}/single-res-meshes/
"""
from functools import partial
import logging
import os
import re
import copy

import pandas as pd
import numpy as np
import requests.exceptions

from neuclease import PrefixFilter
from neuclease.dvid.keyvalue import fetch_key, fetch_keys
from neuclease.util import compute_parallel, dump_json
from ..util.checksum import checksum
from ..caches import cached, SerializerBase

logger = logging.getLogger(__name__)

# TODO:
#   - Need to provide resolution (in nanometers) in the config.
#   - Allow parallel process count to be configured?
#   - Allow user to narrow the set of meshes to export by including or excluding body statuses?

SingleInstanceMeshSchema = {
    "description":
        "Settings for neuroglancer mesh export, from a DVID keyvalue instance "
        "which already stores the meshes in neuroglancer single-resolution format.",
    "type": "object",
    "default": {},
    "properties": {
        "export-meshes": {
            "description": "If true, export the meshes.",
            "type": "boolean",
            "default": False,
        },
        "dvid": {
            "description": "DVID server/UUID and instance to export neuroglancer single-resolution meshes from.",
            "type": "object",
            "additionalProperties": False,
            "default": {},
            "properties": {
                "server": {
                    "type": "string",
                    "default": ""
                },
                "uuid": {
                    "type": "string",
                    "default": ""
                },
                "instance": {
                    "type": "string",
                    "default": ""
                }
            }
        },
        "exclude-statuses": {
            "description": "List of body statuses to exclude from export.",
            "type": "array",
            "items": {
                "type": "string"
            },
            "default": ["Unimportant", "Glia"]
        }
    }
}

MeshesSchema = {
    "description": "Settings for exporting one or more mesh instances from DVID.",
    "type": "object",
    "default": {},
    "properties": {
        "processes": {
            "description":
                "How many processes should be used to export meshes?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    },
    "additionalProperties": SingleInstanceMeshSchema,
}


def export_meshes(cfg, snapshot_tag, ann=None, pointlabeler=None):
    cfg = copy.copy(cfg)
    processes = cfg['processes']
    del cfg['processes']
    for name, instance_cfg in cfg.items():
        export_mesh_instance(name, instance_cfg, snapshot_tag, ann, pointlabeler, processes=processes)


class MeshSerializer(SerializerBase):
    """
    Serializer that just stores the table of exported meshes vs. missing meshes.
    Avoids downloading meshes repeatedly for a release snapshot whose segmentation hasn't changed,
    but attempts to re-export meshes that weren't successfully exported in the last run.
    """

    def get_cache_key(self, cfg_name, cfg, snapshot_tag, ann, pointlabeler=None, subset_bodies=None, processes=0):
        self.cfg_name = cfg_name
        self.cfg = cfg
        self.snapshot_tag = snapshot_tag
        self.ann = ann
        self.pointlabeler = pointlabeler
        self.processes = processes
        cfg_hash = hex(checksum(cfg))

        if ann is not None:
            ann_body_hash = hex(checksum(np.sort(ann.index.values)))
        else:
            ann_body_hash = '0x0'

        if pointlabeler is None:
            mutid = 0
        else:
            mutid = pointlabeler.last_mutation["mutid"]

        return f'{snapshot_tag}-seg-{mutid}-ann-{ann_body_hash}-{self.name}-{cfg_hash}.csv'

    def save_to_file(self, results, path):
        if results is None and os.path.exists(path):
            os.remove(path)
            return

        results.to_csv(path, index=True, header=True)
        if not results['success'].all():
            num_failed = (~results['success']).sum()
            logger.warning(
                f"Failed to export meshes for {num_failed} body IDs. See {os.path.abspath(path)}"
            )

    def load_from_file(self, path):
        results = pd.read_csv(path)
        if not results['success'].all():
            failed_bodies = results.loc[~results['success']].index.str[:-len('.ngmesh')].map(int).unique()
            new_results = export_meshes(
                self.cfg_name,
                self.cfg,
                self.snapshot_tag,
                self.ann,
                self.pointlabeler,
                failed_bodies,
                processes=self.processes
            )
            results.loc[new_results.index, 'success'] = new_results['success']
        return results


@PrefixFilter.with_context('meshes-{cfg_name}')
@cached(MeshSerializer('meshes-{cfg_name}'))
def export_mesh_instance(cfg_name, cfg, snapshot_tag, ann, pointlabeler=None, subset_bodies=None, processes=0):
    """
    Export neuroglancer precomputed single-resolution meshes from a DVID keyvalue instance.
    The set of meshes to export is taken from the body annotations table.

    The MeshSerializer ensures that we don't rerun the full export if it's already been
    completed, but it does call this function if there are missing meshes from the last run.
    """
    del snapshot_tag
    del pointlabeler

    mesh_src = (
        cfg['dvid']['server'],
        cfg['dvid']['uuid'],
        cfg['dvid']['instance'],
    )
    if not (cfg['export-meshes'] and all(mesh_src)):
        return None

    if subset_bodies is not None:
        logger.info(f"Exporting meshes for a subset of {len(subset_bodies)} bodies")
        keys = [f'{body}.ngmesh' for body in subset_bodies]
    elif ann is None:
        logger.info(f"Fetching all keys from {'/'.join(mesh_src)}")
        keys = fetch_keys(*mesh_src)
        keys = [k for k in keys if re.match(r"\d+.ngmesh$", k)]
        if not keys:
            logger.warning(
                "Not exporting meshes: No mesh keys found in the DVID instance "
                f"({'/'.join(mesh_src)})"
            )
            return
    elif len(ann) == 0:
        raise RuntimeError("Can't export meshes: No body IDs in the annotations table.")
    else:
        exclude_statuses = cfg['exclude-statuses']
        ann = ann.query('status not in @exclude_statuses')
        assert ann.index.name == 'body'
        keys = ann.index.astype(str) + ".ngmesh"

    dirname = f'meshes-{cfg_name}/single-res-meshes'
    os.makedirs(f'{dirname}', exist_ok=True)
    with open(f'{dirname}/info', 'w', encoding='utf-8') as f:
        f.write('{"@type": "neuroglancer_legacy_mesh"}\n')

    logger.info(f"Exporting meshes for {len(keys)} body IDs")

    results = compute_parallel(
        partial(_process_single_mesh, *mesh_src, dirname),
        keys,
        processes=processes,
    )
    results = pd.DataFrame(results, columns=['key', 'success']).set_index('key')
    return results


def _process_single_mesh(server, uuid, instance, dirname, key):
    """
    Fetch a single mesh from the DVID server.
    Writes the mesh and also the appropriate pointer JSON file for the mesh e.g. "123:0"

    Returns:
        (key, success) tuple, where success is True if the mesh
        exists on the DVID server and was written successfully.
    """
    assert key.endswith('.ngmesh')
    body = key[:-len('.ngmesh')]

    try:
        mesh_bytes = fetch_key(server, uuid, instance, key)
    except requests.exceptions.HTTPError:
        return key, False
    except Exception as e:
        logger.warning(f"An exception of type {type(e).__name__} occurred. Arguments:\n{e.args}")
        return key, False

    if not mesh_bytes:
        return key, False

    fname = f"{dirname}/{body}.ngmesh"
    with open(fname, "wb") as f:
        f.write(mesh_bytes)

    dump_json({"fragments": [f"{body}.ngmesh"]}, f"{dirname}/{body}:0")

    return key, True
