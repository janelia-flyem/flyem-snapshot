"""
Import "elements" (point objects) from disk, filter them, and associate each point with a body ID.
"""
import os
import json
import logging

import numpy as np
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, encode_coords_to_uint64, decode_coords_from_uint64, dump_json
from neuclease.dvid.labelmap import fetch_mapping, fetch_mutations, fetch_complete_mappings, fetch_bodies_for_many_points


logger = logging.getLogger(__name__)

ElementTableSchema = {
    "description": "Table of elements and where to find them.",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "point-table": {
            "description":
                "A feather file containing the element points, optionally with a 'body' column.\n"
                "If an 'sv' column is also present, it can be used to much more efficiently update the body column if needed.\n",
            "type": "string",
            # NO DEFAULT
        },
    }
}

ElementTablesSchema = {
    "description": "Set of Elements configs.",
    "type": "object",
    "additionalProperties": ElementTableSchema,
    "properties": {
        "processes": {
            "description":
                "How many processes should be used to update synapse labels?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    },
    "default": {
        "my-elements": {
            "point-table": "/path/to/my-elements.feather"
        },
        "processes": None
    }
}


@PrefixFilter.with_context('elements')
def load_elements(cfg, pointlabeler):
    element_dfs = {}
    for name, table_cfg in cfg.items():
        if name == "processes":
            continue

        df = _load_elements_table(name, table_cfg)

        # Inserts new columns for 'body' and 'sv'
        pointlabeler.update_bodies_for_points(df, cfg['processes'])

        element_dfs[name] = df
    return element_dfs


def _load_elements_table(name, table_cfg):
    path = table_cfg['point-table']
    if not path:
        raise RuntimeError("Element config contains an item with an invalid path.")
    os.makedirs('tables', exist_ok=True)
    with Timer(f"Loading '{name}' elements from disk", logger):
        element_df = feather.read_feather(path)

    if not {*element_df.columns} >= {*'xyz'}:
        raise RuntimeError(f"Element table '{name}' doesn't have xyz columns.")

    if 'point_id' in element_df:
        logger.warning(f"Discarding 'point_id' column from element table '{name}' "
                       "and regenerating the point_ids from scratch.")

    point_ids = encode_coords_to_uint64(element_df[[*'zyx']].values)
    element_df.index = point_ids

    return element_df


