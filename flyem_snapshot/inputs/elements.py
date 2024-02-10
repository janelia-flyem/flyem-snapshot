"""
Import "elements" (point objects) from disk, filter them, and associate each point with a body ID.
"""
import os
import logging

import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, encode_coords_to_uint64


logger = logging.getLogger(__name__)

ElementTableSchema = {
    "description": "Table of elements and where to find them.",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "point-table": {
            "description":
                "Optional. A feather or CSV file containing the element points, optionally with a 'body' column.\n"
                "If an 'sv' column is also present, it can be used to much more efficiently update the body column if needed.\n"
                "Required columns are x,y,z,type.  Other columns may be present and will be loaded in neuprint outputs.\n",
            "type": "string",
            "default": ""
        },
        "type": {
            "description":
                "Optional. Adds a column named 'type' to the table, with the given value in all rows.\n"
                "If no type is provided here in the config, then the data itself should have a type column.\n",
            "type": "string",
            "default": "",
        },
        "distance-table": {
            "description":
                "Optional. A feather file containing a table of source and target point pairs, with optional 'distance' column.\n"
                "Required columns are source_x, source_y, source_z, target_x, target_y, target_z\n",
            "type": "string",
            "default": ""
        },
        "roi-set-names": {
            "description":
                "The list of ROI sets to include as columns in the synapse table.\n"
                "If nothing is listed here, all ROI sets are used.",
            "default": None,
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"}
                },
                {
                    "type": "null"
                }
            ]
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
    "default": {}
}


@PrefixFilter.with_context('elements')
def load_elements(cfg, pointlabeler):
    element_dfs = {}
    for name, table_cfg in cfg.items():
        if name == "processes":
            continue

        point_df = _load_element_points(name, table_cfg)
        distance_df = _load_element_distances(name, table_cfg)

        # FIXME: This would be more convenient to mutate if it were a DataClass.
        element_dfs[name] = (point_df, distance_df)

        if point_df is not None:
            # Inserts new columns for 'body' and 'sv', in-place.
            pointlabeler.update_bodies_for_points(point_df, cfg['processes'])

    return element_dfs


def _load_element_points(name, table_cfg):
    path = table_cfg['point-table']
    if not path:
        return
    os.makedirs('tables', exist_ok=True)
    with Timer(f"Loading '{name}' elements from disk", logger):
        assert path.split('.')[-1] in ('feather', 'csv')
        if path.endswith('.csv'):
            element_df = pd.read_csv(path)
        else:
            element_df = feather.read_feather(path)

    if (cfg_type := table_cfg['type']):
        if 'type' in element_df.columns and (element_df['type'] != cfg_type).any():
            raise RuntimeError(f"Element table '{name}' has a type column that does "
                               f"not entirely match its config value ('{cfg_type}')")
        element_df['type'] = cfg_type

    if not {*element_df.columns} >= {*'xyz', 'type'}:
        raise RuntimeError(f"Element table '{name}' doesn't have xyz and/or type columns.")

    if 'point_id' in element_df:
        logger.warning(f"Discarding 'point_id' column from element table '{name}' "
                       "and regenerating the point_ids from scratch.")
        del element_df['point_id']

    point_ids = encode_coords_to_uint64(element_df[[*'zyx']].values)
    element_df.index = pd.Index(point_ids, name='point_id')

    return element_df


def _load_element_distances(name, table_cfg):
    path = table_cfg['distance-table']
    if not path:
        return

    os.makedirs('tables', exist_ok=True)
    with Timer(f"Loading '{name}' distances from disk", logger):
        distance_df = feather.read_feather(path)
        required_cols = {
            'source_x', 'source_y', 'source_z',
            'target_x', 'target_y', 'target_z',
        }
        if not {*distance_df.columns} >= {*required_cols}:
            raise RuntimeError(f"Element distance table '{name}' doesn't have the correct columns.")

        if 'distance' not in distance_df.columns:
            logger.warning(f"Element distance table '{name}' does not contain a 'distance' column")

        source_points = distance_df[['source_x', 'source_y', 'source_z']].values
        target_points = distance_df[['target_x', 'target_y', 'target_z']].values

        distance_df['source_id'] = encode_coords_to_uint64(source_points)
        distance_df['target_id'] = encode_coords_to_uint64(target_points)

    return distance_df
