"""
Import "elements" (point objects) from disk, filter them, and associate each point with a body ID.
"""
import copy
import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease.util import Timer, encode_coords_to_uint64
from neuclease.dvid import fetch_all_elements

from .annotations import PointAnnotationSchema

ElementPointAnnotationSchema = copy.deepcopy(PointAnnotationSchema)
ElementPointAnnotationSchema['properties']['instance']['default'] = ""

logger = logging.getLogger(__name__)

ElementTableSchema = {
    "description": "Table of elements and where to find them.",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "point-table": {
            "description":
                "A feather or CSV file containing the element points, optionally with a 'body' column.\n"
                "If an 'sv' column is also present, it can be used to much more efficiently update the body column if needed.\n"
                "Required columns are x,y,z,type.  Other columns may be present and will be loaded in neuprint outputs.\n"
                "(If your annotation points are stored in a DVID point annotation instance, use the point-annotations config instead of this.)\n",
            "type": "string",
            "default": ""
        },
        "dvid-point-annotations": {
            **ElementPointAnnotationSchema,
            "description":
                "Point annotation dataset from DVID to use for populating properties on the :Element nodes.\n"
                "(If your annotation points are stored in a CSV or feather file, use the point-table config instead of this.)\n"
                "Should look something like this:\n"
                "dvid-point-annotations:\n"
                "  instance: nuclei-centroids\n"
                "  column-name: soma_position\n"
                "  extract-properties:\n"
                "    radius: nucleus_radius\n",
        },
        "permit-body-0": {
            "description":
                "If False, filter out any elements that reside on body 0.\n"
                "Otherwise, leave them in, which means neuprint may include :Segment/:Neuron for body 0 and it will have elements.\n",
            "type": "boolean",
            "default": False,
        },
        "type": {
            "description":
                "Optional. Adds a column named 'type' to the table, with the given value in all rows.\n"
                "If no type is provided here in the config, then the data itself should have a type column/property.\n",
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
        }
    }
}

ElementTablesSchema = {
    "description": "Set of Elements configs.",
    "type": "object",
    "additionalProperties": ElementTableSchema,
    "properties": {
        "processes": {
            "description":
                "How many processes should be used to update element labels?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    },
    "default": {}
}


def load_elements(cfg, pointlabeler):
    element_dfs = {}
    for name, table_cfg in cfg.items():
        if name == "processes":
            continue

        dvid_server = dvid_uuid = None
        if pointlabeler:
            dvid_server = pointlabeler.dvidseg.server
            dvid_uuid = pointlabeler.dvidseg.uuid

        point_df = _load_element_points(name, table_cfg, dvid_server, dvid_uuid)
        distance_df = _load_element_distances(name, table_cfg)

        if point_df is not None:
            # Inserts new columns for 'body' and 'sv', in-place.
            # FIXME: This assumes DVID is used. We need a config option to just trust the input body column.
            pointlabeler.update_bodies_for_points(point_df, cfg['processes'])
            point_df = point_df.astype({'body': np.int64, 'sv': np.int64})

        if not table_cfg['permit-body-0']:
            point_df = point_df.loc[point_df['body'] != 0].copy()

        # FIXME: This would be more convenient to mutate if it were a DataClass.
        element_dfs[name] = (point_df, distance_df)

    return element_dfs


def _load_element_points(name, table_cfg, dvid_server, dvid_uuid):
    table_path = table_cfg['point-table']
    ann_instance = table_cfg['dvid-point-annotations']['instance']

    if ann_instance and table_path:
        raise RuntimeError(f"Element table '{name}' cannot use both a point-annotation instance and a point-table file.")

    if not table_path and not ann_instance:
        return
    
    if table_path:
        element_df = _load_element_points_from_table(name, table_path)
    if ann_instance:
        pa_cfg = table_cfg['dvid-point-annotations']
        element_df = _load_element_points_from_dvid(
            name,
            pa_cfg,
            dvid_server,
            dvid_uuid
        )

    if (cfg_type := table_cfg['type']):
        if 'type' in element_df.columns and (element_df['type'] != cfg_type).any():
            raise RuntimeError(f"Element table '{name}' has a type column that does "
                               f"not entirely match its config value ('{cfg_type}')")
        element_df['type'] = cfg_type

    if not {*element_df.columns} >= {*'xyz', 'type'}:
        raise RuntimeError(f"Element table '{name}' doesn't have xyz and/or type columns.")

    if 'point_id' in element_df:
        element_df = (
            element_df
            .astype({'point_id': np.uint64})
            .set_index('point_id')
        )
    else:
        point_ids = encode_coords_to_uint64(element_df[[*'zyx']].values)
        element_df.index = pd.Index(point_ids, name='point_id')

    return element_df


def _load_element_points_from_table(name, table_path):
    with Timer(f"Loading '{name}' elements from disk", logger):
        assert table_path.split('.')[-1] in ('feather', 'csv')
        if table_path.endswith('.csv'):
            element_df = pd.read_csv(table_path)
        else:
            element_df = feather.read_feather(table_path)
    return element_df


def _load_element_points_from_dvid(name, pa_cfg, dvid_server, dvid_uuid):
    with Timer(f"Loading '{name}' elements from DVID instance '{pa_cfg['instance']}'", logger):
        df = fetch_all_elements(
            dvid_server,
            dvid_uuid,
            pa_cfg['instance'],
            format='pandas'
        )
    df = df.sort_values([*'zyx'])

    # Assign a column for each of the extracted point properties.
    for prop, propcol in pa_cfg['extract-properties'].items():
        if prop.lower() not in df:
            logger.warning(f"Annotation instance {pa_cfg['instance']} contains no properties named '{prop}'")
            continue

        # DVID annotation properties are stored as strings,
        # even if they would ideally be stored as int or float.
        # Attempt to convert back to float if possible,
        # otherwise leave as string.
        try:
            df[prop.lower()] = df[prop.lower()].astype(np.float32)
        except ValueError:
            logger.warning(f"Annotation instance {pa_cfg['instance']} contains non-numeric property '{prop}'; keeping as string.")

        df[propcol] = df[prop.lower()]

    df = df[[*'zyx', *pa_cfg['extract-properties'].values()]]
    return df


def _load_element_distances(name, table_cfg):
    path = table_cfg['distance-table']
    if not path:
        return

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

        source_points = distance_df[['source_z', 'source_y', 'source_x']].values
        target_points = distance_df[['target_z', 'target_y', 'target_x']].values

        distance_df['source_id'] = encode_coords_to_uint64(source_points)
        distance_df['target_id'] = encode_coords_to_uint64(target_points)

    return distance_df
