"""
Import synapses from disk, filter them, and associate each point with a body ID.
"""
import os
import copy
import glob
import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease.util import Timer, timed, encode_coords_to_uint64, decode_coords_from_uint64

from ..util.checksum import checksum
from ..caches import cached, SerializerBase, cache_dataframe

logger = logging.getLogger(__name__)

MALECNS_VNC_CUTOFF_Z = 45056

# As luck would have it, we it's easy to filter by Z values directly from the
# point_id since Z occupies the most-significant position in the encoded point_id.
MALECNS_VNC_CUTOFF_POINT_ID = encode_coords_to_uint64(np.array([[MALECNS_VNC_CUTOFF_Z, 0, 0]], np.uint64))[0]


LabelmapSchema = {
    "description": "dvid labelmap location",
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
        }
    }
}

SnapshotSynapsesSchema = {
    "description": "The synapse tables to use.",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "syndir": {
            "description":
                "Optional. This value can be used within format strings in the other synapse settings.",
            "type": "string",
            "default": ""
        },
        "synapse-points": {
            "description":
                "A feather file containing the synapse points, with a 'body' column.\n"
                "If an 'sv' column is also present, it can be used to much more efficiently update the body column if needed.\n",
            "type": "string",
            # NO DEFAULT
        },
        "synapse-partners": {
            "description":
                "A feather file containing the synapse partners, with columns for pre_id, post_id",
            "type": "string",
            # NO DEFAULT
        },
        "zone": {
            "description":
                "Specifically for the male CNS. Whether to PRE-FILTER the synapses to include the brain only, vnc only, or whole cns",
            "type": "string",
            "enum": ["brain", "vnc", ""],
            "default": "",
        },
        "min-confidence": {
            "description":
                "Before generating any results, exclude synapse predictions which fall below this confidence level.\n"
                "Note: This pre-filters the synapse table, so it is effectively a minimum bound on the confidence threshold for neuprint exports.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.0,
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
        "processes": {
            "description":
                "How many processes should be used to update synapse labels?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    }
}


class SynapseSerializerBase(SerializerBase):

    def save_to_file(self, result, path):
        point_df, partner_df = result
        os.makedirs(path, exist_ok=True)

        # It's not really necessary to put the snapshot tag, etc. in the file names,
        # since that stuff is already in the directory name.
        # But it's convenient if we want to copy these files for other people to use.
        name = os.path.split(path)[-1]
        cache_dataframe(
            point_df.reset_index(),
            f'{path}/point_df-{name}.feather'
        )
        cache_dataframe(
            partner_df,
            f'{path}/partner_df-{name}.feather'
        )

    def load_from_file(self, path):
        point_path = glob.glob(f'{path}/point_df-*.feather')[0]
        partner_path = glob.glob(f'{path}/partner_df-*.feather')[0]
        point_df = feather.read_feather(point_path).set_index('point_id')
        partner_df = feather.read_feather(partner_path)
        return point_df, partner_df


class RawSynapseSerializer(SynapseSerializerBase):

    def get_cache_key(self, cfg, snapshot_tag, pointlabeler):
        cfg = copy.deepcopy(cfg)
        cfg['processes'] = 0
        cfg_hash = hex(checksum(cfg))

        if pointlabeler is None:
            return f'synapses-{snapshot_tag}-cfg-{cfg_hash}'

        mutid = pointlabeler.last_mutation["mutid"]
        return f'{snapshot_tag}-seg-{mutid}-syn-{cfg_hash}'


@cached(RawSynapseSerializer('labeled-synapses'))
def load_synapses(cfg, snapshot_tag, pointlabeler):  # noqa
    point_df, partner_df = _load_raw_synapses(cfg)

    point_df, partner_df = _filter_for_zone(point_df, partner_df, cfg['zone'])
    point_df, partner_df = _filter_for_confidence(point_df, partner_df, cfg['min-confidence'])
    point_df, partner_df = _filter_for_self_consistency(point_df, partner_df)
    logger.info(f"Kept {len(point_df)} points and {len(partner_df)} partners after filtering")

    point_df, partner_df = _update_body_columns(point_df, partner_df, pointlabeler, cfg['processes'])
    return point_df, partner_df


def _load_raw_synapses(cfg):
    points_path = cfg['synapse-points'].format(syndir=cfg['syndir'])
    partners_path = cfg['synapse-partners'].format(syndir=cfg['syndir'])

    with Timer("Loading synapses from disk", logger):
        point_df = feather.read_feather(points_path)
        partner_df = feather.read_feather(partners_path).astype({'pre_id': np.uint64, 'post_id': np.uint64})

    # Temporarily ensure point_id is in the columns to simplify the logic below.
    # (We'll move it to the index below.)
    if 'point_id' not in point_df.columns and point_df.index.name == 'point_id':
        point_df = point_df.reset_index()

    point_df = point_df.astype({'point_id': np.uint64})

    if 'point_id' not in point_df.columns and not {*'zyx'} <= {*point_df.columns}:
        raise RuntimeError("Synapse point table doesn't have coordinates or point_id")

    # If point_df is missing either point_id or zyx columns,
    # then use one to generate the other.
    if not ({*'zyx'} <= {*point_df.columns}):
        logger.warning("synapse (x,y,z) columns not provided. I'm assuming they can be decoded from 'point_id'")
        point_df[[*'zyx']] = decode_coords_from_uint64(point_df['point_id'].values)

    if 'point_id' not in point_df.columns:
        point_df['point_id'] = encode_coords_to_uint64(point_df[[*'zyx']].values)

    if not {'point_id', *'zyx', 'kind', 'conf'} <= {*point_df.columns}:
        raise RuntimeError(f"Synapse point table does not have all required columns. Found: {point_df.columns}")

    if not {'pre_id', 'post_id'} <= {*partner_df.columns}:
        raise RuntimeError(f"Synpase partner table does not have the required columns. Found: {partner_df.columns}")

    point_df = point_df.set_index('point_id')
    logger.info(f"Loaded {len(point_df)} points and {len(partner_df)} partners")

    point_df, partner_df = _streamline_synapse_tables(point_df, partner_df)
    return point_df, partner_df


@timed("Dropping unecessary columns and narrowing dtypes", logger)
def _streamline_synapse_tables(point_df, partner_df):
    id_cols = {'sv', 'body'} & set(point_df.columns)
    if id_cols and (point_df[[*id_cols]] <= 2**32).all().all():
        label_dtype = np.uint32
    else:
        label_dtype = np.int64

    point_df = point_df.drop(
        errors='ignore',
        columns=[
            'roi_label',
            'user',
            'tbar_region',
        ]
    )

    t = {
        'conf': np.float32,
        'sv': label_dtype,
        'body': label_dtype,
        'kind': 'category',
        'roi': 'category',
    }
    point_df = point_df.astype({
        k:v for k,v in t.items()
        if k in point_df.columns
    })

    point_df, partner_df = _drop_fake_synapses(point_df, partner_df)

    if point_df['kind'].isnull().any():
        raise RuntimeError("Synapse 'kind' column must not contain null values")

    syn_kinds = set(point_df['kind'].dtype.categories)
    if invalid_kinds := syn_kinds - {'PreSyn', 'PostSyn'}:
        raise RuntimeError(f"Synapse 'kind' column includes unexpected values: {invalid_kinds}")

    t = {
        'conf_pre': np.float32,
        'conf_post': np.float32,
    }
    partner_df = partner_df.astype({
        k:v for k,v in t.items()
        if k in partner_df.columns
    })
    return point_df, partner_df


def _drop_fake_synapses(point_df, partner_df):
    """
    Sometimes synapse tables that were exported from DVID using
    ``neuclease.dvid.annotation.fetch_synapses_in_batches()``
    may contain points whose kind='Fake'.
    This indicates that at least one synapse had no relationships, so it was
    added to the table as a point with kind='Fake'.
    We exclude these points from the output, and filter out the corresponding
    partners by excluding partners whose pre_id or post_id is 0.

    (This means the stored synapses in DVID are inconsistent, and we generally
    try to fix the source problem, rather than accommodating it downstream.
    But we want to support such datasets here if needed.)
    """
    syn_kinds = set(point_df['kind'].dtype.categories)
    if 'Fake' not in syn_kinds:
        return point_df, partner_df

    if not (point_df['kind'] == 'Fake').any():
        point_df['kind'] = point_df['kind'].cat.remove_unused_categories()
        return point_df, partner_df

    logger.warning(
        "Synapse table contains points whose kind='Fake'. "
        "This indicates that at least one synapse had no relationships. "
        "Those synapses will be excluded from the output."
    )
    num_fake_points = (point_df['kind'] == 'Fake').sum()
    num_orphaned_partners = ((partner_df['pre_id'] == 0) | (partner_df['post_id'] == 0)).sum()
    logger.warning(f"Excluding {num_fake_points} fake points and {num_orphaned_partners} orphaned partners")
    point_df = point_df.loc[point_df['kind'] != 'Fake']
    partner_df = partner_df.loc[partner_df['pre_id'] != 0]
    partner_df = partner_df.loc[partner_df['post_id'] != 0]
    point_df['kind'] = point_df['kind'].cat.remove_unused_categories()
    return point_df, partner_df


@timed("Filtering for zone {zone}", logger)
def _filter_for_zone(point_df, partner_df, zone):
    assert zone in ('', 'brain', 'vnc')
    if zone == 'brain':
        point_df = point_df.loc[point_df['z'] < MALECNS_VNC_CUTOFF_Z].copy()
        partner_df = partner_df.loc[
            (partner_df['pre_id'] < MALECNS_VNC_CUTOFF_POINT_ID) &  # noqa
            (partner_df['post_id'] < MALECNS_VNC_CUTOFF_POINT_ID)].copy()
    elif zone == 'vnc':
        point_df = point_df.loc[point_df['z'] >= MALECNS_VNC_CUTOFF_Z].copy()
        partner_df = partner_df.loc[
            (partner_df['pre_id'] >= MALECNS_VNC_CUTOFF_POINT_ID) &  # noqa
            (partner_df['post_id'] >= MALECNS_VNC_CUTOFF_POINT_ID)].copy()

    return point_df, partner_df


def _filter_for_confidence(point_df, partner_df, min_conf):
    """
    1. Ensure the partner_df contains both 'conf_pre' and 'conf_post'
    2. Filter both tables to exclude low-confidence synapses, according to min_conf
    """
    if 'conf_pre' not in partner_df.columns:
        logger.info("Merging conf_pre column onto partner_df")
        partner_df = partner_df.merge(
            point_df['conf'].rename('conf_pre').rename_axis('pre_id'),
            'left',
            on='pre_id'
        )

    if 'conf_post' not in partner_df.columns:
        logger.info("Merging conf_post column onto partner_df")
        partner_df = partner_df.merge(
            point_df['conf'].rename('conf_post').rename_axis('post_id'),
            'left',
            on='post_id'
        )

    if min_conf == 0.0:
        return point_df, partner_df

    with Timer(f"Filtering for min-confidence: {min_conf}", logger):
        # Note:
        #   This can result in inconsistent partner/point tables, but that's
        #   okay because we will filter for self-consistency in a later step.
        point_df = point_df.loc[point_df['conf'] >= min_conf]
        partner_df = partner_df.loc[
            (partner_df['conf_pre'] >= min_conf) &  # noqa
            (partner_df['conf_post'] >= min_conf)]

    return point_df.copy(), partner_df.copy()


def _filter_for_self_consistency(point_df, partner_df):
    logger.info("Filtering partners to exclude unlisted points")
    valid_pre = partner_df['pre_id'].isin(point_df.index)
    valid_post = partner_df['post_id'].isin(point_df.index)
    partner_df = partner_df.loc[valid_pre & valid_post]

    # Also filter point list again, to toss out points which had no partner
    logger.info("Filtering points to exclude orphaned tbars/psds")
    valid_ids = pd.concat((partner_df['pre_id'].drop_duplicates().rename('point_id'),  # noqa
                           partner_df['post_id'].drop_duplicates().rename('point_id')),
                          ignore_index=True)
    point_df = point_df.loc[point_df.index.isin(valid_ids)]
    return point_df.copy(), partner_df.copy()


def _update_body_columns(point_df, partner_df, pointlabeler, processes):
    if pointlabeler:
        with Timer(f"Updating supervoxels/bodies for UUID {pointlabeler.dvidseg.uuid[:6]}", logger):
            pointlabeler.update_bodies_for_points(point_df, processes=processes)
            point_df = point_df.astype({'body': np.int64, 'sv': np.int64})

    if 'body' not in point_df.columns:
        raise RuntimeError(
            "point_df must contain a 'body' column or you "
            "must provide a dvid segmentation for 'update-to'")

    with Timer("Adding columns to partner table for body_pre, body_post", logger):
        partner_df = partner_df.drop(columns=['body_pre', 'body_post'], errors='ignore')
        partner_df = partner_df.merge(point_df['body'], 'left', left_on='pre_id', right_index=True)
        partner_df = partner_df.merge(point_df['body'], 'left', left_on='post_id', right_index=True, suffixes=['_pre', '_post'])

    return point_df, partner_df
