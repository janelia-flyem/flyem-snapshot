import os
import logging

import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, decode_coords_from_uint64
from neuclease.misc.completeness import ranked_synapse_counts

logger = logging.getLogger(__name__)

FlatConnectomeSchema = {
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "export-connectome": {
            "description": "If true, export the connectome in heavy but easy-to-use flat feather files.",
            "type": "boolean",
            "default": True,
        },
        "roi-set": {
            "type": "string",
            "default": "primary"
        }
    }
}


@PrefixFilter.with_context('Connectome Export')
def export_flat_connectome(cfg, point_df, partner_df, ann, snapshot_tag, min_conf):
    """
    Export the full list of pre-post partners in not-so-compact form,
    for external users, with full x/y/z columns.
    Also export the full body-to-body weighted connectome, and also
    the abridged body-to-body weighted connectome for only 'primary' bodies.
    """
    if not cfg['export-connectome']:
        return

    roiset = cfg['roi-set']

    os.makedirs('flat-connectome', exist_ok=True)

    # If the connectome export files already exist, then skip this function.
    # FIXME: Don't write this filename in two different places.
    if os.path.exists(f'flat-connectome/connectome-weights-{snapshot_tag}-minconf-{min_conf}-primary-only.feather'):
        return

    with Timer("Constructing synapse partner export", logger):
        partner_export_df = partner_df[['pre_id', 'post_id', 'body_pre', 'body_post', roiset]]

        # Add conf_pre, conf_post
        partner_export_df = (
            partner_export_df
            .merge(
                point_df['conf'].rename('conf_pre').rename_axis('pre_id'),
                'left',
                on='pre_id'
            )
            .merge(
                point_df['conf'].rename('conf_post').rename_axis('post_id'),
                'left',
                on='post_id'
            )
        )
        partner_export_df[['z_pre', 'y_pre', 'x_pre']] = decode_coords_from_uint64(partner_export_df['pre_id'].values)
        partner_export_df[['z_post', 'y_post', 'x_post']] = decode_coords_from_uint64(partner_export_df['post_id'].values)

        # In the original point_df, the roi column was determined according to the post location.
        # Rename with _post suffix to make that clear for consumers of this table.
        partner_export_df = partner_export_df.rename(columns={roiset: f'{roiset}_post'})
        partner_cols = [
            'x_pre', 'y_pre', 'z_pre', 'body_pre', 'conf_pre',
            'x_post', 'y_post', 'z_post', 'body_post', 'conf_post', f'{roiset}_post']
        partner_export_df = partner_export_df[partner_cols]

    with Timer("Writing synapse partner export", logger):
        feather.write_feather(
            partner_export_df,
            f'flat-connectome/syn-partners-{snapshot_tag}-minconf-{min_conf}.feather'
        )

    with Timer("Constructing weighted connectome", logger):
        connectome = (
            partner_export_df[['body_pre', 'body_post']]
            .value_counts()
            .rename('weight')
            .reset_index()
        )

    with Timer("Writing weighted connectome", logger):
        feather.write_feather(
            connectome,
            f'flat-connectome/connectome-weights-{snapshot_tag}-minconf-{min_conf}.feather'
        )

    with Timer("Constructing primary-only synapse partner export", logger):
        primary_bodies = ann.query('status >= "Primary Anchor"').index
        logger.info(f"There are {len(primary_bodies)} with status 'Primary Anchor' or better.")
        primary_partner_export_df = partner_export_df.query('body_pre in @primary_bodies and body_post in @primary_bodies')

    with Timer("Writing primary-only synapse partner export", logger):
        feather.write_feather(
            primary_partner_export_df,
            f'flat-connectome/syn-partners-{snapshot_tag}-minconf-{min_conf}-primary-only.feather'
        )

    with Timer("Writing primary-only weighted connectome", logger):
        primary_connectome = (
            primary_partner_export_df[['body_pre', 'body_post']]
            .value_counts()
            .rename('weight')
            .reset_index()
            .merge(
                ann['type'].rename('type_pre').rename_axis('body_pre'),
                'left',
                on='body_pre'
            )
            .merge(
                ann['type'].rename('type_post').rename_axis('body_post'),
                'left',
                on='body_post'
            )
        )

    with Timer("Writing primary-only weighted connectome", logger):
        feather.write_feather(
            primary_connectome,
            f'flat-connectome/connectome-weights-{snapshot_tag}-minconf-{min_conf}-primary-only.feather'
        )

    with Timer("Computing ranked body stats table", logger):
        # A version of this table is also exported for each 'report' in the config,
        # but we also export it as part of the 'flat' connectome export,
        # and we include type/instance.
        syn_counts_df = ranked_synapse_counts(point_df, partner_df, body_annotations_df=ann)
        syn_counts_df = syn_counts_df.rename(columns={
            'OutputPartners': 'downstream',
            'PreSyn': 'pre',
            'PostSyn': 'post',
            'SynWeight': 'synweight'
        })
        body_stats = syn_counts_df.merge(ann[['type', 'instance', 'group']], 'left', on='body')

    with Timer("Writing ranked body stats table", logger):
        feather.write_feather(
            body_stats.reset_index(),
            f'flat-connectome/body-stats-{snapshot_tag}-minconf-{min_conf}.feather'
        )
