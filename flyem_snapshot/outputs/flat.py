import os
import logging

import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, decode_coords_from_uint64
from neuclease.misc.completeness import ranked_synapse_counts

from ..caches import cached, SentinelSerializer
from ..util import restrict_synapses_to_roi

logger = logging.getLogger(__name__)

MIN_SIGNIFICANT_STATUS = "Sensory Anchor"

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
            "description":
                "Which set of ROI names to include in the output data as a column.\n"
                "This does not specify how the synapses are filtered before export.\n",
            "type": "string",
            "default": "primary"
        },
        "restrict-connectivity-to-roiset": {
            "description":
                "Discard synapses that fall outside the given roiset, i.e. they are <unspecified> in this roiset.\n"
                "The discarded synapses won't be included in the database individually,\n"
                "nor will they contribute to neuron-to-neuron connectome weights.\n"
                "This setting does not indicate which ROI set will be listed in the output as a column in the synapse tables.\n",
            "type": "string",
            "default": ""
        },
    }
}


@PrefixFilter.with_context('Connectome Export')
@cached(SentinelSerializer('flat-connectome'))
def export_flat_connectome(cfg, point_df, partner_df, ann, snapshot_tag, min_conf):
    """
    Export the full list of pre-post partners in not-so-compact form,
    for external users, with full x/y/z columns.
    Also export the full body-to-body weighted connectome, and also
    the abridged body-to-body weighted connectome for only 'significant' bodies.
    """
    if not cfg['export-connectome']:
        return

    os.makedirs('flat-connectome', exist_ok=True)

    point_df, partner_df, file_tag = _filter_synapses(cfg, point_df, partner_df, snapshot_tag, min_conf)

    partner_export_df = _export_synapse_partners(cfg, point_df, partner_df, file_tag)
    _export_weighted_connectome(partner_export_df, file_tag)

    significant_partner_export_df = _export_significant_synapse_partners(ann, partner_export_df, file_tag)
    _export_significant_weighted_connectome(ann, significant_partner_export_df, file_tag)

    _export_ranked_body_stats(ann, point_df, partner_df, file_tag)


def _filter_synapses(cfg, point_df, partner_df, snapshot_tag, min_conf):
    filtering_roiset = cfg['restrict-connectivity-to-roiset']
    if filtering_roiset:
        with Timer(f"Excluding synapses whose ROI is unspecified within roi-set: '{filtering_roiset}'"):
            point_df, partner_df = restrict_synapses_to_roi(filtering_roiset, None, point_df, partner_df)

    if filtering_roiset:
        file_tag = f"{snapshot_tag}-minconf-{min_conf}-{filtering_roiset}"
    else:
        file_tag = f"{snapshot_tag}-minconf-{min_conf}"

    return point_df, partner_df, file_tag


def _export_synapse_partners(cfg, point_df, partner_df, file_tag):
    labeling_roiset = cfg['roi-set']
    with Timer("Constructing synapse partner export", logger):
        partner_export_df = partner_df[['pre_id', 'post_id', 'body_pre', 'body_post', labeling_roiset]]

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
        partner_export_df = partner_export_df.rename(columns={labeling_roiset: f'{labeling_roiset}_post'})
        partner_cols = [
            'x_pre', 'y_pre', 'z_pre', 'body_pre', 'conf_pre',
            'x_post', 'y_post', 'z_post', 'body_post', 'conf_post', f'{labeling_roiset}_post']
        partner_export_df = partner_export_df[partner_cols]

    with Timer("Writing synapse partner export", logger):
        feather.write_feather(
            partner_export_df,
            f'flat-connectome/syn-partners-{file_tag}.feather'
        )
    with Timer("Writing synapse point export", logger):
        feather.write_feather(
            point_df,
            f'flat-connectome/syn-points-{file_tag}.feather'
        )

    return partner_export_df


def _export_weighted_connectome(partner_export_df, file_tag):
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
            f'flat-connectome/connectome-weights-{file_tag}.feather'
        )


def _export_significant_synapse_partners(ann, partner_export_df, file_tag):
    with Timer("Constructing significant-only synapse partner export", logger):
        significant_bodies = ann.query(f'status >= "{MIN_SIGNIFICANT_STATUS}"').index
        logger.info(f"There are {len(significant_bodies)} with status '{MIN_SIGNIFICANT_STATUS}' or better.")
        significant_partner_export_df = partner_export_df.query(
            'body_pre in @significant_bodies and body_post in @significant_bodies')

    msg = f"Writing significant-only synapse partner export (with only {MIN_SIGNIFICANT_STATUS} or better)"
    with Timer(msg, logger):
        feather.write_feather(
            significant_partner_export_df,
            f'flat-connectome/syn-partners-{file_tag}-significant-only.feather'
        )
    return significant_partner_export_df


def _export_significant_weighted_connectome(ann, significant_partner_export_df, file_tag):
    msg = f"Constructing significant-only weighted connectome (with only {MIN_SIGNIFICANT_STATUS} or better)"
    with Timer(msg, logger):
        significant_connectome = (
            significant_partner_export_df[['body_pre', 'body_post']]
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

    with Timer("Writing significant-only weighted connectome", logger):
        feather.write_feather(
            significant_connectome,
            f'flat-connectome/connectome-weights-{file_tag}-significant-only.feather'
        )


def _export_ranked_body_stats(ann, point_df, partner_df, file_tag):
    with Timer("Computing ranked body stats table", logger):
        # A version of this table is also exported for each 'report' in the config,
        # but we also export it here as part of the 'flat' connectome export.
        # Here we also include status/class/type/instance.
        extra_cols = [c for c in ('status', 'class', 'type', 'instance') if c in ann.columns]
        syn_counts_df = ranked_synapse_counts(point_df, partner_df, body_annotations_df=ann[extra_cols])
        syn_counts_df = syn_counts_df.rename(columns={
            'OutputPartners': 'downstream',
            'PreSyn': 'pre',
            'PostSyn': 'post',
            'SynWeight': 'synweight'
        })

    with Timer("Writing ranked body stats table", logger):
        feather.write_feather(
            syn_counts_df.reset_index(),
            f'flat-connectome/body-stats-{file_tag}.feather'
        )
