import os
import logging

import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, decode_coords_from_uint64
from neuclease.misc.completeness import ranked_synapse_counts

from ..caches import cached, SentinelSerializer
from ..util.util import restrict_synapses_to_roi, upload_file_to_gcs

from .neuprint.annotations import neuprint_segment_annotations
from .neuprint.segment import CONNECTION_COMPARTMENT_PROPERTIES, _compartment_in_out_counts, _compartments_available

logger = logging.getLogger(__name__)

MIN_SIGNIFICANT_STATUS = "Sensory Anchor"
MIN_TRACED_STATUS = "Leaves"

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
        "compartment-column": {
            "description":
                "Name of the point_df column which classifies each synapse point into a\n"
                "neuron compartment ('axon', 'dendrite', 'linker', 'cell-body-fiber', 'unknown').\n"
                "If not given here, defaults to whatever is configured in inputs.synapses.compartment-column.\n"
                "If left blank (and no default is available), no compartment-breakdown columns\n"
                "(weightAxonDendrite, etc., axonIn/axonOut/dendriteIn/dendriteOut) are exported.",
            "type": "string",
            "default": ""
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
        "gcp-project": {
            "description": "Google Cloud project.",
            "oneOf": [
                {"type": "string"},
                {"type": "null"}
            ],
            "default": ""
        },
        "gcs-bucket-dir": {
            "description":
                "Google Cloud Storage bucket and directory to export the\n"
                "flat-connectome to, e.g. 'gs://my-bucket/my-dir'\n",
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

    partner_export_df = _export_synapse_partners(cfg, point_df, partner_df, snapshot_tag, file_tag)
    _export_weighted_connectome(cfg, partner_export_df, snapshot_tag, file_tag)

    if ann['status'].dtype != 'category':
        logger.info("Status column is not a category. Skipping significant/traced filtered exports.")
    else:
        # 'significant' bodies
        significant_partner_export_df = _export_significant_synapse_partners(cfg, ann, partner_export_df, snapshot_tag, file_tag, MIN_SIGNIFICANT_STATUS, "significant")
        _export_significant_weighted_connectome(cfg, ann, significant_partner_export_df, snapshot_tag, file_tag, MIN_SIGNIFICANT_STATUS, "significant")

        # 'traced' bodies
        significant_partner_export_df = _export_significant_synapse_partners(cfg, ann, partner_export_df, snapshot_tag, file_tag, MIN_TRACED_STATUS, "traced")
        _export_significant_weighted_connectome(cfg, ann, significant_partner_export_df, snapshot_tag, file_tag, MIN_TRACED_STATUS, "traced")

    _export_neuprint_body_annotations(cfg, ann, snapshot_tag, file_tag)
    _export_ranked_body_stats(cfg, ann, point_df, partner_df, snapshot_tag, file_tag)



def _filter_synapses(cfg, point_df, partner_df, snapshot_tag, min_conf):
    filtering_roiset = cfg['restrict-connectivity-to-roiset']
    if filtering_roiset:
        if filtering_roiset not in point_df.columns:
            raise RuntimeError(
                "Cannot filter flat-connectome according to your 'restrict-connectivity-to-roiset' "
                f"config because roi-set '{filtering_roiset}' is not listed in the synapse table."
            )
        with Timer(f"Excluding synapses whose ROI is unspecified within roi-set: '{filtering_roiset}'"):
            point_df, partner_df = restrict_synapses_to_roi(filtering_roiset, None, point_df, partner_df)

    if filtering_roiset:
        file_tag = f"{snapshot_tag}-minconf-{min_conf}-{filtering_roiset}"
    else:
        file_tag = f"{snapshot_tag}-minconf-{min_conf}"

    return point_df, partner_df, file_tag


def _export_synapse_partners(cfg, point_df, partner_df, snapshot_tag, file_tag):
    labeling_roiset = cfg['roi-set']
    if labeling_roiset not in partner_df.columns:
        raise RuntimeError(
            f"Cannot export flat-connectome with roi-set '{labeling_roiset}' "
            "because it isn't listed in the synapse table."
        )

    # If compartments are available, carry the per-side compartment columns
    # through to the export (and use them for the weighted-connectome breakdown).
    compartment_cols = [c for c in ('compartment_pre', 'compartment_post') if c in partner_df.columns]

    with Timer("Constructing synapse partner export", logger):
        partner_export_df = partner_df[['pre_id', 'post_id', 'body_pre', 'body_post', labeling_roiset, *compartment_cols]]

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
            'x_post', 'y_post', 'z_post', 'body_post', 'conf_post', f'{labeling_roiset}_post',
            *compartment_cols]
        partner_export_df = partner_export_df[partner_cols]

    with Timer("Writing synapse partner export", logger):
        fname = f'flat-connectome/syn-partners-{file_tag}.feather'
        feather.write_feather(
            partner_export_df,
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])
    with Timer("Writing synapse point export", logger):
        fname = f'flat-connectome/syn-points-{file_tag}.feather'
        feather.write_feather(
            point_df,
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])

    return partner_export_df


def _connection_compartment_breakdown(partner_export_df):
    """
    Given a partner table with 'compartment_pre'/'compartment_post' columns,
    compute the per-body-pair breakdown of connection weights into the tracked
    compartment combinations (weightAxonDendrite, weightAxonAxon, etc.).

    Returns a DataFrame indexed by ['body_pre', 'body_post'] with exactly the
    columns from CONNECTION_COMPARTMENT_PROPERTIES.values(), as int64.
    Combinations involving linker/cell-body-fiber/unknown are not tracked
    (but still contribute to the overall 'weight').
    """
    props = list(CONNECTION_COMPARTMENT_PROPERTIES.values())
    pair_to_prop = {f'{a}-{b}': name for (a, b), name in CONNECTION_COMPARTMENT_PROPERTIES.items()}
    compartment_prop = (
        partner_export_df['compartment_pre'].astype(str) + '-' + partner_export_df['compartment_post'].astype(str)
    ).map(pair_to_prop)

    breakdown = (
        partner_export_df[['body_pre', 'body_post']]
        .assign(compartment_prop=compartment_prop)
        .dropna(subset=['compartment_prop'])
        .value_counts()
        .unstack('compartment_prop', fill_value=0)
    )
    for prop in props:
        if prop not in breakdown.columns:
            breakdown[prop] = 0
    return breakdown[props].astype('int64')


def _add_compartment_weights(connectome, partner_export_df):
    """
    If the partner table carries compartment columns, merge the per-body-pair
    compartment weight breakdown onto the given connectome table (in-place-ish),
    returning the augmented connectome. Otherwise return it unchanged.

    'connectome' is expected to have 'body_pre'/'body_post' columns (not index).
    """
    if not ({'compartment_pre', 'compartment_post'} <= set(partner_export_df.columns)):
        return connectome

    props = list(CONNECTION_COMPARTMENT_PROPERTIES.values())
    breakdown = _connection_compartment_breakdown(partner_export_df)
    connectome = connectome.merge(breakdown, 'left', on=['body_pre', 'body_post'])
    connectome[props] = connectome[props].fillna(0).astype('int64')
    return connectome


def _export_weighted_connectome(cfg, partner_export_df, snapshot_tag, file_tag):
    with Timer("Constructing weighted connectome", logger):
        connectome = (
            partner_export_df[['body_pre', 'body_post']]
            .value_counts()
            .rename('weight')
            .reset_index()
        )
        connectome = _add_compartment_weights(connectome, partner_export_df)

    with Timer("Writing weighted connectome", logger):
        fname = f'flat-connectome/connectome-weights-{file_tag}.feather'
        feather.write_feather(
            connectome,
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])


def _export_significant_synapse_partners(cfg, ann, partner_export_df, snapshot_tag, file_tag, min_status, description):
    with Timer(f"Constructing {description}-only synapse partner export", logger):
        assert ann.index.name == 'body'
        significant_bodies = ann.query(f'status >= "{min_status}"').index
        logger.info(f"There are {len(significant_bodies)} with status '{min_status}' or better.")
        significant_partner_export_df = (
            partner_export_df
            .query('body_pre in @significant_bodies and body_post in @significant_bodies')
            .reset_index(drop=True)
        )

    msg = f"Writing {description}-only synapse partner export (with only {min_status} or better)"
    with Timer(msg, logger):
        fname = f'flat-connectome/syn-partners-{file_tag}-{description}-only.feather'
        feather.write_feather(
            significant_partner_export_df,
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])
    return significant_partner_export_df


def _export_significant_weighted_connectome(cfg, ann, significant_partner_export_df, snapshot_tag, file_tag, min_status, description):
    msg = f"Constructing {description}-only weighted connectome (with only {min_status} or better)"
    with Timer(msg, logger):
        significant_connectome = (
            significant_partner_export_df[['body_pre', 'body_post']]
            .value_counts()
            .rename('weight')
            .reset_index()
        )
        significant_connectome = _add_compartment_weights(significant_connectome, significant_partner_export_df)

        if 'type' in ann.columns:
            significant_connectome = (
                    significant_connectome
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

    with Timer(f"Writing {description}-only weighted connectome", logger):
        fname = f'flat-connectome/connectome-weights-{file_tag}-{description}-only.feather'
        feather.write_feather(
            significant_connectome,
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])


def _export_neuprint_body_annotations(cfg, ann, snapshot_tag, file_tag):
    """
    Export the body annotation table, but with with the same filtering
    and column names that we use when exporting to neuprint.
    """
    neuprint_ann = neuprint_segment_annotations({}, ann, convert_points_to_neo4j_spatial=False)
    assert 'bodyId' in neuprint_ann.columns
    neuprint_ann = neuprint_ann.reset_index(drop=True)

    with Timer("Writing body annotation table", logger):
        fname = f'flat-connectome/body-annotations-{file_tag}.feather'
        feather.write_feather(
            neuprint_ann,
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])


def _export_ranked_body_stats(cfg, ann, point_df, partner_df, snapshot_tag, file_tag):
    with Timer("Computing ranked body stats table", logger):
        # A version of this table is also exported for each 'report' in the config,
        # but we also export it here as part of the 'flat' connectome export.
        # Here we also include status/class/type/instance.
        extra_cols = [c for c in ('status', 'superclass', 'class', 'type', 'instance') if c in ann.columns]
        syn_counts_df = ranked_synapse_counts(point_df, partner_df, body_annotations_df=ann[extra_cols])
        syn_counts_df = syn_counts_df.rename(columns={
            'OutputPartners': 'downstream',
            'PreSyn': 'pre',
            'PostSyn': 'post',
            'SynWeight': 'synweight'
        })

        # Add per-compartment breakdown of incoming/outgoing connections, if available.
        if _compartments_available(cfg, point_df, partner_df):
            assert syn_counts_df.index.name == 'body'
            compartment_counts = _compartment_in_out_counts(point_df, partner_df, cfg['compartment-column'])
            syn_counts_df = syn_counts_df.merge(compartment_counts, 'left', on='body')
            io_cols = ['axonIn', 'dendriteIn', 'axonOut', 'dendriteOut']
            syn_counts_df[io_cols] = syn_counts_df[io_cols].fillna(0).astype('int64')

    syn_counts_df = syn_counts_df.rename(columns={'status': 'status_fine'})
    with Timer("Writing ranked body stats table", logger):
        fname = f'flat-connectome/body-stats-{file_tag}.feather'
        feather.write_feather(
            syn_counts_df.reset_index(),
            fname
        )
        if cfg['gcs-bucket-dir']:
            upload_file_to_gcs(f"{cfg['gcs-bucket-dir']}/{snapshot_tag}", fname, cfg['gcp-project'])
