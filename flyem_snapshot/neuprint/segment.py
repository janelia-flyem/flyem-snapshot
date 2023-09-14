import os
import json
import shutil
import logging
from functools import partial

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import timed, Timer, compute_parallel, tqdm_proxy, snakecase_to_camelcase

from .util import append_neo4j_type_suffixes

logger = logging.getLogger(__name__)

# For most fields, we formulaically convert from snake_case to camelCase,
# but for some fields the terminology isn't translated by that formula.
# This list provides explicit translations for the special cases.
# Also, to exclude a DVID/clio body annotation field from neuprint entirely,
# list it here and map it to "".
CLIO_TO_NEUPRINT_PROPERTIES = {
    'bodyid': 'bodyId',
    'status': 'statusLabel',
    'hemibrain_bodyid': 'hemibrainBodyId',

    # Make sure these never appear in neuprint.
    'last_modified_by': '',
    'old_bodyids': '',
    'reviewer': '',
    'to_review': '',
    'typing_notes': '',
    'user': '',
    'notes': '',
    'halfbrainBody': '',

    # These generally won't be sourced from Clio anyway;
    # they should be sourced from the appropriate DVID annotation instance.
    'soma_position': 'somaLocation',
    'tosoma_position': 'tosomaLocation',
    'root_position': 'rootLocation',
}

# Note any 'statusLabel' (DVID status) that isn't
# listed here will appear in neuprint unchanged.
NEUPRINT_STATUSLABEL_TO_STATUS = {
    'Unimportant':              'Unimportant',  # noqa
    'Glia':                     'Glia',         # noqa
    'Hard to trace':            'Orphan',       # noqa
    'Orphan-artifact':          'Orphan',       # noqa
    'Orphan':                   'Orphan',       # noqa
    'Orphan hotknife':          'Orphan',       # noqa

    'Out of scope':             '',             # noqa
    'Not examined':             '',             # noqa
    '':                         '',             # noqa

    '0.5assign':                '0.5assign',    # noqa

    'Anchor':                   'Anchor',       # noqa
    'Cleaved Anchor':           'Anchor',       # noqa
    'Sensory Anchor':           'Anchor',       # noqa
    'Cervical Anchor':          'Anchor',       # noqa
    'Soma Anchor':              'Anchor',       # noqa
    'Primary Anchor':           'Anchor',       # noqa
    'Partially traced':         'Anchor',       # noqa

    'Leaves':                   'Traced',       # noqa
    'PRT Orphan':               'Traced',       # noqa
    'Prelim Roughly traced':    'Traced',       # noqa
    'RT Orphan':                'Traced',       # noqa
    'Roughly traced':           'Traced',       # noqa
    'Traced in ROI':            'Traced',       # noqa
    'Traced':                   'Traced',       # noqa
    'Finalized':                'Traced',       # noqa
}


@PrefixFilter.with_context("Neuron")
def export_neuprint_segments(cfg, point_df, partner_df, ann, body_sizes):
    """
    Two issues:
    - efficiently set the ROI boolean properties
    - efficiently compute the roiInfo
    """
    ann = ann.query('body != 0')
    ann = _neuprint_neuron_annotations(cfg, ann)

    # Filter out low-confidence PSDs before computing weights.
    balanced_confidence = cfg['postHighAccuracyThreshold']
    partner_df = partner_df.query('conf_post >= @balanced_confidence')
    _ = balanced_confidence  # linting fix

    body_stats = _body_synstats(point_df, partner_df)
    roi_syn_df = _body_roi_synstats(cfg, point_df, partner_df)
    roi_info_df = _neuprint_neuron_roi_infos(roi_syn_df, cfg['processes'])

    # We use a left-merge here (not outer) because we deliberately exclude
    # bodies which have no synapses whatsoever, even if there is annotation
    # data for the body.
    # One reason not to use such bodies is that the annotations can (sadly)
    # contain stale body IDs, and also 'Unimportant' bodies which are just
    # fixative or whatever and intended to be excluded from our work.
    neuron_df = body_stats.merge(ann, 'left', on='body')
    neuron_df = neuron_df.merge(roi_info_df, 'left', on='body')
    neuron_df = neuron_df.merge(body_sizes, 'left', on='body')
    _assign_segment_label(cfg, neuron_df)

    # We include bodyId as a property AND as the node ID column for neo4j ingestion.
    neuron_df['bodyId'] = neuron_df.index
    neuron_df = neuron_df.rename_axis(':ID(Body-ID)').reset_index()
    neuron_df = append_neo4j_type_suffixes(cfg, neuron_df, exclude=['roiset_hash'])

    _export_neuron_csvs(neuron_df, cfg['processes'])


@timed
def _body_synstats(point_df, partner_df):
    prepost = point_df[['body', 'kind']].value_counts().unstack(fill_value=0)
    prepost.columns = ['post', 'pre']

    upstream = prepost['post'].rename('upstream')
    downstream = partner_df['body_pre'].value_counts().rename('downstream').rename_axis('body')
    upstream, downstream = upstream.align(downstream, fill_value=0)
    upstream, downstream = upstream.astype(np.int32), downstream.astype(np.int32)
    synweight = (upstream + downstream).rename('synweight')
    body_syn = pd.concat((prepost, upstream, downstream, synweight), axis=1)
    body_syn = body_syn.fillna(0).astype(np.int32)
    body_syn = body_syn.query('body != 0')

    feather.write_feather(body_syn.reset_index(), 'neuprint/body_syn.feather')
    return body_syn


@timed
def _body_roi_synstats(cfg, point_df, partner_df):
    """
    Per-body-per-ROI stats
    """
    roiset_names = list(cfg['roi-sets'].keys())

    roiset_dfs = []
    for i, roiset_name in enumerate(tqdm_proxy(roiset_names)):
        if i == 0:
            # We only include <unspecified> counts for the first ROI set.
            # The actual <unspecified> count will not appear in neuprint,
            # but we need to ensure that each body is listed at least once,
            # so it will be given an roiInfo, even if it's empty, i.e. {}.
            df = _roisyn_for_roiset(point_df, partner_df, roiset_name)
        else:
            # Exclude <unspecified> for all subsequent ROI columns
            df = _roisyn_for_roiset(
                point_df.loc[point_df[roiset_name] != "<unspecified>"],
                partner_df.loc[partner_df[roiset_name] != "<unspecified>"],
                roiset_name
            )
        roiset_dfs.append(df)
    roi_syn = pd.concat(roiset_dfs)

    BATCH_SIZE = 10_000
    bodies = pd.Series(roi_syn.index.get_level_values('body'))
    body_count = bodies.diff().fillna(0).astype(bool).cumsum()
    roi_syn['body_batch'] = (body_count // BATCH_SIZE).values
    roi_syn.set_index('body_batch', append=True, inplace=True)
    roi_syn = roi_syn.reorder_levels(['body_batch', 'body', 'roi'])

    logger.info("Writing roi_syn.feather")
    feather.write_feather(roi_syn.reset_index(), 'neuprint/roi_syn.feather')
    return roi_syn


def _roisyn_for_roiset(point_df, partner_df, roiset_name):
    point_df = point_df.drop(columns=['roi'], errors='ignore')
    partner_df = partner_df.drop(columns=['roi'], errors='ignore')

    point_df = point_df.rename(columns={roiset_name: 'roi'})
    partner_df = partner_df.rename(columns={roiset_name: 'roi'})

    roi_prepost = point_df[['body', 'roi', 'kind']].value_counts().unstack(fill_value=0)
    roi_prepost.columns = ['post', 'pre']

    # Note:
    #   Both 'upstream' AND 'downstream' ROI determined by the post-synaptic point location.
    #   In neuprint, ConnectsTo.roiInfo is determined by the 'post' side, so by setting
    #   'downstream' roiInfo the same way, we ensure that 'downstream' from a given neuron
    #   in a particular ROI is equal to the sum of all outgoing :ConnectsTo weights for the
    #   same neuron and ROI.
    #   BTW, since the partner_df 'roi' column was set according to the post side, we already
    #   have the necessary data loaded here.
    roi_downstream = (
        partner_df.rename(columns={'body_pre': 'body'})[['body', 'roi']]
        .value_counts()
        .rename('downstream')
    )

    roi_syn = pd.concat((roi_prepost, roi_downstream), axis=1).fillna(0).astype(np.int32)
    roi_syn['upstream'] = roi_syn['post']
    roi_syn['synweight'] = roi_syn['upstream'] + roi_syn['downstream']

    assert roi_syn.index.names == ['body', 'roi']
    assert roi_syn.columns.tolist() == ['post', 'pre', 'downstream', 'upstream', 'synweight']
    return roi_syn


@PrefixFilter.with_context("Neuron.roiInfo")
def _neuprint_neuron_roi_infos(roisyn_df, processes):
    # Testing shows that there's a steep penalty to pickling
    # a DataFrame if it has a non-trivial index,
    # which would negate the benefits of multiprocessing here.
    # To avoid that penalty, we reset the index in each batch before processing it.
    # In _make_roi_infos(), the appropriate index is set.
    with Timer("Grouping into batches"):
        batches = [df.reset_index() for (_, df) in roisyn_df.groupby('body_batch', sort=False)]

    roi_info_dfs = compute_parallel(
        _make_roi_infos,
        batches,
        processes=processes,
        leave_progress=True
    )
    return pd.concat(roi_info_dfs)


def _make_roi_infos(batch_df):
    batch_df = batch_df.query('body != 0')
    batch_df = batch_df.set_index('roi')
    assert batch_df.columns[:2].tolist() == ['body_batch', 'body']

    bodies = []
    roi_infos = []
    roi_hashes = []
    for body, df in batch_df.groupby('body', sort=False):
        # Discard body_batch and body.
        # Note: df.iloc[] is faster than df.drop()
        df = df.iloc[:, 2:]

        # This would produce the JSON string in one line if we were
        # okay with zeros being present in the result (such as pre: 0).
        # roi_info = df.to_json(orient='index')

        # This is equivalent to df.to_dict(orient='records'),
        # but faster for simple integer data
        single_roi_infos = [
            dict(zip(df.columns, row))
            for row in df.itertuples(index=False, name=None)
        ]

        # This removes 0 entries, which is how neuprint is
        # currently populated (though I'm not sure why).
        single_roi_infos = [{k:v for k,v in d.items() if v} for d in single_roi_infos]
        roi_info = dict(zip(df.index, single_roi_infos))

        # We filter out non-roi counts here (instead of before the loop)
        # because we want to include purely non-roi bodies in
        # this loop and give them an roi_info dict.
        # (In those cases, it will be an empty dict.)
        roi_info.pop("<unspecified>", None)

        roi_hash = hash(tuple(sorted(roi_info.keys())))
        roi_info = json.dumps(roi_info)

        bodies.append(body)
        roi_infos.append(roi_info)
        roi_hashes.append(roi_hash)

    roi_info_df = pd.DataFrame(
        {
            'roiset_hash': roi_hashes,
            'roiInfo': roi_infos
        },
        index=bodies
    ).rename_axis('body')

    return roi_info_df


def _neuprint_neuron_annotations(cfg, ann):
    # Fetch all clio annotations
    # Translate to neuprint terms
    # If config mentions nuclei, use them.
    #
    renames = {c: snakecase_to_camelcase(c.replace(' ', '_'), False) for c in ann.columns}
    renames.update({c: c.replace('Position', 'Location') for c in renames})
    renames.update(CLIO_TO_NEUPRINT_PROPERTIES)
    renames.update(cfg['annotation-property-names'])

    # Drop the ones that map to ""
    renames = {k:v for k,v in renames.items() if (k in ann) and v}
    ann = ann[[*renames.keys()]]
    ann = ann.rename(columns=renames)

    # Neuprint uses 'simplified' status choices,
    # referring to the original (dvid) status as 'statusLabel'.
    ann['status'] = ann['statusLabel'].replace(NEUPRINT_STATUSLABEL_TO_STATUS)

    # Points must be converted to neo4j spatial points.
    for col in ann.columns:
        if 'location' in col.lower() or 'position' in col.lower():
            valid = ann[col].notnull()
            ann.loc[valid, col] = [
                f"{{x:{x}, y:{y}, z:{z}}}"
                for (x,y,z) in ann.loc[valid, [*'xyz']].values
            ]

    return ann


@timed("Assigning :Segment/:Neuron labels")
def _assign_segment_label(cfg, neuron_df):
    dataset = cfg['dataset']
    crit = cfg['neuron-label-criteria']
    crit_props = crit['properties']
    crit_props = set(neuron_df.columns) & set(crit_props)

    is_neuron = neuron_df['synweight'] >= crit['synweight']
    is_neuron |= neuron_df[[*crit_props]].notnull().any(axis=1)
    is_neuron |= neuron_df['status'].isin(crit['status'])

    neuron_df[':LABEL'] = f'Segment;{dataset}_Segment'
    neuron_df.loc[is_neuron, ':LABEL'] = f'Segment;{dataset}_Segment;Neuron;{dataset}_Neuron'


def _export_neuron_csvs(neuron_df, processes):
    neuron_dir = 'neuprint/Neuprint_Neurons'
    if os.path.exists(neuron_dir):
        shutil.rmtree(neuron_dir)
    os.makedirs(neuron_dir)

    feather.write_feather(
        neuron_df,
        'neuprint/Neuprint_Neurons.feather'
    )

    batches = [
        (i, df)
        for i, (_, df) in
        enumerate(neuron_df.groupby('roiset_hash', sort=False))
    ]

    _fn = partial(_export_neurons_with_shared_roiset, neuron_dir, len(batches))
    compute_parallel(
        _fn,
        batches,
        starmap=True,
        processes=processes,
        leave_progress=True
    )


def _export_neurons_with_shared_roiset(neuron_dir, total_batches, batch_index, neuron_df):
    neuron_df.drop(columns=['roiset_hash'], inplace=True)
    assert all(':' in c for c in neuron_df.columns)

    neo4j_to_numpy = {
        'long': np.int64,
        'int': np.int32
    }

    # Some columns might have the wrong dtype due to the way
    # pandas expresses missing values with NaN.
    # To ensure that the non-missing values will be written
    # correctly in the CSV, first convert to the correct dtype
    # (if this batch contains no missing entries) or to 'object'
    # dtype (if necessary) and cast the available values to the
    # correct type.
    for c, pandas_dtype in neuron_df.dtypes.items():
        neo4j_type = c.split(':')[1]
        export_type = neo4j_to_numpy.get(neo4j_type, None)
        if not export_type or export_type == pandas_dtype:
            continue

        if neuron_df[c].notnull().all():
            neuron_df[c] = neuron_df[c].astype(export_type)
        else:
            # Convert column to dtype 'object' (instead of float)
            # so that we can replace floats with ints while
            # allowing for missing values.
            neuron_df[c] = neuron_df[c].astype(object)
            nn = neuron_df[c].notnull()
            neuron_df.loc[nn, c] = neuron_df.loc[nn, c].astype(export_type)

    # FIXME:
    # Technically, its *possible* (but highly unlikely)
    # that our hash has collided for multiple roi sets.
    # The only way to be 100% safe is to check them all.
    # For now, I'm ignoring that problem and just assuming
    # that all neurons in the batch intersect the same set of ROIs.
    rois = list(json.loads(neuron_df['roiInfo:string'].iloc[0]).keys())

    # This awkward way of assigning the new columns is a way to work around
    # this warning from pands: "PerformanceWarning: DataFrame is highly fragmented".
    # https://stackoverflow.com/a/76344743/162094
    flags = pd.DataFrame([[True]*len(rois)], columns=[f'{roi}:boolean' for roi in rois], index=neuron_df.index)
    neuron_df = pd.concat((neuron_df, flags), axis=1)

    digits = len(str(total_batches-1))
    p = f'{neuron_dir}/{batch_index:0{digits}d}.csv'
    neuron_df.to_csv(p, index=False, header=True)


@PrefixFilter.with_context("connections")
def export_neuprint_segment_connections(cfg, partner_df):
    """
    Export the CSV file for Neuron -[:ConnectsTo]-> Neuron
    """
    balanced_confidence = cfg['postHighAccuracyThreshold']
    hp_confidence = cfg['postHPThreshold']

    partner_df = partner_df.query('body_pre != 0 and body_post != 0').copy()
    partner_df['conf_cat'] = 'low'
    partner_df['conf_cat'] = pd.Categorical(partner_df['conf_cat'], ['low', 'med', 'high'])
    partner_df.loc[partner_df['conf_post'] >= balanced_confidence, 'conf_cat'] = 'med'
    partner_df.loc[partner_df['conf_post'] >= hp_confidence, 'conf_cat'] = 'high'
    connectome = (
        partner_df[['body_pre', 'body_post', 'conf_cat']]
        .value_counts()
        .unstack(fill_value=0)
    )
    connectome['low'] = connectome.get('low', 0)
    connectome['med'] = connectome.get('med', 0)
    connectome['high'] = connectome.get('high', 0)

    connectome['weightHR'] = connectome[['low', 'med', 'high']].sum(axis=1)
    connectome['weight'] = connectome[['med', 'high']].sum(axis=1)
    connectome['weightHP'] = connectome['high']
    connectome = connectome.drop(columns=['low', 'med', 'high'])

    # Note that partner_df['roi'] was created using roi_post.
    # Neuprint defines the location of synapse connection
    # weights according to the 'post' side.

    roiset_names = list(cfg['roi-sets'].keys())
    roiset_conns = []
    for i, roiset_name in enumerate(roiset_names):
        df = partner_df
        if i > 0:
            # We only include '<unspecified>' once.
            # The actual unspecified count doesn't matter since
            #  we don't store it. But we need to make sure every
            # connection pair is listed to ensure that they all
            # get a roiInfo (even if it's an empty roiInfo, i.e. {}).
            df = df.loc[df[roiset_name] != "<unspecified>"]
        roiset_conn = (
            df.loc[
                df['conf_post'] >= balanced_confidence,
                ['body_pre', 'body_post', roiset_name]
            ]
            .rename(columns={roiset_name: 'roi'})
            .value_counts()
            .rename('weight')
        )
        roiset_conns.append(roiset_conn)

    roi_conn = pd.concat(roiset_conns).sort_index().reset_index()

    BATCH_SIZE = 10_000
    pair_count = (
        roi_conn[['body_pre', 'body_post']]
        .diff()
        .any(axis=1)
        .cumsum()
    )
    roi_conn['pair_batch'] = (pair_count // BATCH_SIZE).values
    roi_info = _neuron_connection_roi_infos(roi_conn, cfg['processes'])
    connectome = connectome.merge(roi_info, 'left', on=['body_pre', 'body_post'])
    connectome = connectome.reset_index()
    cols = ['body_pre', 'body_post', 'weightHR', 'weight', 'weightHP', 'roiInfo']
    assert connectome.columns.tolist() == cols, f"{connectome.columns}"

    neo4j_connectome = (
        connectome
        .rename(columns={
            'body_pre': ':START_ID(Body-ID)',
            'body_post': ':END_ID(Body-ID)',
        })
    )
    neo4j_connectome = append_neo4j_type_suffixes(cfg, neo4j_connectome)

    with Timer("Writing Neuprint_Neuron_Connections.feather", logger):
        feather.write_feather(
            neo4j_connectome,
            'neuprint/Neuprint_Neuron_Connections.feather'
        )

    with Timer("Writing Neuprint_Neuron_Connections.csv", logger):
        neo4j_connectome.to_csv("neuprint/Neuprint_Neuron_Connections.csv", index=False, header=True)

    return connectome


@PrefixFilter.with_context("Neuron:ConnectsTo.roiInfo")
def _neuron_connection_roi_infos(roi_conn, processes):
    with Timer("Grouping into batches"):
        # In the connection roiInfo, the weight is named 'post'.
        # In earlier neuprint databases, we also emitted 'pre',
        # which was usually -- but not always -- identical to 'post'.
        # But emitting 'pre' would be misleading for multiple reasons:
        #  - 'post' is what we used to calculate the neuron-to-neuron 'weight'.
        #  - In our datasets, we usually eliminate "redundant" PSDs as a
        #    post-processing step, so 'pre' will always match 'post' UNLESS
        #    the connection happens to fall on a boundary between ROIs,
        #    but in that case the precise ROI for the synapse is arbitrary
        #    anyway.
        # So nowadays we don't bother emitting 'pre' at all.
        roi_conn = roi_conn.rename(columns={'weight': 'post'})
        roi_conn = roi_conn[['pair_batch', 'body_pre', 'body_post', 'roi', 'post']]
        batches = [df for (_, df) in roi_conn.groupby('pair_batch', sort=False)]

    roi_info_dfs = compute_parallel(
        _make_connection_roi_infos,
        batches,
        processes=processes,
        leave_progress=True
    )
    return pd.concat(roi_info_dfs, ignore_index=True).set_index(['body_pre', 'body_post'])['roiInfo']


def _make_connection_roi_infos(batch_df):
    """
    Emits a DataFrame with columns ['body_pre', 'body_post', 'roiInfo'],
    in which roiInfo is a string (JSON).
    """
    batch_df = batch_df.set_index('roi')
    assert batch_df.columns.tolist() == ['pair_batch', 'body_pre', 'body_post', 'post']

    body_pairs = []
    roi_infos = []
    for body_pair, df in batch_df.groupby(['body_pre', 'body_post'], sort=False):
        # Discard columns except 'post'
        df = df.iloc[:, 3:]

        # Filter out non-roi weights.
        # If there are no other rois, we'll emit an empty dict: '{}'
        df = df.query('roi != "<unspecified>"')
        roi_info = df.to_json(orient='index')

        body_pairs.append(body_pair)
        roi_infos.append(roi_info)

    roi_info_df = pd.DataFrame(body_pairs, columns=['body_pre', 'body_post'])
    roi_info_df['roiInfo'] = roi_infos
    return roi_info_df
