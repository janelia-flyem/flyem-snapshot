import os
import ujson
import shutil
import logging
from functools import partial

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import timed, Timer, compute_parallel, tqdm_proxy, snakecase_to_camelcase

from ...util import checksum
from .util import append_neo4j_type_suffixes

logger = logging.getLogger(__name__)


@PrefixFilter.with_context("Segment")
def export_neuprint_segments(cfg, point_df, partner_df, ann, body_sizes, body_nt, inbounds_bodies, inbounds_rois):
    """
    """
    assert ann.index.name == 'body'
    assert 0 not in ann.index

    # Filter out low-confidence PSDs before computing weights.
    balanced_confidence = cfg['meta']['postHighAccuracyThreshold']
    partner_df = partner_df.query('conf_post >= @balanced_confidence')
    _ = balanced_confidence  # linting fix

    body_stats = _body_synstats(point_df, partner_df, inbounds_bodies)
    roi_syn_df = _body_roi_synstats(cfg, point_df, partner_df, inbounds_bodies)
    roi_info_df = _neuprint_neuron_roi_infos(roi_syn_df, cfg['processes'])

    # We use a left-merge here (not outer) because we deliberately exclude
    # bodies which have no synapses whatsoever, even if there is annotation
    # data for the body.
    # One reason not to use such bodies is that the annotations can (sadly)
    # contain stale body IDs. Also, 'Unimportant' bodies which are just
    # fixative or whatever should generally be excluded from results.
    neuron_df = body_stats.merge(ann, 'left', on='body')

    if body_nt is not None:
        assert body_nt.index.name == 'body'
        body_nt = body_nt.rename(columns={
            c: snakecase_to_camelcase(c) for c in body_nt.columns
        })
        # Sometimes the neurotransmitters are present in DVID/clio,
        # but we want to supercede those with up-to-date NT calculations.
        neuron_df = neuron_df.drop(columns=body_nt.columns, errors='ignore')
        neuron_df = neuron_df.merge(body_nt, 'left', on='body')

    neuron_df = neuron_df.merge(roi_info_df, 'left', on='body')
    if body_sizes is not None:
        neuron_df = neuron_df.merge(body_sizes, 'left', on='body')
    _assign_segment_label(cfg, neuron_df)

    # We include bodyId as a property column AND as the node ID column for neo4j ingestion.
    assert neuron_df.index.name == 'body'
    neuron_df['bodyId'] = neuron_df.index
    neuron_df = neuron_df.rename_axis(':ID(Body-ID)').reset_index()
    neuron_df = append_neo4j_type_suffixes(neuron_df, exclude=['roiset_hash'])

    _export_neuron_csvs(neuron_df, cfg['max-segment-files'], cfg['processes'])

    # While we've got this data handy, compute total pre/post
    # in each ROI and also the whole dataset.
    roi_totals = (
        roi_syn_df
        .groupby(level='roi')[['pre', 'post']].sum()
        .query('roi != "<unspecified>"')
    )

    if inbounds_rois is None:
        dataset_totals = body_stats[['pre', 'post']].sum()
    else:
        dataset_totals = roi_syn_df.query('roi in @inbounds_rois')[['pre', 'post']].sum()

    return neuron_df, dataset_totals, roi_totals


@timed
def _body_synstats(point_df, partner_df, inbounds_bodies):
    prepost = point_df[['body', 'kind']].value_counts().unstack(fill_value=0)
    prepost.columns = ['post', 'pre']

    upstream = prepost['post'].rename('upstream')
    downstream = partner_df['body_pre'].value_counts().rename('downstream').rename_axis('body')
    upstream, downstream = upstream.align(downstream, fill_value=0)
    upstream, downstream = upstream.astype(np.int32), downstream.astype(np.int32)
    synweight = (upstream + downstream).rename('synweight')
    body_syn = pd.concat((prepost, downstream, upstream, synweight), axis=1)
    body_syn = body_syn.fillna(0).astype(np.int32)
    body_syn = body_syn.query('body != 0')

    # Above, we had to keep all the partners to ensure accurate counts for
    # in-bounds bodies which are partnered to out-of-bounds bodies.
    # But now that the stats are computed, we can drop the out-of-bounds bodies.
    if inbounds_bodies is not None:
        body_syn = body_syn.loc[body_syn.index.isin(inbounds_bodies)]

    logger.info("Writing body_syn.feather")
    feather.write_feather(body_syn.reset_index(), 'neuprint/body_syn.feather')

    assert body_syn.index.name == 'body'
    assert body_syn.columns.tolist() == ['post', 'pre', 'downstream', 'upstream', 'synweight']
    return body_syn


@timed
def _body_roi_synstats(cfg, point_df, partner_df, inbounds_bodies):
    """
    Per-body-per-ROI stats
    """
    roiset_names = set(cfg['roi-set-names']) & set(point_df.columns)
    roiset_dfs = []
    for i, roiset_name in enumerate(tqdm_proxy(sorted(roiset_names))):
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

    # It's critical that each body occupies contiguous
    # rows before we assign bodies into batches.
    roi_syn.sort_index(inplace=True)

    # Above, we had to keep all the partners to ensure accurate counts for
    # in-bounds bodies which are partnered to out-of-bounds bodies.
    # But now that the stats are computed, we can drop the out-of-bounds bodies.
    if inbounds_bodies is not None:
        bodies = roi_syn.index.get_level_values('body')
        roi_syn = roi_syn.loc[bodies.isin(inbounds_bodies)]

    BATCH_SIZE = 10_000
    bodies = roi_syn.index.get_level_values('body')
    body_count = pd.Series(bodies).diff().fillna(0).astype(bool).cumsum()
    roi_syn['body_batch'] = (body_count // BATCH_SIZE).values
    roi_syn.set_index('body_batch', append=True, inplace=True)
    roi_syn = roi_syn.reorder_levels(['body_batch', 'body', 'roi'])

    logger.info("Writing roi_syn.feather")
    feather.write_feather(roi_syn.reset_index(), 'neuprint/roi_syn.feather')

    assert roi_syn.index.names == ['body_batch', 'body', 'roi']
    assert roi_syn.columns.tolist() == ['post', 'pre', 'downstream', 'upstream', 'synweight']
    return roi_syn


def _roisyn_for_roiset(point_df, partner_df, roiset_name):
    if roiset_name != 'roi':
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


@PrefixFilter.with_context("Segment.roiInfo")
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
    """
    Given a table of ROI statistics for each body and each ROI,
    Aggregate the statistics for each body into a neuprint-style roiInfo,
    JSON-serialize it, and also compute a hash from the (sorted) list of
    ROIs the body touches. (The hash can be used to group each body with
    similar bodies for the purposes of CSV exports.)

    Arg:
        batch_df:
            A DataFrame with columns:
            ['body_batch', 'body', 'roi', 'post', 'pre', 'downstream', 'upstream', 'synweight']
            (The index is ignored/discarded.)

    Returns:
        DataFrame, indexed by body, with columns:
            - roiset_hash (an int)
            - roiInfo (already JSON-serialized into a string)

        Example:
                    roiset_hash roiInfo
            body
            123   9384832742398 '{"ME(R)": {"post": 10, "pre": 2, ...}, "LO(R)": {"post": 16, "pre": 0, ...}}'
            456   -387462398472 '{"ME(R)": {"post": 45, "pre": 9, ...}, "LO(R)": {"post": 213, "pre": 12, ...}}'
        ...
    """
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

        # The following commented-out line would produce the JSON string in one line
        # if we were okay with zeros being present in the result (such as pre: 0).
        # But we don't want zeros, and besides, df.to_json() is super slow.
        # Thus, we'll construct the JSON ourselves.
        # roi_info = df.to_json(orient='index')

        # This is equivalent to df.to_dict(orient='records'),
        # but faster for simple integer data.
        # Result is like this:
        #   [
        #       {'post': 10, 'pre': 2, 'downstream': 9, 'upstream': 10, 'synweight': 19},
        #       {'post': 18, 'pre': 1, 'downstream': 3, 'upstream': 18, 'synweight': 22},
        #       {'post': 10, 'pre': 2, 'downstream': 9, 'upstream': 10, 'synweight': 19},
        #       ...
        #   ]
        single_roi_infos = [
            dict(zip(df.columns, row))
            for row in df.itertuples(index=False, name=None)
        ]

        # This removes 0 entries, which is how neuprint is
        # currently populated (though I'm not sure why).
        single_roi_infos = [{k:v for k,v in d.items() if v} for d in single_roi_infos]

        # Now aggregate into the full roiInfo for the current body, like this:
        # {
        #   'ME(R)': {'post': 10, 'pre': 2, 'downstream': 9, 'upstream': 10, 'synweight': 19},
        #   'LO(R)': {'post': 18, 'pre': 1, 'downstream': 3, 'upstream': 18, 'synweight': 22},
        #   'LOP(R)': {'post': 10, 'pre': 2, 'downstream': 9, 'upstream': 10, 'synweight': 19},
        #   ...
        # }
        roi_info = dict(zip(df.index, single_roi_infos))

        # We filter out non-roi counts here (instead of before the loop)
        # because we want to include purely non-roi bodies in
        # this loop and give them an roi_info dict.
        # (In those cases, it will be an empty dict.)
        roi_info.pop("<unspecified>", None)

        roi_hash = checksum(str(sorted(roi_info.keys())))
        roi_info = ujson.dumps(roi_info)

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


@timed("Assigning :Segment/:Neuron labels")
def _assign_segment_label(cfg, neuron_df):
    """
    Determine which segments qualify to get the :Neuron
    label according to the user's config (neuron-label-criteria),
    and construct the :LABEL column of the neuron table
    accordingly.

    TODO:
        In hemibrain v1.2, there is also a :Cell label
        on each neuron (and segment?), along with duplicate
        indexes for that label.
    """
    dataset = cfg['meta']['dataset']
    crit = cfg['neuron-label-criteria']
    crit_props = crit['properties']
    crit_props = set(neuron_df.columns) & set(crit_props)

    is_neuron = neuron_df['synweight'] >= crit['synweight']
    is_neuron |= neuron_df[[*crit_props]].notnull().any(axis=1)
    is_neuron |= neuron_df['status'].isin(crit['status'])

    neuron_df[':LABEL'] = f'Segment;{dataset}_Segment'
    neuron_df.loc[is_neuron, ':LABEL'] = f'Segment;{dataset}_Segment;Neuron;{dataset}_Neuron'


@timed
def _export_neuron_csvs(neuron_df, max_files, processes):
    neuron_dir = 'neuprint/Neuprint_Neurons'
    if os.path.exists(neuron_dir):
        shutil.rmtree(neuron_dir)
    os.makedirs(neuron_dir)

    feather.write_feather(
        neuron_df,
        'neuprint/Neuprint_Neurons.feather'
    )

    # It's critical that each roiset occupies contiguous
    # rows before we assign roisets into batches.
    # Also, sort by size of largest roiset to smallest.
    neuron_df['roiset_size'] = neuron_df.groupby('roiset_hash').transform('size')
    neuron_df.sort_values(['roiset_size', 'roiset_hash'], ascending=False, inplace=True, ignore_index=True)

    batch_size = (len(neuron_df) + max_files - 1) // max_files

    # Determine how many 'singleton' batches there are which are big enough to get their own batch.
    # If we don't, they would the total batch count below the user's desired count.
    roiset_sizes = neuron_df[['roiset_size', 'roiset_hash']].drop_duplicates()
    singleton_batch_df = roiset_sizes.query('roiset_size >= @batch_size')
    singleton_batches = len(singleton_batch_df)
    singleton_batch_total_size = singleton_batch_df['roiset_size'].sum()

    # Calculate batch_size based on remainder after subtracting out the singleton batches.
    # Of course, by changing the batch size, we'll be changing the number of 'singleton' batches,
    # so we could consider iterating here until the batch size converges.
    # But we won't bother; we'll just adjust the batch size once.
    batch_size = (len(neuron_df) - singleton_batch_total_size + max_files - 1) // max(1, (max_files - singleton_batches))
    neuron_df['batch_index'] = neuron_df.index // batch_size
    neuron_df['batch_index'] = neuron_df.groupby('roiset_hash')['batch_index'].transform('first')

    # Renumber with consecutive batch indexes.
    neuron_df['batch_index'] = neuron_df.groupby('batch_index').ngroup()

    del neuron_df['roiset_size']
    batches = neuron_df.groupby('batch_index', sort=False)

    _fn = partial(
        _export_neuron_batch,
        neuron_dir,
        neuron_df['batch_index'].nunique()
    )
    compute_parallel(
        _fn,
        batches,
        starmap=True,
        processes=processes,
        leave_progress=True
    )


def _export_neuron_batch(neuron_dir, total_batches, batch_index, batch_df):
    subbatch_dfs = []
    for _, subbatch_df in batch_df.groupby('roiset_hash'):
        df = _construct_export_df_for_common_roiset(subbatch_df)
        subbatch_dfs.append(df)

    export_df = pd.concat(subbatch_dfs, ignore_index=True)
    digits = len(str(total_batches-1))
    p = f'{neuron_dir}/{batch_index:0{digits}d}.csv'
    export_df.to_csv(p, index=False, header=True)


def _construct_export_df_for_common_roiset(neuron_df):
    """
    For a batch of segments/neurons which all have the SAME roiset_hash,
    construct format their data for CSV export and neo4j ingestion.
    """
    neuron_df.drop(columns=['roiset_hash', 'batch_index'], inplace=True)
    assert all(':' in c for c in neuron_df.columns), \
        f"Columns aren't all ready for neo4j export: {neuron_df.columns.tolist()}"

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
    rois = list(ujson.loads(neuron_df['roiInfo:string'].iloc[0]).keys())

    # Two notes:
    #
    # 1. This awkward way of assigning the new columns (instead of assigning columns one at a time)
    #    avoids this warning from pandas: "PerformanceWarning: DataFrame is highly fragmented".
    #    https://stackoverflow.com/a/76344743/162094
    #
    # 2. Furthermore, we assign them as the string 'true' (instead of boolean True),
    #    because neo4j requires booleans to exactly match 'true' (lowercase),
    #    and pandas would normally write bools as 'True'.
    #    https://neo4j.com/docs/operations-manual/4.4/tools/neo4j-admin/neo4j-admin-import/#import-tool-header-format-properties
    flags = pd.DataFrame(
        [['true']*len(rois)],
        columns=[f'{roi}:boolean' for roi in rois],
        index=neuron_df.index
    )
    neuron_df = pd.concat((neuron_df, flags), axis=1)
    return neuron_df


@PrefixFilter.with_context("ConnectsTo")
def export_neuprint_segment_connections(cfg, partner_df):
    """
    Export the CSV file for Neuron -[:ConnectsTo]-> Neuron
    """
    balanced_confidence = cfg['meta']['postHighAccuracyThreshold']
    hp_confidence = cfg['meta']['postHPThreshold']

    partner_df = partner_df.query('body_pre != 0 and body_post != 0').copy()
    partner_df['conf_cat'] = 'low'
    partner_df['conf_cat'] = pd.Categorical(partner_df['conf_cat'], ['low', 'med', 'high'])
    partner_df.loc[partner_df['conf_post'] >= balanced_confidence, 'conf_cat'] = 'med'
    partner_df.loc[partner_df['conf_post'] >= hp_confidence, 'conf_cat'] = 'high'

    # Connection weights by category (low/med/high)
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

    # Note that partner_df[roiset_name] was created using roi_post.
    # Neuprint defines the location of synapse connection
    # weights according to the 'post' side.

    roiset_names = set(cfg['roi-set-names']) & set(partner_df.columns)
    roiset_conns = []
    for i, roiset_name in enumerate(sorted(roiset_names)):
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

    # In cases where a connection has has a 'weight' of 0
    # but has non-zero 'weightHR', the roiInfo hasn't yet,
    # been created. Fill it with an empty JSON object.
    connectome['roiInfo'].fillna('{}', inplace=True)

    cols = ['body_pre', 'body_post', 'weightHR', 'weight', 'weightHP', 'roiInfo']
    assert connectome.columns.tolist() == cols, f"{connectome.columns}"

    neo4j_connectome = (
        connectome
        .rename(columns={
            'body_pre': ':START_ID(Body-ID)',
            'body_post': ':END_ID(Body-ID)',
        })
    )
    neo4j_connectome = append_neo4j_type_suffixes(neo4j_connectome)

    with Timer("Writing Neuprint_Neuron_Connections.feather", logger):
        feather.write_feather(
            neo4j_connectome,
            'neuprint/Neuprint_Neuron_Connections.feather'
        )

    with Timer("Writing Neuprint_Neuron_Connections.csv", logger):
        neo4j_connectome.to_csv("neuprint/Neuprint_Neuron_Connections.csv", index=False, header=True)

    return connectome


@PrefixFilter.with_context("roiInfo")
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

        # We could just use the following line, but to_json()
        # is super slow, so we construct the JSON the hard way.
        # roi_info = df.to_json(orient='index')

        # Result is like this:
        # {
        #    'ME(R)': {'post': 10},
        #    'LO(R)': {'post': 18},
        #    'LOP(R)': {'post': 10}
        # }
        roi_info = {roi: {'post': post} for roi, post in df['post'].items()}
        roi_info = ujson.dumps(roi_info)

        body_pairs.append(body_pair)
        roi_infos.append(roi_info)

    roi_info_df = pd.DataFrame(body_pairs, columns=['body_pre', 'body_post'])
    roi_info_df['roiInfo'] = roi_infos
    return roi_info_df
