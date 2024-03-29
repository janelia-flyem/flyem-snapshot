"""
Export a connectivity snapshot from a DVID segmentation,
along with other denormalizations.
"""
import os
import json
import logging
from functools import cache

import requests
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import holoviews as hv
import hvplot.pandas

from neuclease import PrefixFilter
from neuclease.util import Timer
from neuclease.dvid.keyvalue import DEFAULT_BODY_STATUS_CATEGORIES
from neuclease.misc.neuroglancer import format_nglink
from neuclease.misc.completeness import (
    completeness_forecast,
    plot_connectivity_forecast,
    plot_categorized_connectivity_forecast,
    variable_width_hbar,
)

from ..util import export_bokeh
from ..caches import cached, SentinelSerializer

_ = hvplot.pandas  # linting

logger = logging.getLogger(__name__)

CAPTURE_STATS = ['traced_presyn_frac', 'traced_postsyn_frac', 'traced_synweight_frac', 'traced_conn_frac']
STATUS_DTYPE = pd.CategoricalDtype(DEFAULT_BODY_STATUS_CATEGORIES, ordered=True)

#
# TODO
# - emit a stacked bar graph of completeness fraction per ROI (stacked by status)
# - create a simple html overview for report PNGs.
# - consider a multiprocessing approach
#

ReportSchema = {
    "description": "Settings for a single connectivity report with one or more ROIs.",
    "type": "object",
    "default": {},
    "additionalProperties": False,
    # "required": []
    "properties": {
        "rois": {
            "description":
                "List of ROI names to aggregate when computing connectivity metrics.\n"
                "Other ROIs are excluded. If you list nothing here, then no ROI\n"
                "filtering is performed at all (all synapses are used).",
            "type": "array",
            "items": { "type": "string" },
            "default": []
        },
        "name": {
            "description":
                "A name for the ROI grouping (e.g. 'right optic lobe')\n"
                "If not provided, then a name will be chosen automatically from the selected ROI(s)",
            "type": "string",
            "default": ""
        }
    }
}

ReportSetSchema = {
    "description": "Specs for connectivity reports to generate from the snapshot data.\n",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "roiset": {
            "description":
                "Reports can be ROI-based, but only from the ROIs in just\n"
                "one of your ROI sets (a column in point_df).\n"
                "Specify which roiset the report ROIs will be selected from.\n",
            "type": "string",
            "default": ""
        },
        "reports": {
            "description":
                "A list of forecasts to produce.\n"
                "Each forecast is computed on a subset of the ROIs in the roiset.\n",
            "type": "array",
            "items": ReportSchema,
            "default": [],
        },
        "stop-at-rank": {
            "description": "When calculating completeness forecast curves, save time by only computing the result for N ranked bodies.",
            "type": "number",
            "default": 1e6,
        },
        "neuroglancer-base-state": {
            "description":
                "A json file containing neuroglancer state.\n"
                "You can provide a local path or a remote https:// or gs:// path.\n"
                "The neuron segmentation MUST be the currently selected layer in the state.\n",
            "default": ""
        },
        "capture-statuses": {
            "description":
                "A list of statuses which count as being in the 'capture set'\n"
                "(essentially the traced-ish set) for downstream capture analysis\n"
                "and captured connectivity summaries.\n",
            "type": "array",
            "items": {"type": "string"},
            # By default, we count all statuses which are Primary Anchor or better
            # as being "good enough" to count towards our capture numbers.
            # In a published connectome, only 'Roughly traced' (or similar) should count,
            # but during the main thrust of a reconstruction, it's usually more informative
            # to look at all bodies which will eventually become Roughly traced,
            # i.e. all Primary Anchor bodies.
            "default": DEFAULT_BODY_STATUS_CATEGORIES[DEFAULT_BODY_STATUS_CATEGORIES.index('Primary Anchor'):]
        }
    }
}

ReportsSchema = {
    "description": "Specs for sets of connectivity reports (each set uses a different roiset layer)\n",
    "default": [],
    "type": "array",
    "items": ReportSetSchema
}


@PrefixFilter.with_context('reports')
@cached(SentinelSerializer('reports'))
def export_reports(cfg, point_df, partner_df, ann, snapshot_tag):
    if len(cfg) == 0:
        logger.info("No reports requested.")
        return

    os.makedirs("png", exist_ok=True)
    os.makedirs("html", exist_ok=True)

    logger.info(f"point_df.columns: {point_df.columns.tolist()}")
    logger.info(f"partner_df.columns: {partner_df.columns.tolist()}")

    for reportset_cfg in cfg:
        roiset = reportset_cfg['roiset']
        _export_reportset(reportset_cfg, point_df, partner_df, ann, snapshot_tag, roiset=roiset)


@PrefixFilter.with_context('{roiset}')
def _export_reportset(cfg, point_df, partner_df, ann, snapshot_tag, *, roiset):
    # Make sure our roiset column is named 'roi' since that's what completeness_forecast() expects.
    assert roiset in point_df.columns, f"roiset not found in point_df: {roiset}.  Columns are: {point_df.columns.tolist()}"
    point_df = point_df.drop(columns=['roi'], errors='ignore').rename(columns={roiset: 'roi'})
    partner_df = partner_df.drop(columns=['roi'], errors='ignore').rename(columns={roiset: 'roi'})

    if 'roi' not in partner_df.columns:
        partner_df = partner_df.merge(point_df['roi'].rename_axis('post_id'), 'left', on='post_id')

    with Timer("Flagging captured bodies", logger):
        capture_statuses = np.array(cfg['capture-statuses'])
        _ = capture_statuses  # linting nonsense
        capture_bodies = ann.query('status in @capture_statuses').index
        partner_df['captured_pre'] = partner_df['body_pre'].isin(capture_bodies)
        partner_df['captured_post'] = partner_df['body_post'].isin(capture_bodies)

        # Converting to categorical ensures that value_counts() always gives
        # results for both False and True, even if there are no True (or False) entries.
        # (An unlikely scenario, but it guards against an ROI having NO traced neurons at
        # all, which could happen early in a reconstruction.)
        partner_df['captured_pre'] = partner_df['captured_pre'].astype(pd.CategoricalDtype([False, True]))
        partner_df['captured_post'] = partner_df['captured_post'].astype(pd.CategoricalDtype([False, True]))

    try:
        assert 'roi' in point_df, f"point_df columns are: {point_df.columns}"
        assert 'roi' in partner_df, f"partner_df columns are: {partner_df.columns}"

        with Timer("Grouping by ROI", logger):
            roi_point_dfs = {roi: df for roi, df in point_df.groupby('roi')}
            roi_partner_dfs = {roi: df for roi, df in partner_df.groupby('roi')}
    except Exception:
        # I'm trying to debug some weird problem...
        feather.write_feather(point_df, "tables/point_df-DEBUG.feather")
        feather.write_feather(partner_df, "tables/partner_df-DEBUG.feather")
        raise

    all_status_stats = {}
    all_syncounts = {}
    for report in cfg['reports']:
        name = report['name']
        if report['rois']:
            report_point_df = pd.concat((roi_point_dfs[roi] for roi in report['rois']))
            report_partner_df = pd.concat(
                (roi_partner_dfs[roi] for roi in report['rois']),
                ignore_index=True
            )
        else:
            report_point_df = point_df
            report_partner_df = partner_df

        syncounts, status_stats = _export_report(
            cfg,
            snapshot_tag,
            report_point_df,
            report_partner_df,
            ann,
            report['rois'],
            name=name
        )
        all_syncounts[name] = syncounts
        all_status_stats[name] = status_stats

    import pickle
    with open(f'tables/{roiset}-all_syncounts.pkl', 'wb') as f:
        pickle.dump(all_syncounts, f)

    with open(f'tables/{roiset}-all_status_stats.pkl', 'wb') as f:
        pickle.dump(all_status_stats, f)

    _export_capture_summaries(cfg, all_syncounts, all_status_stats)


@PrefixFilter.with_context('{name}')
def _export_report(cfg, snapshot_tag, report_point_df, report_partner_df, ann, roi, *, name):
    syncounts = (
        report_point_df['kind']
        .value_counts()
        .rename('count')
        .rename_axis('kind')
        .reset_index()
        .assign(name=name)
    )

    status_stats = _completeness_forecast(
        cfg,
        snapshot_tag,
        report_point_df,
        report_partner_df,
        ann,
        roi,
        name=name,
    )

    _export_downstream_capture(
        cfg,
        snapshot_tag,
        report_partner_df,
        name=name
    )

    return syncounts, status_stats


def _export_capture_summaries(cfg, all_syncounts, all_status_stats):
    if next(iter(all_status_stats.values())) is None:
        # We can't export any capture summaries if no body statuses were given.
        return

    roiset = cfg['roiset']
    names = [report['name'] for report in cfg['reports']]
    names_df = pd.DataFrame({'report_index': np.arange(len(names)), 'name': names})

    all_syncounts = (
        pd.concat(all_syncounts.values(), ignore_index=True)
        .set_index(['name', 'kind'])['count']
        .astype(np.float32)
        .unstack()
        .fillna(0.0)
    )
    # We don't plot anything with empty status or worse.
    all_status_stats = {name: s.query('status > ""') for name, s in all_status_stats.items()}

    # All present statuses, in priority-sorted order.
    relevant_statuses = (
        pd.concat(all_status_stats.values())
        ['status']
        .astype(STATUS_DTYPE)
        .sort_values(ascending=False)
        .drop_duplicates()
        .tolist()
    )

    # Must make sure all relevant statuses are present (with default values)
    # in all dataframes before concatenating
    all_status_stats = {
        name: s.set_index('status')
               .reindex(relevant_statuses)
               .assign(name=name)
               .reset_index()
               .ffill()
               .fillna(0.0)
        for name, s in all_status_stats.items()
    }
    all_status_stats = pd.concat(all_status_stats.values(), ignore_index=True)
    all_status_stats['status'] = all_status_stats['status'].astype(STATUS_DTYPE)
    all_status_stats.to_csv(f'tables/{roiset}-all-status-stats.csv', index=False, header=True)

    # This unstack() will result in multi-level columns:
    # level 0: traced_presyn_frac                            traced_postsyn_frac                        ...
    # level 1: Roughly traced 	Prelim Roughly traced   ...  Roughly traced 	Prelim Roughly traced   ...
    all_status_stats = (
        all_status_stats
        .set_index(['name', 'status'])
        .unstack()
        # Make sure cumulative capture stats list 'good' statuses first.
        .sort_index(axis=1, level='status', ascending=False)
    )
    # This might be overly pedantic, but we want to scale the ROI
    # bars according to the relevant synapse count (pre or post),
    # they aren't always perfectly proportional. We specify PreSyn
    # or PostSyn depending on the statistic being summarized in the plot.
    size_cols = {
        'traced_presyn_frac': 'PreSyn',
        'traced_postsyn_frac': 'PostSyn',
        'traced_conn_frac': 'PostSyn',

        # technically, SynWeight = inputs + outputs = 2 * PostSyn,
        # but that doesn't change the relative proportions among ROIs.
        'traced_synweight_frac': 'PostSyn',
    }
    titles = {
        'traced_presyn_frac': f'{roiset} Captured PreSyn by Status and ROI',
        'traced_postsyn_frac': f'{roiset} Captured PostSyn by Status and ROI',
        'traced_conn_frac': f'{roiset} Captured Connectivity by Status and ROI',
        'traced_synweight_frac': f'{roiset} Captured SynWeight by Status and ROI',
    }

    for level0 in CAPTURE_STATS:
        df = (
            # This will select all columns (status names) under the specified level.
            all_status_stats[level0]
            .reset_index()
            .merge(all_syncounts, 'left', on='name')
            .merge(names_df, 'left', on='name')
            .sort_values('report_index')
        )
        df.to_csv(f'tables/{roiset}-cumulative-{level0}-by-status.csv', index=False, header=True)

        # The dataframe has cumulative connectivity,
        # but for the stacked bar chart we don't want cumulative.
        df[relevant_statuses] -= df[relevant_statuses].shift(1, axis=1, fill_value=0)
        df.to_csv(f'tables/{roiset}-{level0}-by-status.csv', index=False, header=True)

        p = variable_width_hbar(
            df,
            'name',
            size_cols[level0],
            relevant_statuses,
            legend='bottom_right',
            width=800,
            height=1200,
            flip_yaxis=True,
            vlim=(0, 1),
            title=titles[level0]
        )

        fname = '-'.join(titles[level0].lower().split())
        export_bokeh(
            p,
            f"{fname}.html",
            titles[level0]
        )

        # Export again, but sorting the bars by total
        p = variable_width_hbar(
            df.assign(total=df[relevant_statuses].sum(axis=1)).sort_values('total'),
            'name',
            size_cols[level0],
            relevant_statuses,
            legend='bottom_right',
            width=800,
            height=1200,
            flip_yaxis=True,
            vlim=(0, 1),
            title=titles[level0]
        )
        export_bokeh(
            p,
            f"{fname}-sorted.html",
            titles[level0]
        )


@PrefixFilter.with_context("capture forecast")
def _completeness_forecast(cfg, snapshot_tag, point_df, partner_df, ann, roi, *, name):
    stop_at_rank = int(cfg['stop-at-rank'])
    selection_link = _get_neuroglancer_base_link(cfg['neuroglancer-base-state'])
    _name = '-'.join(name.split())

    sort_by = ['SynWeight', 'PreSyn', 'PostSyn']
    if 'status' not in ann.columns:
        ann = None
    else:
        # Pass status along and make sure it's categorical if possible.
        ann = ann[['status']]
        if ann['status'].dtype != 'category' and set(ann['status'].unique()) <= {np.nan, *DEFAULT_BODY_STATUS_CATEGORIES}:
            ann['status'] = pd.Categorical(ann['status'], DEFAULT_BODY_STATUS_CATEGORIES, ordered=True)

        if ann['status'].dtype == 'category':
            sort_by.insert(0, 'status')

    # This takes 10-15 minutes for the whole brain, even with precomputed body_pre, body_post.
    with Timer("Generating sorted connection completeness table and synapse counts", logger):
        conn_df, syn_counts_df = completeness_forecast(
            point_df, partner_df, None, ann, roi=roi,
            sort_by=sort_by,
            stop_at_rank=stop_at_rank
        )

        feather.write_feather(
            conn_df,
            f'tables/{_name}-conn_df-{snapshot_tag}.feather'
        )

        feather.write_feather(
            syn_counts_df.reset_index(),
            f'tables/{_name}-syn_counts_df-{snapshot_tag}.feather'
        )

    with Timer("Generating completeness curve plot"):
        title = f'{name}: Cumulative connectivity by ranked body ({snapshot_tag})'
        if 'status' in ann.columns:
            p = plot_categorized_connectivity_forecast(
                conn_df, 'status', max_rank=None, plotted_points=20_000, hover_cols=['presyn', 'postsyn', 'synweight'],
                title=title, selection_link=selection_link,
                secondary_range=[0, 250]
            )
            export_bokeh(
                p,
                f'{_name}-cumulative-connectivity-with-links-{snapshot_tag}.html',
                title
            )
            status_stats = conn_df.drop_duplicates('status', keep='last')[['status', *CAPTURE_STATS]]
            status_stats['name'] = name
            return status_stats
        else:
            p = plot_connectivity_forecast(
                conn_df, max_rank=None, plotted_points=20_000, hover_cols=['presyn', 'postsyn', 'synweight'],
                title=title
            )
            export_bokeh(
                p,
                f'{_name}-cumulative-connectivity-{snapshot_tag}.html',
                title
            )
            return None


@cache
def _get_neuroglancer_base_link(state_path):
    if not state_path:
        return None

    if state_path.startswith('http') and 'gs://' in state_path:
        state_path = 'gs://' + state_path.split('gs://')[-1]

    if state_path.startswith('gs://'):
        state_path = 'https://storage.googleapis.com/' + state_path[len('gs://'):]

    if state_path.startswith('https://') or state_path.startswith('http://'):
        state = requests.get(state_path, timeout=60).json()
    else:
        state = json.load(open(state_path, 'r'))  # noqa

    return format_nglink('https://clio-ng.janelia.org', state)


@PrefixFilter.with_context("downstream capture")
def _export_downstream_capture(cfg, snapshot_tag, partner_df, *, name):
    _name = '-'.join(name.split())

    capture_statuses = pd.Series(cfg['capture-statuses']).astype(STATUS_DTYPE)
    min_capture_status = capture_statuses.min()

    # We're only interested in the downstream capture of
    # upstream bodies which are themselves in the capture set.
    df = (
        partner_df
        .query('captured_pre')
        .groupby('body_pre')['captured_post'].value_counts()
        .unstack()
        .fillna(0.0)
        .astype(np.int32)
    )
    df = df.rename(columns={True: 'captured', False: 'not_captured'})
    df['capture_frac'] = (df['captured'] / (df['captured'] + df['not_captured'])).astype(np.float32)

    logger.info(f"Exporting {name}-downstream-capture-{snapshot_tag} table and plot")
    feather.write_feather(
        df.reset_index(),
        f"tables/{name}-downstream-capture-{snapshot_tag}.feather"
    )

    p = df['capture_frac'].hvplot.hist(
        title=f"{name}: Downstream capture, {min_capture_status} or better ({snapshot_tag})",
        xlabel=''
    )
    export_bokeh(
        hv.render(p),
        f"{_name}-downstream-capture-{snapshot_tag}.html",
        f"{name} downstream-capture-{snapshot_tag}"
    )
