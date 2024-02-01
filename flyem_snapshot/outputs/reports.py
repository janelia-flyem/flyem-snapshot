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
    plot_categorized_connectivity_forecast,
    variable_width_hbar,
)

from ..util import export_bokeh

_ = hvplot.pandas  # linting

logger = logging.getLogger(__name__)

CAPTURE_STATS = ['traced_presyn_frac', 'traced_postsyn_frac', 'traced_synweight_frac', 'traced_conn_frac']


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

ReportsSchema = {
    "description": "Specs for connectivity reports to generate from the snapshot data.\n",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "report-roiset": {
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


@PrefixFilter.with_context('Report')
def export_reports(cfg, point_df, partner_df, ann, snapshot_tag):
    if len(cfg['reports']) == 0:
        logger.info("No reports requested.")
        return

    os.makedirs("png", exist_ok=True)
    os.makedirs("html", exist_ok=True)

    # Make sure our roiset column is named 'roi' since that's what completeness_forecast() expects.
    roiset = cfg['report-roiset']
    point_df = point_df.drop(columns=['roi'], errors='ignore').rename({roiset: 'roi'})
    partner_df = partner_df.drop(columns=['roi'], errors='ignore').rename({roiset: 'roi'})

    if roiset not in partner_df.columns:
        partner_df = partner_df.merge(point_df['roi'].rename_axis('post_id'), 'left', on='post_id')

    with Timer("Flagging captured bodies", logger):
        capture_statuses = np.array(cfg['capture-statuses'])
        _ = capture_statuses  # linting nonsense
        capture_bodies = ann.query('status in @capture_statuses').index
        partner_df['captured_pre'] = partner_df['body_pre'].isin(capture_bodies)
        partner_df['captured_post'] = partner_df['body_post'].isin(capture_bodies).astype(pd.CategoricalDtype([False, True]))

        # Converting to categorical ensures that value_counts() always gives
        # results for both False and True, even if there are no True (or False) entries.
        # (An unlikely scenario, but it guards against an ROI having NO traced neurons at
        # all, which could happen early in a reconstruction.)
        partner_df['captured_pre'] = partner_df['captured_pre'].astype(pd.CategoricalDtype([False, True]))
        partner_df['captured_post'] = partner_df['captured_post'].astype(pd.CategoricalDtype([False, True]))

    with Timer("Grouping by ROI", logger):
        roi_point_dfs = {roi: df for roi, df in point_df.groupby('roi')}
        roi_partner_dfs = {roi: df for roi, df in partner_df.groupby('roi')}

    all_status_stats = []
    all_syncounts = []
    for report in cfg['reports']:
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
            name=report['name']
        )
        all_syncounts.append(syncounts)
        all_status_stats.append(status_stats)

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
    names = [report['name'] for report in cfg['reports']]
    names_df = pd.DataFrame({'report_index': np.arange(len(names)), 'name': names})

    all_syncounts = (
        pd.concat(all_syncounts, ignore_index=True)
        .set_index(['name', 'kind'])['count']
        .astype(np.float32)
        .unstack()
        .fillna(0.0)
    )
    status_dtype = pd.CategoricalDtype(DEFAULT_BODY_STATUS_CATEGORIES, ordered=True)
    all_status_stats = pd.concat(all_status_stats, ignore_index=True)
    all_status_stats['status'] = all_status_stats['status'].astype(status_dtype)

    # We don't plot anything with empty status or worse.
    all_status_stats = all_status_stats.query('status > ""')
    relevant_statuses = all_status_stats['status'].unique().astype(str)

    # This unstack() will result in multi-level columns:
    # level 0: traced_presyn_frac                            traced_postsyn_frac                        ...
    # level 1: Roughly traced 	Prelim Roughly traced   ...  Roughly traced 	Prelim Roughly traced   ...
    all_status_stats = (
        all_status_stats
        .set_index(['name', 'status'])
        .unstack()
        .ffill(axis=1)
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
        'traced_presyn_frac': 'Captured PreSyn by Status and ROI',
        'traced_postsyn_frac': 'Captured PostSyn by Status',
        'traced_conn_frac': 'Captured Connectivity by Status and ROI',
        'traced_synweight_frac': 'Captured SynWeight by Status and ROI',
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
        df.to_csv(f'tables/{level0}-by-status.csv', index=False, header=True)

        # The dataframe has cumulative connectivity,
        # but for the stacked bar chart we don't want cumulative.
        df[relevant_statuses] -= df[relevant_statuses].shift(1, axis=1, fill_value=0)

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


@PrefixFilter.with_context("capture forecast")
def _completeness_forecast(cfg, snapshot_tag, point_df, partner_df, ann, roi, *, name):
    stop_at_rank = int(cfg['stop-at-rank'])
    selection_link = _get_neuroglancer_base_link(cfg['neuroglancer-base-state'])
    _name = '-'.join(name.split())

    # This takes 10-15 minutes for the whole brain, even with precomputed body_pre, body_post.
    with Timer("Generating sorted connection completeness table and synapse counts", logger):
        conn_df, syn_counts_df = completeness_forecast(
            point_df, partner_df, None, ann[['status']], roi=roi,
            sort_by=['status', 'SynWeight', 'PreSyn', 'PostSyn'],
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
        title=f"{name}: Downstream capture ({snapshot_tag})",
        xlabel=''
    )
    export_bokeh(
        hv.render(p),
        f"{_name}-downstream-capture-{snapshot_tag}.html",
        f"{name} downstream-capture-{snapshot_tag}"
    )
