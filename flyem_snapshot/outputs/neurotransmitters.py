import os
import logging

import numpy as np
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, timed
from neuclease.dvid import fetch_body_annotations, post_keyvalues
from ..caches import cached, SentinelSerializer

logger = logging.getLogger(__name__)


NeurotransmitterExportSchema = {
    "type": "object",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "export-neurotransmitters": {
            "description":
                "If true, export the neurotransmitter predictions in a table\n"
                "that includes columns for body and synapse confidence.\n",
            "type": "boolean",
            "default": False,
        },
        "roi-sets": {
            "description": "Which roi-set columns to include in the output table, if any.",
            "type": "array",
            "default": [],
            "items": {
                "type": "string"
            }
        },
        "restrict-export": {
            "description":
                "Optionally name an roi-set which will be used to filter\n"
                "the set of bodies to include in the export.\n"
                "Only bodies which touch a valid ROI in the given roi-set will be included.\n"
                "Note that the filtering occurs AFTER the per-body aggregations were computed,\n"
                "so the final per-body averages will incorporate information from the tbars outside\n"
                "the restriction ROI.\n",
            "default": "",
            "type": "string",
        },
        "dvid": {
            "type": "object",
            "additionalProperties": False,
            "default": {},
            "properties": {
                "perform-backport": {
                    "description":
                        "If true, backport the body-level neurotransmitter predictions to DVID.\n"
                        "Note: If you specify a locked DVID node, then the export will fail unless the DVID_ADMIN_TOKEN is set in your environment.\n",
                    "type": "boolean",
                    "default": False
                },
                "server": {
                    "type": "string",
                    "default": ""
                },
                "uuid": {
                    "type": "string",
                    "default": ""
                },
                "neuronjson_instance": {
                    "type": "string",
                    "default": ""
                }
            }
        }
    }
}


@PrefixFilter.with_context('neurotransmitters')
@cached(SentinelSerializer('neurotransmitter-export'))
def export_neurotransmitters(cfg, tbar_nt, body_nt, nt_confusion, point_df):
    if cfg['export-neurotransmitters']:
        _write_flat_tables(cfg, tbar_nt, body_nt, nt_confusion, point_df)

    if cfg['dvid']['perform-backport']:
        _backport_to_dvid(cfg, body_nt)


@PrefixFilter.with_context('flat')
@timed("Exporting flat neurotransmitter tables")
def _write_flat_tables(cfg, tbar_nt, body_nt, nt_confusion, point_df):
    with Timer("Constructing tbar neurotransmitter table", logger):
        tbar_nt = tbar_nt[[c for c in tbar_nt.columns if c.startswith('nt')]]
        tbar_df = point_df.query('kind == "PreSyn"')
    tbar_cols = [*'xyz', 'conf', 'sv', 'body', *cfg['roi-sets']]
    tbar_cols = [c for c in tbar_cols if c in tbar_df.columns]
    tbar_df = tbar_df[tbar_cols]
    tbar_df = tbar_df.merge(tbar_nt, 'left', on='point_id')

    if (roiset := cfg['restrict-export']):
        with Timer("Filtering neurotransmitter export", logger):
            bodies = point_df.loc[point_df[roiset] != "<unspecified>", 'body'].unique()
            tbar_df = tbar_df.loc[tbar_df['body'].isin(bodies)]
            body_nt = body_nt.loc[body_nt.index.isin(bodies)]

    with Timer("Writing flat neurotransmitter tables", logger):
        os.makedirs('nt', exist_ok=True)
        assert tbar_df.index.name == 'point_id'
        feather.write_feather(tbar_df.reset_index(), 'nt/tbar-neurotransmitters.feather')

        assert body_nt.index.name == 'body'
        feather.write_feather(body_nt.reset_index(), 'nt/body-neurotransmitters.feather')

        if nt_confusion is not None:
            nt_confusion.to_csv('nt/neurotransmitter-confusion.csv', index=True, header=True)


@PrefixFilter.with_context('dvid-backport')
@timed("Backporting neurotransmitter predictions to DVID")
def _backport_to_dvid(cfg, body_nt):
    assert body_nt.index.name == 'body'
    clio_nt_cols = [
        'total_nt_predictions', 'predicted_nt', 'predicted_nt_confidence',
        'celltype_predicted_nt', 'celltype_predicted_nt_confidence', 'celltype_total_nt_predictions',
        'consensus_nt'
    ]

    # Fetch current body annotations from DVID (clio)
    dvid_inst = (cfg['dvid']['server'], cfg['dvid']['uuid'], cfg['dvid']['neuronjson_instance'])
    orig_ann = fetch_body_annotations(*dvid_inst)
    ann = (
        orig_ann
        .drop(columns=clio_nt_cols, errors='ignore')
        .merge(body_nt, 'left', on='body')
    )

    # We cannot trust the `consensus_nt` column for any cell type in DVID which
    # no longer matches the type at the time we generated the NT predictions.
    # Erase those predictions explicitly.
    retyped_bodies = ann.query('cell_type.notnull() and type.notnull() and cell_type != type')[['type', 'cell_type']].index
    ann.loc[retyped_bodies, ['celltype_predicted_nt', 'consensus_nt', 'ground_truth']] = None
    ann.loc[retyped_bodies, ['celltype_predicted_nt_confidence', 'celltype_total_nt_predictions']] = np.nan

    # Select the columns we want.
    ann_nt = ann[clio_nt_cols].copy().replace([np.nan], [None])
    orig_ann_nt = orig_ann.reindex(columns=clio_nt_cols).replace([np.nan], [None])

    # Find the rows that have changed.
    changed = ((orig_ann_nt != ann_nt) & (orig_ann_nt.notnull() | ann_nt.notnull())).any(axis=1)
    if not changed.any():
        logger.info("No neurotransmitter predictions differ from those already in DVID.")
        return

    with Timer(f"Posting backported neurotransmitter predictions to DVID for {len(ann_nt)} bodies", logger):
        ann_nt = ann_nt[changed]
        ann_nt['bodyid'] = ann_nt.index
        kv = ann_nt.to_dict(orient='index')
        post_keyvalues(*dvid_inst, kv)
