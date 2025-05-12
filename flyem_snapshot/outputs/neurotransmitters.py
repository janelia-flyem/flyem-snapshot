import os
import logging

import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer

from google.cloud import storage

from ..caches import cached, SentinelSerializer
from ..util.util import upload_file_to_gcs

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
        "gc-project": {
            "description":
                "Google Cloud project.\n",
            "type": "string",
            "default": "FlyEM-Private"
        },
        "gcs-bucket": {
            "description":
                "Google Cloud Storage bucket to export files to.\n",
            "type": "string",
            "default": "flyem-snapshots"
        },
    }
}


@PrefixFilter.with_context('neurotransmitters')
@cached(SentinelSerializer('neurotransmitter-export'))
def export_neurotransmitters(cfg, tbar_nt, body_nt, nt_confusion, point_df, snapshot_tag):
    if not cfg['export-neurotransmitters']:
        return
    # Initialize Google cloud client and select project
    client = storage.Client()
    if cfg['gc-project'] and cfg['gcs-bucket']:
        client.project = cfg['gc-project']
    else:
        cfg['gcs-bucket'] = ""

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

    with Timer("Writing neurotransmitter tables", logger):
        os.makedirs('nt', exist_ok=True)
        assert tbar_df.index.name == 'point_id'
        fname = 'nt/tbar-neurotransmitters.feather'
        feather.write_feather(tbar_df.reset_index(), fname)
        upload_file_to_gcs(cfg['gcs-bucket'], fname, f"{snapshot_tag}/{fname}")

        assert body_nt.index.name == 'body'
        fname = 'nt/body-neurotransmitters.feather'
        feather.write_feather(body_nt.reset_index(), fname)
        upload_file_to_gcs(cfg['gcs-bucket'], fname, f"{snapshot_tag}/{fname}")

        if nt_confusion is not None:
            fname = 'nt/neurotransmitter-confusion.csv'
            nt_confusion.to_csv(fname, index=True, header=True)
            upload_file_to_gcs(cfg['gcs-bucket'], fname, f"{snapshot_tag}/{fname}")
