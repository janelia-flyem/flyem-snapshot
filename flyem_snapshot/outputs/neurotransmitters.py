import os
import logging

import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer

from ..caches import cached, SentinelSerializer

logger = logging.getLogger(__name__)


NeurotransmiterExportSchema = {
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
        }
    }
}


@PrefixFilter.with_context('neurotransmitters')
@cached(SentinelSerializer('neurotransmitter-export'))
def export_neurotransmitters(cfg, tbar_nt, body_nt, nt_confusion, point_df):
    if not cfg['export-neurotransmitters']:
        return

    with Timer("Constructing tbar neurotransmitter table", logger):
        tbar_nt = tbar_nt[[c for c in tbar_nt.columns if c.startswith('nt')]]
        tbar_df = point_df.query('kind == "PreSyn"')[[*'xyz', 'conf', 'sv', 'body', *cfg['roi-sets']]]
        tbar_df = tbar_df.merge(tbar_nt, 'left', on='point_id')

    if (roiset := cfg['restrict-export']):
        with Timer("Filtering neurotransmitter export", logger):
            bodies = point_df.loc[point_df[roiset] != "<unspecified>", 'body'].unique()
            tbar_df = tbar_df.loc[tbar_df['body'].isin(bodies)]
            body_nt = body_nt.loc[body_nt.index.isin(bodies)]

    with Timer("Writing neurotransmitter tables", logger):
        os.makedirs('nt', exist_ok=True)
        feather.write_feather(tbar_df, 'nt/tbar-neurotransmitters.feather')
        feather.write_feather(body_nt, 'nt/body-neurotransmitters.feather')

        if nt_confusion is not None:
            nt_confusion.to_csv('nt/neurotransmitter-confusion.csv', index=True, header=True)
