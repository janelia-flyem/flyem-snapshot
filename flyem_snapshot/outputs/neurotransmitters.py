import logging

import pyarrow.feather as feather
from neuclease.util import Timer

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
        }
    }
}


def export_neurotransmitters(cfg, tbar_nt, point_df):
    if not cfg['export-neurotransmitters']:
        return

    with Timer("Exporting tbar neurotransmitter table", logger):
        tbar_nt = tbar_nt[[c for c in tbar_nt.columns if c.startswith('nt')]]
        tbar_df = point_df.query('kind == "PreSyn"')[[*'xyz', 'conf', 'sv', 'body', *cfg['roi-sets']]]
        tbar_df = tbar_df.merge(tbar_nt, 'left', on='point_id')
        feather.write_feather(tbar_df, 'nt/tbar-bodies-neurotransmitters.feather')
