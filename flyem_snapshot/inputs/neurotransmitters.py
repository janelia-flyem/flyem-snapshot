import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, encode_coords_to_uint64

logger = logging.getLogger(__name__)

NeurotransmittersSchema = {
    "description": "How to load neurotransmitter data",
    "type": "object",
    "additionalProperties": False,
    "default": {},
    "properties": {
        "synister-feather": {
            "description":
                "Path to an Apache Feather file with neurotransmitter\n"
                "predictions as produced via the 'synister' tool/method.",
            "type": "string",
            "default": ""
        },
        "groundtruth": {
            "description": "Path to a CSV file containing the celltype groundtruth neurotransmitters",
            "type": "string",
            "default": ""
        },
        "rescale-coords": {
            "description":
                "If the synister file has x,y,z coordinates in nanometers instead of voxels,\n"
                "use this setting to rescale them to voxel units. (You'll have to check the file.)\n"
                "Example: 0.125 will convert nm units to units of 8nm voxels.\n",
            "default": [1, 1, 1],
            "oneOf": [
                {
                    "type": "number"
                },
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            ]
        },
        "translate-names": {
            "description":
                "If desired, you can translate the neurotransmitter names from the source file to alternate names.\n"
                "In particular, synister uses the term 'neither' for indeterminate predictions, but FlyEM uses the term 'unknown'.\n",
            "type": "object",
            "default": {
                "neither": "unknown",
            },
            "additionalProperties": {
                "type": "string"
            }
        }
    }
}


@PrefixFilter.with_context('neurotransmitters')
def load_neurotransmitters(cfg, point_df, ann):
    """
    Load the synister neurotransmitter predictions, but tweak the column
    names into the form nt_{transmitter}_prob and exclude columns other than
    the predictions and xyz.

    Also, a 'body' column is added to the table,
    and the point_id is stored in the index.

    Furthermore, a table of bodywise aggregate scores is generated,
    using the 'body' column in point_df.
    """
    if not (path := cfg['synister-feather']):
        return None, None

    with Timer("Loading neurotransmitters", logger):
        tbar_nt, body_nt = _load_neurotransmitters(path, cfg['rescale-coords'], cfg['translate-names'], point_df)

    if cfg['groundtruth']:
        if 'split' not in tbar_nt.columns or not (tbar_nt['split'] == 'test').any():
            logger.warning("Can't compute neurotransmitter confidences without a confusion matrix.")
        assert False, 'fixme here'

    return tbar_nt, body_nt


def _load_neurotransmitters(path, rescale, translations, point_df):
    ##
    ## TODO: Provide config values for excluding body-level predictions based on tbar count.
    ##
    tbar_nt = feather.read_feather(path)

    # Rename columns pre_x, pre_y, pre_z -> x,y,z
    tbar_nt = tbar_nt.rename(columns={f'pre_{k}':k for k in 'xyz'})
    tbar_nt = tbar_nt.rename(columns={f'{k}_pre':k for k in 'xyz'})
    nt_cols = [col for col in tbar_nt.columns if col.startswith('nts')]
    tbar_nt = tbar_nt[[*'xyz', *nt_cols]]

    # Apply user's coordinate scaling factor.
    tbar_nt[[*'xyz']] = (tbar_nt[[*'xyz']] * rescale).astype(np.int32)

    # The original table has names like 'nts_8.glutamate',
    # but we'll convert that to 'nt_glutamate_prob'.
    nt_names = [c.split('.')[1] for c in nt_cols]
    nt_names = [translations.get(n, n) for n in nt_names]
    renames = {
        c: 'nt_' + name + '_prob'
        for c, name in zip(nt_cols, nt_names)
    }
    tbar_nt = tbar_nt.rename(columns=renames)
    nt_cols = list(renames.values())

    tbar_nt['point_id'] = encode_coords_to_uint64(tbar_nt[[*'zyx']].values)
    tbar_nt = tbar_nt.set_index('point_id')

    # Drop predictions which correspond to synapses we don't have (anymore?)
    presyn_df = point_df.query('kind == "PreSyn"')
    tbar_nt = presyn_df[['body']].merge(tbar_nt, 'left', on='point_id')

    col_to_nt = {c: c.split('_')[1] for c in nt_cols}
    tbar_nt['predicted_nt'] = tbar_nt[nt_cols].idxmax(axis=1).map(col_to_nt)

    if (nullcount := tbar_nt[nt_cols[0]].isnull().sum()):
        logger.warning(
            f"Neurotransmitter predictions do not cover {nullcount} "
            f"({100 * nullcount / len(presyn_df):.2f}%) of the PreSyn points!")

    body_nt = tbar_nt.groupby('body')[nt_cols].mean()
    body_nt['predicted_nt'] = body_nt.idxmax(axis=1).map(col_to_nt)

    # FIXME
    #body_nt['predicted_nt_prob'] = body_nt.idxmax(axis=1).map(col_to_nt)
    return tbar_nt, body_nt
