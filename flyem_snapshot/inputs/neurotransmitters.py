import logging

import numpy as np
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
        "rescale-coords": {
            "description":
                "If the synister file has x,y,z coordinates in nanometers instead of voxels,\n"
                "use this setting to rescale them to voxel units. (You'll have to check the file.)\n"
                "Example: 0.125 will convert nm to 8nm voxels.\n",
            "type": "number",
            "default": 1
        }
    }
}


@PrefixFilter.with_context('neurotransmitters')
def load_neurotransmitters(cfg, point_df):
    """
    Load the synister neurotransmitter predictions, but tweak the column
    names into the form nt_{transmitter}_prob and exclude columns other than
    the predictions and xyz.
    Also, a 'body' column is added to the table, and the point_id is stored in the index.
    """
    if not (path := cfg['synister-feather']):
        return None, None

    with Timer("Loading neurotransmitters", logger):
        return _load_neurotransmitters(path, cfg['rescale-coords'], point_df)


def _load_neurotransmitters(path, rescale, point_df):
    tbar_nt = feather.read_feather(path)
    tbar_nt = tbar_nt.rename(columns={f'pre_{k}':k for k in 'xyz'})
    tbar_nt = tbar_nt.rename(columns={f'{k}_pre':k for k in 'xyz'})
    nt_cols = [col for col in tbar_nt.columns if col.startswith('nts')]
    tbar_nt = tbar_nt[[*'xyz', *nt_cols]]

    tbar_nt[[*'xyz']] = (tbar_nt[[*'xyz']] * rescale).astype(np.int32)

    renames = {
        c: 'nt_' + c.split('.')[1] + '_prob' for c in nt_cols
    }
    tbar_nt = tbar_nt.rename(columns=renames)
    nt_cols = list(renames.values())

    tbar_nt['point_id'] = encode_coords_to_uint64(tbar_nt[[*'zyx']].values)
    tbar_nt = tbar_nt.set_index('point_id')

    # Drop predictions which correspond to synapses we don't have (anymore?)
    presyn_df = point_df.query('kind == "PreSyn"')
    tbar_nt = presyn_df[['body']].merge(tbar_nt, 'left', on='point_id')

    if (nullcount := tbar_nt[nt_cols[0]].isnull().sum()):
        logger.warning(
            f"Neurotransmitter predictions do not cover {nullcount} "
            f"({100 * nullcount / len(presyn_df):.2f}%) of the PreSyn points!")

    body_nt = tbar_nt.groupby('body')[nt_cols].mean()
    nt_names = {c: c.split('_')[1] for c in body_nt.columns}
    body_nt['predicted_nt'] = body_nt.idxmax(axis=1).map(nt_names)
    return tbar_nt, body_nt
