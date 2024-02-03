import os
import logging

from neuclease import PrefixFilter
from neuclease.util import timed, Timer

logger = logging.getLogger(__name__)


@PrefixFilter.with_context("ElementSet")
@timed
def export_neuprint_elementsets(cfg, element_tables):
    for config_name, (points, _) in element_tables:
        _export_elementsets(cfg, config_name, points)


def _export_elementsets(cfg, config_name, point_df):
    if point_df is None:
        return

    dataset = cfg['meta']['dataset']
    point_df = point_df.reset_index()
    point_df['elmset_id'] = point_df['body'].astype(str) + '_' + point_df['type']

    elmset_df = point_df.drop_duplicates('elmset_id')[['elmset_id', 'type']]

    os.makedirs('Neuprint_ElementSets')
    fname = f"Neuprint_ElementSets/Neuprint_ElementSet_{config_name}.csv"
    with Timer(f"Writing {fname}", logger):
        (
            elmset_df[['elmset_id', 'type']]
            .assign(label=f"ElementSet;{dataset}_ElementSet")
            .rename(columns={
                'elmset_id': ':ID(ElementSet-ID)',
                'label': ':Label',
                'type': 'type:string'
            })
            .to_csv(f'neuprint/{fname}', index=False, header=True)
        )

    fname = f"Neuprint_Neuron_to_ElementSet_{config_name}.csv"
    with Timer(f"Writing {fname}", logger):
        (
            elmset_df[['body', 'elmset_id']].rename(columns={
                'body': ':START_ID(Body-ID)',
                'elmset_id': ':END_ID(ElementSet-ID)'
            })
            .to_csv(f'neuprint/{fname}', index=False, header=True)
        )

    fname = f"Neuprint_ElementSet_to_Element_{config_name}.csv"
    with Timer(f"Writing {fname}", logger):
        (
            point_df[['elmset_id', 'point_id']].rename(columns={
                'elmset_id': ':START_ID(Body-ID)',
                'point_id': 'END_ID(Element-ID)'
            })
            .to_csv(f'neuprint/{fname}', index=False, header=True)
        )
