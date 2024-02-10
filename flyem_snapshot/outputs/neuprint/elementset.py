import os
import logging

import pandas as pd

from neuclease import PrefixFilter
from neuclease.util import timed, Timer

logger = logging.getLogger(__name__)


@PrefixFilter.with_context("ElementSet")
@timed
def export_neuprint_elementsets(cfg, element_tables, connectome):
    synaptic_bodies = pd.concat(
        (connectome['body_pre'].rename('body'),
         connectome['body_post'].rename('body')), ignore_index=True).drop_duplicates()

    for config_name, (points, _) in element_tables.items():
        _export_elementsets(cfg, config_name, points, synaptic_bodies)


def _export_elementsets(cfg, config_name, point_df, synaptic_bodies):
    if point_df is None:
        return

    dataset = cfg['meta']['dataset']
    label = f"ElementSet;{dataset}_ElementSet"

    specific_label = cfg['element-labels'].get(config_name, '')
    if specific_label.startswith(':'):
        specific_label = specific_label[1:]
    if specific_label:
        label += f";{specific_label}Set;{dataset}_{specific_label}Set"

    point_df = point_df.reset_index()
    point_df['elmset_id'] = point_df['body'].astype(str) + '_' + point_df['type']

    elmset_df = point_df.drop_duplicates('elmset_id')

    os.makedirs('neuprint/Neuprint_ElementSets', exist_ok=True)
    fname = f"Neuprint_ElementSet_{config_name}.csv"
    with Timer(f"Writing {fname}", logger):
        (
            elmset_df[['elmset_id', 'type']]
            .assign(label=label)
            .rename(columns={
                'elmset_id': ':ID(ElementSet-ID)',
                'label': ':Label',
                'type': 'type:string'
            })
            .to_csv(f'neuprint/Neuprint_ElementSets/{fname}', index=False, header=True)
        )

    fname = f"Neuprint_Neuron_to_ElementSet_{config_name}.csv"
    with Timer(f"Writing {fname}", logger):
        # Currently, neuprint does not create any :Segment nodes
        # for non-synaptic bodies (including body 0).
        # (We also discard all Synapses that fall on body 0.)
        #
        # But generic Elements are intended to be versatile and serve
        # diverse use-cases, including those that don't require an
        # enclosing :Segment.
        #
        # Therefore, unlike synapses, we DO allow Elements to exist
        # even if they come from non-synaptic bodies (including
        # body 0).
        #
        # The only "catch" is that we must not create the edge
        # :Segment-[:Contains]->:ElementSet in such cases,
        # since the :Segment doesn't exist.  The individual :Elements
        # can still be accessed in Cypher queries, indepedent of any
        # :Segment.
        #
        # Note:
        #   To support reasonable query performance, special care
        #   must be taken to generate appropriate indexes
        #   (e.g. for Element ROI properties).
        (
            elmset_df.loc[
                # Synaptic bodies only.
                elmset_df['body'].isin(synaptic_bodies),
                ['body', 'elmset_id']
            ]
            .rename(columns={
                'body': ':START_ID(Body-ID)',
                'elmset_id': ':END_ID(ElementSet-ID)'
            })
            .to_csv(f'neuprint/{fname}', index=False, header=True)
        )

    fname = f"Neuprint_ElementSet_to_Element_{config_name}.csv"
    with Timer(f"Writing {fname}", logger):
        (
            point_df[['elmset_id', 'point_id']].rename(columns={
                'elmset_id': ':START_ID(ElementSet-ID)',
                'point_id': ':END_ID(Element-ID)'
            })
            .to_csv(f'neuprint/{fname}', index=False, header=True)
        )
