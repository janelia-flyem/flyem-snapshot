import os
import shutil
import logging
import warnings
from functools import partial

from neuclease import PrefixFilter
from neuclease.util import timed, compute_parallel, snakecase_to_camelcase

from .util import append_neo4j_type_suffixes

from .element import export_element_group_csv

logger = logging.getLogger(__name__)


@PrefixFilter.with_context("Synapse")
def export_neuprint_synapses(cfg, point_df, tbar_nt):
    synapse_dir = 'neuprint/Neuprint_Synapses'
    if os.path.exists(synapse_dir):
        shutil.rmtree(synapse_dir)
    os.makedirs(synapse_dir)

    dataset = cfg['meta']['dataset']

    # Check that the 'kind' column has the correct categorical dtype.
    # Sometimes our tables have a third category in the dtype,
    # but it shouldn't be present in the data values.
    assert set(point_df['kind'].cat.categories) >= {'PostSyn', 'PreSyn'}
    if len(point_df['kind'].cat.categories) > 2:
        assert point_df['kind'].value_counts().loc[['PreSyn', 'PostSyn']].sum() == len(point_df), \
            "point_df['kind'] must not contain any categories other than PreSyn and PostSyn"

    if tbar_nt is not None:
        tbar_nt = tbar_nt.drop(columns=['body', *'xyz'])
        tbar_nt = tbar_nt.rename(columns={
            c: snakecase_to_camelcase(c) for c in tbar_nt.columns
        })
        tbar_nt = append_neo4j_type_suffixes(tbar_nt)
        point_df = point_df.merge(tbar_nt, 'left', on='point_id')

    point_df = point_df.reset_index()
    point_df[':Label'] = f'Synapse;{dataset}_Synapse;Element;{dataset}_Element'
    point_df['kind'] = point_df['kind'].cat.rename_categories({'PreSyn': 'pre', 'PostSyn': 'post'})

    roisets = cfg['roi-set-names']
    if (missing_roisets := set(roisets) - set(point_df.columns)):
        raise RuntimeError(
            f"roi-set-names includes {missing_roisets} which aren't\n"
            "present as columns in the synapse point table"
        )

    # All non-ROI columns from the input table are exported as :Element properties.
    roicols = (*roisets, *(f'{c}_label' for c in roisets))
    prop_cols = set(point_df.columns) - set(roicols) - {*'xyz', 'point_id'}
    roi_syn_props = {k:v for k,v in cfg['roi-synapse-properties'].items() if k in point_df.columns}

    # Note that :Synapses are :Elements and must be referenced that way in :CloseTo relationships.
    # Therefore, we use an ID space named "Element-ID", which is shared by Element nodes.
    point_df = (
        point_df
        .drop(columns=['sv'], errors='ignore')
        .rename(columns={
            'point_id': ':ID(Element-ID)',
            'kind': 'type:string',
            'conf': 'confidence:float',
            'body': 'bodyId',
        })
    )

    logger.info(f"Non-ROI Synapse properties: {prop_cols}")
    logger.info(f"ROI Synapse properties from the following roi-sets: {roisets}")
    point_df = append_neo4j_type_suffixes(point_df, exclude=(*'xyz', *roicols))

    _export_fn = partial(
        export_element_group_csv,
        'neuprint/Neuprint_Synapses',
        roi_syn_props
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*groupby with a grouper equal to a list of length 1.*")
        groups = point_df.groupby(roisets, dropna=False, observed=True)
        batches = ((i, group_rois, df) for i, (group_rois, df) in enumerate(groups))
        compute_parallel(
            _export_fn,
            batches,
            starmap=True,
            processes=cfg['processes'],
            total=groups.ngroups,
            leave_progress=True,
        )


@timed("Writing Neuprint_Synapse_Connections.csv")
def export_neuprint_synapse_connections(partner_df):
    df = partner_df[['pre_id', 'post_id']]
    df.columns = [':START_ID(Element-ID)', ':END_ID(Element-ID)']
    df.to_csv('neuprint/Neuprint_Synapse_Connections.csv', index=False, header=True)
