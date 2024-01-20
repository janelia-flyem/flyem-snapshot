import os
import shutil
import warnings
from functools import partial

from neuclease import PrefixFilter
from neuclease.util import timed, compute_parallel, snakecase_to_camelcase

from .util import neo4j_column_names, append_neo4j_type_suffixes


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

    # Note that :Synapses are :Elements and must be referenced that way in :CloseTo relationships.
    # Therefore, we use an ID space named "Element-ID", which is shared by Element nodes.
    point_df = point_df.rename(columns={
        'point_id': ':ID(Element-ID)',
        'kind': 'type:string',
        'conf': 'confidence:float',
    })

    _export_fn = partial(_export_synapse_group_csv, cfg['roi-synapse-properties'])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*groupby with a grouper equal to a list of length 1.*")
        groups = point_df.groupby(cfg['roi-set-names'], dropna=False, observed=True)
        batches = ((i, group_rois, df) for i, (group_rois, df) in enumerate(groups))
        compute_parallel(
            _export_fn,
            batches,
            starmap=True,
            processes=cfg['processes'],
            total=groups.ngroups,
            leave_progress=True,
        )


def _export_synapse_group_csv(roi_syn_props, i, group_rois, df):
    # A pandas bug causes groupby to unwrap single-item lists into a string.
    # It will be fixed in a new version of pandas, but for now we need this workaround.
    if isinstance(group_rois, str):
        group_rois = [group_rois]

    # The weird srid:9157 means 'cartesian-3d' according to the neo4j docs.
    # https://neo4j.com/docs/cypher-manual/current/values-and-types/spatial/#spatial-values-crs-cartesian
    df['location:point{srid:9157}'] = (
        '{x:' + df['x'].astype(str) +   # noqa
        ', y:' + df['y'].astype(str) +  # noqa
        ', z:' + df['z'].astype(str) + '}'
    )

    extra_props = []
    for roiset, prop_cfg in roi_syn_props.items():
        assert (df[roiset] == df[roiset].iloc[0]).all(), \
            "_export_synapse_group_csv is supposed to be called exclusively with homogenous ROI columns."
        roi_segment_id = df[f'{roiset}_label'].iloc[0]
        if roi_segment_id == 0:
            continue
        for prop_name, formula in prop_cfg.items():
            df[prop_name] = eval(formula, None, {'x': roi_segment_id})  # pylint: disable=eval-used
            extra_props.append(prop_name)

    # The ROI boolean flags are all :boolean
    # Note that we must use 'true' here instead of True because pandas
    # doesn't write lowercase for bools, but neo4j requires lowercase.
    # https://neo4j.com/docs/operations-manual/4.4/tools/neo4j-admin/neo4j-admin-import/#import-tool-header-format-properties
    df[[f'{roi}:boolean' for roi in group_rois if roi != '<unspecified>']] = 'true'

    # Give types to the extra properties, too.
    typed_renames = neo4j_column_names(df[extra_props])
    df = df.rename(columns=typed_renames)

    # Only export the columns which we intend for neo4j (not x,y,z)
    df = df[[c for c in df.columns if ':' in c]]
    df.to_csv(f'neuprint/Neuprint_Synapses/{i:06d}.csv', index=False, header=True)


@timed("Writing Neuprint_Synapse_Connections.csv")
def export_neuprint_synapse_connections(partner_df):
    df = partner_df[['pre_id', 'post_id']]
    df.columns = [':START_ID(Element-ID)', ':END_ID(Element-ID)']
    df.to_csv('neuprint/Neuprint_Synapse_Connections.csv', index=False, header=True)
