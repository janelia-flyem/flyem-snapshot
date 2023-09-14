import os
import shutil
import warnings
from functools import partial

from neuclease import PrefixFilter
from neuclease.util import timed, compute_parallel

from .util import neo4j_column_names


@PrefixFilter.with_context("Synapse")
def export_neuprint_synapses(cfg, point_df):
    synapse_dir = 'neuprint/Neuprint_Synapses'
    if os.path.exists(synapse_dir):
        shutil.rmtree(synapse_dir)
    os.makedirs(synapse_dir)

    dataset = cfg['dataset']
    roiset_names = list(cfg['roi-set-meta'].keys())

    point_df = point_df.reset_index()
    point_df[':Label'] = f'Synapse;{dataset}_Synapse'
    assert (point_df['kind'].cat.categories == ['PostSyn', 'PreSyn']).all()
    point_df['kind'].cat.rename_categories(['post', 'pre'])
    point_df = point_df.rename(columns={
        'point_id': ':ID(Syn-ID)',
        'kind': 'type:string',
        'conf': 'confidence:float',
    })

    roi_syn_props = {
        roiset_name: rs['synapse-properties']
        for roiset_name, rs in cfg['roi-set-meta'].items()
    }
    _export_fn = partial(_export_synapse_group_csv, roi_syn_props)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*groupby with a grouper equal to a list of length 1.*")
        groups = point_df.groupby(roiset_names, dropna=False, observed=True)
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
            df[prop_name] = eval(formula, {}, {'x': roi_segment_id})  # pylint: disable=eval-used
            extra_props.append(prop_name)

    # The ROI boolean flags are all :boolean
    df[[f'{roi}:boolean' for roi in group_rois if roi != '<unspecified>']] = True

    # Give types to the extra properties, too.
    typed_renames = neo4j_column_names(None, df[extra_props])
    df = df.rename(columns=typed_renames)

    # Only export the columns which we intend for neo4j (not x,y,z)
    df = df[[c for c in df.columns if ':' in c]]
    df.to_csv(f'neuprint/Neuprint_Synapses/{i:06d}.csv', index=False, header=True)


@timed("Writing Neuprint_Synapse_Connections.csv")
def export_neuprint_synapse_connections(partner_df):
    df = partner_df[['pre_id', 'post_id']]
    df.columns = [':START_ID(Syn-ID)', ':END_ID(Syn-ID)']
    df.to_csv('neuprint/Neuprint_Synapse_Connections.csv', index=False, header=True)
