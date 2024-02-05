import os
import shutil
import warnings
from functools import partial

from neuclease import PrefixFilter
from neuclease.util import timed, compute_parallel

from .util import neo4j_column_names, append_neo4j_type_suffixes


@PrefixFilter.with_context("Synapse")
def export_neuprint_elements(cfg, element_tables):
    """
    Export generic (non-Synapse) Element nodes and relationships.
    (Synapses are exported elsewhere.)
    """
    element_dir = 'neuprint/Neuprint_Elements'
    if os.path.exists(element_dir):
        shutil.rmtree(element_dir)
    os.makedirs(element_dir)

    for config_name, (point_df, _) in element_tables.values():
        if point_df is not None:
            _export_neuprint_elements(cfg, config_name, point_df)


@PrefixFilter.with_context("{config_name}")
def _export_neuprint_elements(cfg, config_name, point_df):
    dataset = cfg['meta']['dataset']
    specific_label = cfg['elements'][config_name]['neuprint-label']
    if specific_label.startswith(':'):
        specific_label = specific_label[1:]
    point_df = point_df.reset_index()
    point_df[':Label'] = f'Element;{dataset}_Element;{specific_label};{dataset}_{specific_label}'
    point_df['type'] = point_df['type'].astype('category')

    point_df = point_df.rename(columns={
        'point_id': ':ID(Element-ID)'
    })

    # All columns from the input table are exported as :Element properties.
    point_df = append_neo4j_type_suffixes(point_df, exclude=(*'xyz',))

    export_subdir = f'neuprint/Neuprint_Elements/{config_name}'
    os.makedirs(export_subdir, exist_ok=True)
    _export_fn = partial(
        _export_element_group_csv,
        export_subdir,
        cfg['roi-synapse-properties']
    )

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


def _export_element_group_csv(subdir, roi_syn_props, i, group_rois, df):
    """
    Element a single CSV file of Element nodes, in which the ROI for all
    points in the group are homogenous (i.e. all ROI columns are the same
    in every row).
    """
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
            "_export_element_group_csv is supposed to be called exclusively with homogenous ROI columns."
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
    df.to_csv(f'{subdir}/{i:06d}.csv', index=False, header=True)


@timed("Exporting neuprint :Element:CloseTo files")
def export_neuprint_elements_closeto(element_tables):
    for config_name, (_, distance_df) in element_tables.items():
        df = distance_df.rename(columns={
            'source_id': ':START_ID(Element-ID)',
            'target_id': ':END_ID(Element-ID)',
        })
        df = df.drop(columns=['source_x', 'source_y', 'source_z', 'target_x', 'target_y', 'target_z'])

        # All other columns from the input table are exported as :CloseTo properties.
        df = append_neo4j_type_suffixes(df, exclude=(*'xyz',))
        df.to_csv(f'neuprint/Neuprint_Elements_CloseTo_{config_name}.csv', index=False, header=True)
