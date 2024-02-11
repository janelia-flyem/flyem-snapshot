import os
import shutil
import logging
import warnings
from functools import partial

from neuclease import PrefixFilter
from neuclease.util import timed, compute_parallel

from .util import neo4j_column_names, append_neo4j_type_suffixes

logger = logging.getLogger(__name__)


@PrefixFilter.with_context("Synapse")
def export_neuprint_elements(cfg, element_tables):
    """
    Export generic (non-Synapse) Element nodes and relationships.
    (Synapses are exported elsewhere.)
    """
    if not element_tables:
        return
    element_dir = 'neuprint/Neuprint_Elements'
    if os.path.exists(element_dir):
        shutil.rmtree(element_dir)
    os.makedirs(element_dir)

    for config_name, (point_df, _) in element_tables.items():
        if point_df is not None:
            _export_neuprint_elements(cfg, point_df, config_name=config_name)


@PrefixFilter.with_context("{config_name}")
def _export_neuprint_elements(cfg, point_df, *, config_name):
    dataset = cfg['meta']['dataset']

    point_df = point_df.reset_index()
    point_df['type'] = point_df['type'].astype('category')

    point_df[':Label'] = f'Element;{dataset}_Element'

    specific_label = cfg['element-labels'].get(config_name, '')
    if specific_label.startswith(':'):
        specific_label = specific_label[1:]
    if specific_label:
        point_df[':Label'] += f';{specific_label};{dataset}_{specific_label}'

    # All non-ROI columns from the input table are exported as :Element properties.
    roicols = (*cfg['roi-set-names'], *(f'{c}_label' for c in cfg['roi-set-names']))
    prop_cols = set(point_df.columns) - set(roicols) - {*'xyz', 'point_id'}
    roi_syn_props = {k:v for k,v in cfg['roi-synapse-properties'].items() if k in point_df.columns}

    point_df = point_df.rename(columns={
        'point_id': ':ID(Element-ID)'
    })

    logger.info(f"Exporting {specific_label} Elements")
    logger.info(f"Non-ROI properties: {prop_cols}")
    logger.info(f"ROI properties from the following roi-sets: {cfg['roi-set-names']}")
    point_df = append_neo4j_type_suffixes(point_df, exclude=(*'xyz', *roicols))

    export_subdir = f'neuprint/Neuprint_Elements/{config_name}'
    os.makedirs(export_subdir, exist_ok=True)
    _export_fn = partial(
        _export_element_group_csv,
        export_subdir,
        roi_syn_props,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*groupby with a grouper equal to a list of length 1.*")
        roisets = sorted(set(cfg['roi-set-names']) & set(point_df.columns))
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
        if distance_df is None:
            continue
        df = distance_df.rename(columns={
            'source_id': ':START_ID(Element-ID)',
            'target_id': ':END_ID(Element-ID)',
        })
        df = df.drop(columns=['source_x', 'source_y', 'source_z', 'target_x', 'target_y', 'target_z'])

        # All other columns from the input table are exported as :CloseTo properties.
        df = append_neo4j_type_suffixes(df, exclude=(*'xyz',))
        df.to_csv(f'neuprint/Neuprint_Elements_CloseTo_{config_name}.csv', index=False, header=True)
