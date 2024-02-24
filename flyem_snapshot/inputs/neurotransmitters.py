import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import Timer, encode_coords_to_uint64

from ..util import restrict_synapses_to_roi

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
                "predictions as produced via the 'synister' tool/method.\n"
                "If body-level (and type-level) confidences are desired, then the table\n"
                "must also include a 'split' column indicating which synapses were in the 'train' and 'validation' sets\n"
                "(so we can discard them before computing the confusion matrix).\n",
                # FIXME: Specify required columns...
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
        },
        "override-celltype-before-consensus": {
            "description":
                "Some NT predictions may be considered unreliable unless there is experimental data to back them up.\n"
                "This column can be used to override the celltype prediction immediately before the 'consensus' column is produced.\n"
                "So, the celltype prediction column (property) will NOT be overridden in the end result, but the 'consensus'\n"
                "column will only use the override value, if at all.\n",
            "type": "object",
            "default": {},
            "additionalProperties": {
                "type": "string"
            }
        },
        "restrict-to-roi": {
            "description": "Drop synapses outside the given region before computing aggregate body and cell type scores and confidences.",
            "default": {},
            "type": "object",
            "additional-properties": False,
            "properties": {
                "roiset": {
                    "type": "string",
                    "default": ""
                },
                "roi": {
                    "type": "string",
                    "default": ""
                }
            }
        },
        "ground-truth": {
            "description":
                "Neurotransmitter groundtruth table CSV with columns cell_type,ground_truth\n"
                "If not provided, then no body-level or type-level confidences will be computed.\n",
            "type": "string",
            "default": ""
        },
        "experimental-groundtruth": {
            "description":
                "Optional. Table of high-confidence experimental groundtruth, used to override type-level\n"
                "predictions in the 'consensus' preduction column.\n"
                "Columns: cell_type, ground_truth, reference, [other_gt], [other_ref]",
            "type": "string",
            "default": ""
        },
        "min-body-confidence": {
            "description":
                "After computing body confidence scores, bodies with lower confidence scores than\n"
                "this threshold will not be assigned a body-level NT prediction.\n"
                "Instead, they'll be assigned 'unclear' as their NT prediction.\n",
            "type": "number",
            "default": 0.5
        },
        "min-body-presyn": {
            "description":
                "Bodies with fewer tbars than this cutoff will not be assigned a body-level NT prediction.\n"
                "Instead, they'll be assigned 'unclear' as their NT prediction.\n",
            "type": "integer",
            "default": 50
        },
        "min-celltype-presyn": {
            "description":
                "If the total number of tbars across all cells of a type does not meet this threshold,\n"
                "the type-level prediction will be 'unclear.\n",
            "type": "integer",
            "default": 100
        },
        "export-mean-tbar-scores": {
            "description":
                "For older datasets (MANC, hemibrain), we didn't compute a top body prediction\n"
                "and associated confidence score.  Instead, we just exported the mean NT prediction\n"
                "scores for each body as Neuron properties.\n",
            "type": "boolean",
            "default": True
        }
    }
}


@PrefixFilter.with_context('neurotransmitters')
def load_neurotransmitters(cfg, point_df, partner_df, ann):
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

    # Filter the points according to the ROI, which will cause the tbar predictions to be filtered, too.
    roiset = cfg['restrict-to-roi']['roiset']
    roi = cfg['restrict-to-roi']['roi']
    point_df, _ = restrict_synapses_to_roi(roiset, roi, point_df, partner_df)

    # We only care about annotations with a 'type', and only for bodies which exist in point_df.
    # Note:
    #   We do include bodies which contain only post-synapses.  Even though they lack
    #   pre-synapses (and therefore lack tbar NT predictions), we will still assign
    #   them a celltype_nt and consensus_nt.
    bodies = point_df['body'].unique()
    ann = ann.loc[ann['type'].notnull() & ann.index.isin(bodies), ['type']]

    with Timer("Loading tbar NT predictions", logger):
        tbar_nt = _load_tbar_neurotransmitters(path, cfg['rescale-coords'], cfg['translate-names'], point_df)

    if not cfg['ground-truth']:
        gt_df = None
    else:
        gt_df = pd.read_csv(cfg['ground-truth'])
        if not {*gt_df.columns} >= {'cell_type', 'ground_truth'}:
            raise RuntimeError("Neurotransmitter ground-truth table does not supply the necessary columns.")
        if 'split' not in tbar_nt:
            raise RuntimeError("Can't make use of your ground-truth because your point data does not contain a 'split' column.")

    with Timer("Computing groupwise NT predictions for bodies and cell types", logger):
        body_nt = _compute_body_neurotransmitters(
            tbar_nt, gt_df, ann,
            cfg['min-body-confidence'], cfg['min-body-presyn'], cfg['min-celltype-presyn']
        )
    tbar_nt = tbar_nt.drop(columns=['split'], errors='ignore')

    if cfg['export-mean-tbar-scores']:
        # For consistency with MANC, we also list ALL mean neurotransmitter
        # predictions as separate columns (neuprint properties).
        nt_cols = [col for col in tbar_nt.columns if col.startswith('nt')]
        body_nt = body_nt.merge(
            tbar_nt.groupby('body')[nt_cols].mean(),
            'left',
            on='body'
        )

        # NOTE:
        #   For manc, the body prediction was the one with the highest mean tbar score.
        #   But going forward, we pick the most frequent max tbar score.
        #
        # col_to_nt = {c: c.split('_')[1] for c in body_nt.columns}
        # body_nt['predicted_nt'] = body_nt.idxmax(axis=1).map(col_to_nt)

    if (path := cfg['experimental-groundtruth']):
        # Start with the celltype majority prediction...
        body_nt['consensus_nt'] = body_nt['celltype_predicted_nt']

        # Apply user-provided overrides (e.g. replace 'octopamine' with 'unclear').
        body_nt['consensus_nt'].update(body_nt['consensus_nt'].map(cfg['override-celltype-before-consensus']))

        # Overwrite cases where experimental groundtruth is available.
        exp_df = pd.read_csv(path)
        exp_map = exp_df.set_index('cell_type')['ground_truth']
        body_nt['consensus_nt'].update(body_nt['cell_type'].map(exp_map))

        if 'reference' in exp_df.columns:
            ref_map = exp_df.set_index('cell_type')['reference'].dropna()
            body_nt['nt_reference'] = body_nt['cell_type'].map(ref_map)

        if 'other_gt' in exp_df.columns:
            other_map = exp_df.set_index('cell_type')['other_gt'].dropna()
            body_nt['other_nt'] = body_nt['cell_type'].map(other_map)

        if 'other_ref' in exp_df.columns:
            other_ref_map = exp_df.set_index('cell_type')['other_ref'].dropna()
            body_nt['other_nt_reference'] = body_nt['cell_type'].map(other_ref_map)

    body_nt = body_nt.drop(columns=['cell_type'], errors='ignore')
    return tbar_nt, body_nt


def _load_tbar_neurotransmitters(path, rescale, translations, point_df):
    tbar_nt = feather.read_feather(path)

    # Rename columns pre_x, pre_y, pre_z -> x,y,z
    tbar_nt = tbar_nt.rename(columns={f'pre_{k}':k for k in 'xyz'})
    tbar_nt = tbar_nt.rename(columns={f'{k}_pre':k for k in 'xyz'})
    nt_cols = [col for col in tbar_nt.columns if col.startswith('nts')]

    # Discard extraneous columns
    cols = [*'xyz', *nt_cols]
    if 'split' in tbar_nt.columns:
        cols.append('split')
    tbar_nt = tbar_nt[cols]

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

    # Drop predictions which correspond to synapses we don't have
    # (due to ROI selection, for example).
    # Note:
    #   If there are synapses in point_df which are not present in the tbar
    #   predictions, they will have NaN predictions after this merge.
    presyn_df = point_df.query('kind == "PreSyn"')
    tbar_nt = presyn_df[['body']].merge(tbar_nt, 'left', on='point_id')
    return tbar_nt


def _compute_body_neurotransmitters(tbar_nt, gt_df, ann, min_body_conf, min_body_presyn, min_type_presyn):
    """
    FIXME: The output column names still need to be finalized.
    """
    # Append 'cell_type' column to point table
    tbar_nt = tbar_nt.merge(ann['type'].rename('cell_type'), 'left', on='body')

    # Determine top prediction (pred1) for each point
    nt_cols = [c for c in tbar_nt.columns if c.startswith('nt_')]
    nts = [c.split('_')[1] for c in nt_cols]
    tbar_nt['pred1'] = (
        tbar_nt[nt_cols]
        .rename(columns=dict(zip(nt_cols, nts)))
        .idxmax(axis=1)
    )
    confusion_df = _confusion_matrix(tbar_nt, gt_df, nts)
    tbar_nt = tbar_nt[['body', 'cell_type', 'pred1']]
    body_nt = _calc_group_predictions(tbar_nt, ann, confusion_df, gt_df, 'body')
    type_df = _calc_group_predictions(tbar_nt, ann, confusion_df, gt_df, 'cell_type')

    if 'confidence' in body_nt:
        # Apply thresholds to erase unreliable predictions.
        body_nt.loc[body_nt['confidence'] < min_body_conf, 'top_pred'] = 'unclear'
        body_nt.loc[body_nt['num_tbar_nt_predictions'] < min_body_presyn, 'top_pred'] = 'unclear'
        type_df.loc[type_df['num_tbar_nt_predictions'] < min_type_presyn, 'top_pred'] = 'unclear'

    # We don't return a separate table for celltype predictions.
    # Instead, append celltype prediction columns to the body table
    # (with duplicated values for bodies with matching cell types).
    type_cols = ['cell_type', 'num_tbar_nt_predictions', 'top_pred']
    if 'confidence' in type_df.columns:
        type_cols.append('confidence')

    body_nt = (
        body_nt
        .merge(
            type_df[type_cols],
            'left',
            on='cell_type',
            suffixes=['', '_celltype']
        )
        .rename(columns={
            'top_pred': 'predicted_nt',
            'top_pred_celltype': 'celltype_predicted_nt',
            'confidence': 'predicted_nt_confidence',
            'confidence_celltype': 'celltype_predicted_nt_confidence',
            'num_tbar_nt_predictions': 'total_nt_predictions',
            'num_tbar_nt_predictions_celltype': 'celltype_total_nt_predictions',
        })
    )
    body_nt = body_nt.set_index('body')
    return body_nt


def _confusion_matrix(tbar_nt, gt_df, all_nts):
    if gt_df is None:
        return None

    # Generate 'ground_truth' column using cell_type and GT table
    gt_mapping = gt_df.set_index('cell_type')['ground_truth']
    tbar_gt = tbar_nt['cell_type'].map(gt_mapping)

    # Compute confusion matrix for the 'test' set only.
    confusion_df = (
        tbar_nt
        .assign(ground_truth=tbar_gt)
        .query('split != "train" and split != "validation" and not ground_truth.isnull()')
        .groupby(['ground_truth', 'pred1'])
        .size()
        .unstack(-1, 0.0)
        # We reindex to ensure that the confusion matrix has rows/columns
        # for all neurotransmitters in the ground_truth, even if some of
        # them aren't in the 'test' set.  (This can happen when working
        # with a small set of tbars, such as when working with a small ROI
        # or when using testing datasets.)
        .reindex(all_nts)
        .reindex(all_nts, axis=1)
    )

    # Normalize the rows
    confusion_df /= confusion_df.sum(axis=1).values[:, None]

    # NaNs might exist due to the reindex() above.
    confusion_df = confusion_df.fillna(0.0)
    return confusion_df


def _calc_group_predictions(pred_df, ann, confusion_df, gt_df, groupcol):
    """
    Args:
        pred_df:
            synapse prediction table with columns:
                ['body', 'cell_type', 'pred1']

        ann:
            body annotations table, just for the `type` column.
            This is used to ensure that the results include all bodies/types,
            even if some annotated bodies had no tbars and therefore are not
            present in pred_df.

        confusion_df:
            Confusion matrix of NT predictions, as a DataFrame
            with ground truth NT in the index and predicted NT in the columns.

        gt_df:
            Cell type NT ground truth, as a DataFrame with columns:
                ['cell_type', 'ground_truth']

        groupcol:
            Indicates whether to aggregate by 'body' or 'cell_type'
            to produce group-wise predications.

    Returns:
        DataFrame of group-wise predictions with columns (but 'body' is omitted if groupcol='cell_type'):
        ['cell_type', 'body', 'num_tbar_nt_predictions', 'confidence', 'top_pred', 'ground_truth']
    """
    assert groupcol in ('body', 'cell_type')

    # Compute the group NT predictions
    # (the most common NT prediction among its synapses)
    #
    # Notes:
    #   - value_counts().groupby().head(1) is much faster
    #     than groupby().apply(pd.Series.mode).
    #   - sort_values() isn't necessary, but it's used here
    #     to ensure predictable ordering in case of a tie.
    group_pred = (
        pred_df[[groupcol, 'pred1']]
        .value_counts(dropna=False)
        .rename('count')
        .reset_index(1)
        .sort_values(['count', 'pred1'], ascending=[False, True])
        .groupby(groupcol)
        .head(1)
        ['pred1']
        .rename('group_pred')
    )
    df = group_pred.to_frame()

    # Ensure that our final results will include all bodies (or types)
    # from the annotations, even if they weren't present in pred_df
    ann_types = ann['type'].rename('cell_type')
    if groupcol == 'body':
        # Append the 'cell_type' column
        assert df.index.name == ann_types.index.name == 'body'
        df = df.merge(ann_types, 'outer', on='body')
        assert df.index.name == ann_types.index.name == 'body'
    else:
        # Make sure all cell_types from ann are listed in df
        ann_types = ann_types.drop_duplicates()
        df = df.merge(ann_types, 'outer', on='cell_type').set_index('cell_type')

    df['num_tbar_nt_predictions'] = pred_df.groupby(groupcol)['pred1'].count()
    df['num_tbar_nt_predictions'].fillna(0, inplace=True)

    assert df.index.name == groupcol
    df = df.sort_index()

    # Without groundtruth, all we can provide are
    # the aggregated values -- no confidences
    if gt_df is None:
        # If there were no tbar NT predictions provided at all
        # for some bodies, those bodies get 'unclear' NT.
        df['group_pred'].fillna('unclear', inplace=True)

        # Rearrange/rename columns to match expected output
        df = df.rename(columns={'group_pred': 'top_pred'})
        cols = ['cell_type', 'body', 'num_tbar_nt_predictions', 'top_pred']
        if groupcol == 'cell_type':
            cols.remove('body')
        return df.reset_index()[cols]

    pred_df = pred_df.merge(group_pred, 'left', on=groupcol)

    # Extract each synapse's confusion score from confusion matrix,
    # using the group prediction as the 'ground_truth' NT.
    # (There is a 10x faster way to do this using Categoricals and numpy slicing,
    # but it's more verbose. This is good enough.)

    valid_rows = pred_df[['group_pred', 'pred1']].notnull().all(axis=1)
    group_pred_and_syn_pred = pred_df.loc[valid_rows, ['group_pred', 'pred1']]
    group_pred_and_syn_pred = pd.MultiIndex.from_frame(group_pred_and_syn_pred)
    pred_df.loc[valid_rows, 'confusion_score'] = confusion_df.stack().loc[group_pred_and_syn_pred].values
    df['mean_confusion'] = pred_df.groupby(groupcol)['confusion_score'].mean().fillna(0.0)

    # If there were no tbar NT predictions provided at all
    # for some bodies, those bodies get 'unclear' NT.
    df['group_pred'].fillna('unclear', inplace=True)

    # Append 'ground_truth' column where possible
    # (First reset index so it isn't lost in this merge.)
    df = df.reset_index().merge(gt_df, 'left', on='cell_type')

    # Rearrange/rename columns to match expected output
    df = df.rename(columns={'mean_confusion': 'confidence',
                            'group_pred': 'top_pred'})
    cols = ['cell_type', 'body', 'num_tbar_nt_predictions', 'confidence', 'top_pred', 'ground_truth']
    if groupcol == 'cell_type':
        cols.remove('body')
    return df[cols]
