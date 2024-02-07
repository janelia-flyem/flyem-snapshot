import sys
import pandas as pd


# FIXME: This should be a CLI argument or something.
OL_ROIS = ['LA(R)', 'ME(R)', 'LO(R)', 'LOP(R)', 'AME(R)']


def main():
    pred_df = pd.read_parquet(sys.argv[1])
    body_df, celltype_df = neuron_and_celltype_predictions(pred_df, OL_ROIS)
    body_df.to_csv('nt_pred_confidence_by_neuron.csv')
    celltype_df.to_csv('nt_pred_confidence_by_celltype.csv')


def neuron_and_celltype_predictions(pred_df, rois=None):
    """
    Given a table of synapses and their neurotransmitter predictions,
    produce a table of group-wise predictions (either body-wise or celltype-wise).
    Also compute a confidence score for each group, based
    on the mean confusion score for all synapses in each group.

    Args:
        pred_df:
            synapse prediction table with the following columns,
            where 'roi' is optional unless the 'roi' argument is used:
                ['bodyId', 'cell_type', 'pred1', 'ground_truth', 'split', 'roi']
        rois:
            Optional.  A list of rois to which the predictions will be restricted.
            If provided, then pred_df must have an 'roi' column.
    Returns:
        body_df, celltype_df
        Two DataFrames, for neuron-level predictions and celltype-level predictions.
    """
    # Compute confusion matrix for the 'test' set only.
    confusion_df = (
        pred_df
        .query('split != "train" and split != "validation" and not ground_truth.isnull()"')
        .groupby(['ground_truth', 'pred1'])
        .size()
        .unstack(-1, 0.0)
    )

    # Normalize the rows
    confusion_df /= confusion_df.sum(axis=1).values[:, None]

    gt_df = (
        pred_df[['cell_type', 'ground_truth']]
        .drop_duplicates()
        .query('not ground_truth.isnull() and not cell_type.isnull() and cell_type != "unnamed"')
        .reset_index(drop=True)
    )
    assert not gt_df['cell_type'].duplicated().any(), \
        "Each cell type must have only one ground truth transmitter or the code below won't work."

    if rois:
        pred_df = pred_df.query('roi in @rois')

    body_df = _calc_group_predictions(pred_df, confusion_df, gt_df, 'bodyId')
    celltype_df = _calc_group_predictions(pred_df, confusion_df, gt_df, 'cell_type')
    return body_df, celltype_df


def _calc_group_predictions(pred_df, confusion_df, gt_df, groupcol):
    """
    Args:
        pred_df:
            synapse prediction table with columns:
                ['bodyId', 'cell_type', 'pred1']

        confusion_df:
            Confusion matrix of NT predictions, as a DataFrame
            with ground truth NT in the index and predicted NT in the columns.

        gt_df:
            Cell type NT ground truth, as a DataFrame with columns:
                ['cell_type', 'ground_truth']

        groupcol:
            Indicates whether to aggregate by 'bodyId' or 'cell_type'
            to produce group-wise predications.

    Returns:
        DataFrame of group-wise predictions
    """
    assert groupcol in ('bodyId', 'cell_type')

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
        .value_counts()
        .rename('count')
        .reset_index(1)
        .sort_values(['count', 'pred1'], ascending=[False, True])
        .groupby(groupcol)
        .head(1)
        ['pred1'].rename('group_pred')
    )
    pred_df = pred_df.merge(group_pred, 'left', on=groupcol)

    # Extract each synapse's confusion score from confusion matrix,
    # using the group prediction as the 'ground_truth' NT.
    # (There is a 10x faster way to do this using Categoricals and numpy slicing,
    # but it's more verbose and this is good enough.)
    group_pred_and_syn_pred = pd.MultiIndex.from_frame(pred_df[['group_pred', 'pred1']])
    pred_df['confusion_score'] = confusion_df.stack().loc[group_pred_and_syn_pred].values

    # Assemble all desired output columns into a single DataFrame
    df = group_pred.to_frame()
    df['num_presyn'] = pred_df.groupby(groupcol).size()
    df['mean_confusion'] = pred_df.groupby(groupcol)['confusion_score'].mean()
    if groupcol == 'bodyId':
        df['cell_type'] = pred_df.groupby(groupcol)['cell_type'].agg('first')
    df = df.reset_index().merge(gt_df, 'left', on='cell_type')
    df = df.sort_values(groupcol, ignore_index=True)

    # Rearrange/rename columns to match expected output
    df = df.rename(columns={'mean_confusion': 'confidence_val',
                            'group_pred': 'top_pred'})
    cols = ['cell_type', 'bodyId', 'num_presyn', 'confidence_val', 'top_pred', 'ground_truth']
    if groupcol == 'cell_type':
        cols.remove('bodyId')

    return df[cols]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_file = 'nt_20230821_124849_pred_all_presyn_split_parsed_annotated.parquet'
        print(f'No file provided.  Trying {default_file}')
        sys.argv.append(default_file)

    main()
