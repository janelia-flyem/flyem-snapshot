"""
This file contains the implementation for loading synaptic-level neurotransmitter predictions
produced from the 'synful' method (Eckstein et al.), and computing body-level aggregations,
including 'confidence' scores (if cell type annotations are present and type-level groundtruth
is provided as a separate input table).

Additionally, a 'consensusNt' column is produced for each body, based on type-level aggregations
and (optionally) yet more type-level groundtruth ("experimental groundtruth") which is meant to
override any automated predictions.

The implementation below was developed in part to meet the specification shown below.
(Source: https://github.com/reiserlab/optic-lobe-connectome/issues/334#issuecomment-1906944951)

    Add multiple fields/columns to neuprint:

        1. synapse level predictions: already in neuprint, no changes needed?

        2. add column with confidence score at the bodyID and also cell type level.

        3. 'cell' (bodyID) level NT predictions:
           report the transmitter with largest share of predictions for the synapses of the body,
           but with some filtering: only include predictions for bodies with at least 50 pre-synapses
           and with a confidence score >= 0.5. If a bodyID doesn't meet both thresholds, then use NT = 'unclear'

        4. cell type level NT predictions:
           use same rule as above, but threshold at 100 synapses pooled across pre-synapses of all neurons of type.
           confidence score >=0.5. If a cell type doesn't meet both thresholds, then NT = 'unclear'
           (we expect >100 cell types like this, most are VPNs, this is fine, and central brain data will fix these
           in later releases). In the cases where there is only 1 neuron per type (per side), then should use the
           50 synapse per-cell threshold so there is no conflict between cell and cell type level for these cells.

        5. experimental data: transmitter information from high confidence experimental data, for the initial
           release this will be the ground truth data and new FISH data + Dm3a/b (accidentally left out of grown
           truth data). We have a table with source that will go in the paper. Jan suggested that data source
           should also go into neuprint. In which case the sources will be: "Nern et al. 2024" or "Davis et al. 2020"
           and to future proof this we should expect that more than one source could go into this string,
           e.g. "Nern et al. 2024; Nern et al. 2025"

        6. consensus transmitter: this is the cell type level prediction (4) but replaced by the experimental
           result if there is a conflict with the experimental data. Based on GMR comment -- "these should be
           at least as reliable as an antibody" **therefore the OA/5HT/DA should all go to "unclear" unless we
           have experimental data.**

        7. "other transmission" this field will only contain experimental data, and will be used to handle the
           few known cases of co-transmission, including peptidergic transmission. We expect more data like these
           in the near future. This field could also be a comma or semi-colon separated string, as we could have
           more than one peptide, etc. Simplest implementation would have an additional source field for "other
           transmission" or we could put everything into the source field requested in (5) above.
"""
import os
import re
import copy
import shutil
import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter
from neuclease.util import timed, encode_coords_to_uint64, camelcase_to_snakecase

from ..util.util import replace_object_nan_with_none, restrict_synapses_to_roi
from ..util.checksum import checksum
from ..caches import cached, SerializerBase, cache_dataframe

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
                "Must contain columns like nt_gaba_prob or nts_8.gaba\n"
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
                "glut": "gluatmate",
                "oct": "octopamine",
                "da": "dopmaine",
                "ser": "serotonin",
                "5ht": "serotonin",
                "ach": "acetylcholine",
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
                "column will only use the override value, if at all.\n"
                "Example: octomamine: unclear",
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
                "The expected columns match the columns in our final output (but we also accept their camelCase equivalents).\n"
                "type, consensusNt, ntReference, otherNt, otherNtReference\n"
                "We also accept the same column names used in the groud-truth table: cell_type,ground_truth",
            "type": "string",
            "default": ""
        },
        "training-set-split-indicators": {
            "description":
                "Values in the 'split' column which indicate that a synapse is in the training set\n"
                "and thus should be excluded before calculating the confusion matrix.\n",
            "type": "array",
            "items": {"type": "string"},
            "default": ["train", "validation"]
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


class NeurotransmitterSerializer(SerializerBase):
    """
    Cache handling for load_transmitters()
    """

    def get_cache_key(self, cfg, synpoint_df, synpartner_df, ann):
        cfg = copy.copy(cfg)
        cfg['processes'] = 0
        cfg_hash = hex(checksum(cfg))
        arg_hash = hex(checksum([synpoint_df, synpartner_df, ann]))
        return f'nt-{cfg_hash}-{arg_hash}'

    def save_to_file(self, result, path):
        tbar_df, body_df, confusion_df = result
        if tbar_df is None:
            shutil.rmtree(path, ignore_errors=True)
            return
        os.makedirs(path, exist_ok=True)
        assert tbar_df.index.name == 'point_id'
        assert body_df.index.name == 'body'
        cache_dataframe(tbar_df.reset_index(), f'{path}/nt-tbar.feather')
        cache_dataframe(body_df.reset_index(), f'{path}/nt-body.feather')
        if confusion_df is not None:
            assert confusion_df.index.name == 'ground_truth'
            cache_dataframe(confusion_df.reset_index(), f'{path}/nt-confusion.feather')

    def load_from_file(self, path):
        tbar_df = feather.read_feather(f'{path}/nt-tbar.feather').set_index('point_id')
        body_df = feather.read_feather(f'{path}/nt-body.feather').set_index('body')
        if os.path.exists(f'{path}/nt-confusion.feather'):
            confusion_df = feather.read_feather(f'{path}/nt-confusion.feather').set_index('ground_truth')
        else:
            confusion_df = None
        return tbar_df, body_df, confusion_df


@PrefixFilter.with_context('neurotransmitters')
@cached(NeurotransmitterSerializer('neurotransmitters'))
def load_neurotransmitters(cfg, synpoint_df, synpartner_df, ann):
    """
    Load neurotransmitter tbar predictions from disk, filter/rescale/translate
    according to the user's configuration, and aggregate them into per-body
    and per-celltype predictions.

    Args:
        cfg:
            The ['inputs']['neurotransmitter'] section of the config.
        synpoint_df:
            Synapse point DataFrame, with 'point_id' index and 'body' column.
            Used to assign a body to each tbar prediction.
        synpartner_df:
            Synapse partner DataFrame, used to filter synapses by ROI
            before processing NT predictions.
        ann:
            Annotation table. Must have a 'type' column, indexed by body,

    Returns:
        tbar_df, body_df, confusion_df
    """
    if not (path := cfg['synister-feather']):
        return None, None, None

    # Filter the points according to the ROI, which will cause the tbar predictions to be filtered, too.
    roiset = cfg['restrict-to-roi']['roiset']
    roi = cfg['restrict-to-roi']['roi']
    synpoint_df, _ = restrict_synapses_to_roi(roiset, roi, synpoint_df, synpartner_df)

    # We only care about annotations with a 'type',
    # and only for bodies which exist in synpoint_df.
    #
    # Note:
    #   We DO include bodies which contain only post-synapses.  Even though they lack
    #   pre-synapses (and therefore lack tbar NT predictions), we will still assign
    #   them a celltype_nt and consensus_nt.
    bodies = synpoint_df['body'].unique()
    ann = ann.loc[ann['type'].notnull() & ann.index.isin(bodies), ['type']]

    tbar_df = _load_tbar_neurotransmitters(path, cfg['rescale-coords'], cfg['translate-names'], synpoint_df)
    body_df, confusion_df = _compute_body_neurotransmitters(cfg, tbar_df, ann)
    tbar_df = tbar_df.drop(columns=['split'], errors='ignore')
    return tbar_df, body_df, confusion_df


@timed("Loading tbar NT predictions", logger)
def _load_tbar_neurotransmitters(path, rescale, translations, synpoint_df):
    """
    Load the tbar NT predictions from the given path to a feather
    file with synister-style column names.
    Drop predictions which fall outside of the known synapse set
    (as listed in synpoint_df).
    """
    tbar_df = feather.read_feather(path)
    if tbar_df.index.name:
        tbar_df = tbar_df.reset_index()

    # Rename columns pre_x, pre_y, pre_z -> x,y,z
    tbar_df = tbar_df.rename(columns={f'pre_{k}':k for k in 'xyz'})
    tbar_df = tbar_df.rename(columns={f'{k}_pre':k for k in 'xyz'})
    nt_cols = [col for col in tbar_df.columns if re.match('nts|nt_', col)]
    if len(nt_cols) == 0:
        raise RuntimeError(f"Could not find any neurotransmiter probability columns in {path}")

    # Discard extraneous columns
    keep_cols = [*'xyz', *nt_cols, 'split', 'point_id']
    keep_cols = [*filter(lambda c: c in tbar_df.columns, keep_cols)]
    tbar_df = tbar_df[keep_cols]

    # Apply user's coordinate scaling factor.
    tbar_df.loc[:, [*'xyz']] = (tbar_df[[*'xyz']] * rescale).astype(np.int32)

    # The original table may have names like 'nts_8.glutamate' or 'nt_glut_prob',
    # but we'll convert that to standard form: 'nt_glutamate_prob'.
    nt_names = [re.sub(r"_prob.*$", "", c) for c in nt_cols]        # remove suffix (if any)
    nt_names = [re.sub(r".*[._]", "", c) for c in nt_names]         # remove prefix (if any)
    nt_names = [translations.get(n.lower(), n) for n in nt_names]   # standardize name
    renames = {
        c: 'nt_' + name + '_prob'
        for c, name in zip(nt_cols, nt_names)
    }
    tbar_df = tbar_df.rename(columns=renames)
    nt_cols = list(renames.values())

    if 'point_id' in tbar_df.columns:
        tbar_df['point_id'] = tbar_df['point_id'].astype(np.uint64)
    else:
        tbar_df['point_id'] = encode_coords_to_uint64(tbar_df[[*'zyx']].values)

    tbar_df = tbar_df.set_index('point_id')

    # Drop predictions which correspond to synapses we don't have
    # (for example, due to ROI selection or min-confidence).
    # Note:
    #   If there are synapses in synpoint_df which are not present in the tbar
    #   predictions, they will have NaN predictions after this merge.
    presyn_df = synpoint_df.query('kind == "PreSyn"')

    # The merge below will silently produce incorrect results if one index is signed and the other is unsigned.
    # In that case, pandas must be converting to float64 and losing precision.
    assert tbar_df.index.name == presyn_df.index.name == 'point_id'
    assert tbar_df.index.dtype == presyn_df.index.dtype == np.uint64
    tbar_df = presyn_df[['body', *'xyz']].merge(tbar_df.drop(columns=[*'xyz']), 'left', on='point_id')
    return tbar_df


@timed("Computing groupwise NT predictions for bodies and cell types", logger)
def _compute_body_neurotransmitters(cfg, tbar_df, ann):
    """
    Compute aggregate per-body NT prediction columns, including celltype prediction columns.
    The body prediction is defined as the most frequent NT prediction from the set of tbars
    in the body.  Similarly, the celltype prediction is the most frequent NT prediction
    among all tbars in all bodies belonging to celltype.

    If a table of 'ground-truth' (cell type -> NT) is provided in the config,
    then a confusion matrix is computed for the tbar predictions, and that matrix is used
    to compute a 'confidence' for each body and celltype aggregate prediction.

    Additionally, if high-confidence 'experimental-groundtruth' is provided in the config,
    then a 'consensus' column is populated with:

        - the groundtruth NT prediction (when the cell type is known and listed
          in the groundtruth table) OR
        - the celltype prediction (when the cell type is known but without groundtruth) OR
        - the body prediction (when no cell type is available)
    """
    if not cfg['ground-truth']:
        gt_df = None
    else:
        gt_df = pd.read_csv(cfg['ground-truth'])
        if not {*gt_df.columns} >= {'cell_type', 'ground_truth'}:
            raise RuntimeError("Neurotransmitter ground-truth table does not supply the necessary columns.")
        if 'split' not in tbar_df:
            msg = "Can't make use of your ground-truth because your point data does not contain a 'split' column."
            raise RuntimeError(msg)
        if gt_df['cell_type'].isnull().any() or gt_df['ground_truth'].isnull().any():
            raise RuntimeError("Neurotransmitter ground-truth table contains null values.")

    # Append 'cell_type' column to point table
    tbar_df = tbar_df.merge(ann['type'].rename('cell_type'), 'left', on='body')

    # Determine top prediction (pred1) for each point
    nt_cols = [c for c in tbar_df.columns if c.startswith('nt_')]
    nts = [c.split('_')[1] for c in nt_cols]
    tbar_df['pred1'] = (
        tbar_df[nt_cols]
        .rename(columns=dict(zip(nt_cols, nts)))
        .idxmax(axis=1)
    )
    confusion_df = _confusion_matrix(tbar_df, gt_df, nts, cfg['training-set-split-indicators'])
    tbar_df = tbar_df[['body', 'cell_type', 'pred1']]
    body_df = _calc_group_predictions(tbar_df, ann, confusion_df, gt_df, 'body')
    type_df = _calc_group_predictions(tbar_df, ann, confusion_df, gt_df, 'cell_type')

    if 'confidence' in body_df:
        # Apply thresholds to erase unreliable predictions.
        body_df.loc[body_df['confidence'] < cfg['min-body-confidence'], 'top_pred'] = 'unclear'
        body_df.loc[body_df['num_tbar_nt_predictions'] < cfg['min-body-presyn'], 'top_pred'] = 'unclear'
        type_df.loc[type_df['num_tbar_nt_predictions'] < cfg['min-celltype-presyn'], 'top_pred'] = 'unclear'

    body_df = _append_celltype_predictions_to_body_df(body_df, type_df)
    body_df = body_df.set_index('body')

    # For consistency with MANC, we can optionally list ALL mean neurotransmitter
    # predictions as separate columns (which become neuprint properties).
    if cfg['export-mean-tbar-scores']:
        nt_cols = [col for col in tbar_df.columns if col.startswith('nt')]
        body_df = body_df.merge(
            tbar_df.groupby('body')[nt_cols].mean(),
            'left',
            on='body'
        )

        # NOTE:
        #   For manc, the body prediction was the one with the highest mean tbar score.
        #   But going forward, we pick the most frequent max tbar score.
        #
        # col_to_nt = {c: c.split('_')[1] for c in body_df.columns}
        # body_df['predicted_nt'] = body_df.idxmax(axis=1).map(col_to_nt)

    _set_body_exp_gt_based_columns(cfg, body_df)
    replace_object_nan_with_none(body_df)
    return body_df, confusion_df


def _confusion_matrix(tbar_df, gt_df, all_nts, training_indicators):
    """
    Compute the confusion matrix for non-training tbar predictions
    in the given table, using given groundtruth NT mapping
    (cell_type, ground_truth).

    Args:
        tbar_df:
            tbar prediction table with columns (cell_type, ground_truth, split, pred1),
            where 'pred1' is the top NT prediction and ground_truth is the true NT.
            The 'split' column is used to distinguish between traning and
            non-training points.  We discard rows which are marked with any of the
            values in the 'training_indicators' list.
        gt_df:
            Table of known NT labels (cell_type, ground_truth).
        all_nts:
            The superset of NT names to include in the output rows/columns.
            We ask for this explicitly instead of merely using the values from the input data.
            This is convenient when we test this code with small subsets of data,
            in which not all neurotransmitters may be present but we want the output
            to have the expected rows/columns.
        training_indicators:
            List of values to look for in the 'split' column which indicate that a synapse is in the training set.
    Returns:
        DataFrame, indexed by 'ground_truth' neurotransmitter, with predicted neurotransmitter ('pred1') in the columns.

         Example (note the names of the index and column index):

            pred1          acetylcholine  dopamine      gaba  glutamate  histamine  octopamine  serotonin
            ground_truth
            acetylcholine       0.933063  0.034370  0.008257   0.012560   0.004381    0.001211   0.006158
            dopamine            0.000000  0.971299  0.002590   0.001295   0.000432    0.000000   0.024385
            gaba                0.029723  0.036046  0.893162   0.023048   0.001155    0.001538   0.015328
            glutamate           0.030444  0.032100  0.030863   0.879269   0.007699    0.004446   0.015179
            histamine           0.009424  0.010358  0.005934   0.010852   0.960560    0.001360   0.001511
            octopamine          0.003435  0.028827  0.002041   0.005029   0.001046    0.907194   0.052427
            serotonin           0.002505  0.095601  0.002104   0.003107   0.000100    0.003407   0.893176
    """
    if gt_df is None:
        return None

    # Generate 'ground_truth' column using cell_type and GT table
    gt_mapping = gt_df.set_index('cell_type')['ground_truth']
    tbar_gt = tbar_df['cell_type'].map(gt_mapping)

    # Compute confusion matrix for the 'test' set only.
    confusion_df = (
        tbar_df
        .assign(ground_truth=tbar_gt)
        .query('split not in @training_indicators and ground_truth.notnull()')
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
    Calculate aggregate predictions, either for each body or each celltype.
    (Since the calculation steps are the same in each case, we use this
    function for both per-body and per-celltype aggregation.)

    The aggregate body (or celltype) prediction is defined as the most
    frequent tbar NT prediction in each body (or celltype).

    Additionally a 'confidence' is computed for the aggregate prediction,
    defined as the mean confusion for the tbar predictions within the body
    (or celltype), according to the given confusion matrix.

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
    #   - To obtain the most common prediction in each group,
    #     value_counts().reset_index().drop_duplicates(groupcol)
    #     is much faster than groupby(groupcol).apply(pd.Series.mode).
    #   - sort_values() isn't strictly necessary here, but we use
    #     it to ensure predictable ordering in case of a tie.
    group_pred = (
        pred_df[[groupcol, 'pred1']]
        .value_counts(dropna=False)
        .rename('count')
        .reset_index()
        .sort_values(['count', 'pred1'], ascending=[False, True])
        .drop_duplicates(groupcol)
        .set_index(groupcol)
        ['pred1']
        .rename('group_pred')
    )
    df = group_pred.to_frame()

    # Cells with no type at all should not be given an aggregate celltype prediction.
    df.loc[df.index.isnull(), 'group_pred'] = None

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

    df['num_tbar_nt_predictions'] = pred_df.groupby(groupcol)['pred1'].count().fillna(0)

    assert df.index.name == groupcol
    df = df.sort_index()

    # Without groundtruth, all we can provide are
    # the aggregated values -- no confidences
    if gt_df is None:
        # If there were no tbar NT predictions provided at all
        # for some bodies, those bodies get 'unclear' NT.
        df['group_pred'] = df['group_pred'].fillna('unclear')

        # Rearrange/rename columns to match expected output
        df = df.rename(columns={'group_pred': 'top_pred'})
        cols = ['cell_type', 'body', 'num_tbar_nt_predictions', 'top_pred']
        if groupcol == 'cell_type':
            cols.remove('body')
        return df.reset_index()[cols]

    pred_df = pred_df.merge(group_pred, 'left', on=groupcol)

    # Extract each synapse's confusion score from confusion matrix,
    # assuming that the group prediction is considered the 'ground_truth' NT
    # for the purpose of calculating 'confusion'.

    # (BTW, there is a 10x faster way to compute this using Categoricals
    # and numpy slicing, but it's more verbose. This is good enough.)
    valid_rows = pred_df[['group_pred', 'pred1']].notnull().all(axis=1)
    group_pred_and_syn_pred = pred_df.loc[valid_rows, ['group_pred', 'pred1']]
    group_pred_and_syn_pred = pd.MultiIndex.from_frame(group_pred_and_syn_pred)
    pred_df.loc[valid_rows, 'confusion_score'] = confusion_df.stack().loc[group_pred_and_syn_pred].values

    assert df.index.name == groupcol
    df['mean_confusion'] = pred_df.groupby(groupcol)['confusion_score'].mean().fillna(0.0)

    # If there were no tbar NT predictions provided at all
    # for some bodies, those bodies get 'unclear' NT.
    df['group_pred'] = df['group_pred'].fillna('unclear')

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


def _append_celltype_predictions_to_body_df(body_df, type_df):
    """
    We don't return a separate table for celltype predictions.
    Instead, we append celltype prediction columns to the body table
    (with duplicated values for bodies with matching cell types).
    """
    type_cols = ['cell_type', 'num_tbar_nt_predictions', 'top_pred']
    if 'confidence' in type_df.columns:
        type_cols.append('confidence')

    body_df = (
        body_df
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
        .fillna({'total_nt_predictions': 0})
    )

    # If a cell has no assigned type yet, consider it to be the only member of its type.
    # Therefore, its celltype_total_nt_predictions is the same as its own prediction count.
    body_df = body_df.fillna({'celltype_total_nt_predictions': body_df['total_nt_predictions']})

    body_df = body_df.astype({
        'total_nt_predictions': np.int32,
        'celltype_total_nt_predictions': np.int32,
    })

    return body_df


def _set_body_exp_gt_based_columns(cfg, body_df):
    """
    Some columns in the body table depend on the 'experimental-groundtruth' input table.
    If that table is available, this function adds the corresponding columns,
    the most notable of which is 'consensus_nt'.

    Modifies body_df IN-PLACE.
    """
    if not (path := cfg['experimental-groundtruth']):
        return

    # Start with the celltype majority prediction...
    body_df['consensus_nt'] = body_df['celltype_predicted_nt']

    # But if a cell has no assigned type yet, consider it to be the only member of its type.
    # Therefore, give it a celltype_predicted_nt according to its own predicted_nt.
    body_df.loc[body_df['cell_type'].isnull(), 'consensus_nt'] = body_df.loc[body_df['cell_type'].isnull(), 'predicted_nt']

    # Apply user-provided overrides (e.g. replace 'octopamine' with 'unclear').
    body_df['consensus_nt'].update(body_df['consensus_nt'].map(cfg['override-celltype-before-consensus']))

    # Overwrite cases where experimental groundtruth is available.
    exp_df = pd.read_csv(path)
    exp_df = exp_df.rename(columns={'type': 'cell_type', 'ground_truth': 'consensus_nt'})
    exp_df = exp_df.rename(columns={c: camelcase_to_snakecase(c) for c in exp_df.columns})
    exp_df = exp_df.set_index('cell_type')

    body_df['consensus_nt'].update(body_df['cell_type'].map(exp_df['consensus_nt']))

    if 'nt_reference' in exp_df.columns:
        ref_map = exp_df['nt_reference'].dropna()
        body_df['nt_reference'] = body_df['cell_type'].map(ref_map)

    if 'other_nt' in exp_df.columns:
        other_map = exp_df['other_nt'].dropna()
        body_df['other_nt'] = body_df['cell_type'].map(other_map)

    if 'other_nt_reference' in exp_df.columns:
        other_ref_map = exp_df['other_nt_reference'].dropna()
        body_df['other_nt_reference'] = body_df['cell_type'].map(other_ref_map)
