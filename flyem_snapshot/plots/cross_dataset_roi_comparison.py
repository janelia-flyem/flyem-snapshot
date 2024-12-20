import numpy as np
import pandas as pd


def roi_completeness_comparison_bars(dataset_dfs, label_fudge=13):
    """
    Produce a bar chart for comparing datasets with ROIs in common.
    For left/right ROIs pairs, the mean is plotted, with whiskers to
    indicate the spread between the two.  Text labels are added to
    the whiskers to indicate whether 'L' or 'R' has the better completeness.

    Args:
        dataset_dfs:
            dict of {dataset_name: DataFrame},
            where the dataframe has columns for 'neuropil' and 'completeness'.
            Left/right pairs of neuropils will be inferred via their suffix, e.g. '_L' or '(L)'.
        label_fudge:
            Tragically, the L/R annotations can't be easily placed using
            normalized units that would work regardless of plot height.
            If we want those annotations, we'll have to tweak their exact
            positioning after the height of the plot has already been set.

    Returns:
        plotly Figure
        You will likely desire to update the layout to adjust the title, legend position, etc.
    """
    paired_dfs = {}
    for dset_name, df in dataset_dfs.items():
        paired_dfs[dset_name] = _make_paired_df(df)

    combined_df = pd.concat(
        paired_dfs.values(),
        axis=0,
        keys=paired_dfs.keys(),
        names=['dataset', 'roi_base']
    )

    return _make_comparison_plot(combined_df, label_fudge)


def _make_paired_df(df):
    """
    Condense the roi completeness table (having separate rows for each roi)
    into a 'paired' table in which corresponding left/right ROIs appear on
    the same row, with separate columns for 'L' and 'R' (or 'central' for
    ROIs with no L/R suffix).
    """
    assert set(df.columns) >= {'neuropil', 'completeness'}

    # Replace (L) -> _L, (R) -> _R
    df['neuropil'] = df['neuropil'].str.replace(r'\((L|R)\)', r'_\1', regex=True)
    df[['roi_base', 'roi_side']] = df['neuropil'].str.extract(r'^(.*?)(_([^_]+))?$')[[0, 2]].values
    df = df.set_index('neuropil')

    paired_df = 100 * df.fillna('central').set_index(['roi_base', 'roi_side'])['completeness'].unstack()

    paired_df['mean'] = paired_df.mean(axis=1)
    paired_df['residual'] = paired_df[['L', 'R', 'central']].max(axis=1) - paired_df['mean']
    paired_df = paired_df.loc[paired_df.index != "None"]

    paired_df['min_side'] = np.where(paired_df.eval('R < L'), 'R', 'L')
    paired_df['max_side'] = np.where(paired_df.eval('R > L'), 'R', 'L')

    paired_df.loc[paired_df['L'].isnull(), 'min_side'] = ''
    paired_df.loc[paired_df['L'].isnull(), 'max_side'] = ''

    return paired_df


def _make_comparison_plot(combined_df, label_fudge=13):
    import plotly.express as px

    fig = px.bar(
        combined_df.reset_index(),
        y='roi_base',
        x='mean',
        color='dataset',
        title='L/R Mean Completeness',
        barmode='group',
        error_x='residual',
        orientation='h'
    )

    fig.update_layout(
        height=1000,
        width=700,
        title={
            'x': 0.5,
            'text': 'Synaptic Connection Completeness',
            'xanchor': 'center',
            'xref': 'paper',
        },
        yaxis={'autorange': 'reversed', 'title': None},
        yaxis2={'autorange': 'reversed', 'title': None},
        legend=dict(
            title_text='',
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.96
        )
    )

    # Add subtitle as an annotation
    fig.add_annotation(
        text="mean of (L) & (R) - whiskers show difference",
        xref="paper", yref="paper",
        xanchor='center',
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=12)
    )

    yshifts = {
        1: (0,),
        2: (label_fudge, -label_fudge),
        3: (label_fudge, 0, -label_fudge),
        4: (2 * label_fudge, label_fudge, -label_fudge, -2 * label_fudge)
    }

    # Add text labels to the right side of the whiskers
    datasets = combined_df.index.get_level_values(0).unique()
    for dset, yshift in zip(datasets, yshifts[len(datasets)]):
        for i, row in combined_df.loc[dset].iterrows():
            if row['max_side']:
                fig.add_annotation(
                    text=row['max_side'],
                    x=row['mean'] + row['residual'],
                    y=i,
                    yshift=yshift,
                    showarrow=False,
                    font=dict(size=11),
                    xanchor='left',
                    yanchor='middle',
                    # yref='foo'
                )
            if row['min_side']:
                fig.add_annotation(
                    text=row['min_side'],
                    x=row['mean'] - row['residual'],
                    y=i,
                    yshift=yshift,
                    showarrow=False,
                    font=dict(size=11),
                    xanchor='right',
                    yanchor='middle',
                    # yref='paper'
                )

    return fig
