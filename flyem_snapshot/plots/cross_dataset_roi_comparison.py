import numpy as np
import pandas as pd


def roi_completeness_comparison_bars(dataset_dfs, include_labels=True, legend_position='first', label_fudge=13):
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
        include_labels:
            Whether to include L/R text labels on the error bars.
        legend_position:
            Where to place the legend. Either 'first' (top-right of first subplot) 
            or 'second' (top-right of second subplot).
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

    roi_bases = sorted(set(chain(*(df.index for df in paired_dfs.values()))))
    for key, df in list(paired_dfs.items()):
        paired_dfs[key] = df.reindex(roi_bases)

    combined_df = pd.concat(
        paired_dfs.values(),
        axis=0,
        keys=paired_dfs.keys(),
        names=['dataset', 'roi_base']
    )

    return _make_comparison_plot(combined_df, include_labels, legend_position, label_fudge)


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
    df = df.sort_index()

    paired_df = (
        100 *
        df.assign(roi_side=df['roi_side'].fillna('central'))
        .set_index(['roi_base', 'roi_side'])
        ['completeness']
        .unstack()
    )

    paired_df['mean'] = paired_df.mean(axis=1)
    paired_df['residual'] = paired_df[['L', 'R', 'central']].max(axis=1) - paired_df['mean']
    paired_df = paired_df.loc[paired_df.index != "None"]

    paired_df['min_side'] = np.where(paired_df.eval('R < L'), 'R', 'L')
    paired_df['max_side'] = np.where(paired_df.eval('R > L'), 'R', 'L')

    paired_df.loc[paired_df['L'].isnull(), 'min_side'] = ''
    paired_df.loc[paired_df['L'].isnull(), 'max_side'] = ''

    return paired_df


def _make_comparison_plot(combined_df, include_labels=True, legend_position='first', label_fudge=13):
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Get unique ROI bases and split them into two halves
    roi_bases = combined_df.index.get_level_values(1).unique()
    mid_point = len(roi_bases) // 2
    first_half_rois = roi_bases[:mid_point]
    second_half_rois = roi_bases[mid_point:]
    
    # Split the data
    first_half_df = combined_df[combined_df.index.get_level_values(1).isin(first_half_rois)]
    second_half_df = combined_df[combined_df.index.get_level_values(1).isin(second_half_rois)]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.1
    )
    
    datasets = combined_df.index.get_level_values(0).unique()
    colors = px.colors.qualitative.Plotly[:len(datasets)]
    
    # Add traces for first subplot
    for i, dataset in enumerate(datasets):
        if dataset in first_half_df.index.get_level_values(0):
            data = first_half_df.loc[dataset].reset_index()
            error_x = data['residual']
            fig.add_trace(
                go.Bar(
                    y=data['roi_base'],
                    x=data['mean'],
                    error_x=dict(type='data', array=error_x, visible=True),
                    name=dataset,
                    orientation='h',
                    marker_color=colors[i],
                    showlegend=True,
                    legendgroup=dataset
                ),
                row=1, col=1
            )
    
    # Add traces for second subplot
    for i, dataset in enumerate(datasets):
        if dataset in second_half_df.index.get_level_values(0):
            data = second_half_df.loc[dataset].reset_index()
            error_x = data['residual']
            fig.add_trace(
                go.Bar(
                    y=data['roi_base'],
                    x=data['mean'],
                    error_x=dict(type='data', array=error_x, visible=True),
                    name=dataset,
                    orientation='h',
                    marker_color=colors[i],
                    showlegend=False,  # Don't show legend for second subplot traces
                    legendgroup=dataset
                ),
                row=1, col=2
            )

    # Calculate legend position based on legend_position parameter
    if legend_position == 'first':
        # Position legend in top-right of first subplot
        legend_x = 0.43  # Just to the right of first subplot
        legend_xanchor = "right"
    elif legend_position == 'second':
        # Position legend in top-right of second subplot
        legend_x = 0.98  # Far right of second subplot
        legend_xanchor = "right"
    else:
        # Default to first subplot
        legend_x = 0.48
        legend_xanchor = "right"

    # Update layout
    fig.update_layout(
        height=1000,
        width=1400,  # Increased width for two subplots
        title={
            'x': 0.5,
            'text': 'Synaptic Connection Completeness',
            'xanchor': 'center',
            'xref': 'paper',
        },
        barmode='group',
        legend=dict(
            title_text='',
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor=legend_xanchor,
            x=legend_x
        )
    )
    
    # Update y-axes to reverse order
    fig.update_yaxes(autorange='reversed', title=None, row=1, col=1)
    fig.update_yaxes(autorange='reversed', title=None, row=1, col=2)

    # # Add subtitle as an annotation
    # fig.add_annotation(
    #     text="mean of (L) & (R) - whiskers show difference",
    #     xref="paper", yref="paper",
    #     xanchor='center',
    #     x=0.5, y=1.03,
    #     showarrow=False,
    #     font=dict(size=12)
    # )

    yshifts = {
        1: (0,),
        2: (label_fudge, -label_fudge),
        3: (label_fudge, 0, -label_fudge),
        4: (2 * label_fudge, label_fudge, -label_fudge, -2 * label_fudge)
    }

    if not include_labels:
        return fig

    # Add text labels to the whiskers for first subplot
    for dset, yshift in zip(datasets, yshifts[len(datasets)]):
        if dset in first_half_df.index.get_level_values(0):
            for i, row in first_half_df.loc[dset].iterrows():
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
                        xref='x1', yref='y1'
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
                        xref='x1', yref='y1'
                    )

    # Add text labels to the whiskers for second subplot
    for dset, yshift in zip(datasets, yshifts[len(datasets)]):
        if dset in second_half_df.index.get_level_values(0):
            for i, row in second_half_df.loc[dset].iterrows():
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
                        xref='x2', yref='y2'
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
                        xref='x2', yref='y2'
                    )

    return fig
