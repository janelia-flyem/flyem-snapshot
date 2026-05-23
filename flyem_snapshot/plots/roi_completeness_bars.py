"""
Misc. bar plots for displaying synaptic capture ("completeness") rates by ROI.

A couple of these are used in the MaleCNS paper, in Fig 1(d) and Fig S9(g).
"""
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

#SNAPSHOT_TAG = 'male-cns-v0.9'
SNAPSHOT_TAG = 'male-cns-v1.0'
SNAPSHOT_DIR = f'/groups/flyem/data/scratchspace/flyemflows/cns-full/snapshots/{SNAPSHOT_TAG}'

def load_completeness_stats(stat='conn'):
    assert stat in ('conn', 'presyn', 'postsyn', 'synweight')
    major_traced_conn = pd.read_csv(f'{SNAPSHOT_DIR}/reports/major/major-compartments/csv/major-compartments-cumulative-traced_{stat}_frac-by-status.csv')
    major_traced_conn = major_traced_conn.rename(columns={'name': 'roi'})

    cb_traced_conn = pd.read_csv(f'{SNAPSHOT_DIR}/reports/primary/central-brain/csv/central-brain-cumulative-traced_{stat}_frac-by-status.csv')
    oll_traced_conn = pd.read_csv(f'{SNAPSHOT_DIR}/reports/primary/optic-left/csv/optic-left-cumulative-traced_{stat}_frac-by-status.csv')
    olr_traced_conn = pd.read_csv(f'{SNAPSHOT_DIR}/reports/primary/optic-right/csv/optic-right-cumulative-traced_{stat}_frac-by-status.csv')
    vnc_traced_conn = pd.read_csv(f'{SNAPSHOT_DIR}/reports/primary/vnc/csv/vnc-cumulative-traced_{stat}_frac-by-status.csv')

    cb_traced_conn = cb_traced_conn.rename(columns={'name': 'roi'})
    oll_traced_conn = oll_traced_conn.rename(columns={'name': 'roi'})
    olr_traced_conn = olr_traced_conn.rename(columns={'name': 'roi'})
    vnc_traced_conn = vnc_traced_conn.rename(columns={'name': 'roi'})

    # The columns are sorted from best status to worst,
    # and cumulative from left to right.
    # The 'Leaves' status (like PRT, but the neuron leaves the volume)
    # is the lowest status we consider traced, so that's what we'll use.
    major_traced_conn['traced_frac'] = major_traced_conn['Leaves']
    cb_traced_conn['traced_frac'] = cb_traced_conn['Leaves']
    oll_traced_conn['traced_frac'] = oll_traced_conn['Leaves']
    olr_traced_conn['traced_frac'] = olr_traced_conn['Leaves']
    vnc_traced_conn['traced_frac'] = vnc_traced_conn['Leaves']

    major_traced_conn['compartment'] = 'CNS Major Compartments'
    cb_traced_conn['compartment'] = 'Central Brain'
    oll_traced_conn['compartment'] = 'Optic Left'
    olr_traced_conn['compartment'] = 'Optic Right'
    vnc_traced_conn['compartment'] = 'VNC'

    traced_conn = pd.concat([major_traced_conn, cb_traced_conn, oll_traced_conn, olr_traced_conn, vnc_traced_conn], ignore_index=True)
    traced_conn['compartment'] = traced_conn['compartment'].astype(
        pd.CategoricalDtype(categories=['CNS Major Compartments', 'Central Brain', 'Optic Right', 'Optic Left', 'VNC'])
    )

    traced_conn = traced_conn[['roi', 'PreSyn', 'PostSyn', 'traced_frac', 'compartment']]
    traced_conn = traced_conn.sort_values(['compartment', 'traced_frac'], ascending=[True, False])

    (traced_conn
        .sort_values(['compartment', 'roi'])
        .rename(columns={'traced_frac': f'{stat}_traced_frac'})
        .to_csv(f'{SNAPSHOT_TAG}-{stat}-traced-frac-by-roi.csv', index=False, header=True)
    )

    return traced_conn


def pivot_completeness_stats(combined_df):
    """
    Starting with completeness stats like this:

        conn_frac  presyn_frac  postsyn_frac             compartment
        roi                                                                           
        Optic(R)           0.537940     0.949363      0.553027  CNS Major Compartments
        Optic(L)           0.429726     0.928825      0.451147  CNS Major Compartments
        VNC                0.372337     0.910941      0.402623  CNS Major Compartments
        CentralBrain       0.351027     0.945107      0.368483  CNS Major Compartments
        CV                 0.055718     0.394348      0.296827  CNS Major Compartments
        ...                     ...          ...           ...                     ...
        LegNp(T3)(L)       0.345695     0.883404      0.385766                     VNC
        NTct(UTct-T1)(R)   0.329429     0.959145      0.342702                     VNC
        LegNp(T1)(L)       0.289930     0.873428      0.322201                     VNC
        mVAC(T1)(R)        0.282278     0.754983      0.348611                     VNC
        LegNp(T1)(R)       0.246136     0.871619      0.274688                     VNC

        [108 rows x 4 columns]

    Infer the 'side' of each ROI as being either left/right/center, and pivot the values into multi-index columns as shown here:

        stat                          presyn_frac                     postsyn_frac                     conn_frac                    
        roi_side                                C         L         R            C         L         R         C         L         R
        compartment     roi_base                                                                                                    
        optic lobes     AME                   NaN  0.954734  0.977971          NaN  0.454463  0.561643       NaN  0.437951  0.553877
                        LA                    NaN  0.257651  0.330016          NaN  0.324933  0.355772       NaN  0.123471  0.148687
                        LO                    NaN  0.958517  0.978119          NaN  0.406969  0.513519       NaN  0.391761  0.503965
                        LOP                   NaN  0.962538  0.980403          NaN  0.493748  0.606016       NaN  0.476996  0.595859
                        ME                    NaN  0.948552  0.974824          NaN  0.469888  0.569827       NaN  0.450187  0.558654
        central brain      AB                    NaN  0.924528  0.945310          NaN  0.406398  0.395088       NaN  0.380121  0.374550
                        AL                    NaN  0.866817  0.920170          NaN  0.550216  0.599870       NaN  0.485075  0.555988
                        AOTU                  NaN  0.969175  0.980567          NaN  0.536734  0.604495       NaN  0.523200  0.595287
                        ATL                   NaN  0.957735  0.951397          NaN  0.278415  0.242573       NaN  0.267590  0.231621
                        AVLP                  NaN  0.972581  0.983461          NaN  0.358179  0.468686       NaN  0.349585  0.462604
                        BU                    NaN  0.956040  0.965253          NaN  0.361743  0.352135       NaN  0.347197  0.341303
                        CA                    NaN  0.951366  0.949432          NaN  0.499993  0.565230       NaN  0.485370  0.547163
                        CAN                   NaN  0.949436  0.950837          NaN  0.363678  0.340472       NaN  0.346012  0.325070
                        CRE                   NaN  0.967733  0.954339          NaN  0.324912  0.312167       NaN  0.315946  0.299533
                        EB               0.972892       NaN       NaN     0.793210       NaN       NaN  0.774029       NaN       NaN
                        EPA                   NaN  0.964804  0.960466          NaN  0.325423  0.307614       NaN  0.314769  0.296096
                        FB               0.970318       NaN       NaN     0.563674       NaN       NaN  0.548905       NaN       NaN
                        FLA                   NaN  0.894876  0.920523          NaN  0.287803  0.287457       NaN  0.260076  0.265069
                        GNG              0.907709       NaN       NaN     0.353665       NaN       NaN  0.322026       NaN       NaN
                        GOR                   NaN  0.948815  0.953737          NaN  0.318491  0.317876       NaN  0.303077  0.304685
                        IB               0.964741       NaN       NaN     0.279752       NaN       NaN  0.271007       NaN       NaN
                        ICL                   NaN  0.969658  0.962211          NaN  0.300879  0.289492       NaN  0.292648  0.279613
                        IPS                   NaN  0.966834  0.963782          NaN  0.340494  0.348903       NaN  0.329979  0.337145
                        LAL                   NaN  0.977091  0.974662          NaN  0.345307  0.342736       NaN  0.338593  0.335499
                        LH                    NaN  0.963376  0.967241          NaN  0.253664  0.320778       NaN  0.245543  0.312166
                        NO               0.974762       NaN       NaN     0.828064       NaN       NaN  0.811647       NaN       NaN
                        PB               0.977757       NaN       NaN     0.445908       NaN       NaN  0.439121       NaN       NaN
                        PED                   NaN  0.976325  0.978548          NaN  0.740817  0.734891       NaN  0.723479  0.720526
                        PLP                   NaN  0.962210  0.959706          NaN  0.298341  0.348526       NaN  0.287752  0.335938
                        PRW              0.898313       NaN       NaN     0.293214       NaN       NaN  0.264451       NaN       NaN
                        PVLP                  NaN  0.969862  0.978217          NaN  0.413819  0.504168       NaN  0.402688  0.495007
                        SAD              0.867386       NaN       NaN     0.386553       NaN       NaN  0.335674       NaN       NaN
                        SCL                   NaN  0.961653  0.961377          NaN  0.241337  0.276987       NaN  0.232781  0.268155
                        SIP                   NaN  0.959827  0.937293          NaN  0.288677  0.290042       NaN  0.278414  0.275134
                        SLP                   NaN  0.960532  0.950938          NaN  0.242644  0.284109       NaN  0.233978  0.273059
                        SMP                   NaN  0.966456  0.963440          NaN  0.287475  0.273789       NaN  0.278915  0.264743
                        SPS                   NaN  0.967362  0.966164          NaN  0.312597  0.312544       NaN  0.303077  0.302688
                        VES                   NaN  0.967481  0.963840          NaN  0.335162  0.320363       NaN  0.324873  0.309695
                        WED                   NaN  0.968040  0.978054          NaN  0.325704  0.383740       NaN  0.316043  0.376803
                        a'L                   NaN  0.968601  0.950491          NaN  0.645553  0.564167       NaN  0.630362  0.541342
                        aL                    NaN  0.964031  0.953579          NaN  0.745931  0.734469       NaN  0.721983  0.704599
                        b'L                   NaN  0.960599  0.916917          NaN  0.774057  0.660667       NaN  0.742733  0.608526
                        bL                    NaN  0.966974  0.951023          NaN  0.787287  0.689425       NaN  0.762493  0.657661
                        gL                    NaN  0.966055  0.957938          NaN  0.716374  0.723997       NaN  0.692602  0.695566
        ventral nerve cord ANm              0.943687       NaN       NaN     0.478288       NaN       NaN  0.455329       NaN       NaN
                        HTct(UTct-T3)         NaN  0.973933  0.962951          NaN  0.415793  0.391380       NaN  0.405930  0.378268
                        IntTct           0.945243       NaN       NaN     0.371332       NaN       NaN  0.351906       NaN       NaN
                        LTct             0.949516       NaN       NaN     0.405758       NaN       NaN  0.387810       NaN       NaN
                        LegNp(T1)             NaN  0.873428  0.871619          NaN  0.322201  0.274688       NaN  0.289930  0.246136
                        LegNp(T2)             NaN  0.915489  0.927507          NaN  0.432337  0.414964       NaN  0.398713  0.386225
                        LegNp(T3)             NaN  0.883404  0.883880          NaN  0.385766  0.393780       NaN  0.345695  0.351523
                        NTct(UTct-T1)         NaN  0.967472  0.959145          NaN  0.358517  0.342702       NaN  0.347613  0.329429
                        Ov                    NaN  0.968664  0.960612          NaN  0.570467  0.521585       NaN  0.554129  0.502642
                        WTct(UTct-T2)         NaN  0.964503  0.951595          NaN  0.503928  0.470692       NaN  0.487257  0.449298
                        mVAC(T1)              NaN  0.822140  0.754983          NaN  0.412142  0.348611       NaN  0.354610  0.282278
                        mVAC(T2)              NaN  0.879573  0.883558          NaN  0.471492  0.484291       NaN  0.423622  0.437161
                        mVAC(T3)              NaN  0.869278  0.934035          NaN  0.565869  0.588327       NaN  0.499859  0.552791
    """
    combined_df = combined_df.rename(
        columns={
            'conn_traced_frac': 'conn_frac',
            'presyn_traced_frac': 'presyn_frac',
            'postsyn_traced_frac': 'postsyn_frac',
        }
    )
    combined_df['compartment'] = combined_df['compartment'].map({
        'CNS Major Compartments': 'major compartments',
        'Central Brain': 'central brain',
        'VNC': 'ventral nerve cord',
        'Optic Left': 'optic lobes',
        'Optic Right': 'optic lobes',
    })

    side_df = (
        combined_df
        .reset_index()
        .set_index(['roi_base', 'roi_side'])[['presyn_frac', 'postsyn_frac']].rename_axis('stat', axis=1).unstack()
    #    .merge(cns_df[['compartment', 'roi_base']], 'left', on='roi_base')
    )
    # side_df['compartment'] = side_df['compartment'].astype(pd.CategoricalDtype(categories=['Optic Right', 'Optic Left', 'Central Brain', 'VNC'], ordered=True))
    # side_df = side_df.sort_values(['compartment', 'roi_base'], ascending=True, ignore_index=True)
    base_to_compartment = combined_df.drop_duplicates('roi_base').set_index('roi_base')['compartment']

    compartment_dtype = pd.CategoricalDtype(categories=['major compartments', 'optic lobes', 'central brain', 'ventral nerve cord'], ordered=True)
    compartment_index = side_df.index.map(base_to_compartment).rename('compartment')
    compartment_index = compartment_index.astype(compartment_dtype)

    side_df = side_df.set_index(compartment_index, append=True)
    side_df = side_df.swaplevel(0, 1)
    side_df = side_df.sort_index()
    return side_df


def roi_completeness(traced_conn, title=None, ylabel='completion rate [%]'):
    # For this plot, we just want the primary ROIs.
    traced_conn = traced_conn.query('compartment != "CNS Major Compartments"')

    #traced_conn = traced_conn.sort_values('traced_frac', ascending=False)
    widths = traced_conn['PostSyn'] / traced_conn['PostSyn'].sum()
    weighted_avg = 100 * (traced_conn['traced_frac'] * traced_conn['PostSyn']).sum() / traced_conn['PostSyn'].sum()

    x_positions = np.cumsum([0] + list(widths[:-1])) + (widths / 2)

    # Define color mapping for compartments - dark theme
    compartment_colors = {
        'Central Brain': '#228c43',
        'Optic Right': '#572c87',
        'Optic Left': '#8b67ad',
        'VNC': '#37658b'
    }
    
    # Map colors to each ROI based on its compartment
    bar_colors = [compartment_colors.get(comp, 'purple') for comp in traced_conn['compartment']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_positions,
        y=100 * traced_conn['traced_frac'],
        width=widths,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=0.1),
        ),
        name='Traced Fraction',
        hovertext=traced_conn['roi'],
        hovertemplate='<b>%{hovertext}</b><br>Compartment: %{text}<br>Traced Fraction: %{y:.3f}<br>PostSyn: %{customdata}<extra></extra>',
        customdata=traced_conn['PostSyn'],
        #text=traced_conn['compartment']
    ))

    # Add horizontal line for weighted average
    fig.add_hline(
        y=weighted_avg,
        line=dict(color='gray', width=1, dash='dot'),
        annotation_text=f'                       average: {weighted_avg:.1f} %',
        annotation_position="top left"
    )

    # Calculate midpoints for each compartment group
    compartment_midpoints = {}
    for compartment in traced_conn['compartment'].cat.categories:
        comp_mask = traced_conn['compartment'] == compartment
        if comp_mask.any():
            comp_widths = widths[comp_mask]
            comp_positions = x_positions[comp_mask]
            # Find the range of this compartment's bars
            min_pos = comp_positions.min() - comp_widths[comp_mask].iloc[0] / 2
            max_pos = comp_positions.max() + comp_widths[comp_mask].iloc[-1] / 2
            compartment_midpoints[compartment] = (min_pos + max_pos) / 2

    # Add custom compartment labels
    for compartment, midpoint in compartment_midpoints.items():
        fig.add_annotation(
            x=midpoint,
            y=-4,  # Position below x-axis
            text=compartment.lower(),
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            xanchor="center"
        )

    # Update layout
    y_max = 100 * np.ceil(traced_conn['traced_frac'].max() / .2) * .2

    fig.update_layout(
        title_text=title,
        xaxis=dict(
            title='',  # Remove default x-axis title
            showticklabels=False,  # Remove x-axis labels
            showgrid=False,
            ticks='',
            ticklen=0
        ),
        yaxis=dict(
            title=ylabel,
            # Extended lower bound to accommodate labels
            range=[-15, y_max],
            showgrid=False,  # No grid lines
            ticks='outside',
            tickvals=list(range(0, int(y_max + 1), 20)),
            ticklen=4
        ),
        width=600,
        height=400,
        template='plotly_white',
        margin=dict(b=60),  # Extra bottom margin for custom labels
        bargap=0  # No gaps between bars
    )

    return fig


def create_overlaid_bars(presyn_data, postsyn_data, conn_data, title=None):
    """Create overlaid bar chart with three different statistics"""

    # For this plot, we just want the primary ROIs.
    presyn_data = presyn_data.query('compartment != "CNS Major Compartments"')
    postsyn_data = postsyn_data.query('compartment != "CNS Major Compartments"')
    conn_data = conn_data.query('compartment != "CNS Major Compartments"')

    conn_data = conn_data.sort_values(['compartment', 'traced_frac'], ascending=[True, False])
    presyn_data = presyn_data.set_index('roi').reindex(conn_data['roi']).reset_index()
    postsyn_data = postsyn_data.set_index('roi').reindex(conn_data['roi']).reset_index()

    # Use conn_data as the base for positioning (assuming all have same ROIs)
    widths = conn_data['PostSyn'] / conn_data['PostSyn'].sum()
    x_positions = np.cumsum([0] + list(widths[:-1])) + (widths / 2)
    
    # Define color mapping for compartments
    compartment_colors = {
        'Central Brain': 'purple',
        'Optic Left': 'green',
        'Optic Right': 'blue',
        'VNC': 'orange'
    }
    
    fig = go.Figure()

    palette = px.colors.qualitative.Set2
    datasets = [
        (presyn_data, 'PreSyn', 1.0, palette[0]),      # (data, name, opacity, color)
        (postsyn_data, 'PostSyn', 1.0, palette[1]),
        (conn_data, 'Connections', 1.0, palette[2])
    ]
    
    for data, name, opacity, color in datasets:
        fig.add_trace(go.Bar(
            x=x_positions,
            y=100 * data['traced_frac'],
            width=widths,
            marker=dict(
                color=color,
                opacity=opacity,
                line=dict(color='white', width=0.1),
            ),
            name=name,
            hovertext=data['roi'],
            hovertemplate=f'<b>%{{hovertext}}</b><br>Type: {name}<br>Traced Fraction: %{{y:.3f}}<br>PostSyn: %{{customdata}}<extra></extra>',
            customdata=data['PostSyn']
        ))
    
    weighted_avg = 100 * (conn_data['traced_frac'] * conn_data['PostSyn']).sum() / conn_data['PostSyn'].sum()
    fig.add_hline(
        y=weighted_avg,
        line=dict(color='black', width=1, dash='dash'),
        annotation_text=f'              connection avg: {weighted_avg:.1f}%',
        annotation_position="top left",
        annotation_font=dict(color='black')
    )
    
    # Add compartment labels (same as original)
    compartment_midpoints = {}
    for compartment in conn_data['compartment'].cat.categories:
        comp_mask = conn_data['compartment'] == compartment
        if comp_mask.any():
            comp_widths = widths[comp_mask]
            comp_positions = x_positions[comp_mask]
            min_pos = comp_positions.min() - comp_widths[comp_mask].iloc[0] / 2
            max_pos = comp_positions.max() + comp_widths[comp_mask].iloc[-1] / 2
            compartment_midpoints[compartment] = (min_pos + max_pos) / 2

    for compartment, midpoint in compartment_midpoints.items():
        fig.add_annotation(
            x=midpoint,
            y=-4,
            text=compartment.lower(),
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            xanchor="center"
        )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis=dict(
            title='',
            showticklabels=False,
            showgrid=False,
            ticks='',
            ticklen=0
        ),
        yaxis=dict(
            title='completion rate [%]',
            range=[-15, 100],
            showgrid=False,
            ticks='outside',
            tickvals=[0, 20, 40, 60, 80, 100],
            ticklen=4
        ),
        width=600,
        height=400,
        template='plotly_white',
        margin=dict(b=60),
        bargap=0,
        barmode='overlay'  # This is key for overlaying bars
    )
    
    return fig


def roi_presyn_and_inverted_postsyn(traced_presyn, traced_postsyn, title=None):
    """
    Create a bar chart with presyn data above x-axis and postsyn data below (inverted).
    Both sets of bars use the same width scaling based on PostSyn values from traced_postsyn.
    
    Args:
        traced_presyn: DataFrame with presyn data (displayed above x-axis)
        traced_postsyn: DataFrame with postsyn data (displayed below x-axis)
        title: Optional title for the plot
    """
    # For this plot, we just want the primary ROIs.
    traced_presyn = traced_presyn.query('compartment != "CNS Major Compartments"')
    traced_postsyn = traced_postsyn.query('compartment != "CNS Major Compartments"')

    # Sort both dataframes to match the same ROI order
    traced_presyn = traced_presyn.sort_values(['compartment', 'traced_frac'], ascending=[True, False])
    traced_postsyn = traced_postsyn.set_index('roi').reindex(traced_presyn['roi']).reset_index()
    
    # Use PostSyn from traced_postsyn for width scaling of both sets of bars
    widths = traced_postsyn['PostSyn'] / traced_postsyn['PostSyn'].sum()
    x_positions = np.cumsum([0] + list(widths[:-1])) + (widths / 2)
    
    # Calculate weighted averages
    presyn_weighted_avg = 100 * (traced_presyn['traced_frac'] * traced_postsyn['PostSyn']).sum() / traced_postsyn['PostSyn'].sum()
    postsyn_weighted_avg = 100 * (traced_postsyn['traced_frac'] * traced_postsyn['PostSyn']).sum() / traced_postsyn['PostSyn'].sum()
    
    # Define color mapping for compartments
    compartment_colors = {
        'Central Brain': 'purple',
        'Optic Left': 'green', 
        'Optic Right': 'blue',
        'VNC': 'orange'
    }
    
    # Map colors to each ROI based on its compartment
    bar_colors = [compartment_colors.get(comp, 'purple') for comp in traced_postsyn['compartment']]
    
    fig = go.Figure()
    
    # Add presyn bars (above x-axis)
    fig.add_trace(go.Bar(
        x=x_positions,
        y=100 * traced_presyn['traced_frac'],
        width=widths,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=0.1),
            opacity=0.8
        ),
        name='PreSyn',
        hovertext=traced_presyn['roi'],
        hovertemplate='<b>%{hovertext}</b><br>Type: PreSyn<br>Traced Fraction: %{y:.3f}<br>PostSyn (width): %{customdata}<extra></extra>',
        customdata=traced_postsyn['PostSyn']
    ))
    
    # Add postsyn bars (below x-axis, inverted)
    fig.add_trace(go.Bar(
        x=x_positions,
        y=-100 * traced_postsyn['traced_frac'],  # Negative values for below x-axis
        width=widths,
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=0.1),
            opacity=0.6
        ),
        name='PostSyn',
        hovertext=traced_postsyn['roi'],
        hovertemplate='<b>%{hovertext}</b><br>Type: PostSyn<br>Traced Fraction: %{y:.3f}<br>PostSyn (width): %{customdata}<extra></extra>',
        customdata=traced_postsyn['PostSyn']
    ))
    
    # Add horizontal lines for weighted averages
    fig.add_hline(
        y=presyn_weighted_avg,
        line=dict(color='gray', width=1, dash='dash'),
        annotation_text=f'                       presyn avg: {presyn_weighted_avg:.1f} %',
        annotation_position="top left"
    )
    
    fig.add_hline(
        y=-postsyn_weighted_avg,
        line=dict(color='gray', width=1, dash='dash'),
        annotation_text=f'                       postsyn avg: {postsyn_weighted_avg:.1f} %',
        annotation_position="bottom left"
    )
    
    # Calculate midpoints for each compartment group
    compartment_midpoints = {}
    for compartment in traced_postsyn['compartment'].cat.categories:
        comp_mask = traced_postsyn['compartment'] == compartment
        if comp_mask.any():
            comp_widths = widths[comp_mask]
            comp_positions = x_positions[comp_mask]
            # Find the range of this compartment's bars
            min_pos = comp_positions.min() - comp_widths[comp_mask].iloc[0] / 2
            max_pos = comp_positions.max() + comp_widths[comp_mask].iloc[-1] / 2
            compartment_midpoints[compartment] = (min_pos + max_pos) / 2
    
    # Add custom compartment labels
    for compartment, midpoint in compartment_midpoints.items():
        fig.add_annotation(
            x=midpoint,
            y=-130,  # Position below the inverted bars
            text=compartment.lower(),
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=12, color="black"),
            xanchor="center"
        )
    
    # Update layout
    # Calculate y-axis range to accommodate both positive and negative bars
    max_presyn = traced_presyn['traced_frac'].max()
    max_postsyn = traced_postsyn['traced_frac'].max()
    y_max = 100 * np.ceil(max_presyn / 0.2) * 0.2
    y_min = -100 * np.ceil(max_postsyn / 0.2) * 0.2
    
    fig.update_layout(
        title_text=title,
        xaxis=dict(
            title='',  # Remove default x-axis title
            showticklabels=False,  # Remove x-axis labels
            showgrid=False,
            ticks='',
            ticklen=0
        ),
        yaxis=dict(
            title='completion rate [%]',
            range=[y_min - 15, y_max],  # Extended lower bound for labels
            showgrid=True,  # Show grid for better readability with positive/negative
            gridcolor='lightgray',
            gridwidth=0.5,
            ticks='outside',
            tickvals=list(range(int(y_min), int(y_max + 1), 20)),
            ticklen=4,
            zeroline=True,  # Show x-axis line
            zerolinecolor='black',
            zerolinewidth=1
        ),
        width=600,
        height=500,  # Slightly taller to accommodate both positive and negative bars
        template='plotly_white',
        margin=dict(b=80),  # Extra bottom margin for custom labels
        bargap=0  # No gaps between bars
    )
    
    return fig


def side_by_side_presyn_postsyn(traced_presyn, traced_postsyn, title=None):
    """
    Create side-by-side horizontal bar charts for presyn (left) and postsyn (right) data.
    Bars are ordered alphabetically by ROI and have normal bar chart lengths (no width scaling).
    
    Args:
        traced_presyn: DataFrame with presyn data
        traced_postsyn: DataFrame with postsyn data  
        title: Optional title for the plot
    """
    from plotly.subplots import make_subplots
    
    # For this plot, we just want the primary ROIs.
    traced_presyn = traced_presyn.query('compartment != "CNS Major Compartments"')
    traced_postsyn = traced_postsyn.query('compartment != "CNS Major Compartments"')

    # Sort both dataframes alphabetically by ROI
    traced_presyn = traced_presyn.sort_values('roi')
    traced_postsyn = traced_postsyn.sort_values('roi')
    
    # Define color mapping for compartments
    compartment_colors = {
        'Central Brain': 'purple',
        'Optic Left': 'green',
        'Optic Right': 'blue', 
        'VNC': 'orange'
    }
    
    # Map colors for each dataset
    presyn_colors = [compartment_colors.get(comp, 'purple') for comp in traced_presyn['compartment']]
    postsyn_colors = [compartment_colors.get(comp, 'purple') for comp in traced_postsyn['compartment']]
    
    # Create subplots with shared y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('PreSyn Completeness', 'PostSyn Completeness'),
        shared_yaxis=True,
        horizontal_spacing=0.15
    )
    
    # Add presyn bars (left subplot)
    fig.add_trace(
        go.Bar(
            x=100 * traced_presyn['traced_frac'],
            y=traced_presyn['roi'],
            orientation='h',
            marker=dict(
                color=presyn_colors,
                line=dict(color='white', width=0.5)
            ),
            name='PreSyn',
            hovertemplate='<b>%{y}</b><br>PreSyn Completion: %{x:.1f}%<br>PostSyn Count: %{customdata}<extra></extra>',
            customdata=traced_presyn['PostSyn'],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add postsyn bars (right subplot)  
    fig.add_trace(
        go.Bar(
            x=100 * traced_postsyn['traced_frac'],
            y=traced_postsyn['roi'],
            orientation='h',
            marker=dict(
                color=postsyn_colors,
                line=dict(color='white', width=0.5)
            ),
            name='PostSyn',
            hovertemplate='<b>%{y}</b><br>PostSyn Completion: %{x:.1f}%<br>PostSyn Count: %{customdata}<extra></extra>',
            customdata=traced_postsyn['PostSyn'],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Calculate weighted averages
    presyn_avg = 100 * (traced_presyn['traced_frac'] * traced_presyn['PostSyn']).sum() / traced_presyn['PostSyn'].sum()
    postsyn_avg = 100 * (traced_postsyn['traced_frac'] * traced_postsyn['PostSyn']).sum() / traced_postsyn['PostSyn'].sum()
    
    # Add vertical lines for averages
    fig.add_vline(
        x=presyn_avg,
        line=dict(color='gray', width=1, dash='dash'),
        annotation_text=f'avg: {presyn_avg:.1f}%',
        annotation_position="top",
        row=1, col=1
    )
    
    fig.add_vline(
        x=postsyn_avg, 
        line=dict(color='gray', width=1, dash='dash'),
        annotation_text=f'avg: {postsyn_avg:.1f}%',
        annotation_position="top",
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        width=1000,
        height=max(400, len(traced_presyn) * 15 + 100),  # Dynamic height based on number of ROIs
        template='plotly_white',
        showlegend=False
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Completion Rate [%]",
        range=[0, 100],
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.5
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="ROI",
        showgrid=False,
        tickfont=dict(size=10)
    )
    
    return fig


def grouped_presyn_postsyn_bars_with_whiskers(side_df, bar_orientation='vertical'):
    """
    Produce a bar plot of the pre/post-synaptic completion rates grouped in pairs of bars
    for each ROI (but not the overall major compartments),
    in which each bar represents the mean of the left/right ROI pairs,
    and the whiskers represent the individual left/right ROI values.

    Args:
        side_df:
            As produced by pivot_completeness_stats(), above.
    Returns:
        plotly figure
    """
    side_df = side_df.query('compartment != "major compartments"').copy()

    # Prepare data for bar chart
    plot_data = []

    for (compartment, roi_base) in side_df.index:
        row = side_df.loc[(compartment, roi_base)]
        
        for stat in ['presyn_frac', 'postsyn_frac']:
            L = row[(stat, 'L')]
            R = row[(stat, 'R')]
            C = row[(stat, 'C')]
            
            # Calculate bar value: mean of L and R if they exist, otherwise C
            if pd.notna(L) and pd.notna(R):
                bar_value = (L + R) / 2
                # Calculate whiskers (error bars)
                error_minus = bar_value - min(L, R)
                error_plus = max(L, R) - bar_value
                has_whiskers = True
            elif pd.notna(C):
                bar_value = C
                error_minus = 0
                error_plus = 0
                has_whiskers = False
            else:
                continue  # Skip if no data
            
            plot_data.append({
                'roi_base': roi_base,
                'compartment': compartment,
                'stat': 'Presynaptic' if stat == 'presyn_frac' else 'Postsynaptic',
                'value': bar_value * 100,  # Convert to percentage
                'error_minus': error_minus * 100 if has_whiskers else None,
                'error_plus': error_plus * 100 if has_whiskers else None,
            })

    plot_df = pd.DataFrame(plot_data)

    # Preserve original order from side_df
    original_order = [roi_base for (compartment, roi_base) in side_df.index]

    if bar_orientation == 'horizontal':
        # (reversed to show first rows at top)
        original_order = original_order[::-1]

    # Define colors by compartment and stat
    # Using first 3 colors of Plotly Dark2 palette for postsynaptic
    # And desaturated/grayed versions for presynaptic
    color_map = {
        ('optic lobes', 'Postsynaptic'): '#1b9e77',  # Dark2 teal (1st)
        ('optic lobes', 'Presynaptic'): '#8bb3a2',   # Grayed teal
        ('central brain', 'Postsynaptic'): '#d95f02',  # Dark2 orange (2nd)
        ('central brain', 'Presynaptic'): '#c99872',   # Grayed orange
        ('ventral nerve cord', 'Postsynaptic'): '#7570b3',  # Dark2 purple (3rd)
        ('ventral nerve cord', 'Presynaptic'): '#a39fbe',   # Grayed purple
    }

    # Add color column to plot_df
    plot_df['color'] = plot_df.apply(lambda row: color_map.get((row['compartment'], row['stat']), 'gray'), axis=1)

    # Create bar chart with configurable orientation
    fig = go.Figure()

    bar_width = 0.44
    whisker_width = 3
    whisker_thickness = 1.0

    for stat in ['Postsynaptic', 'Presynaptic']:
        stat_data = plot_df[plot_df['stat'] == stat]
        
        # Separate data with and without whiskers
        with_whiskers = stat_data[stat_data['error_minus'].notna()]
        without_whiskers = stat_data[stat_data['error_minus'].isna()]

        if bar_orientation == 'horizontal':
            # Add bars with whiskers
            if len(with_whiskers) > 0:
                fig.add_trace(go.Bar(
                    name=stat,
                    y=with_whiskers['roi_base'],
                    x=with_whiskers['value'],
                    orientation='h',
                    width=bar_width,
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=with_whiskers['error_plus'],
                        arrayminus=with_whiskers['error_minus'],
                        width=whisker_width,
                        thickness=whisker_thickness,
                    ),
                    marker=dict(color=with_whiskers['color']),
                    legendgroup=stat,
                    offsetgroup=stat,
                    showlegend=False,
                ))
            # Add bars without whiskers
            if len(without_whiskers) > 0:
                fig.add_trace(go.Bar(
                    name=stat,
                    y=without_whiskers['roi_base'],
                    x=without_whiskers['value'],
                    orientation='h',
                    width=bar_width,
                    marker=dict(color=without_whiskers['color']),
                    legendgroup=stat,
                    offsetgroup=stat,
                    showlegend=False,
                ))
        else:
            # Add bars with whiskers (vertical)
            if len(with_whiskers) > 0:
                fig.add_trace(go.Bar(
                    name=stat,
                    x=with_whiskers['roi_base'],
                    y=with_whiskers['value'],
                    orientation='v',
                    width=bar_width,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=with_whiskers['error_plus'],
                        arrayminus=with_whiskers['error_minus'],
                        width=whisker_width,
                        thickness=whisker_thickness,
                    ),
                    marker=dict(color=with_whiskers['color']),
                    legendgroup=stat,
                    offsetgroup=stat,
                    showlegend=False,
                ))
            # Add bars without whiskers (vertical)
            if len(without_whiskers) > 0:
                fig.add_trace(go.Bar(
                    name=stat,
                    x=without_whiskers['roi_base'],
                    y=without_whiskers['value'],
                    orientation='v',
                    width=bar_width,
                    marker=dict(color=without_whiskers['color']),
                    legendgroup=stat,
                    offsetgroup=stat,
                    showlegend=False,
                ))

    # Layout adjustments depending on orientation
    num_rois = len(plot_df['roi_base'].unique())
    if bar_orientation == 'horizontal':
        fig.update_layout(
            xaxis_title='% on traced neurons',
            barmode='group',
            height=max(600, num_rois * 20),
            width=400,
            xaxis=dict(
                range=[0, 100],
                tickmode='array',
                tickvals=[0, 25, 50, 75, 100],
            ),
            yaxis=dict(
                categoryorder='array',
                categoryarray=original_order,
                tickfont=dict(size=15),
            ),
            bargap=0.05,
            bargroupgap=0.0,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=20, r=20),
        )
    else:
        fig.update_layout(
            yaxis_title='% on traced neurons',
            barmode='group',
            height=300,
            width=max(800, num_rois * 18),
            yaxis=dict(
                range=[0, 100],
                tickmode='array',
                tickvals=[0, 25, 50, 75, 100],
            ),
            xaxis=dict(
                categoryorder='array',
                categoryarray=original_order,
                tickfont=dict(size=12),
                tickangle=-90,
            ),
            bargap=0.05,
            bargroupgap=0.0,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=20, r=20),
        )

    # Add compartment labels as rectangles with text
    # Group ROIs by compartment
    compartment_groups = []
    current_comp = None
    start_idx = None

    for idx, (compartment, roi_base) in enumerate(side_df.index):
        if compartment != current_comp:
            if current_comp is not None:
                compartment_groups.append((current_comp, start_idx, idx - 1))
            current_comp = compartment
            start_idx = idx

    # Add the last group
    if current_comp is not None:
        compartment_groups.append((current_comp, start_idx, len(side_df) - 1))

    # Reverse to match the plot order (since we reversed original_order)
    compartment_groups_reversed = []
    total_rois = len(side_df)
    for comp, start, end in compartment_groups:
        # Reverse the indices
        new_start = total_rois - 1 - end
        new_end = total_rois - 1 - start
        compartment_groups_reversed.append((comp, new_start, new_end))

    return fig


def main():
    # Load data for all three statistics
    traced_conn = load_completeness_stats('conn')
    traced_presyn = load_completeness_stats('presyn')
    traced_postsyn = load_completeness_stats('postsyn')
    traced_synweight = load_completeness_stats('synweight')

    combined_df = pd.concat(
        (
            traced_presyn.set_index('roi').rename(columns={'traced_frac': 'presyn_traced_frac'})[['compartment', 'PreSyn', 'PostSyn', 'presyn_traced_frac']],
            traced_postsyn.set_index('roi').rename(columns={'traced_frac': 'postsyn_traced_frac'})[['postsyn_traced_frac']],
            traced_conn.set_index('roi').rename(columns={'traced_frac': 'conn_traced_frac'})[['conn_traced_frac']],
            traced_synweight.set_index('roi').rename(columns={'traced_frac': 'synweight_traced_frac'})[['synweight_traced_frac']]
        ),
        axis=1
    )
    combined_df[['roi_base', 'roi_side']] = combined_df.reset_index()['roi'].str.extract(r'(.*?)(\(([LR])\))?$')[[0, 2]].values
    combined_df['roi_side'] = combined_df['roi_side'].fillna('C')

    cols = [
        'compartment', 'roi', 'roi_base', 'roi_side',
        'PreSyn', 'PostSyn',
        'presyn_traced_frac', 'postsyn_traced_frac', 'conn_traced_frac', 'synweight_traced_frac'
    ]
    combined_df = combined_df.reset_index()[cols]
    combined_df = combined_df.astype({'PreSyn': int, 'PostSyn': int})
    combined_df = combined_df.sort_values(['compartment', 'roi'])
    combined_df.to_csv(f'{SNAPSHOT_TAG}-traced-synapse-capture-by-roi.csv', index=False, header=True)

    combined_minimal = combined_df.rename(columns={
        'roi': 'name',
        'presyn_traced_frac': 'pre',
        'postsyn_traced_frac': 'post',
        'synweight_traced_frac': 'synweight',
        'conn_traced_frac': 'connections',
    })[['compartment', 'name', 'pre', 'synweight', 'post', 'connections']]
    combined_minimal.to_csv(f'{SNAPSHOT_TAG}-traced-synapse-capture-by-roi-fractions-only.csv', index=False, header=True)

    # Create individual figures
    fig_conn = roi_completeness(traced_conn, None, 'connection completion [%]')
    fig_synweight = roi_completeness(traced_synweight, None, 'synapse weight completion [%]')
    fig_presyn = roi_completeness(traced_presyn, None, 'presynapse completion [%]')
    fig_postsyn = roi_completeness(traced_postsyn, None, 'postsynapse completion [%]')
    
    # Create overlaid figure
    fig_overlaid = create_overlaid_bars(traced_presyn, traced_postsyn, traced_conn)
    
    # Create presyn/postsyn inverted figure
    fig_presyn_postsyn = roi_presyn_and_inverted_postsyn(
        traced_presyn,
        traced_postsyn, 
        title="PreSyn vs PostSyn Completeness"
    )
    
    print(combined_df)
    side_df = pivot_completeness_stats(combined_df)
    print(side_df)

    bar_orientation = 'vertical'
    fig_grouped = grouped_presyn_postsyn_bars_with_whiskers(side_df.query('compartment != "major compartments"'), bar_orientation)
    
    # Export all figures as PDFs
    fig_conn.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_conn.pdf")
    fig_presyn.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_presyn.pdf")
    fig_postsyn.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_postsyn.pdf")
    fig_synweight.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_synweight.pdf")
    fig_overlaid.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_overlaid.pdf")
    fig_presyn_postsyn.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_presyn_postsyn.pdf")
    fig_grouped.write_image(f'{SNAPSHOT_TAG}-roi-presyn-postsyn-completeness-{bar_orientation}.pdf')

    # # Create side-by-side presyn/postsyn figure
    # fig_side_by_side = side_by_side_presyn_postsyn(traced_presyn, traced_postsyn,
    #                                                 "PreSyn and PostSyn Completeness by ROI")
    # fig_side_by_side.write_image(f"{SNAPSHOT_TAG}-roi_completeness_bars_side_by_side.pdf")


if __name__ == '__main__':
    main()
