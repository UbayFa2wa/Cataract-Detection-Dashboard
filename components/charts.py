"""Two compact, high-contrast charts shared by the Overview and Analysis pages."""
import plotly.graph_objects as go

NORMAL = '#0F9D8A'
CATARACT = '#F0644F'
INK = '#0B2E3F'
BOLD_FONT = 'Plus Jakarta Sans, Inter, Arial, sans-serif'


def class_chart():
    fig = go.Figure(go.Pie(
        labels=['Normal', 'Cataract'], values=[300, 100],
        hole=0.62, sort=False, marker_colors=[NORMAL, CATARACT],
        textinfo='label+percent', textposition='inside',
        insidetextorientation='horizontal',
        textfont=dict(family=BOLD_FONT, size=17, weight=800,
                      color=['#FFFFFF', INK]),
        hovertemplate='<b>%{label}</b>: %{value} images (%{percent})<extra></extra>',
    ))
    fig.update_layout(
        height=300, margin=dict(l=5, r=5, t=10, b=10),
        showlegend=False, font=dict(family=BOLD_FONT, color=INK, size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        uniformtext=dict(minsize=14, mode='show'),
    )
    return fig


def confusion_chart(row):
    # Same values and ordering as the final notebook.
    z = [[int(row['TN']), int(row['FP'])],
         [int(row['FN']), int(row['TP'])]]
    x = ['Predicted normal', 'Predicted cataract']
    y = ['Actual normal', 'Actual cataract']
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, '#E3F5F2'], [1, NORMAL]],
        showscale=False, xgap=4, ygap=4,
        hovertemplate='%{y} / %{x}: <b>%{z}</b><extra></extra>',
    ))
    for i, row_values in enumerate(z):
        for j, value in enumerate(row_values):
            fig.add_annotation(
                x=x[j], y=y[i], text=f'<b>{value}</b>', showarrow=False,
                font=dict(family=BOLD_FONT, size=26,
                          color='#FFFFFF' if value >= 30 else INK),
            )
    fig.update_xaxes(
        tickfont=dict(family=BOLD_FONT, size=13, color=INK),
        showline=False, showgrid=False, ticks='', fixedrange=True,
    )
    fig.update_yaxes(
        autorange='reversed',
        tickfont=dict(family=BOLD_FONT, size=13, color=INK),
        showline=False, showgrid=False, ticks='', fixedrange=True,
    )
    fig.update_layout(
        height=300, margin=dict(l=20, r=5, t=10, b=10),
        font=dict(family=BOLD_FONT, color=INK),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig
