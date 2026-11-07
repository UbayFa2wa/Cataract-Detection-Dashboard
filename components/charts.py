"""Plotly figures reused on several pages."""
import plotly.graph_objects as go

from components.data import N_CATARACT, N_NORMAL
from components.ui import CATARACT, NAVY, NORMAL, style


def donut_fig(title="Class distribution", height=340):
    total = N_NORMAL + N_CATARACT
    fig = go.Figure(go.Pie(
        labels=["Normal", "Cataract"], values=[N_NORMAL, N_CATARACT], hole=0.62, sort=False,
        marker=dict(colors=[NORMAL, CATARACT], line=dict(color="white", width=3)),
        textinfo="percent", textfont=dict(color="white", size=14),
        hovertemplate="%{label}: %{value} images (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total}</b><br>images", showarrow=False, font=dict(size=18, color=NAVY))
    return style(fig, height=height, legend=False, title=title)


def confusion_fig(tp, fn, tn, fp, title="Confusion matrix", height=340):
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted<br>normal", "Predicted<br>cataract"], y=["Actually<br>normal", "Actually<br>cataract"],
        colorscale=[[0, "#EAF6F4"], [0.5, "#7FCBC0"], [1, NORMAL]], showscale=False, xgap=4, ygap=4,
        text=z, texttemplate="%{text}", textfont=dict(size=24, color=NAVY),
        hovertemplate="%{y}, %{x}: %{z} images<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(tickangle=0, automargin=True)
    return style(fig, height=height, legend=False, title=title)
