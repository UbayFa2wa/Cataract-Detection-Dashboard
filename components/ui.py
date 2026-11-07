"""Shared look: palette, Plotly styling and small HTML blocks used by every page."""
import re

import streamlit as st

# ---- palette: teal (normal, primary) + coral (cataract, alert) on deep navy ----
NAVY = "#0B2E3F"
NORMAL = "#0F9D8A"
CATARACT = "#F0644F"
CLASSICAL = "#5B7DB1"
NEURAL = "#F2A541"
VIOLET = "#7C6FD0"
ACCENT = NORMAL
GREY = "#8D99AE"
FAMILY_COLOUR = {"Classical": CLASSICAL, "Neural": NEURAL}
CLASS_COLOUR = {"Normal": NORMAL, "Cataract": CATARACT}
QUAL = [NORMAL, CATARACT, NEURAL, CLASSICAL, VIOLET, "#2BB5D8", "#9BA7B4"]
TINT = {"teal": ("#E3F5F2", "#0F9D8A"), "mint": ("#E4F6EC", "#1FA463"), "violet": ("#EEEBFB", "#7C6FD0"),
        "peach": ("#FDEFE6", "#E8793A"), "sky": ("#E3F2FA", "#2B8FC4"), "coral": ("#FDE8E4", "#F0644F")}

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def style(fig, height=380, legend=True, title=None, legend_below=False, right=10):
    """Apply one consistent Plotly look. The title is only set when given (an empty title shows 'undefined')."""
    top = 20
    if title and legend and not legend_below:
        top = 100
    elif title:
        top = 62
    elif legend and not legend_below:
        top = 50
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=10, r=right, t=top, b=70 if legend_below else 10),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=NAVY),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=NORMAL),
        showlegend=legend,
        legend=(dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0) if legend_below
                else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=QUAL,
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", y=1, yref="container",
                                     yanchor="top", pad=dict(t=12), font=dict(size=16, color=NAVY)))
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def hero(tag, title, text=None):
    body = f"<div class='hero-text'>{text}</div>" if text else ""
    st.markdown(
        f"""<div class="hero"><div class="hero-bar"></div><div>
              <div class="page-tag">{tag}</div><div class="hero-title">{title}</div>{body}
            </div></div>""",
        unsafe_allow_html=True,
    )


def kpi_tiles(items):
    """items: list of (label, value, icon, tint_key, caption)."""
    cells = ""
    for label, value, icon, tint, caption in items:
        bg, fg = TINT[tint]
        cells += (f"<div class='kpi' style='background:linear-gradient(135deg,{bg},#ffffff);"
                  f"border:1px solid {bg};'><div class='kpi-top'><span class='kpi-icon' style='background:{fg}1A;"
                  f"color:{fg};'>{icon}</span><span class='kpi-label'>{label}</span></div>"
                  f"<div class='kpi-value' style='color:{fg};'>{value}</div>"
                  f"<div class='kpi-cap'>{caption}</div></div>")
    st.markdown(f"<div class='kpi-row'>{cells}</div>", unsafe_allow_html=True)


def panel_title(text, sub=None):
    extra = f"<span class='panel-sub'>{sub}</span>" if sub else ""
    st.markdown(f"<div class='panel-title'>{text}{extra}</div>", unsafe_allow_html=True)


def chapter(number, title, question):
    """Section header that frames each chart as one step of the story."""
    st.markdown(
        f"""
        <div class="chapter">
          <div class="chapter-no">{number}</div>
          <div>
            <div class="chapter-title">{title}</div>
            <div class="chapter-q">{question}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def takeaway(text, kind="finding"):
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)  # markdown bold does not render inside HTML
    label = {"finding": "What this shows", "next": "Where this leads"}[kind]
    st.markdown(
        f"""<div class="takeaway {kind}"><b>{label}.</b> {text}</div>""",
        unsafe_allow_html=True,
    )


def goto(flag):
    """Set the navigation flag read in app.py, then rerun."""
    st.session_state[flag] = True
    st.rerun()
