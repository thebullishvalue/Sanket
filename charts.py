"""
SANKET — Chart builders, in Pragyam's grammar.

Every chart takes its layout from ``ui.theme.chart_layout`` / ``style_axes`` and
every colour from ``COLORS`` below, which resolves against the ACTIVE theme on each
lookup — the arrangement Pragyam's charts.py uses, so the two apps' plots read as one
system and flip together with the Slate / Paper toggle.

COLOUR IS NEVER ALONE. The grid's tones are the Graphite SEMANTIC hues (favourable,
caution, watched, unfavourable), designed as meanings rather than as a categorical
set — run through a categorical validator they do not separate well enough on their
own (cyan and emerald especially). So every tone-coded mark here also carries a
distinct marker SHAPE, and the tones are named in the legend and on the plane.

Author: @thebullishvalue
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import cvgrid as cg
import samanvaya as sv
from ui.components import table_tokens
from ui.theme import chart_color, chart_layout, chart_rgba, diverging_scale, panel_bg, style_axes


class _LivePalette:
    """Semantic chart colours for the ACTIVE theme (Pragyam's _LivePalette).

    A plain dict would bind at import, before a session exists, and then keep the
    graphite hexes on Paper while the axes flipped. ``_dim`` / ``_glow`` suffixes
    return the same hue at a fill's opacity.
    """

    _ALPHA = {"_dim": 0.45, "_glow": 0.18}

    def __getitem__(self, key: str) -> str:
        for suffix, alpha in self._ALPHA.items():
            if key.endswith(suffix):
                return chart_rgba(key[: -len(suffix)], alpha)
        return chart_color(key)


COLORS = _LivePalette()

#: One shape per tone — the redundant channel that keeps a tone readable without colour.
TONE_SYMBOL = {"emerald": "triangle-up", "cyan": "circle", "amber": "square",
               "rose": "triangle-down", "slate": "diamond"}
#: Legend wording for each tone, in the order a book reads them.
TONE_LABEL = {"emerald": "Buy — capitulation or a turn",
              "cyan": "Accumulate — washout or base",
              "slate": "Wait — no edge",
              "amber": "Hold / Trim — building, stalling or paid",
              "rose": "Exit — distribution"}
#: The trace's push in five levels, for the hover.
_PUSH5 = {2: "impulse ↑", 1: "push ↑", 0: "no push", -1: "push ↓", -2: "impulse ↓"}
TONE_ORDER = ("emerald", "cyan", "slate", "amber", "rose")

_MONO = "JetBrains Mono, monospace"
_LIM = 100.0


def _cell_edges():
    th = sv.THETA_OSC
    return (-_LIM, -30.0, 30.0, _LIM), (-_LIM, -th, th, _LIM)


def create_conviction_value_map(df: "pd.DataFrame | None") -> go.Figure:
    """Every name placed by its two tapes — the grid as a plane (Pragyam's CVG map, 3 × 3).

    Conviction across (sellers ← → buyers), value up (+ rich). The dotted lines are
    each tape's own knee — ±30 conviction, ±θ value — and they cut the plane into the
    nine cells; the faint solid lines are each tape's zero. A cell is tinted with its
    tone and named in its corner. A point sits where its TAPES are; its colour and shape
    are its STATE. The histogram moves a name's row only on a confirmed push, so a
    HOLLOW point — a held row — can sit in a region it is not coloured for. A point
    ringed in the accent fired an event on this bar.
    """
    fig = go.Figure()
    if df is None or getattr(df, "empty", True):
        fig.update_layout(**chart_layout(height=460, show_legend=False))
        return fig
    u = df.copy()
    u["_c"] = pd.to_numeric(u.get("PRG_CTape"), errors="coerce")
    u["_v"] = pd.to_numeric(u.get("PRG_VTape"), errors="coerce")
    u = u[u["_c"].notna() & u["_v"].notna()]
    xs, ys = _cell_edges()

    # Region tints, faint: the plane's structure without competing with the points.
    for r in range(cg.N_ROWS):
        for c in range(cg.N_COLS):
            cell = r * cg.N_COLS + c
            tone = cg.TONES[cell]
            if tone == "slate":
                continue                      # a cell with nothing to say stays panel
            fig.add_shape(type="rect", x0=xs[r], x1=xs[r + 1], y0=ys[c], y1=ys[c + 1],
                          layer="below", line=dict(width=0), fillcolor=chart_rgba(tone, 0.07))
            fig.add_annotation(x=xs[r] + 2, y=ys[c + 1] - 2, xanchor="left", yanchor="top",
                               showarrow=False, text=cg.action(cell).upper(),
                               font=dict(size=8, family=_MONO, color=chart_rgba(tone, 0.85)))
    for x in (-30.0, 30.0):
        fig.add_vline(x=x, line=dict(color=chart_rgba("slate", 0.55), width=1, dash="dot"))
    for y in (-sv.THETA_OSC, sv.THETA_OSC):
        fig.add_hline(y=y, line=dict(color=chart_rgba("slate", 0.55), width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=chart_rgba("slate", 0.30), width=1))
    fig.add_hline(y=0, line=dict(color=chart_rgba("slate", 0.30), width=1))

    cells = pd.to_numeric(u.get("CVG_Cell"), errors="coerce").fillna(cg.UNREAD).astype(int)
    tones = cells.map(lambda k: cg.TONES[k])
    held = u.get("CVG_Held", pd.Series(False, index=u.index)).fillna(False).astype(bool)
    names = u.get("SimpleName", u.get("Symbol", pd.Series("", index=u.index))).astype(str)
    show_text = len(u) <= 40
    for tone in TONE_ORDER:
        part = u[tones == tone]
        if part.empty:
            continue
        col = COLORS[tone]
        h = held.loc[part.index]
        fig.add_trace(go.Scatter(
            x=part["_c"], y=part["_v"], name=TONE_LABEL[tone],
            mode="markers+text" if show_text else "markers",
            text=names.loc[part.index] if show_text else None, textposition="top center",
            textfont=dict(size=8, family=_MONO, color=chart_rgba("slate", 0.95)),
            marker=dict(symbol=TONE_SYMBOL[tone], size=10,
                        color=[chart_rgba(tone, 0.0) if x else col for x in h],
                        line=dict(width=1.6, color=col)),
            customdata=np.stack([
                names.loc[part.index],
                [cg.NAMES[k] for k in cells.loc[part.index]],
                pd.to_numeric(part.get("Signal_Score"), errors="coerce").round(0).fillna(np.nan),
                [_PUSH5.get(int(p) if pd.notna(p) else 0, "no push")
                 for p in pd.to_numeric(part.get("PRG_Push"), errors="coerce")],
                ["<br>row HELD — the push is not behind the tape yet" if x else "" for x in h],
            ], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}"
                           "<br>Conviction %{x:+.0f} · Value %{y:+.0f} · Trace %{customdata[2]}"
                           "<br>%{customdata[3]}%{customdata[4]}<extra></extra>"),
        ))
    fired = u[u.get("Side", pd.Series("—", index=u.index)).isin(["Buy", "Sell"])]
    if len(fired):
        fig.add_trace(go.Scatter(
            x=fired["_c"], y=fired["_v"], name="Event on this bar", mode="markers",
            marker=dict(symbol="circle-open", size=20, color=COLORS["accent"],
                        line=dict(width=2, color=COLORS["accent"])),
            hoverinfo="skip"))
    fig.update_layout(**chart_layout(height=500, show_legend=True,
                                     margin=dict(t=16, l=60, r=16, b=64)))
    fig.update_layout(hovermode="closest")
    style_axes(fig, y_title="Value tape · rich ↑  cheap ↓",
               x_title="Conviction tape · sellers ←  → buyers",
               y_range=[-_LIM - 4, _LIM + 4])
    fig.update_xaxes(range=[-_LIM - 4, _LIM + 4], zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def create_tone_history(tone_share: pd.DataFrame) -> go.Figure:
    """Share of the readable universe in each grid tone, per session — stacked to 100%.

    One stacked area per tone in the book's reading order (build → watched → no edge →
    caution → cut), each named in the legend, so a drift from green toward red is the
    universe moving down the grid.
    """
    fig = go.Figure()
    if tone_share is None or tone_share.empty:
        fig.update_layout(**chart_layout(height=320))
        return fig
    for tone in TONE_ORDER:
        if tone not in tone_share.columns:
            continue
        fig.add_trace(go.Scatter(
            x=tone_share.index, y=tone_share[tone], name=TONE_LABEL[tone], mode="lines",
            stackgroup="tone", line=dict(width=0.5, color=COLORS[tone]),
            fillcolor=chart_rgba(tone, 0.55 if tone != "slate" else 0.30),
            hovertemplate=f"{TONE_LABEL[tone]}: %{{y:.0f}}%<extra></extra>"))
    fig.update_layout(**chart_layout(height=340))
    fig.update_layout(legend_traceorder="normal")   # stacked areas otherwise list bottom-last
    style_axes(fig, y_title="% of readable universe", y_range=[0, 100])
    return fig


def create_correlation_heatmap(names, corr_values) -> go.Figure:
    """Current correlation to the target, one row per name, on the panel-midpoint diverging scale."""
    fig = go.Figure(data=go.Heatmap(
        z=np.asarray(corr_values, dtype=float).reshape(-1, 1), x=["Correlation"], y=list(names),
        colorscale=diverging_scale("rose", "emerald"), zmid=0, zmin=-1, zmax=1,
        text=np.asarray(corr_values, dtype=float).reshape(-1, 1), texttemplate="%{text:+.2f}",
        # Values sit ON the fill, so they take primary ink — the one text colour that
        # holds contrast on both poles and on the panel-coloured midpoint.
        textfont={"size": 9, "family": _MONO, "color": table_tokens()["ink_primary"]},
        xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b> · ρ %{z:+.3f}<extra></extra>",
        colorbar=dict(title="ρ", thickness=12, len=0.7, outlinewidth=0)))
    fig.update_layout(**chart_layout(height=max(320, 18 * len(names) + 80), show_legend=False,
                                     margin=dict(l=140, r=40, t=16, b=32)))
    fig.update_layout(plot_bgcolor=panel_bg())
    return fig


__all__ = ["COLORS", "TONE_LABEL", "TONE_ORDER", "TONE_SYMBOL", "create_conviction_value_map",
           "create_correlation_heatmap", "create_tone_history"]
