"""
SANKET — the Conviction-Value Grid, 4 × 4 (pragati.pine section 10)
══════════════════════════════════════════════════════════════════════════════

Pragyam's CVG, grown from its 3 × 3 seed into the 4 × 4 the indicator now draws.
Where a name stands BETWEEN signals, as something to DO about it. A state, not a
signal: signals are events, the grid is where the name stands between them, and
neither reads the other.

FOUR READINGS, FOUR JOBS
    conviction tape   the ROW     who controls
    value tape        the COLUMN  where price stands
    histogram         the PUSH    whether the row may change
    trace's parts     the CHART   whether the chart already stands elsewhere

THE AXES are the tapes, split at their knee and at zero — their step and their
hue — so every cell is a pair of colours actually drawn on the pane:
    rows     buyers firm ≥ +30 · buyers edge · sellers edge · sellers firm ≤ −30
    columns  cheap ≤ −θ · below fair · above fair · rich ≥ +θ

HOW A NAME MOVES. Columns move freely — price is where it is. Rows are a claim
about control, so the push must stand behind the claim in proportion to its
size: a push moves the row ONE step toward the tape, an impulse all the way, no
push holds it. A row the tape has left is HELD. Before the histogram is
calibrated the row follows the tape.

WHAT EACH CELL SAYS TO DO. The seed's nine states became nine ACTIONS, each
keeping its seed units, so the word is the weight:

    Buy 3 · Add 3 · Hold 1.5 · Accumulate 1.5 · Wait 1 · Watch 1 ·
    Trim 0.75 · Reduce 0.5 · Exit 0.25

                   CHEAP              BELOW FAIR         ABOVE FAIR         RICH
    buyers firm    Buy·turn           Add·trend          Hold·extended      Hold·don't add
    buyers edge    Accumulate·basing  Accumulate·        Wait·drifting      Trim·stalling
                                      early turn
    sellers edge   Accumulate·        Wait·no edge       Trim·rolling over  Trim·topping
                   deep value
    sellers firm   Buy·capitulation   Accumulate·washout Reduce·breakdown   Exit·distribution

v7 — THREE CELLS CHANGED BY MEASUREMENT (pine_audit.py, studies/pine_audit.md). The
v6 map was chosen by meaning. Measured on 380 instruments in six asset classes over
~20 years, split before / after 2018:
  · sellers firm × cheap  was Watch 1u   → Buy · capitulation 3u
  · sellers firm × below  was Reduce ½u  → Accumulate · washout 1½u
    Both cells were followed by gains in BOTH eras on NSE, US, indices, commodities
    and FX (holdout +0.10 to +0.35σ over 10–20 bars, significant on four of the five);
    the stack adds to plain oversold — oversold names OUTSIDE these cells lagged.
  · buyers firm × above fair  was Add 3u → Hold · extended 1½u — no support in
    either era (negative after 2018 on US, indices and FX).
Every other cell is unchanged: nothing measured held in both eras. Crypto trends
and is the exception — the capitulation cells carried nothing there. The units are
a weight, not a forecast.

Author: @thebullishvalue
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNREAD = 16

# Cell = row · 4 + column. Rows: 0 sellers firm, 1 sellers edge, 2 buyers edge,
# 3 buyers firm. Columns: 0 cheap, 1 below fair, 2 above fair, 3 rich. 16 UNREAD.
NAMES = (
    "Buy · capitulation",      "Accumulate · washout",    "Reduce · breakdown",  "Exit · distribution",
    "Accumulate · deep value", "Wait · no edge",          "Trim · rolling over", "Trim · topping",
    "Accumulate · basing",     "Accumulate · early turn", "Wait · drifting",     "Trim · stalling",
    "Buy · turn",              "Add · trend",             "Hold · extended",     "Hold · don't add",
    "Unread",
)
# Units are the action's, and each action's are its seed state's.
UNITS = (
    3.00, 1.50, 0.50, 0.25,
    1.50, 1.00, 0.75, 0.75,
    1.50, 1.50, 1.00, 0.75,
    3.00, 3.00, 1.50, 1.50,
    1.00,
)
# The side each action works: +1 builds the position, −1 cuts it, 0 neither.
SIDES = (
    1,  1, -1, -1,
    1,  0, -1, -1,
    1,  1,  0, -1,
    1,  1,  0,  0,
    0,
)
MEANING = (
    "sellers firmly in control at a cheap price - capitulation; measured, this cell was followed by gains in both eras on every asset class but crypto",
    "sellers firmly in control below fair - a washout; measured, followed by gains in both eras outside crypto",
    "sellers firmly in control while price is still above fair - the market is breaking down",
    "sellers firmly in control at a rich price - distribution; the floor",
    "cheap, and the sellers are down to an edge - build slowly on value",
    "sellers edging, price below fair - nothing decided; no edge either way",
    "sellers edging in while price is above fair - the move is rolling over; take some off",
    "rich, and the sellers are edging in - a top forming",
    "cheap, and buyers edging in - the base is forming",
    "buyers edging in below fair - the earliest turn; start building",
    "buyers edging, price above fair - drifting without conviction",
    "rich, and control has faded to an edge - the move has stalled",
    "cheap, buyers now firmly in control - a dislocation that turned",
    "buyers firmly in control below fair - the trend, with room to run",
    "buyers firmly in control and price already above fair - extended; measured, adding here earned nothing in either era",
    "buyers firmly in control of a rich price - hold it, do not add",
    "a tape not yet calibrated, or switched off",
)
# ── TONES — Pragyam's inference, one mapping for every surface ────────────────
# Pragyam colours a grid state by what it means, not by the side it trades
# (ui/shared.py · CVG_TONE), and a state is the same colour everywhere it appears —
# the map, the census, the tables. Each tone keeps its app-wide meaning:
#   emerald  the favourable end — buyers firmly in control below a rich price
#   amber    CAUTION — a price already rich, the move stalling or topping
#   cyan     information — a cheap name being watched for its turn
#   slate    unclaimed — no edge either way, or unread
#   rose     the unfavourable end — sellers in control, breaking down or distributing
TONES = (
    "emerald", "cyan",    "rose",    "rose",       # sellers firm: buy·capitulation · accumulate·washout · reduce · exit
    "cyan",    "slate",   "rose",    "amber",      # sellers edge: accumulate · wait · trim·rolling · trim·topping
    "cyan",    "cyan",    "slate",   "amber",      # buyers edge: accumulate · accumulate · wait · trim·stalling
    "emerald", "emerald", "amber",   "amber",      # buyers firm: buy · add · hold·extended · hold
    "slate",                                        # unread
)
#: The same, in render_chip / render_metric_card's vocabulary.
TONE_CHIP = {"emerald": "success", "amber": "warning", "cyan": "info",
             "slate": "neutral", "rose": "danger"}
#: Actions in the order a book reads them — build, hold, cut — with their tone.
ACTION_TONE = {"Buy": "emerald", "Add": "emerald", "Accumulate": "cyan", "Hold": "amber",
               "Wait": "slate", "Watch": "cyan", "Trim": "amber", "Reduce": "rose",
               "Exit": "rose", "Unread": "slate"}

ROW_LABELS = ("sellers firm", "sellers edge", "buyers edge", "buyers firm")
COL_LABELS = ("cheap", "below fair", "above fair", "rich")
ACTIONS = ("Buy", "Add", "Accumulate", "Hold", "Wait", "Watch", "Trim", "Reduce", "Exit")
ACTION_UNITS = {"Buy": 3.0, "Add": 3.0, "Hold": 1.5, "Accumulate": 1.5, "Wait": 1.0,
                "Watch": 1.0, "Trim": 0.75, "Reduce": 0.5, "Exit": 0.25, "Unread": 1.0}

PUSH_TEXT = {2: "impulse ↑", 1: "push ↑", 0: "no push", -1: "push ↓", -2: "impulse ↓"}
PUSH_GLYPH = {2: "↑↑", 1: "↑", 0: "·", -1: "↓", -2: "↓↓"}

READ_THE_PUSH = ("The push says which way the row may move next. Measured: inside the "
                 "capitulation cells its direction made no consistent difference - the state "
                 "carried the edge, not the timing - so Buy · capitulation and Accumulate · "
                 "washout do not wait for a push ↑. Hold, Wait and Watch change nothing.")

COLUMNS = ("cvg_cell", "cvg_target_row", "cvg_held", "cvg_since", "cvg_from",
           "cvg_chart", "cvg_lead")


def action(cell: int) -> str:
    return NAMES[int(cell)].split(" · ")[0]


def reason(cell: int) -> str:
    parts = NAMES[int(cell)].split(" · ")
    return parts[1] if len(parts) > 1 else "needs both tapes"


def row_of(c: float, z1: float) -> int:
    """f_gRow: the conviction tape's row."""
    return 3 if c >= z1 else 2 if c >= 0.0 else 1 if c > -z1 else 0


def col_of(v: float, theta: float) -> int:
    """f_gCol: the value tape's column."""
    return 0 if v <= -theta else 1 if v < 0.0 else 2 if v < theta else 3


def classify(c_tape, v_tape, push, hist_ready, read, chart_conv, chart_value, chart_ok,
             z1: float, theta: float, index=None) -> pd.DataFrame:
    """Section 10, bar by bar: the grid state, its age, and the chart cell.

    c_tape / v_tape  the MTF conviction and value tapes (the axes)
    push             the histogram in five levels, −2 … +2
    hist_ready       the histogram's σ window is clean (else rows follow the tape)
    read             both tapes can be read (else UNREAD)
    chart_conv / chart_value / chart_ok   the trace's own ingredients on this
                     chart — conviction ±100 and value ±100 — and whether both
                     are calibrated, for the chart cell

    Columns: cvg_cell (0-15, 16 unread), cvg_target_row (the tape's row),
    cvg_held (1 when the row stands against its tape for want of a push),
    cvg_since (bar the current cell began), cvg_from (the cell before),
    cvg_chart (the chart cell) and cvg_lead (+1 the chart cell carries more
    units than the state, −1 fewer, 0 the same or unread).
    """
    c = np.asarray(c_tape, dtype=float).tolist()
    v = np.asarray(v_tape, dtype=float).tolist()
    g = np.asarray(push, dtype=float).tolist()
    rdy = np.asarray(hist_ready, dtype=bool).tolist()
    rd = np.asarray(read, dtype=bool).tolist()
    cc = np.asarray(chart_conv, dtype=float).tolist()
    cv = np.asarray(chart_value, dtype=float).tolist()
    cok = np.asarray(chart_ok, dtype=bool).tolist()
    T = len(c)
    cell = np.full(T, UNREAD, dtype=int)
    tgt_row = np.full(T, -1, dtype=int)
    held = np.zeros(T, dtype=bool)
    since = np.zeros(T, dtype=int)
    frm = np.full(T, UNREAD, dtype=int)
    chart = np.full(T, UNREAD, dtype=int)
    lead = np.zeros(T, dtype=int)

    g_row = None
    g_cell, g_from, g_since = UNREAD, UNREAD, 0
    for t in range(T):
        now = UNREAD
        if rd[t] and np.isfinite(c[t]) and np.isfinite(v[t]):
            tgt = row_of(c[t], z1)
            gp = int(g[t]) if np.isfinite(g[t]) else 0
            if g_row is None or not rdy[t]:
                g_row = tgt
            elif tgt > g_row and gp >= 1:
                g_row = tgt if gp == 2 else g_row + 1
            elif tgt < g_row and gp <= -1:
                g_row = tgt if gp == -2 else g_row - 1
            held[t] = g_row != tgt
            tgt_row[t] = tgt
            now = g_row * 4 + col_of(v[t], theta)
        else:
            g_row = None
        if now != g_cell:
            g_from, g_cell, g_since = g_cell, now, t
        cell[t], frm[t], since[t] = g_cell, g_from, g_since
        if cok[t] and np.isfinite(cc[t]) and np.isfinite(cv[t]):
            chart[t] = row_of(cc[t], z1) * 4 + col_of(cv[t], theta)
        if g_cell != UNREAD and chart[t] != UNREAD:
            du = UNITS[chart[t]] - UNITS[g_cell]
            lead[t] = 1 if du > 0 else -1 if du < 0 else 0
    return pd.DataFrame({"cvg_cell": cell, "cvg_target_row": tgt_row, "cvg_held": held,
                         "cvg_since": since, "cvg_from": frm, "cvg_chart": chart,
                         "cvg_lead": lead}, index=index)


def tooltip(cell: int, units: float, bars: int, frm: int, chart: int, lead: int,
            held: bool) -> str:
    """The Pine's gTip: what the state means, its units, and how to read the push."""
    cell, frm, chart = int(cell), int(frm), int(chart)
    if cell == UNREAD:
        return MEANING[UNREAD]
    s = (f"{NAMES[cell]} - {MEANING[cell]}.\n\n{units:.4g} units · {bars} bars here"
         + (f" · came from {NAMES[frm]}" if frm not in (UNREAD, cell) else ""))
    if chart != UNREAD:
        s += (f"\n\nCHART CELL: {NAMES[chart]} - where the trace's own conviction and value "
              "place it" + (". The chart is already in a better cell than the ladder: it LEADS ↑."
                            if lead > 0 else
                            ". The chart is already in a weaker cell than the ladder: it LEADS ↓."
                            if lead < 0 else "."))
    s += "\n\nREAD THE PUSH AGAINST THE ACTION. " + READ_THE_PUSH
    if held:
        s += ("\n\nHELD: the conviction tape has moved to another row, but the push has not "
              "confirmed it, so the state stands.")
    return s


__all__ = [
    "ACTIONS", "ACTION_TONE", "ACTION_UNITS", "COLUMNS", "COL_LABELS", "MEANING", "NAMES", "PUSH_GLYPH",
    "PUSH_TEXT", "READ_THE_PUSH", "ROW_LABELS", "SIDES", "TONES", "TONE_CHIP", "UNITS", "UNREAD",
    "action", "classify", "col_of", "reason", "row_of", "tooltip",
]
