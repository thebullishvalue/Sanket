"""
SANKET — the Conviction-Value Grid, 3 × 3 · Pragati v8 (pragati.pine, the "CVG" block)
══════════════════════════════════════════════════════════════════════════════

Pragyam's grid, read for one chart — the grid pragati.pine v8 draws and Pragyam sizes
its book from. A state, not a signal: where a name stands BETWEEN signals, as something
to DO about it, at a weight.

THE AXES are the two ladder tapes:
    rows     UP (buyers in control) past +30 · FAINT between · DOWN (sellers) past −30
    columns  cheap past −θ · fair between · rich past +θ

CONVICTION'S OWN HISTOGRAM RUNS THE ROWS. A row moves to its tape only while
conviction's histogram confirms a push that way — on the move's side, not TURNING, not
QUIET. Otherwise the row is HELD against its tape.

THE 5 × 5 PHASES. Each momentum tape confirms its own instrument's edge: conviction's
row edge while the faster view RUNS on the row's side, value's column edge once the fast
end is REVERTING toward fair. An unconfirmed edge sits half-way to the middle.

GRADED UNITS. The name's units are the nine cells' units read at its shaded position on
the map (the Pine's own tape ramps; a held row keeps between half and all of its cell
by how firmly the push holding it is drawn) — Pragyam's graded map, bit for bit.

THE UNITS — v8, measured (studies/pine_audit.md, research in Pragyam):

                   CHEAP                FAIR                  RICH
    UP  buyers     Buy · turned 3       Hold · building 1½    Trim · paid ¾
    FAINT          Accumulate · basing 1½  Wait · idle 1      Trim · stalling ¾
    DOWN sellers   Buy · capitulation 4 Accumulate · washout 1½  Exit · distribution ¼

v5 / Pragyam's seed had DOWN·cheap Watch 1, DOWN·fair Reduce ½, UP·fair Add 3 and
UP·rich Hold 1½. Chosen on data before 2018 across 380 instruments in six classes and
confirmed after it; in Pragyam's own allocator (monthly, every name held) the same four
moves beat the seed units in both eras on Nifty 50 and Dow 30, at lower turnover.
Crypto trends and is the stated exception.

v9.1: Buy · capitulation 3 → 4. The one cell positive in every era of the v9 audit
(look-ahead-free scoring); at 4 units the grid read as a position improved or tied in all
three eras on both scorers, and in Pragyam's allocator it beat 3 in all three eras on
Nifty 50 and Dow 30 (every-name and top-30 books, net of costs).

Author: @thebullishvalue
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pine_v5

N_ROWS, N_COLS = 3, 3
UNREAD = N_ROWS * N_COLS          # 9

# Cell = row · 3 + column. Rows: 0 DOWN (sellers), 1 FAINT, 2 UP (buyers).
# Columns: 0 cheap, 1 fair, 2 rich.
NAMES = (
    "Buy · capitulation",  "Accumulate · washout", "Exit · distribution",
    "Accumulate · basing", "Wait · idle",          "Trim · stalling",
    "Buy · turned",        "Hold · building",      "Trim · paid",
    "Unread",
)
#: Pragyam's name for each cell — the family its book sizes from.
FAMILY = ("DISLOCATED", "FADING", "DISTRIBUTION", "BASING", "IDLE", "STALLING",
          "TURNED", "BUILDING", "PAID", "UNREAD")
UNITS = (
    4.00, 1.50, 0.25,
    1.50, 1.00, 0.75,
    3.00, 1.50, 0.75,
    1.00,
)
#: The same units keyed as pine_v5.grid reads them: (row −1 DOWN / 0 FAINT / +1 UP, column).
UNITS_RC = {(r - 1, c): UNITS[r * 3 + c] for r in range(3) for c in range(3)}
# The side each action works: +1 builds (Buy, Accumulate), −1 cuts (Trim, Exit), 0 neither.
SIDES = (
    1,  1, -1,
    1,  0, -1,
    1,  0, -1,
    0,
)
MEANING = (
    "sellers in control at a cheap price - capitulation; measured, followed by gains in both eras on every asset class but crypto",
    "sellers in control at a fair price - a washout; measured, followed by gains in both eras outside crypto",
    "sellers in control of a rich price - distribution",
    "cheap, control not yet decided - the base is forming",
    "fair price, control not yet decided - nothing to do",
    "rich, and control has faded - the move has stalled",
    "cheap, and buyers now in control - a dislocation that turned",
    "buyers in control at a fair price - hold it; measured, adding here earned nothing in either era",
    "buyers in control of a price already rich - paid for; take some off",
    "a tape not yet calibrated, or switched off",
)
ROW_LABELS = ("sellers in control", "control undecided", "buyers in control")
COL_LABELS = ("cheap", "fair", "rich")

# ── TONES — one colour per meaning, everywhere the grid appears (Pragyam's CVG_TONE) ──
#   emerald  build — capitulation, a turn        cyan   accumulate — washout, base
#   amber    hold or trim — building, rich        slate  wait / unread
#   rose     exit — distribution
TONES = (
    "emerald", "cyan",  "rose",
    "cyan",    "slate", "amber",
    "emerald", "amber", "amber",
    "slate",
)
TONE_CHIP = {"emerald": "success", "amber": "warning", "cyan": "info",
             "slate": "neutral", "rose": "danger"}
ACTIONS = ("Buy", "Accumulate", "Hold", "Wait", "Trim", "Exit")
ACTION_TONE = {"Buy": "emerald", "Accumulate": "cyan", "Hold": "amber", "Wait": "slate",
               "Trim": "amber", "Exit": "rose", "Unread": "slate"}
ACTION_UNITS = {"Buy": 4.0, "Accumulate": 1.5, "Hold": 1.5, "Wait": 1.0, "Trim": 0.75,
                "Exit": 0.25, "Unread": 1.0}

PUSH_TEXT = {1: "push ↑", 0: "no push", -1: "push ↓"}
PUSH_GLYPH = {1: "↑", 0: "·", -1: "↓"}
READ_THE_PUSH = ("Conviction's histogram runs the rows: a row moves only on a push that way, "
                 "and is HELD otherwise. Measured: inside the capitulation cells the push's "
                 "direction made no consistent difference - the state carried the edge, not "
                 "the timing - so Buy · capitulation and Accumulate · washout do not wait for "
                 "a push ↑.")

COLUMNS = ("cvg_cell", "cvg_units", "cvg_held", "cvg_since", "cvg_from", "cvg_chart",
           "cvg_lead", "cvg_cph", "cvg_vph", "cvg_push")


def action(cell: int) -> str:
    return NAMES[int(cell)].split(" · ")[0]


def reason(cell: int) -> str:
    parts = NAMES[int(cell)].split(" · ")
    return parts[1] if len(parts) > 1 else "needs both tapes"


def row_of(c: float, z1: float) -> int:
    """The conviction tape's row: 2 UP, 1 FAINT, 0 DOWN."""
    return 2 if c >= z1 else 0 if c <= -z1 else 1


def col_of(v: float, theta: float) -> int:
    """The value tape's column: 0 cheap, 1 fair, 2 rich."""
    return 0 if v <= -theta else 2 if v >= theta else 1


def classify(out: pd.DataFrame, cv: pd.Series, raw_sd: pd.Series, cv_ready: np.ndarray,
             p) -> pd.DataFrame:
    """The grid, bar by bar, on pragati.compute's output (pine_v5.grid with v8's units).

    ``cv`` is chart conviction (the Nishchaya v3 oscillator), ``raw_sd`` its raw share's σ,
    ``cv_ready`` its calibration gate. Columns: cvg_cell (0-8, 9 unread), cvg_units (the
    graded units), cvg_held, cvg_since, cvg_from, cvg_chart (where this chart alone would
    place the name), cvg_lead (+1 the chart cell carries more units, −1 fewer), cvg_cph /
    cvg_vph (the 5 × 5 phases: +1 confirmed, −1 not, 0 no edge) and cvg_push (the drawn
    push that runs the rows, −1 … +1).
    """
    ld = (out["c_ladder"] == "down").to_numpy() if "c_ladder" in out.columns else False
    g = pine_v5.grid(out, cv, raw_sd, cv_ready, p, UNITS_RC, ladder_down=ld)
    T = len(g)
    row = g["v5_row"].to_numpy(dtype=float)
    col = g["v5_col"].to_numpy(dtype=float)
    ok = np.isfinite(row) & np.isfinite(col)
    cell = np.where(ok, (np.nan_to_num(row) + 1) * 3 + np.nan_to_num(col), UNREAD).astype(int)
    since = np.zeros(T, dtype=int)
    frm = np.full(T, UNREAD, dtype=int)
    cur, prev, st = UNREAD, UNREAD, 0
    for t in range(T):
        if cell[t] != cur:
            prev, cur, st = cur, cell[t], t
        since[t], frm[t] = st, prev
    th = float(p.theta)
    cc = cv.to_numpy(dtype=float)
    vv = out["value"].to_numpy(dtype=float)
    cok = cv_ready & out["value_built"].fillna(False).to_numpy(bool) & np.isfinite(cc) & np.isfinite(vv)
    chart = np.where(cok, np.array([row_of(a, p.z1) * 3 + col_of(b, th) if k else UNREAD
                                    for a, b, k in zip(cc, vv, cok)]), UNREAD).astype(int)
    u = np.asarray(UNITS)
    lead = np.where((cell != UNREAD) & (chart != UNREAD), np.sign(u[chart] - u[cell]), 0).astype(int)
    return pd.DataFrame({"cvg_cell": cell, "cvg_units": g["v5_units"].to_numpy(),
                         "cvg_held": g["v5_held"].to_numpy(bool), "cvg_since": since,
                         "cvg_from": frm, "cvg_chart": chart, "cvg_lead": lead,
                         "cvg_cph": g["v5_cph"].to_numpy(), "cvg_vph": g["v5_vph"].to_numpy(),
                         "cvg_push": g["v5_push"].to_numpy()}, index=out.index)


def tooltip(cell: int, units: float, bars: int, frm: int, chart: int, lead: int,
            held: bool) -> str:
    """The Pine's gTip: what the state means, its units, and how to read the push."""
    cell, frm, chart = int(cell), int(frm), int(chart)
    if cell == UNREAD:
        return MEANING[UNREAD]
    s = (f"{NAMES[cell]} ({FAMILY[cell]}) - {MEANING[cell]}.\n\n{units:.3g} units, graded · "
         f"{UNITS[cell]:g} flat · {bars} bars here"
         + (f" · came from {NAMES[frm]}" if frm not in (UNREAD, cell) else ""))
    if chart != UNREAD:
        s += (f"\n\nCHART CELL: {NAMES[chart]} - where this chart's own conviction and value "
              "place it" + (". The chart is already in a cell carrying more units: it LEADS ↑."
                            if lead > 0 else
                            ". The chart is already in a cell carrying fewer units: it LEADS ↓."
                            if lead < 0 else "."))
    s += "\n\n" + READ_THE_PUSH
    if held:
        s += ("\n\nHELD: the conviction tape has moved to another row, but conviction's "
              "histogram has not confirmed a push that way, so the row stands.")
    return s


__all__ = [
    "ACTIONS", "ACTION_TONE", "ACTION_UNITS", "COLUMNS", "COL_LABELS", "FAMILY", "MEANING",
    "NAMES", "N_COLS", "N_ROWS", "PUSH_GLYPH", "PUSH_TEXT", "READ_THE_PUSH", "ROW_LABELS",
    "SIDES", "TONES", "TONE_CHIP", "UNITS", "UNITS_RC", "UNREAD", "action", "classify",
    "col_of", "reason", "row_of", "tooltip",
]
