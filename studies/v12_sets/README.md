# Can the evidence make sets A, B and C better? Yes, for set C.

`PROTOCOL.md` was committed before any variant was scored. Stage 1 chose on the research set
(the 151 instruments, daily and hourly). Stage 2 confirmed on fresh data, where the sets had been
measured but never tuned:

- **Fresh daily:** 572 S&P 500 and Nifty 200 stocks, 2005–2026.
- **Fresh hourly:** 595 of the same stocks over the last 730 days, fetched for the first time here.

The variant code rebuilds the current B and C rules bar for bar. They match `ledger.py` on every
instrument in all four datasets.

## The result

**Change set C's trend condition to the opposing side, in its strong state.**

| | ▲ | ▼ |
|---|---|---|
| **Set C today** | Set B ▲ with the trend tape agreeing (T ≥ 0) | Set B ▼ with T < 0 |
| **Set C proposed** | Set B ▲ with the trend tape **opposing, in its down state (T < −0.5)** | Set B ▼ with the trend tape **in its up state (T > +0.5)** |

Results, edge over 10 bars in σ (stage 2 at 97.5%, Bonferroni over the two changes):

| | Research daily | Research hourly | **Fresh daily** | **Fresh hourly** | Verdict |
|---|---|---|---|---|---|
| C ▼ today | −0.035 | −0.016 | +0.011 | −0.000 | |
| **C ▼ proposed** | **+0.085** | **+0.168** | **+0.080** [+0.022, +0.142], 53% hit | **+0.132** [+0.084, +0.182], 57% hit | **Confirmed: works and is better** |
| C ▲ today | −0.033 | −0.058 | −0.053 (significantly wrong) | −0.059 (significantly wrong) | |
| **C ▲ proposed** | **+0.087** | **+0.102** | **+0.029** [−0.115, +0.154] | **+0.107** [+0.042, +0.177] | **Better on both; works on hourly, not proven on daily** |

The proposed C ▲ is rarer. It fired 1,202 times on fresh daily data, against 26,720 for today's
rule, so its daily interval is wide.

## What stays

| Set | Verdict |
|---|---|
| **B ▲** | Both B variants are negative on at least one research timeframe, so the current rule stays. Fresh: −0.048 daily, −0.014 hourly. |
| **B ▼** | The current rule beats the variant without the risk filter, so the calm-risk filter earns its place. Fresh: +0.051 daily, +0.046 hourly (significant at 97.5%). |
| **A** | No variant was backed by evidence, and it is still too rare to judge. Fresh: −0.07 / −0.14 daily on 162 / 120 events. |

## Why C was backwards

The pane's readings have no directional edge on their own, but the **trend tape tells you which
way a flip will fail.**

- A flip *with* the trend comes late: the move is already priced in, and it reverts.
- A flip *against* a strong trend is the market breaking the trend's grip, and it continues.

This is the same mechanism as the confirmed band-break signal (`studies/v11_lab`): a downtrend
name breaking up through its band. Today's C asked for agreement, so it selected the wrong half.

## Caveats

- **Partial overlap with earlier evidence.** The fresh daily stocks were also used to confirm the
  v11 band-break signal, and the proposed C ▲ (an up-flip in a downtrend) overlaps with it. Fresh
  hourly is fully independent of everything before it.
- **Survivorship.** Fresh stocks are today's index members. The proposed C ▼ is a short, so
  survivorship works against it; it holds anyway.
- **One more variant, recorded but not chosen.** Cxn, the opposing trend without the risk filter,
  also did well on fresh data (C ▲ +0.067 daily, +0.093 hourly). It lost the stage-1 choice by a
  hair on ▼ and clearly on ▲, so it is not proposed.
