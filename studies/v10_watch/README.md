# Does the Watchlist mean what it says? (v10.2.1)

**Question:** are the names on the Watchlist actually about to fire the event the list says they
will?

**Data:** the Indicator repo's 151 instruments, daily from 2003 to 2026 and hourly from 2024 to
2026.

**Method:** for every bar on which a name would sit on the list, did set B fire within the next
1, 5, 10 or 20 bars? Each group is compared with names on no list. Signals come from `ledger.py`,
the same code the screen runs.

Run it with:
`python3 studies/v10_watch/watch_check.py <Indicator/research> [data|data1h]`

## Answer: half wrong in v10.2.0, fixed in v10.2.1

### What was wrong in v10.2.0

"Ready" checked only the slow conditions: risk calm and not rising, and the ladder (and for C,
the trend) agreeing. It did not check the trigger itself.

For 85% of ready bars, the surprise column was already past zero. The flip is that column crossing
zero, so these names could not fire on the next bar, and none did (0.0%). The column first had to
cross back the other way before it could flip again. Results before the fix are in
`watch_check_daily_before.txt`.

### Fix 1: ready means armed

Ready now also requires the column to be **armed**: still on the far side of zero, so that its
next crossing is the flip. The signals themselves are unchanged; the event counts are identical
before and after.

### Fix 2: a new "near" group

Half of all B events were not on the list the bar before they fired. In 81–87% of those cases,
the only missing condition was the stretch ladder.

The ladder averages the column with three other horizons, so the flip itself often tips it over.
The new **near** group covers names that are armed, have calm risk that is not rising, and have
the ladder on the wrong side by at most 0.2.

## Measured rates after the fix

Percentage of names in each group that fired B, daily data (hourly in brackets):

| Group | Fired next bar | Fired within 10 bars |
|---|---|---|
| ready B·C | 20–24% (19–20%) | 51–60% (50–53%) |
| ready B | 27% (23%) | 60–62% (54–55%) |
| near | 10–12% (9–10%) | 41–44% (37–40%) |
| on no list | 0.3–0.4% | 9–10% |

**In a proven state (set A):** these names are not "about to fire" B. They are already in the
state, and the A triangle has fired. They re-enter the state, and A fires again, 20–28% of the
time within 5 bars. The Watchlist now describes them that way.

**Coverage:** ready and near together held about 80% of all B and C events on the bar before they
fired, on both daily and hourly data. Ready alone held about 50%.

**What this does not measure:** whether the event pays. That question belongs to the Edge Study.
