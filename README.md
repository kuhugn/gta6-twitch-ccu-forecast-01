# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kuhugn/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kuhugn/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An hour-by-hour quantile forecaster for a game launch day's Twitch category
viewers, entered in make.fun's [“Forecast GTA VI's launch-day Twitch viewers
hour by hour”](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty.
Given a case — everything recorded up to the end of release day's first hour,
that hour's viewer count included, plus a library of past launches — the
program writes, for each of hours 2 to 24, five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's concurrent viewers. It is scored by mean pinball loss on
the log scale, so it aims to state the quantiles it honestly believes.

## Method

Everything is derived at run time from the case's own `comparables`; nothing
about any specific launch is hardcoded.

- For each forecast hour, the log ratio ln(viewers_h / viewers_1) is modelled
  with a small ridge regression on two checkpoint features: the first hour's
  size, saturated near 60k viewers (below that, a small first hour means the
  big regions have not unlocked yet and the day ramps far above hour 1; above
  it, the launch is already live worldwide and the day follows the diurnal
  cycle), and the pre-release trajectory (the mean of the three hours before
  the day, relative to hour 1).
- The spread at each hour is the empirical distribution of the regression's
  leave-one-out residuals across the comparables — honest, neither fitted to a
  normal shape nor tuned on the answer — contracted 10% toward its median,
  because the raw residual spread over-covers in practice.
- For large first hours the quantiles are blended (up to 60%) with the pooled
  quantiles of the “hot” comparables alone: launches whose release-day peak
  stayed within 3× their hour 1.
- Anything unexpected falls back to the house persistence forecast for that
  hour, so a case always gets a valid file.

Checked by leave-one-out over the 37 usable past launches in the practice
kit (each launch forecast from the others, through this exact program): mean
pinball loss 0.381 over all of them, 0.238 over the 25 large launches, against
the house baseline's 0.930 / 0.309 on the same harness. (First submitted
version, with a 100k hinge, a 50% blend and no spread contraction, scored
0.385 / 0.245 on the same harness and a 0.241 practice loss on the house's
board.)

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. The program is a pure
function of the case bytes: two runs agree byte for byte.
