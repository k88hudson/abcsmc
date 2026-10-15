# abcsmc viewer

A static page for watching an ABC-SMC calibration as it runs, or reviewing
one afterwards. It reads the `run.jsonl` event log that `JsonlObserver`
writes and never talks to the engine directly.

## Use

```sh
plz ui dev        # http://localhost:5173
plz ui build      # static site in dist/
```

Then, in the page:

- **Open run.jsonl** or drop the file on the page, in Chromium browsers:
  opens the file with the File System Access API and re-reads it every
  second, so a calibration that is still appending to it shows generations
  filling in, and a finished one is simply read. The file is reopened on the
  next visit.
- **Load a finished run** or drop the file, in other browsers: a one-shot
  read. A copy is kept so the next visit shows the same run.
- **Clear** empties the page and forgets the remembered run.

The renewal example writes `examples/output/runs/<timestamp>/run.jsonl`:

```sh
plz renewal
```

## Run log format

One JSON object per line, in this order. `JsonlObserver` in
`src/observer.rs` is the writer and the reference for this section.

| type                   | fields                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_started`          | `version`, `id`, `description` and `distance_description` (each a string or `null`; absent in older logs), `n_particles`, `n_generations`, `quantiles` (or `null`), `adaptive` (`{quantile, floor, min_acceptance, max_generations}` or `null`; absent in older logs), `params` (`[{name, kind, prior}]`, kind `real` or `int`, `prior` a type-tagged document such as `{"type":"Uniform","a":0.5,"b":3.0}`, absent in older logs), `observed` (`[{name, values}]`, the distance's observed series, name a string or `null`; older logs hold one series' values or `null`), `markers` (`[{x, label}]`, label a string or `null`; absent in older logs), `ranges` (`[{x, lower, upper, value, series}]`, the ranges the distance fits series to, `value` a number or `null`, `series` a string or `null` for the unnamed series; absent in older logs), `calendar` (`{start, step_days}` or `null`: index `i` is the date `start + i * step_days` days, `start` a `yyyy-mm-dd` string; absent in older logs) |
| `generation_started`   | `generation`, `tolerance` (`null` when infinite)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `progress`             | `generation`, running `accepted` and `attempts` counts, `batch`: `[{params, distance}]` accepted since the previous progress line. At most one every 100 ms, plus one final flush right before `generation_completed`, so every accepted particle is in exactly one batch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `generation_completed` | `generation`, `stats` (`tolerance`, `accepted`, `attempts`, `acceptance_ratio`, `ess`, `perplexity`, `duration_seconds`), `particles` (`[{params, weight, distance, seed}]`), `trajectories` (`[{particle, series}]`, an even sample of at most 200, `series` the model's series as `[{name, values}]`), `rejected` (`[{distance, series}]`, the first 200 simulations the generation rejected; absent in older logs). Older logs have `values` of one unnamed series in place of `series`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `generation_abandoned` | `generation`; the run stops here                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `run_finished`         | `generations`, the number completed since `run_started`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `run_resumed`          | `completed` (generations kept from before) and `n_generations` (the total now planned); the run is under way again, and a generation that was in progress starts over with a second `generation_started`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `projection`           | `label`, `trajectories` (`[{weight, series}]`, older logs `values`): posterior particles simulated under a scenario, appended after the run by `Projection::append_to`. A later line with the same label replaces the earlier one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `comparison`           | `label`, `series` (`[{name, values}]`): data to set the projections against that the run was not fitted to (a held-out stretch of the observed series, say), appended after the run by `Comparison::append_to`. Each series is drawn on the projections of its name. A later line with the same label replaces the earlier one.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `markers`              | `markers` (`[{x, label}]`): positions to mark on the series charts, appended after the run by `append_markers`. A marker with a label replaces an earlier one of the same label; one without is added.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `ranges`               | `ranges` (`[{x, lower, upper, value, series}]`): target ranges to draw as whiskers on the series' charts, appended after the run by `append_ranges`. A range replaces an earlier one on the same series at the same `x`; any other is added.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `calendar`             | `calendar` (`{start, step_days}`): the dates of the series' indices, appended after the run by `append_calendar`. The latest replaces the run's.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

`params` are positional in prior-declaration order; integers are written as
JSON integers. `seed` is a string because it is a `u64`. Trajectories and
`observed` are only present when the model implements `CalibrationModel::trajectory`
(or `trajectories`, for several named series) and the run's distance provides
`Distance::observed` (or `observed_series`). A series is drawn against the
observed series of the same name; the viewer draws every series chart once
per series, observed ones first.

## Layout

- `src/run.ts`: event types and the reducer that folds lines into a
  `RunState`, plus `LineSplitter` for chunked reads.
- `src/sources.ts`: file drop, one-shot file read, and file-handle tailing.
- `src/recent.ts`: remembers the last run in IndexedDB so the next visit
  reopens it (the file handle of a watched run, a saved copy of a loaded one).
- `src/stats.ts`: weighted histogram, KDE, quantiles, the per-parameter
  summaries behind the Posteriors table, and the weighted correlation behind
  the pairwise scatters.
- `src/ChartTip.vue`: the hover tooltip body shared by every chart, a heading
  plus one named row per series. The trajectory and band charts build their
  own rows: the observed point, the weighted mean, and the median and range
  of the accepted and rejected trajectories at that index, or the mean, the
  median, and each band's bounds.
- `src/App.vue`: the page, in tabs: Calibration, and Target data, Priors, and
  Projections when the log has them. Charts are `LineChart` and `BarChart` from
  `cfasim-ui`.
- `e2e/fixtures/renewal.jsonl`: a renewal run shrunk with
  `scripts/shrink-run.mjs` for the Playwright test.

## Test

```sh
plz ui unit       # vitest, the reducer and splitter
plz ui e2e        # playwright, loads the fixture through the file input
plz ui lint       # prettier --check
```
