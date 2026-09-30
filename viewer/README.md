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

- **Open run.jsonl (live)**, in Chromium browsers: picks the file with the
  File System Access API and re-reads it every second, so a calibration that
  is still appending to it shows generations filling in.
- **Load a copy** or drop the file on the page, in any browser: a one-shot
  read of a finished run.

The renewal example writes `examples/output/runs/<timestamp>/run.jsonl`:

```sh
plz renewal
```

## Run log format

One JSON object per line, in this order. `JsonlObserver` in
`src/observer.rs` is the writer and the reference for this section.

| type                   | fields                                                                                                                                                                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_started`          | `version`, `id`, `n_particles`, `n_generations`, `quantiles` (or `null`), `params` (`[{name, kind}]`, kind `real` or `int`), `observed` (or `null`)                                                                                                                        |
| `generation_started`   | `generation`, `tolerance` (`null` when infinite)                                                                                                                                                                                                                           |
| `progress`             | `generation`, running `accepted` and `attempts` counts, `batch`: `[{params, distance}]` accepted since the previous progress line. At most one every 100 ms, plus one final flush right before `generation_completed`, so every accepted particle is in exactly one batch. |
| `generation_completed` | `generation`, `stats` (`tolerance`, `accepted`, `attempts`, `acceptance_ratio`, `ess`, `perplexity`, `duration_seconds`), `particles` (`[{params, weight, distance, seed}]`), `trajectories` (`[{particle, values}]`, an even sample of at most 200)                       |
| `generation_abandoned` | `generation`; the run stops here                                                                                                                                                                                                                                           |
| `run_finished`         | `generations`                                                                                                                                                                                                                                                              |

`params` are positional in prior-declaration order; integers are written as
JSON integers. `seed` is a string because it is a `u64`. Trajectories and
`observed` are only present when the model implements `Model::trajectory`
and `Model::observed`.

## Layout

- `src/run.ts`: event types and the reducer that folds lines into a
  `RunState`, plus `LineSplitter` for chunked reads.
- `src/sources.ts`: file drop, one-shot file read, and file-handle tailing.
- `src/stats.ts`: weighted histogram and KDE.
- `src/App.vue`: the page. Charts are `LineChart` and `BarChart` from
  `cfasim-ui`.
- `e2e/fixtures/renewal.jsonl`: a renewal run shrunk with
  `scripts/shrink-run.mjs` for the Playwright test.

## Test

```sh
plz ui unit       # vitest, the reducer and splitter
plz ui e2e        # playwright, loads the fixture through the file input
plz ui lint       # prettier --check
```
