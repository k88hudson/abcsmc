<script setup lang="ts">
import {
  computed,
  nextTick,
  onMounted,
  onScopeDispose,
  reactive,
  ref,
  watch,
} from "vue";
import {
  Button,
  Hint,
  SelectBox,
  SidebarLayout,
  Toggle,
  ToggleGroup,
} from "cfasim-ui/components";
import { BarChart, DataTable, LineChart } from "cfasim-ui/charts";
import ChartTip from "./ChartTip.vue";
import CorrelationTable, { type CorrelationRow } from "./CorrelationTable.vue";
import Pager from "./Pager.vue";
import {
  correlationPairs,
  pageCount,
  pageOf,
  sortPairs,
  type CorrelationPair,
  type CorrelationSort,
} from "./correlations";
import PosteriorTable, { type PosteriorRow } from "./PosteriorTable.vue";
import { columnsToCsv, fileStem } from "./csv";
import { downloadBytes, exportOutputs } from "./export";
import {
  applyLine,
  completedGenerations,
  currentGeneration,
  emptyRun,
  priorLabel,
  seriesNamed,
  seriesNames,
  type Generation,
  type Series,
  type RunState,
  type TargetRange,
} from "./run";
import {
  canRead,
  clearRecent,
  loadRecent,
  requestRead,
  saveRecent,
} from "./recent";
import {
  pickRunFile,
  readFile,
  supportsFilePicker,
  tailHandle,
} from "./sources";
import {
  axisScale,
  extent,
  finitePoints,
  normalizedWeights,
  parameterSummaries,
  weightedSummary,
  type ParamSummary,
  quantileBands,
  spreadAt,
  weightedHistogram,
  weightedKde,
} from "./stats";
import { chartPalette, generationColor, useDark } from "./theme";

const isDark = useDark();
const palette = computed(() => chartPalette(isDark.value));

const run = reactive<RunState>(emptyRun());
const source = ref<string | null>(null);
const live = ref(false);
// The run shown is the copy kept from an earlier visit, not a fresh read.
const savedCopy = ref(false);
const error = ref<string | null>(null);
const dragging = ref(false);
const selectedGeneration = ref<number | null>(null);
const logScale = ref(false);
const showAccepted = ref(true);
const showRejected = ref(false);
const showObserved = ref(true);
const showRanges = ref(true);
const allGenerations = ref(false);
const tab = ref("calibration");
const facet = ref(false);
// The trajectories section draws the logged trajectories themselves or the
// credible intervals they give.
const trajectoryMode = ref("trajectories");
const TRAJECTORY_MODES = [
  { value: "trajectories", label: "Trajectories" },
  { value: "intervals", label: "Credible intervals" },
];
const main = ref<HTMLElement | null>(null);
const exporting = ref(false);
const yScaleType = computed(() => (logScale.value ? "log" : "linear"));
let stopTail: (() => void) | null = null;

function reset() {
  stopTail?.();
  stopTail = null;
  Object.assign(run, emptyRun());
  error.value = null;
  live.value = false;
  savedCopy.value = false;
  reopenable.value = null;
  selectedGeneration.value = null;
  allGenerations.value = false;
  tab.value = "calibration";
  correlationParam.value = ALL_PARAMS;
}

function onLines(lines: string[]) {
  for (const line of lines) applyLine(run, line);
}

function watchHandle(handle: FileSystemFileHandle) {
  reset();
  source.value = handle.name;
  live.value = true;
  stopTail = tailHandle(handle, onLines, (e) => {
    error.value = e === null ? null : String(e);
  });
}

function openHandle(handle: FileSystemFileHandle) {
  watchHandle(handle);
  void saveRecent({ kind: "handle", name: handle.name, handle });
}

async function openPicked() {
  const handle = await pickRunFile();
  if (handle) openHandle(handle);
}

async function readSnapshot(file: File, saved = false) {
  reset();
  source.value = file.name;
  savedCopy.value = saved;
  try {
    await readFile(file, onLines);
  } catch (e) {
    error.value = String(e);
  }
}

async function loadFile(file: File) {
  void saveRecent({ kind: "file", name: file.name, file });
  await readSnapshot(file);
}

// A handle from an earlier visit that the browser will not read until the
// user allows it again.
const reopenable = ref<FileSystemFileHandle | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);

async function reopen() {
  const handle = reopenable.value;
  if (!handle) return;
  try {
    if (!(await requestRead(handle))) return;
  } catch (e) {
    error.value = String(e);
    return;
  }
  watchHandle(handle);
}

// Empties the page and forgets the remembered run.
function clearRun() {
  reset();
  source.value = null;
  if (fileInput.value) fileInput.value.value = "";
  void clearRecent();
}

onMounted(async () => {
  const recent = await loadRecent();
  // Something was opened while the lookup was pending.
  if (!recent || source.value) return;
  if (recent.kind === "file") {
    await readSnapshot(recent.file, true);
  } else if (await canRead(recent.handle).catch(() => false)) {
    watchHandle(recent.handle);
  } else {
    reopenable.value = recent.handle;
  }
});

function onFileInput(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) void loadFile(file);
}

// A dropped file comes with a handle where the browser has them, so it is
// followed and remembered like a picked one.
function onDrop(event: DragEvent) {
  dragging.value = false;
  const item = event.dataTransfer?.items?.[0] as
    | (DataTransferItem & {
        getAsFileSystemHandle?: () => Promise<FileSystemHandle | null>;
      })
    | undefined;
  const file = event.dataTransfer?.files?.[0];
  // Must be asked for before the handler returns.
  const pending = item?.getAsFileSystemHandle?.();
  if (!pending) {
    if (file) void loadFile(file);
    return;
  }
  void pending
    .then((handle) => {
      if (handle?.kind === "file") openHandle(handle as FileSystemFileHandle);
      else if (file) void loadFile(file);
    })
    .catch(() => {
      if (file) void loadFile(file);
    });
}

async function exportAll() {
  if (!main.value || exporting.value) return;
  exporting.value = true;
  try {
    const { name, zip } = await exportOutputs(run, main.value);
    downloadBytes(zip, name, "application/zip");
  } catch (e) {
    error.value = String(e);
  } finally {
    exporting.value = false;
  }
}

onScopeDispose(() => stopTail?.());

// Target data, priors, and projections each get a tab once the log has them.
const tabs = computed(() => {
  const extra = [
    ...(run.observed.length > 0
      ? [{ value: "target", label: "Target data" }]
      : []),
    ...(run.params.some((p) => p.prior)
      ? [{ value: "priors", label: "Priors" }]
      : []),
    ...(run.projections.length > 0
      ? [{ value: "projections", label: "Projections" }]
      : []),
  ];
  return extra.length > 0
    ? [{ value: "calibration", label: "Calibration" }, ...extra]
    : undefined;
});
const activeTab = computed(() =>
  tabs.value?.some((t) => t.value === tab.value) ? tab.value : "calibration",
);

const completed = computed(() => completedGenerations(run));
const current = computed(() => currentGeneration(run));
const paramCount = computed(() => run.params.length);

const statusLabel = computed(() => {
  if (run.status === "empty")
    return source.value ? "Waiting for events" : "No run loaded";
  if (run.status === "finished") return "Finished";
  const g = current.value;
  return g
    ? `Running ${generationLabel(g)}, ${g.generation + 1} of ${run.nGenerations}`
    : "Running";
});

const shownGeneration = computed<Generation | null>(() => {
  const gens = completed.value;
  if (gens.length === 0) return null;
  const wanted = selectedGeneration.value;
  return gens.find((g) => g.generation === wanted) ?? gens[gens.length - 1]!;
});

// Shared bin edges per parameter across every generation, so prior and
// posterior histograms line up. Generation 0 (the prior sample) usually
// sets the range. Only completed generations count, so progress events do
// not re-bin finished charts; the live batch is used only before any
// generation has completed.
const paramRanges = computed(() =>
  run.params.map((param, i) => {
    const values: number[] = [];
    const done = completed.value;
    if (done.length > 0) {
      for (const g of done)
        for (const p of g.particles) values.push(p.params[i]!);
    } else {
      for (const g of run.generations)
        for (const p of g.live) values.push(p.params[i]!);
    }
    const [lo, hi] = extent(values);
    if (param.kind === "int") {
      const a = Math.floor(lo);
      const b = Math.ceil(hi);
      return { lo: a - 0.5, hi: b + 0.5, bins: Math.max(1, b - a + 1) };
    }
    return { lo, hi: hi > lo ? hi : lo + 1, bins: 20 };
  }),
);

function paramHistogram(
  paramIndex: number,
  particles: { params: number[] }[],
  weights: number[],
) {
  const r = paramRanges.value[paramIndex]!;
  return weightedHistogram(
    particles.map((p) => p.params[paramIndex]!),
    weights,
    r.bins,
    r.lo,
    r.hi,
  );
}

function categoriesFor(paramIndex: number): string[] {
  const r = paramRanges.value[paramIndex]!;
  const kind = run.params[paramIndex]!.kind;
  const width = (r.hi - r.lo) / r.bins;
  return Array.from({ length: r.bins }, (_, i) => {
    const c = r.lo + width * (i + 0.5);
    return kind === "int" ? c.toFixed(0) : c.toFixed(2);
  });
}

// KDE x positions projected into BarChart category-index space so the line
// sits over the right bars.
function kdeOverlay(
  paramIndex: number,
  particles: { params: number[] }[],
  weights: number[],
) {
  const r = paramRanges.value[paramIndex]!;
  const width = (r.hi - r.lo) / r.bins;
  const kde = weightedKde(
    particles.map((p) => p.params[paramIndex]!),
    weights,
    r.lo,
    r.hi,
    100,
  );
  return { x: kde.x.map((v) => (v - r.lo) / width - 0.5), data: kde.y };
}

const liveCharts = computed(() => {
  const g = current.value;
  if (!g || g.live.length === 0) return [];
  const weights = g.live.map(() => 1 / g.live.length);
  return run.params.map((param, i) => {
    const categories = categoriesFor(i);
    const data = paramHistogram(i, g.live, weights).map((b) => b.weight);
    return {
      name: param.name,
      index: i,
      categories,
      data,
      title: `${param.name}: accepted so far in generation ${g.generation} (unweighted)`,
      filename: fileStem(run.id, `gen ${g.generation}`, param.name, "live"),
      csv: () =>
        columnsToCsv([
          { header: param.name, values: categories },
          { header: "mass", values: data },
        ]),
    };
  });
});

const liveFraction = computed(() => {
  const g = current.value;
  if (!g || run.nParticles === 0) return 0;
  return Math.min(1, g.liveAccepted / run.nParticles);
});

// The run's series, observed ones first: every series chart is drawn once
// per entry. `key` is for v-for, since the one unnamed series has no name.
const seriesList = computed(() =>
  seriesNames(run).map((name) => ({ name, key: name ?? "" })),
);

function observedOf(name: string | null): (number | null)[] | null {
  return seriesNamed(run.observed, name);
}

// The `name` series of each trajectory, for those that have it.
function seriesValues<T extends { series: Series[] }>(
  trajectories: T[],
  name: string | null,
): { trajectory: T; values: (number | null)[] }[] {
  const out: { trajectory: T; values: (number | null)[] }[] = [];
  for (const trajectory of trajectories) {
    const values = seriesNamed(trajectory.series, name);
    if (values) out.push({ trajectory, values });
  }
  return out;
}

// A chart title that names the series it shows, when the series has a name:
// "Trajectories: ..." or "Cases trajectories: ...".
function titled(base: string, name: string | null, rest: string): string {
  return name === null
    ? `${base}: ${rest}`
    : `${name} ${base.toLowerCase()}: ${rest}`;
}

// Weighted median and central 50% and 90% bands of `trajectories`, with the
// observed series on top. Shared by the projections and the credible-interval
// view of the trajectories.
function bandChart(
  trajectories: { weight: number; values: (number | null)[] }[],
  name: string | null,
  title: string,
  filename: string,
  comparisons: { label: string; values: (number | null)[] }[] = [],
) {
  {
    const observed = observedOf(name);
    const { x, bands } = quantileBands(
      trajectories,
      [0.05, 0.25, 0.5, 0.75, 0.95],
    );
    const series: {
      x: number[];
      data: number[];
      color: string;
      strokeWidth: number;
      dots: boolean;
      legend: string;
      line?: boolean;
      dotRadius?: number;
      dotFill?: string;
      dotStroke?: string;
    }[] = [
      {
        x,
        data: bands[2]!,
        color: palette.value.median,
        strokeWidth: 2,
        dots: false,
        legend: "Median",
      },
    ];
    // Hollow points, under the observed series where the two overlap.
    for (const comparison of comparisons) {
      series.push({
        ...finitePoints(comparison.values),
        color: palette.value.comparison,
        strokeWidth: 1.5,
        line: false,
        dots: true,
        dotRadius: 3,
        dotFill: "var(--color-bg-0, #fff)",
        dotStroke: palette.value.comparison,
        legend: comparison.label,
      });
    }
    if (observed) {
      series.push({
        ...finitePoints(observed),
        color: palette.value.observed,
        strokeWidth: 2.5,
        dots: true,
        dotRadius: OBSERVED_DOT,
        dotStroke: palette.value.observedOutline,
        legend: "Observed",
      });
    }
    return {
      title,
      filename,
      name,
      bands,
      observed,
      comparisons,
      intervals: rangeIntervals(name),
      csv: () =>
        columnsToCsv([
          { header: "index", values: x },
          ...(observed ? [{ header: "observed", values: observed }] : []),
          ...comparisons.map((c) => ({ header: c.label, values: c.values })),
          { header: "q05", values: bands[0]! },
          { header: "q25", values: bands[1]! },
          { header: "median", values: bands[2]! },
          { header: "q75", values: bands[3]! },
          { header: "q95", values: bands[4]! },
        ]),
      series,
      areas: [
        {
          x,
          lower: bands[0]!,
          upper: bands[4]!,
          color: palette.value.band,
          opacity: palette.value.bandOpacity[0],
          legend: "90%",
        },
        {
          x,
          lower: bands[1]!,
          upper: bands[3]!,
          color: palette.value.band,
          opacity: palette.value.bandOpacity[1],
          legend: "50%",
        },
      ],
    };
  }
}

// One band chart per projection and series, so the fitted window and the
// projected part read together.
const projectionCharts = computed(() =>
  run.projections.flatMap((projection) =>
    seriesList.value.flatMap(({ name, key }) => {
      const trajectories = seriesValues(projection.trajectories, name).map(
        ({ trajectory, values }) => ({ weight: trajectory.weight, values }),
      );
      if (trajectories.length === 0) return [];
      const label =
        name === null ? projection.label : `${projection.label}, ${name}`;
      return [
        {
          key: `${projection.label}/${key}`,
          ...bandChart(
            trajectories,
            name,
            `Projection: ${label} (${trajectories.length} particles)`,
            fileStem(run.id, "projection", projection.label, name),
            run.comparisons.flatMap((c) => {
              const values = seriesNamed(c.series, name);
              return values ? [{ label: c.label, values }] : [];
            }),
          ),
        },
      ];
    }),
  ),
);

// The credible intervals of each generation in view, weighted by the
// trajectories' particles: a row per series with one chart per generation,
// never pooled.
const intervalGroups = computed(() => {
  const gens = allGenerations.value
    ? completed.value
    : shownGeneration.value
      ? [shownGeneration.value]
      : [];
  return seriesList.value.flatMap(({ name, key }) => {
    const charts = gens.flatMap((g) => {
      const trajectories = seriesValues(g.trajectories, name).map(
        ({ trajectory, values }) => ({
          weight: g.particles[trajectory.particle]?.weight ?? 0,
          values,
        }),
      );
      if (trajectories.length === 0) return [];
      const label = generationLabel(g);
      return [
        {
          key: g.generation,
          ...bandChart(
            trajectories,
            name,
            titled(
              "Credible intervals",
              name,
              `${label}, ${trajectories.length} of ${g.particles.length} accepted particles`,
            ),
            fileStem(run.id, "intervals", name, label),
          ),
        },
      ];
    });
    return charts.length > 0 ? [{ key, charts }] : [];
  });
});

// The run's markers as every series chart draws them: fixed vertical lines
// in the theme's marker colour, labelled where the log labels them.
const chartMarkers = computed(() =>
  run.markers.map((marker) => ({
    x: marker.x,
    label: marker.label ?? undefined,
    color: palette.value.marker,
    draggable: false,
  })),
);

// The target ranges of one series as its charts draw them: a whisker from
// the floor to the ceiling at each position.
function rangeIntervals(name: string | null) {
  return run.ranges
    .filter((r) => r.series === name)
    .map((r) => ({ x: r.x, lower: r.lower, upper: r.upper }));
}

const OBSERVED_DOT = 3;

const intervalStyle = computed(() => ({
  color: palette.value.range,
  lineWidth: 1.5,
  capWidth: 6,
  dot: false,
}));

function rangeAt(name: string | null, index: number): TargetRange | null {
  return run.ranges.find((r) => r.series === name && r.x === index) ?? null;
}

// The observed series the distance fits, each on its own.
const targetCharts = computed(() =>
  run.observed.map((observed) => {
    const points = finitePoints(observed.values);
    const ranges = rangeIntervals(observed.name);
    const bound = (pick: (r: TargetRange) => number) =>
      points.x.map((x) => {
        const r = rangeAt(observed.name, x);
        return r ? pick(r) : null;
      });
    return {
      key: observed.name ?? "",
      name: observed.name,
      values: observed.values,
      intervals: ranges,
      points: points.x.length,
      title:
        observed.name === null
          ? "Target data"
          : `Target data: ${observed.name}`,
      filename: fileStem(run.id, "target", observed.name),
      series: [
        {
          ...points,
          color: palette.value.observed,
          strokeWidth: 2,
          dots: true,
          dotRadius: OBSERVED_DOT,
          dotStroke: palette.value.observedOutline,
          legend: "Observed",
        },
      ],
      csv: () =>
        columnsToCsv([
          { header: "index", values: points.x },
          { header: observed.name ?? "observed", values: points.data },
          ...(ranges.length > 0
            ? [
                { header: "range_lower", values: bound((r) => r.lower) },
                { header: "range_upper", values: bound((r) => r.upper) },
              ]
            : []),
        ]),
    };
  }),
);

// Every observed series side by side, one column each.
const targetTable = computed(() => {
  if (run.observed.length === 0) return null;
  const length = Math.max(...run.observed.map((s) => s.values.length));
  const index = Array.from({ length }, (_, i) => i);
  const data: Record<string, (number | string)[]> = { index };
  const columnConfig: Record<
    string,
    { label: string; align?: "right"; width: "small" }
  > = { index: { label: "Index", width: "small" } };
  run.observed.forEach((s, i) => {
    const column = `s${i}`;
    data[column] = index.map((j) => s.values[j] ?? "");
    columnConfig[column] = {
      label: s.name ?? "Observed",
      align: "right",
      width: "small",
    };
    if (!run.ranges.some((r) => r.series === s.name)) return;
    for (const bound of ["lower", "upper"] as const) {
      data[`${column}_${bound}`] = index.map(
        (j) => rangeAt(s.name, j)?.[bound] ?? "",
      );
      columnConfig[`${column}_${bound}`] = {
        label: `${s.name ?? "Observed"} ${bound}`,
        align: "right",
        width: "small",
      };
    }
  });
  return { data, columnConfig, filename: fileStem(run.id, "target") };
});

// One series of the trajectories in view: the logged accepted and rejected
// trajectories' values for it, and the observed series of its name.
interface TrajectoryView {
  label: string;
  // The series shown; null for the one unnamed series.
  name: string | null;
  observed: (number | null)[] | null;
  nParticles: number;
  // `name` is the series' CSV column.
  trajectories: { name: string; values: (number | null)[] }[];
  rejected: { name: string; values: (number | null)[] }[];
}

function trajectoryViewOf(
  gens: Generation[],
  label: string,
  prefixed: boolean,
  name: string | null,
): TrajectoryView {
  const prefix = (g: Generation) => (prefixed ? `gen_${g.generation}_` : "");
  return {
    label,
    name,
    observed: observedOf(name),
    nParticles: gens.reduce((n, g) => n + g.particles.length, 0),
    trajectories: gens.flatMap((g) =>
      seriesValues(g.trajectories, name).map(({ trajectory, values }) => ({
        name: `${prefix(g)}particle_${trajectory.particle}`,
        values,
      })),
    ),
    rejected: gens.flatMap((g) =>
      seriesValues(g.rejected, name).map(({ values }, i) => ({
        name: `${prefix(g)}rejected_${i}`,
        values,
      })),
    ),
  };
}

// What the trajectories section draws: the shown generation, or every
// completed generation together.
const trajectoryGens = computed(() => {
  if (allGenerations.value)
    return completed.value.length > 0
      ? { gens: completed.value, label: "All generations", prefixed: true }
      : null;
  const g = shownGeneration.value;
  return g ? { gens: [g], label: generationLabel(g), prefixed: false } : null;
});

// The first series' view of it, which stands for all of them where only the
// counts matter: the subtitle, the legend, and whether there is anything.
const trajectoryView = computed<TrajectoryView | null>(() => {
  const g = trajectoryGens.value;
  if (!g) return null;
  const first = seriesList.value[0];
  return trajectoryViewOf(g.gens, g.label, g.prefixed, first?.name ?? null);
});

// A row per series: one chart of all of it, or one per generation when
// faceted, on the series' shared axes.
const trajectoryGroups = computed(() => {
  const g = trajectoryGens.value;
  if (!g) return [];
  return seriesList.value.map(({ name, key }) => {
    const all = trajectoryViewOf(g.gens, g.label, g.prefixed, name);
    const frame = trajectoryFrame(all);
    const views =
      allGenerations.value && facet.value
        ? g.gens.map((gen) =>
            trajectoryViewOf([gen], generationLabel(gen), false, name),
          )
        : [all];
    const charts = views.map((view) => ({
      key: view.label,
      view,
      series: trajectorySeries(view, frame),
      intervals: showRanges.value ? rangeIntervals(name) : [],
      ...trajectoryDownload(view),
      // Facets are narrow, so their titles are short.
      ...(views.length > 1
        ? {
            title: `${name === null ? "" : `${name}, `}${view.label}: ${view.trajectories.length} accepted, ${view.rejected.length} rejected`,
          }
        : {}),
    }));
    return { key, charts };
  });
});

const ALL_GENERATIONS = "all";

const generationOptions = computed(() => [
  { value: ALL_GENERATIONS, label: "All generations" },
  ...completed.value.map((g) => ({
    value: String(g.generation),
    label:
      generationLabel(g) === "Prior" ? "Prior" : `Generation ${g.generation}`,
  })),
]);

const generationChoice = computed({
  get: () =>
    allGenerations.value
      ? ALL_GENERATIONS
      : String(shownGeneration.value?.generation ?? ""),
  set: (value: string | undefined) => {
    if (value === undefined) return;
    allGenerations.value = value === ALL_GENERATIONS;
    if (value !== ALL_GENERATIONS) selectedGeneration.value = Number(value);
  },
});

// The extent of the series that are turned on, across a series' whole view.
// The legend's key series span it, so facets share axes that fit what is
// showing. With everything off it falls back to the full extent, which keeps
// the legend there to turn things back on.
type Frame = { xMax: number; yMin: number; yMax: number } | null;

function trajectoryFrame(g: TrajectoryView): Frame {
  const collect = (onlyShown: boolean) => {
    const all: (number | null)[][] = [];
    if (g.observed && (showObserved.value || !onlyShown)) all.push(g.observed);
    if (showAccepted.value || !onlyShown)
      for (const t of g.trajectories) all.push(t.values);
    if (showRejected.value || !onlyShown)
      for (const t of g.rejected) all.push(t.values);
    if (showRanges.value || !onlyShown)
      for (const r of run.ranges)
        if (r.series === g.name) all.push([r.lower, r.upper]);
    return all;
  };
  const shown = collect(true);
  const all = shown.length > 0 ? shown : collect(false);
  let xMax = 0;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const values of all) {
    xMax = Math.max(xMax, values.length - 1);
    const [lo, hi] = extent(values);
    yMin = Math.min(yMin, lo);
    yMax = Math.max(yMax, hi);
  }
  return yMin <= yMax ? { xMax, yMin, yMax } : null;
}

function trajectorySeries(g: TrajectoryView, frame: Frame) {
  const series: {
    x: number[];
    data: number[];
    color: string;
    strokeWidth: number;
    dots: boolean;
    legend?: string;
    showInTooltip?: boolean;
  }[] = [];
  if (frame) {
    // The chart reads the hovered index off its first series, so the first
    // one has every index: the key series below have only their two ends.
    series.push({
      x: Array.from({ length: frame.xMax + 1 }, (_, i) => i),
      data: new Array<number>(frame.xMax + 1).fill(frame.yMin),
      color: "transparent",
      strokeWidth: 0,
      dots: false,
      showInTooltip: false,
    });
    // One zero-width series per legend item: it puts the item in the chart's
    // legend (so exports carry it) whether or not its lines are drawn.
    for (const item of trajectoryLegend.value) {
      series.push({
        x: [0, frame.xMax],
        data: [frame.yMin, frame.yMax],
        color: item.shown.value ? item.color : palette.value.legendOff,
        strokeWidth: 0,
        dots: false,
        legend: item.label,
        showInTooltip: false,
      });
    }
  }
  // Rejected first, so accepted trajectories draw over them.
  if (showRejected.value) {
    for (const t of g.rejected) {
      series.push({
        ...finitePoints(t.values),
        color: palette.value.rejected,
        strokeWidth: 1,
        dots: false,
        showInTooltip: false,
      });
    }
  }
  if (showAccepted.value) {
    for (const t of g.trajectories) {
      series.push({
        ...finitePoints(t.values),
        color: palette.value.trajectory,
        strokeWidth: 1,
        dots: false,
        showInTooltip: false,
      });
    }
  }
  if (g.observed && showObserved.value) {
    series.push({
      ...finitePoints(g.observed),
      color: palette.value.observed,
      strokeWidth: 2.5,
      dots: true,
      dotRadius: OBSERVED_DOT,
      dotStroke: palette.value.observedOutline,
    });
  }
  return series;
}

// Whether the section has anything to draw, whatever the toggles say, so
// turning both off does not hide the toggles.
const hasTrajectoryData = computed(() => {
  const g = trajectoryView.value;
  return (
    run.observed.length > 0 ||
    (g !== null && g.trajectories.length + g.rejected.length > 0)
  );
});

// The trajectories chart's legend items, each turning its series off and on.
const trajectoryLegend = computed(() => {
  const g = trajectoryView.value;
  const items = [];
  if (run.observed.length > 0)
    items.push({
      label: "Observed",
      color: palette.value.observed,
      shown: showObserved,
    });
  if (g && g.trajectories.length > 0)
    items.push({
      label: "Accepted",
      color: solid(palette.value.trajectory),
      shown: showAccepted,
    });
  if (g && g.rejected.length > 0)
    items.push({
      label: "Rejected",
      color: solid(palette.value.rejected),
      shown: showRejected,
    });
  if (run.ranges.length > 0)
    items.push({
      label: "Target range",
      color: palette.value.range,
      shown: showRanges,
    });
  return items;
});

// An rgba() line color at full opacity, for a legend swatch.
function solid(color: string): string {
  return color.replace(/rgba\(([^)]*),[^,)]*\)/, "rgb($1)");
}

// The charts' legend is plain svg, so clicks are matched to its labels by
// position: the label text plus the swatch to its left. The listeners sit on
// each chart's own wrapper, which is what moves to the document body when a
// chart is expanded, so the legend works there too.
const trajectoryChart = ref<HTMLElement | null>(null);

function legendLabels(root: ParentNode): SVGTextElement[] {
  const labels = new Set(trajectoryLegend.value.map((item) => item.label));
  return [...root.querySelectorAll<SVGTextElement>("svg text")].filter((text) =>
    labels.has(text.textContent?.trim() ?? ""),
  );
}

function legendItemAt(event: MouseEvent) {
  for (const text of legendLabels(event.currentTarget as HTMLElement)) {
    const box = text.getBoundingClientRect();
    if (
      event.clientX >= box.left - 22 &&
      event.clientX <= box.right + 4 &&
      event.clientY >= box.top - 4 &&
      event.clientY <= box.bottom + 4
    ) {
      const label = text.textContent!.trim();
      return trajectoryLegend.value.find((item) => item.label === label);
    }
  }
  return undefined;
}

function onLegendClick(event: MouseEvent) {
  const item = legendItemAt(event);
  if (item) item.shown.value = !item.shown.value;
}

function onLegendHover(event: MouseEvent) {
  (event.currentTarget as HTMLElement).style.cursor = legendItemAt(event)
    ? "pointer"
    : "";
}

// Strike through the labels of hidden items. Set on the elements so exported
// images show it too.
watch(
  [trajectoryGroups, trajectoryChart],
  () => {
    for (const chart of document.querySelectorAll("[data-trajectory-chart]"))
      for (const text of legendLabels(chart)) {
        const label = text.textContent!.trim();
        const item = trajectoryLegend.value.find((i) => i.label === label);
        const off = item ? !item.shown.value : false;
        text.style.textDecoration = off ? "line-through" : "";
        text.style.opacity = off ? "0.55" : "";
      }
  },
  { flush: "post" },
);

function trajectoryDownload(g: TrajectoryView) {
  const observed = showObserved.value ? g.observed : null;
  const trajectories = showAccepted.value ? g.trajectories : [];
  const rejected = showRejected.value ? g.rejected : [];
  let length = observed?.length ?? 0;
  for (const t of trajectories) length = Math.max(length, t.values.length);
  for (const t of rejected) length = Math.max(length, t.values.length);
  return {
    title: titled("Trajectories", g.name, trajectorySubtitle(g)),
    filename: fileStem(run.id, "trajectories", g.name, g.label),
    csv: () =>
      columnsToCsv([
        { header: "index", values: Array.from({ length }, (_, i) => i) },
        ...(observed ? [{ header: "observed", values: observed }] : []),
        ...trajectories.map((t) => ({
          header: t.name,
          values: t.values,
        })),
        ...rejected.map((t) => ({
          header: t.name,
          values: t.values,
        })),
      ]),
  };
}

function trajectorySubtitle(g: TrajectoryView): string {
  const accepted = `${g.trajectories.length} of ${g.nParticles} accepted particles`;
  return g.rejected.length > 0
    ? `${g.label}, ${accepted}, ${g.rejected.length} rejected simulations`
    : `${g.label}, ${accepted}`;
}

// ---- Pairwise parameter scatters

// Dots drawn per cloud; the correlation always uses every particle.
const MAX_SCATTER_DOTS = 500;

// The parameter every pair must include, on the x axis, or ALL_PARAMS for
// every pair.
const ALL_PARAMS = "*";
const correlationParam = ref<string>(ALL_PARAMS);
const correlationSort = ref<CorrelationSort>("strongest");
const correlationView = ref<"plots" | "table">("table");
const correlationPage = ref(0);

const CORRELATION_SORTS = [
  { value: "strongest", label: "Strongest |r| first" },
  { value: "weakest", label: "Weakest |r| first" },
  { value: "name", label: "By name" },
];
const CORRELATION_VIEWS = [
  { value: "plots", label: "Plots" },
  { value: "table", label: "Table" },
];
const CORRELATION_PAGE_SIZES = { plots: 6, table: 20 };

const correlationParamOptions = computed(() => [
  { value: ALL_PARAMS, label: "All parameters" },
  ...run.params.map((p) => ({ value: p.name, label: p.name })),
]);

// Every pair (or the chosen parameter's) for the shown generation, ranked.
// Correlations are cheap; only the shown page of plots is drawn.
const correlationPairList = computed(() => {
  const g = shownGeneration.value;
  if (!g || run.params.length < 2) return [];
  const first = completed.value[0];
  const focus = run.params.findIndex((p) => p.name === correlationParam.value);
  const pairs = correlationPairs(
    run.params.length,
    g.particles,
    first && first !== g ? first.particles : null,
    focus >= 0 ? focus : null,
  );
  return sortPairs(
    pairs,
    correlationSort.value,
    run.params.map((p) => p.name),
  );
});

// What the grey correlations come from; null when the shown generation is
// generation 0 itself.
const correlationReference = computed(() => {
  const first = completed.value[0];
  if (!first || first === shownGeneration.value) return null;
  return first.tolerance === null ? "prior" : generationLabel(first);
});

const correlationPages = computed(() =>
  pageCount(
    correlationPairList.value.length,
    CORRELATION_PAGE_SIZES[correlationView.value],
  ),
);

watch(
  [correlationParam, correlationSort, correlationView, shownGeneration],
  () => (correlationPage.value = 0),
);
watch(correlationPages, (pages) => {
  if (correlationPage.value >= pages) correlationPage.value = pages - 1;
});

function pairKey(pair: CorrelationPair): string {
  return `${run.params[pair.x]!.name}~${run.params[pair.y]!.name}`;
}

function pairLabel(pair: CorrelationPair): string {
  return `${run.params[pair.x]!.name} ~ ${run.params[pair.y]!.name}`;
}

const correlationRows = computed((): CorrelationRow[] =>
  pageOf(
    correlationPairList.value,
    correlationPage.value,
    CORRELATION_PAGE_SIZES.table,
  ).map((pair) => ({
    key: pairKey(pair),
    label: pairLabel(pair),
    r: pair.r,
    referenceR: pair.referenceR,
  })),
);

// A table row opens the plot page that pair is on.
function showPairPlot(key: string) {
  const index = correlationPairList.value.findIndex((p) => pairKey(p) === key);
  correlationView.value = "plots";
  // The view watcher resets the page; set it after.
  nextTick(() => {
    correlationPage.value = Math.floor(
      Math.max(0, index) / CORRELATION_PAGE_SIZES.plots,
    );
  });
}

function everyNth<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const stride = items.length / limit;
  return Array.from(
    { length: limit },
    (_, i) => items[Math.floor(i * stride)]!,
  );
}

// One scatter per pair on the shown page, the prior cloud behind it. Each
// dot is one particle, so a pair is read off the same draw, which the
// marginal charts cannot show.
const correlationPanels = computed(() => {
  const g = shownGeneration.value;
  const prior = completed.value[0];
  if (!g) return [];
  const scales = paramRanges.value.map((r, i) =>
    axisScale(run.params[i]!.name, Math.max(Math.abs(r.lo), Math.abs(r.hi))),
  );
  const cloud = (generation: Generation, i: number, j: number) => {
    const isPrior = generation === prior;
    const drawn = everyNth(generation.particles, MAX_SCATTER_DOTS);
    return {
      x: drawn.map((p) => p.params[i]! * scales[i]!.factor),
      data: drawn.map((p) => p.params[j]! * scales[j]!.factor),
      // A line through unsorted draws would mean nothing: dots only.
      strokeWidth: 0,
      dots: true,
      dotRadius: isPrior ? 1.6 : 2.2,
      opacity: isPrior ? 0.3 : 0.75,
      color: isPrior ? palette.value.prior : palette.value.posterior,
      legend: generationLabel(generation),
      showInTooltip: false,
    };
  };
  const weights = g.particles.map((p) => p.weight);
  return pageOf(
    correlationPairList.value,
    correlationPage.value,
    CORRELATION_PAGE_SIZES.plots,
  ).map((pair) => {
    const { x: i, y: j } = pair;
    const x = run.params[i]!.name;
    const y = run.params[j]!.name;
    const r = Number.isFinite(pair.r) ? pair.r.toFixed(2) : "–";
    return {
      key: pairKey(pair),
      xLabel: scales[i]!.label,
      yLabel: scales[j]!.label,
      title: `${x} ~ ${y}: r = ${r} (${generationLabel(g)})`,
      filename: fileStem(run.id, "correlation", generationLabel(g), x, y),
      series:
        prior && g !== prior
          ? [cloud(prior, i, j), cloud(g, i, j)]
          : [cloud(g, i, j)],
      csv: () =>
        columnsToCsv([
          { header: x, values: g.particles.map((p) => p.params[i]!) },
          { header: y, values: g.particles.map((p) => p.params[j]!) },
          { header: "weight", values: weights },
        ]),
    };
  });
});

// Draws every chart title line after the first smaller and at normal weight.
// The chart styles the whole title alike, and both exports keep attributes
// but not page CSS, so attributes are set on the lines as they change.
const SUBTITLE = { "font-weight": "normal", "font-size": "12" };
const titleObservers = new WeakMap<HTMLElement, MutationObserver>();
function plainTitleLines(el: HTMLElement) {
  for (const text of el.querySelectorAll("svg text")) {
    const lines = text.querySelectorAll(":scope > tspan");
    lines.forEach((line, i) => {
      if (i === 0) return;
      for (const [attr, value] of Object.entries(SUBTITLE))
        if (line.getAttribute(attr) !== value) line.setAttribute(attr, value);
    });
  }
}
const vPlainSubtitle = {
  mounted(el: HTMLElement) {
    plainTitleLines(el);
    const observer = new MutationObserver(() => plainTitleLines(el));
    observer.observe(el, { childList: true, subtree: true });
    titleObservers.set(el, observer);
  },
  unmounted(el: HTMLElement) {
    titleObservers.get(el)?.disconnect();
  },
};

// A generation's summary of one parameter, as two title lines.
function posteriorLines(g: Generation, i: number): string {
  const s = weightedSummary(
    g.particles.map((p) => p.params[i]!),
    g.particles.map((p) => p.weight),
  );
  return (
    `\nmedian ${fmt(s.median)}, mean ${fmt(s.mean)}, SD ${fmt(s.sd)}` +
    `\n90% ${fmt(s.q05)} – ${fmt(s.q95)}, 50% ${fmt(s.q25)} – ${fmt(s.q75)}`
  );
}

const overlays = computed(() => {
  const gens = completed.value;
  const n = gens.length;
  return run.params.map((param, i) => {
    const title = `Posterior of ${param.name} across generations`;
    if (param.kind === "real") {
      const r = paramRanges.value[i]!;
      const series = gens.map((g, k) => {
        const kde = weightedKde(
          g.particles.map((p) => p.params[i]!),
          normalizedWeights(g.particles.map((p) => p.weight)),
          r.lo,
          r.hi,
          120,
        );
        return {
          x: kde.x,
          data: kde.y,
          color: generationColor(k, n, isDark.value),
          strokeWidth: k === n - 1 ? 2.5 : 1.5,
          dots: false,
          legend: generationLabel(g),
        };
      });
      return {
        name: param.name,
        index: i,
        kind: param.kind,
        series,
        categories: [] as string[],
        title,
        filename: fileStem(run.id, "posterior", param.name),
        csv: () =>
          columnsToCsv([
            { header: param.name, values: series[0]?.x ?? [] },
            ...series.map((s) => ({ header: s.legend, values: s.data })),
          ]),
      };
    }
    const series = gens.map((g, k) => ({
      data: paramHistogram(
        i,
        g.particles,
        normalizedWeights(g.particles.map((p) => p.weight)),
      ).map((b) => b.weight),
      color: generationColor(k, n, isDark.value),
      opacity: 0.35 + (0.6 * k) / Math.max(1, n - 1),
      legend: generationLabel(g),
    }));
    const categories = categoriesFor(i);
    return {
      name: param.name,
      index: i,
      kind: param.kind,
      series,
      categories,
      title,
      filename: fileStem(run.id, "posterior", param.name),
      csv: () =>
        columnsToCsv([
          { header: param.name, values: categories },
          ...series.map((s) => ({ header: s.legend, values: s.data })),
        ]),
    };
  });
});

const priorsTable = computed(() => {
  if (!run.params.some((p) => p.prior)) return null;
  return {
    data: {
      parameter: run.params.map((p) => p.name),
      kind: run.params.map((p) => p.kind),
      prior: run.params.map((p) => (p.prior ? priorLabel(p.prior) : "")),
    },
    columnConfig: {
      parameter: { label: "Parameter" },
      kind: { label: "Kind", width: "small" as const },
      prior: { label: "Prior", width: "large" as const },
    },
    filename: fileStem(run.id, "priors"),
  };
});

// Weighted summary of every parameter in the shown generation.
const summaryTable = computed(() => {
  const g = shownGeneration.value;
  if (!g) return null;
  const summaries = parameterSummaries(run.params.length, g.particles);
  // Shown beside the posterior's spread: generation 0 is a sample of the
  // prior when it ran at infinite tolerance, and otherwise the first filtered
  // one.
  const first = completed.value[0]!;
  const reference = first.tolerance === null ? "prior" : generationLabel(first);
  const base = parameterSummaries(run.params.length, first.particles);
  const range = (lo: number, hi: number) => `${fmt(lo)} – ${fmt(hi)}`;
  const shown = (x: ParamSummary) => [
    fmt(x.mean),
    fmt(x.sd),
    fmt(x.median),
    range(x.q05, x.q95),
    range(x.q25, x.q75),
  ];
  return {
    label: generationLabel(g),
    reference,
    columns: ["Mean", "SD", "Median", "90% CI", "50% CI"],
    rows: summaries.map((s, i): PosteriorRow => ({
      parameter: run.params[i]!.name,
      cells: shown(s).map((value, k) => ({
        value,
        reference: shown(base[i]!)[k]!,
      })),
    })),
  };
});

// One block per generation: every completed one under "All generations",
// otherwise only the shown generation.
const cells = computed(() => {
  const gens = completed.value;
  const prior = gens[0];
  const shown = allGenerations.value
    ? gens
    : gens.filter((g) => g === shownGeneration.value);
  return shown.map((g) => {
    const weights = normalizedWeights(g.particles.map((p) => p.weight));
    const priorWeights = prior
      ? normalizedWeights(prior.particles.map((p) => p.weight))
      : [];
    return {
      generation: g,
      label: generationLabel(g),
      params: run.params.map((param, i) => {
        const categories = categoriesFor(i);
        const priorMass = prior
          ? paramHistogram(i, prior.particles, priorWeights).map(
              (b) => b.weight,
            )
          : [];
        const posteriorMass = paramHistogram(i, g.particles, weights).map(
          (b) => b.weight,
        );
        return {
          name: param.name,
          index: i,
          categories,
          prior: priorMass,
          posterior: posteriorMass,
          kde:
            param.kind === "real" ? kdeOverlay(i, g.particles, weights) : null,
          title:
            (g === prior
              ? `${param.name}: prior${param.prior ? ` ${priorLabel(param.prior)}` : ""}`
              : `${param.name}: prior and ${generationLabel(g)} posterior`) +
            posteriorLines(g, i),
          filename: fileStem(run.id, generationLabel(g), param.name),
          csv: () =>
            columnsToCsv([
              { header: param.name, values: categories },
              { header: "prior", values: priorMass },
              { header: "posterior", values: posteriorMass },
            ]),
        };
      }),
    };
  });
});

// Heading for a histogram tooltip: bins of a real parameter are labeled by
// their center, integer bins by their value.
function binHeading(paramIndex: number, category: string): string {
  const param = run.params[paramIndex]!;
  return `${param.name} ${param.kind === "int" ? "=" : "≈"} ${category}`;
}

// Tooltip rows built here rather than taken from the chart, which reports
// the nearest point of each series even where the series has none.
interface TipRows {
  values: { seriesIndex: number; color: string; value: number | string }[];
  labels: string[];
}

function tipRows(): TipRows & {
  push: (label: string, color: string, value: number | string) => void;
} {
  const values: TipRows["values"] = [];
  const labels: string[] = [];
  return {
    values,
    labels,
    push: (label, color, value) => {
      values.push({ seriesIndex: values.length, color, value });
      labels.push(label);
    },
  };
}

function observedAt(
  observed: (number | null)[] | null,
  index: number,
): number | null {
  const v = observed?.[index];
  return v !== undefined && v !== null && Number.isFinite(v) ? v : null;
}

function range(lo: number, hi: number): string {
  return `${fmt(lo)} – ${fmt(hi)}`;
}

function pushRange(rows: ReturnType<typeof tipRows>, r: TargetRange | null) {
  if (r)
    rows.push("Target range", palette.value.range, range(r.lower, r.upper));
}

// The target data tooltip at `index`: the observed point and its range.
function targetTipRows(
  chart: { name: string | null; values: (number | null)[] },
  index: number,
): TipRows {
  const rows = tipRows();
  const observed = observedAt(chart.values, index);
  if (observed !== null)
    rows.push("Observed", palette.value.observed, observed);
  pushRange(rows, rangeAt(chart.name, index));
  return { values: rows.values, labels: rows.labels };
}

// A band chart's tooltip at `index`: the observed point, the median, and
// each band's bounds.
function bandTipRows(
  chart: {
    name: string | null;
    bands: number[][];
    observed: (number | null)[] | null;
    comparisons: { label: string; values: (number | null)[] }[];
  },
  index: number,
): TipRows {
  const { bands } = chart;
  const rows = tipRows();
  const observed = observedAt(chart.observed, index);
  if (observed !== null)
    rows.push("Observed", palette.value.observed, observed);
  for (const c of chart.comparisons) {
    const value = observedAt(c.values, index);
    if (value !== null) rows.push(c.label, palette.value.comparison, value);
  }
  pushRange(rows, rangeAt(chart.name, index));
  const q = (k: number) => bands[k]![index] ?? NaN;
  rows.push("Median", palette.value.median, q(2));
  rows.push("50% band", palette.value.band, range(q(1), q(3)));
  rows.push("90% band", palette.value.band, range(q(0), q(4)));
  return { values: rows.values, labels: rows.labels };
}

// The trajectories tooltip at `index`: the observed point and, for the
// accepted and rejected trajectories showing, their median and range there.
function trajectoryTipRows(view: TrajectoryView, index: number): TipRows {
  const rows = tipRows();
  const observed = observedAt(view.observed, index);
  if (showObserved.value && observed !== null)
    rows.push("Observed", palette.value.observed, observed);
  if (showRanges.value) pushRange(rows, rangeAt(view.name, index));
  const spread = (
    label: string,
    color: string,
    trajectories: { values: (number | null)[] }[],
  ) => {
    const s = spreadAt(
      trajectories.map((t) => t.values),
      index,
    );
    if (!s) return;
    rows.push(`${label} median`, color, s.median);
    rows.push(`${label} range`, color, range(s.min, s.max));
  };
  if (showAccepted.value)
    spread("Accepted", solid(palette.value.trajectory), view.trajectories);
  if (showRejected.value)
    spread("Rejected", solid(palette.value.rejected), view.rejected);
  return { values: rows.values, labels: rows.labels };
}

function selectGeneration(generation: number) {
  selectedGeneration.value = generation;
  allGenerations.value = false;
}

function generationLabel(g: Generation): string {
  return g.generation === 0 && g.tolerance === null
    ? "Prior"
    : `Gen ${g.generation}`;
}

function tolerance(g: Generation): string {
  return g.tolerance === null ? "∞" : fmt(g.tolerance);
}

function fmt(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return "–";
  return Number.isInteger(x) ? String(x) : x.toFixed(digits);
}
</script>

<template>
  <SidebarLayout v-model:tab="tab" :tabs="tabs">
    <template #topbar>
      <Button
        v-if="completed.length"
        variant="secondary"
        :disabled="exporting"
        @click="exportAll"
        >{{ exporting ? "Exporting…" : "Export outputs" }}</Button
      >
    </template>
    <template #sidebar>
      <section class="side-section">
        <h3>Source</h3>
        <div class="source-buttons">
          <Button v-if="reopenable" data-testid="reopen" @click="reopen"
            >Reopen {{ reopenable.name }}</Button
          >
          <div class="source-row">
            <Button
              v-if="supportsFilePicker()"
              :variant="reopenable ? 'secondary' : undefined"
              @click="openPicked"
              >Open run.jsonl</Button
            >
            <!-- Browsers without file handles can only read the file once. -->
            <label v-else class="file-label">
              <span class="file-label__text">Load a finished run:</span>
              <input
                type="file"
                accept=".jsonl,application/jsonl"
                ref="fileInput"
                aria-label="Load run.jsonl"
                @change="onFileInput"
              />
            </label>
            <Button
              v-if="source || reopenable"
              variant="secondary"
              @click="clearRun"
              >Clear</Button
            >
          </div>
        </div>
        <p v-if="error" class="error">{{ error }}</p>
      </section>

      <section class="side-section">
        <h3>Run</h3>
        <dl class="facts">
          <dt v-if="source">File</dt>
          <dd v-if="source" class="source-name">
            {{ source }}<span v-if="live"> (watching)</span
            ><span v-else-if="savedCopy"> (saved copy)</span>
          </dd>
          <dt>Status</dt>
          <dd data-testid="status">{{ statusLabel }}</dd>
          <template v-if="run.id">
            <dt>Id</dt>
            <dd data-testid="run-id">{{ run.id }}</dd>
            <dt v-if="run.distanceDescription">Distance</dt>
            <dd v-if="run.distanceDescription" data-testid="distance">
              {{ run.distanceDescription }}
            </dd>
            <dt>Particles</dt>
            <dd>{{ run.nParticles }} per generation</dd>
            <dt>Generations</dt>
            <dd>{{ run.nGenerations }}</dd>
            <dt v-if="run.quantiles">Quantiles</dt>
            <dd v-if="run.quantiles">{{ run.quantiles.join(", ") }}</dd>
            <dt>Parameters</dt>
            <dd>
              {{ run.params.map((p) => `${p.name} (${p.kind})`).join(", ") }}
            </dd>
          </template>
        </dl>
      </section>

      <section v-if="run.generations.length" class="side-section">
        <h3>Generations</h3>
        <table class="gen-table">
          <thead>
            <tr>
              <th>
                <span class="th-label"
                  >Gen
                  <Hint
                    text="Generation: 0 is the prior sample, later ones are SMC steps. Click a completed row to show it."
                /></span>
              </th>
              <th class="num">
                <span class="th-label"
                  >Tol.
                  <Hint
                    text="Tolerance: the largest distance from the target a simulation may have to be accepted."
                /></span>
              </th>
              <th class="num">
                <span class="th-label"
                  >Acc.
                  <Hint
                    text="Acceptance ratio: accepted particles over simulations run. While running, accepted so far out of the particles wanted."
                /></span>
              </th>
              <th class="num">
                <span class="th-label"
                  >ESS
                  <Hint
                    text="Effective sample size: 1 / Σ w², the number of equally weighted particles the weights are worth."
                /></span>
              </th>
              <th class="num">
                <span class="th-label"
                  >s <Hint text="Seconds the generation took."
                /></span>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="g in run.generations"
              :key="g.generation"
              :class="{
                selected: shownGeneration?.generation === g.generation,
                running: g.status === 'running',
              }"
              @click="
                g.status === 'completed' && selectGeneration(g.generation)
              "
            >
              <td>{{ g.generation }}</td>
              <td class="num">{{ tolerance(g) }}</td>
              <td class="num">
                <template v-if="g.stats">{{
                  fmt(g.stats.acceptance_ratio)
                }}</template>
                <template v-else-if="g.status === 'abandoned'"
                  >stopped</template
                >
                <template v-else
                  >{{ g.liveAccepted }}/{{ run.nParticles }}</template
                >
              </td>
              <td class="num">{{ g.stats ? fmt(g.stats.ess, 0) : "" }}</td>
              <td class="num">
                {{ g.stats ? fmt(g.stats.duration_seconds, 1) : "" }}
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </template>

    <div
      ref="main"
      class="main"
      :class="{ dragging }"
      @dragover.prevent="dragging = true"
      @dragleave="dragging = false"
      @drop.prevent="onDrop"
    >
      <h1>abcsmc viewer</h1>
      <p v-if="run.description" class="description" data-testid="description">
        {{ run.description }}
      </p>
      <Toggle v-model="logScale" label="Log scale" class="log-toggle" />

      <div
        class="tab-panel"
        :class="{ parked: activeTab !== 'calibration' }"
        :inert="activeTab !== 'calibration'"
      >
        <section v-if="!run.id" class="empty">
          <p v-if="supportsFilePicker()">
            Open the <code>run.jsonl</code> a calibration wrote, or drop it
            here. The page keeps re-reading the file, so a run that is still
            going fills in as its generations complete, and it reopens the file
            on your next visit.
          </p>
          <p v-else>
            Load the <code>run.jsonl</code> a calibration wrote, or drop it
            here. This browser can only read the file once, so load it again to
            see a run's later generations.
          </p>
        </section>

        <section v-if="current" class="live" data-testid="live">
          <h2>
            Generation {{ current.generation }}
            <small>tolerance {{ tolerance(current) }}</small>
          </h2>
          <div
            class="progress"
            role="progressbar"
            :aria-valuenow="current.liveAccepted"
            :aria-valuemax="run.nParticles"
          >
            <div
              class="progress__fill"
              :style="{ width: `${liveFraction * 100}%` }"
            />
          </div>
          <p class="muted">
            {{ current.liveAccepted }} / {{ run.nParticles }} accepted from
            {{ current.liveAttempts }} simulations
            <template v-if="current.liveAttempts > 0">
              ({{ fmt(current.liveAccepted / current.liveAttempts) }} acceptance
              so far)
            </template>
          </p>
          <div v-if="liveCharts.length" class="grid">
            <div v-for="chart in liveCharts" :key="chart.name" class="cell">
              <BarChart
                zoom
                :categories="chart.categories"
                :series="[{ data: chart.data, color: palette.posterior }]"
                :height="160"
                :filename="chart.filename"
                :data-export-name="chart.filename"
                :title="chart.title"
                :csv="chart.csv"
                tooltip-trigger="hover"
              >
                <template #tooltip="t">
                  <ChartTip
                    :heading="`Mass at ${binHeading(chart.index, t.category)}`"
                    :values="t.values"
                    :labels="['Accepted']"
                  />
                </template>
              </BarChart>
            </div>
          </div>
        </section>

        <section v-if="summaryTable" data-testid="posteriors">
          <h2>
            Posteriors
            <small
              >{{ summaryTable.label }}, weighted mean, SD, median, and credible
              intervals</small
            >
          </h2>
          <PosteriorTable
            :columns="summaryTable.columns"
            :rows="summaryTable.rows"
            :reference="summaryTable.reference"
          />
        </section>

        <section
          v-if="trajectoryView && hasTrajectoryData"
          data-testid="trajectories"
        >
          <h2>
            Trajectories
            <small>{{ trajectorySubtitle(trajectoryView) }}</small>
          </h2>
          <div class="chart-controls">
            <ToggleGroup
              v-model="trajectoryMode"
              :options="TRAJECTORY_MODES"
              label="View"
              hide-label
            />
            <SelectBox
              v-model="generationChoice"
              :options="generationOptions"
              label="Generation"
              hide-label
              class="generation-select"
            />
            <Toggle
              v-if="allGenerations && trajectoryMode === 'trajectories'"
              v-model="facet"
              label="Facet by generation"
            />
          </div>
          <p v-if="trajectoryMode === 'intervals'" class="muted legend-hint">
            Weighted median and central 50% and 90% bands of the logged accepted
            trajectories, one chart per generation.
          </p>
          <p v-else class="muted legend-hint">
            Click a legend item to hide or show it.
          </p>
          <div
            v-if="trajectoryMode === 'intervals'"
            class="cells"
            data-testid="intervals"
          >
            <div
              v-for="group in intervalGroups"
              :key="group.key"
              :class="{ grid: group.charts.length > 1 }"
            >
              <div v-for="chart in group.charts" :key="chart.key" class="cell">
                <LineChart
                  zoom
                  :series="chart.series"
                  :areas="chart.areas"
                  :markers="chartMarkers"
                  :intervals="chart.intervals"
                  :interval-style="intervalStyle"
                  :height="group.charts.length > 1 ? 300 : 260"
                  x-label="Index"
                  y-label="Value"
                  :y-scale-type="yScaleType"
                  :filename="chart.filename"
                  :data-export-name="chart.filename"
                  :title="chart.title"
                  :csv="chart.csv"
                  tooltip-trigger="hover"
                >
                  <template #tooltip="t">
                    <ChartTip
                      :heading="`Index ${t.index}`"
                      v-bind="bandTipRows(chart, t.index)"
                    />
                  </template>
                </LineChart>
              </div>
            </div>
          </div>
          <div v-else ref="trajectoryChart" class="cells">
            <div
              v-for="group in trajectoryGroups"
              :key="group.key"
              :class="{ grid: group.charts.length > 1 }"
            >
              <div v-for="chart in group.charts" :key="chart.key" class="cell">
                <LineChart
                  zoom
                  :series="chart.series"
                  :markers="chartMarkers"
                  :intervals="chart.intervals"
                  :interval-style="intervalStyle"
                  :height="group.charts.length > 1 ? 300 : 260"
                  x-label="Index"
                  y-label="Value"
                  :y-scale-type="yScaleType"
                  :filename="chart.filename"
                  :data-export-name="chart.filename"
                  :title="chart.title"
                  :csv="chart.csv"
                  tooltip-trigger="hover"
                  data-trajectory-chart
                  @click="onLegendClick"
                  @mousemove="onLegendHover"
                >
                  <template #tooltip="t">
                    <ChartTip
                      :heading="`Index ${t.index}`"
                      v-bind="trajectoryTipRows(chart.view, t.index)"
                    />
                  </template>
                </LineChart>
              </div>
            </div>
          </div>
        </section>

        <section v-if="completed.length" data-testid="overlays">
          <h2>Posterior across generations</h2>
          <p class="muted">
            Prior in grey, later generations
            {{ isDark ? "lighter" : "darker" }}.
          </p>
          <div class="grid">
            <div v-for="o in overlays" :key="o.name" class="cell">
              <LineChart
                zoom
                v-if="o.kind === 'real'"
                :series="o.series"
                :height="240"
                :x-label="o.name"
                y-label="density"
                :filename="o.filename"
                :data-export-name="o.filename"
                :title="o.title"
                :csv="o.csv"
                tooltip-trigger="hover"
              >
                <template #tooltip="t">
                  <ChartTip
                    :heading="`Density at ${o.name} = ${t.xLabel ?? ''}`"
                    :values="t.values"
                    :labels="o.series.map((s) => s.legend)"
                  />
                </template>
              </LineChart>
              <BarChart
                zoom
                v-else
                :categories="o.categories"
                :series="o.series"
                layout="overlay"
                :height="240"
                :x-label="o.name"
                y-label="mass"
                :filename="o.filename"
                :data-export-name="o.filename"
                :title="o.title"
                :csv="o.csv"
                tooltip-trigger="hover"
              >
                <template #tooltip="t">
                  <ChartTip
                    :heading="`Mass at ${binHeading(o.index, t.category)}`"
                    :values="t.values"
                    :labels="o.series.map((s) => s.legend)"
                  />
                </template>
              </BarChart>
            </div>
          </div>
        </section>

        <section v-if="cells.length" data-testid="cells">
          <h2>Per generation</h2>
          <div class="chart-controls">
            <SelectBox
              v-model="generationChoice"
              :options="generationOptions"
              label="Generation"
              hide-label
              class="generation-select"
            />
          </div>
          <div class="cells">
            <div
              v-for="cell in cells"
              :key="cell.generation.generation"
              class="gen-cell"
              :class="{
                selected:
                  shownGeneration?.generation === cell.generation.generation,
              }"
              @click="selectGeneration(cell.generation.generation)"
            >
              <h4>
                {{ cell.label }}
                <small v-if="cell.generation.stats">
                  tol. {{ tolerance(cell.generation) }}, acceptance
                  {{ fmt(cell.generation.stats.acceptance_ratio) }}, ESS
                  {{ fmt(cell.generation.stats.ess, 0) }}
                </small>
              </h4>
              <div class="grid" :style="{ '--cols': Math.min(paramCount, 3) }">
                <div
                  v-for="p in cell.params"
                  :key="p.name"
                  v-plain-subtitle
                  class="cell"
                >
                  <BarChart
                    zoom
                    :categories="p.categories"
                    :series="[
                      {
                        data: p.prior,
                        color: palette.prior,
                        blendMode: palette.overlapBlend,
                        legend: 'Prior',
                      },
                      {
                        data: p.posterior,
                        color: palette.posterior,
                        blendMode: palette.overlapBlend,
                        legend: 'Posterior',
                      },
                    ]"
                    layout="overlay"
                    :summary-lines="
                      p.kde
                        ? [
                            {
                              x: p.kde.x,
                              data: p.kde.data,
                              color: palette.kde,
                              strokeWidth: 2,
                              dots: false,
                            },
                          ]
                        : []
                    "
                    :height="220"
                    :filename="p.filename"
                    :data-export-name="p.filename"
                    :title="p.title"
                    :csv="p.csv"
                    tooltip-trigger="hover"
                  >
                    <template #tooltip="t">
                      <ChartTip
                        :heading="`Mass at ${binHeading(p.index, t.category)}`"
                        :values="t.values"
                        :labels="['Prior', 'Posterior']"
                      />
                    </template>
                  </BarChart>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section v-if="correlationPairList.length" data-testid="correlations">
          <h2>
            Parameter correlations
            <small v-if="shownGeneration"
              >{{ generationLabel(shownGeneration) }}, weighted Pearson r</small
            >
          </h2>
          <p class="muted">
            Every pair of parameters, or the chosen one's, ranked across all
            pairs before paging. One dot per particle, the prior behind in grey.
            A tilted cloud means the data pins the two parameters' combination
            but not either one alone, which the marginal charts cannot show.
            <template
              v-if="
                shownGeneration &&
                shownGeneration.particles.length > MAX_SCATTER_DOTS
              "
            >
              {{ MAX_SCATTER_DOTS }} of
              {{ shownGeneration.particles.length }} particles are drawn; r uses
              all of them.
            </template>
          </p>
          <div class="correlation-controls">
            <ToggleGroup
              v-model="correlationView"
              :options="CORRELATION_VIEWS"
              label="View"
            />
            <SelectBox
              v-model="correlationParam"
              :options="correlationParamOptions"
              label="Parameter"
            />
            <SelectBox
              v-if="correlationPairList.length > 1"
              v-model="correlationSort"
              :options="CORRELATION_SORTS"
              label="Sort"
            />
          </div>
          <CorrelationTable
            v-if="correlationView === 'table'"
            :rows="correlationRows"
            :reference="correlationReference"
            :color="palette.posterior"
            @select="showPairPlot"
          />
          <div v-else class="grid">
            <div
              v-for="panel in correlationPanels"
              :key="panel.key"
              class="cell"
            >
              <LineChart
                zoom
                :series="panel.series"
                :height="280"
                :x-label="panel.xLabel"
                :y-label="panel.yLabel"
                :filename="panel.filename"
                :data-export-name="panel.filename"
                :title="panel.title"
                :csv="panel.csv"
              />
            </div>
          </div>
          <Pager
            v-model="correlationPage"
            :pages="correlationPages"
            :total="correlationPairList.length"
            noun="pairs"
          />
        </section>
      </div>

      <section
        v-if="targetTable"
        class="tab-panel"
        :class="{ parked: activeTab !== 'target' }"
        :inert="activeTab !== 'target'"
        data-testid="target"
      >
        <div v-for="chart in targetCharts" :key="chart.key">
          <p class="muted">
            <template v-if="chart.name !== null">{{ chart.name }}: </template
            >{{ chart.points }} observed points
          </p>
          <LineChart
            zoom
            :series="chart.series"
            :markers="chartMarkers"
            :intervals="chart.intervals"
            :interval-style="intervalStyle"
            :height="220"
            x-label="Index"
            y-label="Value"
            :y-scale-type="yScaleType"
            :filename="chart.filename"
            :data-export-name="chart.filename"
            :title="chart.title"
            :csv="chart.csv"
            tooltip-trigger="hover"
          >
            <template #tooltip="t">
              <ChartTip
                :heading="`Index ${t.index}`"
                v-bind="targetTipRows(chart, t.index)"
              />
            </template>
          </LineChart>
        </div>
        <DataTable
          class="target-table"
          :data="targetTable.data"
          :column-config="targetTable.columnConfig"
          :filename="targetTable.filename"
        />
      </section>

      <section
        v-if="priorsTable"
        class="tab-panel"
        :class="{ parked: activeTab !== 'priors' }"
        :inert="activeTab !== 'priors'"
        data-testid="priors"
      >
        <DataTable
          :data="priorsTable.data"
          :column-config="priorsTable.columnConfig"
          :filename="priorsTable.filename"
        />
      </section>

      <section
        v-if="projectionCharts.length"
        class="tab-panel"
        :class="{ parked: activeTab !== 'projections' }"
        :inert="activeTab !== 'projections'"
        data-testid="projections"
      >
        <p class="muted">
          Posterior particles simulated forward. Line is the weighted median,
          bands are the central 50% and 90%.
        </p>
        <div class="cells">
          <div
            v-for="chart in projectionCharts"
            :key="chart.key"
            class="projection"
          >
            <LineChart
              zoom
              :series="chart.series"
              :areas="chart.areas"
              :markers="chartMarkers"
              :intervals="chart.intervals"
              :interval-style="intervalStyle"
              :height="280"
              x-label="Index"
              y-label="Value"
              :y-scale-type="yScaleType"
              :filename="chart.filename"
              :data-export-name="chart.filename"
              :title="chart.title"
              :csv="chart.csv"
              tooltip-trigger="hover"
            >
              <template #tooltip="t">
                <ChartTip
                  :heading="`Index ${t.index}`"
                  v-bind="bandTipRows(chart, t.index)"
                />
              </template>
            </LineChart>
          </div>
        </div>
      </section>
    </div>
  </SidebarLayout>
</template>

<style>
/* Non-scoped override of cfasim-ui's SidebarLayout: wider content. */
.MainContent {
  max-width: 1400px !important;
}
</style>

<style scoped>
.main {
  min-height: 80svh;
  min-height: 80vh;
}
/* The tab that is not showing stays laid out at full width, so its charts
   keep their size and "Export outputs" still includes them. */
.tab-panel.parked {
  height: 0;
  overflow: hidden;
  visibility: hidden;
}
/* cfasim-ui's table sizes to its content, so without a cap it is clipped
   on a narrow screen instead of scrolling inside its wrapper. */
:deep(.TableOuter) {
  max-width: 100%;
}
/* The main scroll pane is not positioned in cfasim-ui, so an absolutely
   positioned descendant (a SelectBox's visually hidden label) is contained
   by the whole layout instead, escapes the pane's clipping, and stretches
   the page: the whole layout then scrolls off the top of the window. */
:deep(.MainScroll) {
  position: relative;
}
.target-table {
  margin-top: 1rem;
}
.target-table :deep(.Table th),
.target-table :deep(.Table td) {
  padding-block: 0.3em;
}
.main > h1 {
  margin-top: 0;
}
.main.dragging {
  outline: 3px dashed #2563eb;
  outline-offset: -3px;
}
h2 small,
h4 small {
  font-weight: normal;
  color: var(--color-text-muted, #64748b);
  font-size: 0.8em;
  margin-left: 0.5em;
}
.description {
  margin: 0 0 0.75rem;
  max-width: 80ch;
}
.correlation-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.5rem;
  margin-bottom: 0.75rem;
  font-size: 0.85rem;
}
.correlation-controls > * {
  width: 16rem;
}
.chart-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 1.5rem;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
}
.generation-select {
  width: 16rem;
}
.legend-hint {
  margin: 0 0 0.25rem;
  font-size: 0.85rem;
}
.legend__swatch {
  width: 0.9em;
  height: 0.2em;
  border-radius: 1px;
}
.log-toggle {
  font-size: 0.85rem;
  margin-bottom: 1rem;
}
.side-section {
  margin-bottom: 1.25rem;
}
.side-section h3 {
  margin: 0 0 0.6rem;
}
.side-section:last-child {
  margin-bottom: 0;
}
.source-buttons {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.source-row {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 0.5rem;
}
.file-label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.85rem;
}
.source-name {
  font-family: monospace;
  word-break: break-all;
}
.error {
  color: #dc2626;
}
.facts {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.2rem 0.75rem;
  font-size: 0.9rem;
}
.facts dt {
  color: var(--color-text-muted, #64748b);
}
.facts dd {
  margin: 0;
}
.gen-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.gen-table .th-label {
  display: inline-flex;
  align-items: center;
  gap: 0.15em;
}
.gen-table th,
.gen-table td {
  padding: 0.2rem 0.3rem;
  border-bottom: 1px solid var(--color-border, #e5e7eb);
  text-align: left;
}
.gen-table .num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.gen-table tbody tr {
  cursor: pointer;
}
.gen-table tr.selected {
  background: rgba(37, 99, 235, 0.1);
}
.gen-table tr.running {
  font-style: italic;
}
.muted {
  color: var(--color-text-muted, #64748b);
}
.progress {
  height: 10px;
  background: var(--color-bg-2, #e5e7eb);
  border-radius: 5px;
  overflow: hidden;
}
.progress__fill {
  height: 100%;
  background: #2563eb;
  transition: width 0.3s;
}
.grid {
  display: grid;
  grid-template-columns: repeat(var(--cols, 2), minmax(0, 1fr));
  gap: 1rem;
}
@media (max-width: 800px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
.cells {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.gen-cell {
  border: 1px solid var(--color-border, #e5e7eb);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  cursor: pointer;
}
.gen-cell.selected {
  border-color: #2563eb;
}
.gen-cell h4 {
  margin: 0 0 0.5rem;
}
.empty {
  max-width: 60ch;
}
</style>
