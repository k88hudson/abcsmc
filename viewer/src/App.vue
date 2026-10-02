<script setup lang="ts">
import { computed, onMounted, onScopeDispose, reactive, ref, watch } from "vue";
import { Button, SelectBox, SidebarLayout, Toggle } from "cfasim-ui/components";
import { BarChart, DataTable, LineChart } from "cfasim-ui/charts";
import ChartTip from "./ChartTip.vue";
import { columnsToCsv, fileStem } from "./csv";
import { downloadBytes, exportOutputs, SUMMARY_COLUMNS } from "./export";
import {
  applyLine,
  completedGenerations,
  currentGeneration,
  emptyRun,
  priorLabel,
  type Generation,
  type RunState,
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
  normalizedWeights,
  parameterSummaries,
  quantileBands,
  weightedCorrelation,
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
const showRejected = ref(true);
const showObserved = ref(true);
const allGenerations = ref(false);
const tab = ref("calibration");
const facet = ref(false);
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
  correlationParam.value = undefined;
}

function onLines(lines: string[]) {
  for (const line of lines) applyLine(run, line);
}

function watchHandle(handle: FileSystemFileHandle) {
  reset();
  source.value = handle.name;
  live.value = true;
  stopTail = tailHandle(handle, onLines, (e) => {
    error.value = String(e);
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
    ...(run.observed ? [{ value: "target", label: "Target data" }] : []),
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
    ? `Running generation ${g.generation} of ${run.nGenerations - 1}`
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

// Median and central 50% and 90% bands of each projection, with the observed
// series on top so the fitted window and the projected part read together.
const projectionCharts = computed(() =>
  run.projections.map((projection) => {
    const { x, bands } = quantileBands(
      projection.trajectories,
      [0.05, 0.25, 0.5, 0.75, 0.95],
    );
    const series: {
      x: number[];
      data: number[];
      color: string;
      strokeWidth: number;
      dots: boolean;
      legend: string;
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
    if (run.observed) {
      series.push({
        x: run.observed.map((_, i) => i),
        data: run.observed,
        color: palette.value.observed,
        strokeWidth: 2.5,
        dots: true,
        legend: "Observed",
      });
    }
    const observed = run.observed;
    return {
      label: projection.label,
      title: `Projection: ${projection.label} (${projection.trajectories.length} particles)`,
      filename: fileStem(run.id, "projection", projection.label),
      csv: () =>
        columnsToCsv([
          { header: "index", values: x },
          ...(observed ? [{ header: "observed", values: observed }] : []),
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
  }),
);

// The observed series the distance fits, on its own.
const targetChart = computed(() => {
  const observed = run.observed;
  if (!observed) return null;
  const x = observed.map((_, i) => i);
  return {
    points: observed.length,
    table: { index: x, observed },
    title: "Target data",
    filename: fileStem(run.id, "target"),
    series: [
      {
        x,
        data: observed,
        color: palette.value.observed,
        strokeWidth: 2,
        dots: true,
        legend: "Observed",
      },
    ],
    csv: () =>
      columnsToCsv([
        { header: "index", values: x },
        { header: "observed", values: observed },
      ]),
  };
});

interface TrajectoryView {
  label: string;
  nParticles: number;
  // `name` is the series' CSV column.
  trajectories: { name: string; values: number[] }[];
  rejected: { name: string; values: number[] }[];
}

function trajectoryViewOf(
  gens: Generation[],
  label: string,
  prefixed: boolean,
): TrajectoryView {
  const prefix = (g: Generation) => (prefixed ? `gen_${g.generation}_` : "");
  return {
    label,
    nParticles: gens.reduce((n, g) => n + g.particles.length, 0),
    trajectories: gens.flatMap((g) =>
      g.trajectories.map((t) => ({
        name: `${prefix(g)}particle_${t.particle}`,
        values: t.values,
      })),
    ),
    rejected: gens.flatMap((g) =>
      g.rejected.map((t, i) => ({
        name: `${prefix(g)}rejected_${i}`,
        values: t.values,
      })),
    ),
  };
}

// Everything the trajectories section draws: the shown generation, or every
// completed generation together.
const trajectoryView = computed<TrajectoryView | null>(() => {
  if (allGenerations.value)
    return completed.value.length > 0
      ? trajectoryViewOf(completed.value, "All generations", true)
      : null;
  const g = shownGeneration.value;
  return g ? trajectoryViewOf([g], generationLabel(g), false) : null;
});

// One chart per view: all of it, or one generation each when faceted.
const trajectoryCharts = computed(() => {
  const all = trajectoryView.value;
  if (!all) return [];
  const views =
    allGenerations.value && facet.value
      ? completed.value.map((g) =>
          trajectoryViewOf([g], generationLabel(g), false),
        )
      : [all];
  return views.map((view) => ({
    key: view.label,
    series: trajectorySeries(view),
    ...trajectoryDownload(view),
    // Facets are narrow, so their titles are short.
    ...(views.length > 1
      ? {
          title: `${view.label}: ${view.trajectories.length} accepted, ${view.rejected.length} rejected`,
        }
      : {}),
  }));
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

// The extent of the series that are turned on, across the whole section.
// The legend's key series span it, so facets share axes that fit what is
// showing. With everything off it falls back to the full extent, which keeps
// the legend there to turn things back on.
const trajectoryFrame = computed(() => {
  const g = trajectoryView.value;
  const collect = (onlyShown: boolean) => {
    const all: number[][] = [];
    if (run.observed && (showObserved.value || !onlyShown))
      all.push(run.observed);
    if (g && (showAccepted.value || !onlyShown))
      for (const t of g.trajectories) all.push(t.values);
    if (g && (showRejected.value || !onlyShown))
      for (const t of g.rejected) all.push(t.values);
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
});

function trajectorySeries(g: TrajectoryView) {
  const series: {
    x: number[];
    data: number[];
    color: string;
    strokeWidth: number;
    dots: boolean;
    legend?: string;
    showInTooltip?: boolean;
  }[] = [];
  // One zero-width series per legend item: it puts the item in the chart's
  // legend (so exports carry it) whether or not its lines are drawn.
  const frame = trajectoryFrame.value;
  if (frame) {
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
        x: t.values.map((_, i) => i),
        data: t.values,
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
        x: t.values.map((_, i) => i),
        data: t.values,
        color: palette.value.trajectory,
        strokeWidth: 1,
        dots: false,
        showInTooltip: false,
      });
    }
  }
  if (run.observed && showObserved.value) {
    series.push({
      x: run.observed.map((_, i) => i),
      data: run.observed,
      color: palette.value.observed,
      strokeWidth: 2.5,
      dots: true,
    });
  }
  return series;
}

// Whether the section has anything to draw, whatever the toggles say, so
// turning both off does not hide the toggles.
const hasTrajectoryData = computed(() => {
  const g = trajectoryView.value;
  return (
    !!run.observed ||
    (g !== null && g.trajectories.length + g.rejected.length > 0)
  );
});

// The trajectories chart's legend items, each turning its series off and on.
const trajectoryLegend = computed(() => {
  const g = trajectoryView.value;
  const items = [];
  if (run.observed)
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
  return items;
});

// An rgba() line color at full opacity, for a legend swatch.
function solid(color: string): string {
  return color.replace(/rgba\(([^)]*),[^,)]*\)/, "rgb($1)");
}

// The charts' legend is plain svg, so clicks are matched to its labels by
// position: the label text plus the swatch to its left.
const trajectoryChart = ref<HTMLElement | null>(null);

function legendLabels(): SVGTextElement[] {
  const labels = new Set(trajectoryLegend.value.map((item) => item.label));
  return [
    ...(trajectoryChart.value?.querySelectorAll<SVGTextElement>("svg text") ??
      []),
  ].filter((text) => labels.has(text.textContent?.trim() ?? ""));
}

function legendItemAt(event: MouseEvent) {
  for (const text of legendLabels()) {
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
  if (trajectoryChart.value)
    trajectoryChart.value.style.cursor = legendItemAt(event) ? "pointer" : "";
}

// Strike through the labels of hidden items. Set on the elements so exported
// images show it too.
watch(
  [trajectoryCharts, trajectoryChart],
  () => {
    for (const text of legendLabels()) {
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
  const observed = showObserved.value ? run.observed : null;
  const trajectories = showAccepted.value ? g.trajectories : [];
  const rejected = showRejected.value ? g.rejected : [];
  let length = observed?.length ?? 0;
  for (const t of trajectories) length = Math.max(length, t.values.length);
  for (const t of rejected) length = Math.max(length, t.values.length);
  return {
    title: `Trajectories: ${trajectorySubtitle(g)}`,
    filename: fileStem(run.id, "trajectories", g.label),
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

// The parameter on every panel's x axis; unset means the first one.
const correlationParam = ref<string | undefined>(undefined);
const correlationSort = ref("strongest");

const CORRELATION_SORTS = [
  { value: "strongest", label: "Strongest |r| first" },
  { value: "weakest", label: "Weakest |r| first" },
  { value: "name", label: "By name" },
];

const correlationParamOptions = computed(() =>
  run.params.map((p) => ({ value: p.name, label: p.name })),
);

// Index of the chosen parameter, falling back to the first.
const correlationIndex = computed(() =>
  Math.max(
    0,
    run.params.findIndex((p) => p.name === correlationParam.value),
  ),
);

const correlationChoice = computed({
  get: () => run.params[correlationIndex.value]?.name,
  set: (value: string | undefined) => {
    correlationParam.value = value;
  },
});

function everyNth<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const stride = items.length / limit;
  return Array.from(
    { length: limit },
    (_, i) => items[Math.floor(i * stride)]!,
  );
}

// The chosen parameter against each of the others for the shown generation,
// the prior cloud behind it. Only these pairs are built, so a run with many
// parameters does not draw every combination at once. Each dot is one particle, so a pair is read off the same
// draw, which the marginal charts cannot show.
const correlationPanels = computed(() => {
  const g = shownGeneration.value;
  const prior = completed.value[0];
  if (!g || run.params.length < 2) return [];
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
  const panels = [];
  const i = correlationIndex.value;
  {
    for (let j = 0; j < run.params.length; j++) {
      if (j === i) continue;
      const x = run.params[i]!.name;
      const y = run.params[j]!.name;
      const xs = g.particles.map((p) => p.params[i]!);
      const ys = g.particles.map((p) => p.params[j]!);
      const correlation = weightedCorrelation(xs, ys, weights);
      const r = Number.isFinite(correlation) ? correlation.toFixed(2) : "–";
      panels.push({
        key: `${x}~${y}`,
        x,
        y,
        xLabel: scales[i]!.label,
        yLabel: scales[j]!.label,
        correlation,
        title: `${x} ~ ${y}: r = ${r} (${generationLabel(g)})`,
        filename: fileStem(run.id, "correlation", generationLabel(g), x, y),
        series:
          prior && g !== prior
            ? [cloud(prior, i, j), cloud(g, i, j)]
            : [cloud(g, i, j)],
        csv: () =>
          columnsToCsv([
            { header: x, values: xs },
            { header: y, values: ys },
            { header: "weight", values: weights },
          ]),
      });
    }
  }
  return panels;
});

const shownCorrelationPanels = computed(() => {
  const strength = (r: number) => (Number.isFinite(r) ? Math.abs(r) : 0);
  const panels = [...correlationPanels.value];
  if (correlationSort.value === "strongest")
    panels.sort((a, b) => strength(b.correlation) - strength(a.correlation));
  else if (correlationSort.value === "weakest")
    panels.sort((a, b) => strength(a.correlation) - strength(b.correlation));
  else panels.sort((a, b) => a.key.localeCompare(b.key));
  return panels;
});

const overlays = computed(() => {
  const gens = completed.value;
  const n = gens.length;
  return run.params.map((param, i) => {
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
        title: `Posterior of ${param.name} across generations`,
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
      title: `Posterior of ${param.name} across generations`,
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

const SUMMARY_LABELS: Record<(typeof SUMMARY_COLUMNS)[number], string> = {
  mean: "Mean",
  sd: "SD",
  q05: "5%",
  q25: "25%",
  median: "Median",
  q75: "75%",
  q95: "95%",
};

// Weighted summary of every parameter in the shown generation.
const summaryTable = computed(() => {
  const g = shownGeneration.value;
  if (!g) return null;
  const summaries = parameterSummaries(run.params.length, g.particles);
  const names = run.params.map((p) => p.name);
  const data: Record<string, (string | number)[]> = { parameter: names };
  for (const key of SUMMARY_COLUMNS) data[key] = summaries.map((s) => s[key]);
  return {
    label: generationLabel(g),
    data,
    columnConfig: {
      parameter: { label: "Parameter" },
      ...Object.fromEntries(
        SUMMARY_COLUMNS.map((key) => [
          key,
          {
            label: SUMMARY_LABELS[key],
            align: "right" as const,
            width: 72,
            format: (v: string | number | boolean) => fmt(Number(v)),
          },
        ]),
      ),
    },
    filename: fileStem(run.id, "posteriors", generationLabel(g)),
    // Full precision, unlike the rounded cells.
    csv: () =>
      columnsToCsv(
        Object.entries(data).map(([header, values]) => ({ header, values })),
      ),
  };
});

const cells = computed(() => {
  const gens = completed.value;
  const prior = gens[0];
  return gens.map((g) => {
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
            g === prior
              ? `${param.name}: prior${param.prior ? ` ${priorLabel(param.prior)}` : ""}`
              : `${param.name}: prior and ${generationLabel(g)} posterior`,
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

// Projection tooltips: the chart reports the nearest observed point even
// past the end of the observed series, so drop it there. Observed is series 1.
function withinObserved<T extends { seriesIndex: number }>(
  values: T[],
  index: number,
): T[] {
  const n = run.observed?.length ?? 0;
  return values.filter((v) => v.seriesIndex !== 1 || index < n);
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
              <th>Gen</th>
              <th class="num">Tol.</th>
              <th class="num">Acc.</th>
              <th class="num">ESS</th>
              <th class="num">s</th>
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
              >{{ summaryTable.label }}, weighted mean, SD, and quantiles</small
            >
          </h2>
          <DataTable
            :data="summaryTable.data"
            :column-config="summaryTable.columnConfig"
            :filename="summaryTable.filename"
            :csv="summaryTable.csv"
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
          <div class="trajectory-controls">
            <SelectBox
              v-model="generationChoice"
              :options="generationOptions"
              label="Generation"
              hide-label
              class="generation-select"
            />
            <Toggle
              v-if="allGenerations"
              v-model="facet"
              label="Facet by generation"
            />
          </div>
          <p class="muted legend-hint">
            Click a legend item to hide or show it.
          </p>
          <div
            ref="trajectoryChart"
            :class="{ grid: trajectoryCharts.length > 1 }"
            @click="onLegendClick"
            @mousemove="onLegendHover"
          >
            <div
              v-for="chart in trajectoryCharts"
              :key="chart.key"
              class="cell"
            >
              <LineChart
                zoom
                :series="chart.series"
                :height="trajectoryCharts.length > 1 ? 300 : 260"
                x-label="Index"
                y-label="Value"
                :y-scale-type="yScaleType"
                :filename="chart.filename"
                :data-export-name="chart.filename"
                :title="chart.title"
                :csv="chart.csv"
              />
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
                <div v-for="p in cell.params" :key="p.name" class="cell">
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
                    :height="170"
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

        <section v-if="correlationPanels.length" data-testid="correlations">
          <h2>
            Parameter correlations
            <small v-if="shownGeneration"
              >{{ generationLabel(shownGeneration) }}, weighted Pearson r</small
            >
          </h2>
          <p class="muted">
            The chosen parameter against each of the others. One dot per
            particle, the prior behind in grey. A tilted cloud means the data
            pins the two parameters' combination but not either one alone, which
            the marginal charts cannot show.
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
            <SelectBox
              v-model="correlationChoice"
              :options="correlationParamOptions"
              label="Parameter"
            />
            <SelectBox
              v-if="correlationPanels.length > 1"
              v-model="correlationSort"
              :options="CORRELATION_SORTS"
              label="Sort"
            />
          </div>
          <div class="grid">
            <div
              v-for="panel in shownCorrelationPanels"
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
        </section>
      </div>

      <section
        v-if="targetChart"
        class="tab-panel"
        :class="{ parked: activeTab !== 'target' }"
        :inert="activeTab !== 'target'"
        data-testid="target"
      >
        <p class="muted">{{ targetChart.points }} observed points</p>
        <LineChart
          zoom
          :series="targetChart.series"
          :height="220"
          x-label="Index"
          y-label="Value"
          :y-scale-type="yScaleType"
          :filename="targetChart.filename"
          :data-export-name="targetChart.filename"
          :title="targetChart.title"
          :csv="targetChart.csv"
          tooltip-trigger="hover"
        >
          <template #tooltip="t">
            <ChartTip
              :heading="`Index ${t.index}`"
              :values="t.values"
              :labels="['Observed']"
            />
          </template>
        </LineChart>
        <DataTable
          class="target-table"
          :data="targetChart.table"
          :column-config="{
            index: { label: 'Index', width: 'small' },
            observed: { label: 'Observed', align: 'right', width: 'small' },
          }"
          :filename="targetChart.filename"
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
            :key="chart.label"
            class="projection"
          >
            <LineChart
              zoom
              :series="chart.series"
              :areas="chart.areas"
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
                  :values="withinObserved(t.values, t.index)"
                  :labels="chart.series.map((s) => s.legend)"
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
.trajectory-controls {
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
