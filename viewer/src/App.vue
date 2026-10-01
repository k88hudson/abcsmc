<script setup lang="ts">
import { computed, onScopeDispose, reactive, ref } from "vue";
import { Button, SidebarLayout, Toggle } from "cfasim-ui/components";
import { BarChart, LineChart } from "cfasim-ui/charts";
import { columnsToCsv, fileStem } from "./csv";
import { downloadBytes, exportOutputs } from "./export";
import {
  applyLine,
  completedGenerations,
  currentGeneration,
  emptyRun,
  type Generation,
  type RunState,
} from "./run";
import {
  pickRunFile,
  readFile,
  supportsFilePicker,
  tailHandle,
} from "./sources";
import {
  extent,
  normalizedWeights,
  quantileBands,
  weightedHistogram,
  weightedKde,
} from "./stats";
import { chartPalette, generationColor, useDark } from "./theme";

const isDark = useDark();
const palette = computed(() => chartPalette(isDark.value));

const run = reactive<RunState>(emptyRun());
const source = ref<string | null>(null);
const live = ref(false);
const error = ref<string | null>(null);
const dragging = ref(false);
const selectedGeneration = ref<number | null>(null);
const logScale = ref(false);
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
  selectedGeneration.value = null;
}

function onLines(lines: string[]) {
  for (const line of lines) applyLine(run, line);
}

async function openLive() {
  const handle = await pickRunFile();
  if (!handle) return;
  reset();
  source.value = handle.name;
  live.value = true;
  stopTail = tailHandle(handle, onLines, (e) => {
    error.value = String(e);
  });
}

async function loadFile(file: File) {
  reset();
  source.value = file.name;
  try {
    await readFile(file, onLines);
  } catch (e) {
    error.value = String(e);
  }
}

function onFileInput(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) void loadFile(file);
}

function onDrop(event: DragEvent) {
  dragging.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file) void loadFile(file);
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

const trajectorySeries = computed(() => {
  const g = shownGeneration.value;
  const series: {
    x: number[];
    data: number[];
    color: string;
    strokeWidth: number;
    dots: boolean;
    legend?: string;
    showInTooltip?: boolean;
  }[] = [];
  if (g) {
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
  return series;
});

const trajectoryDownload = computed(() => {
  const g = shownGeneration.value;
  const observed = run.observed;
  const trajectories = g?.trajectories ?? [];
  let length = observed?.length ?? 0;
  for (const t of trajectories) length = Math.max(length, t.values.length);
  return {
    title: g
      ? `Trajectories: ${generationLabel(g)}, ${trajectories.length} of ${g.particles.length} particles`
      : "Trajectories",
    filename: fileStem(run.id, "trajectories", g ? generationLabel(g) : null),
    csv: () =>
      columnsToCsv([
        { header: "index", values: Array.from({ length }, (_, i) => i) },
        ...(observed ? [{ header: "observed", values: observed }] : []),
        ...trajectories.map((t) => ({
          header: `particle_${t.particle}`,
          values: t.values,
        })),
      ]),
  };
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
          categories,
          prior: priorMass,
          posterior: posteriorMass,
          kde:
            param.kind === "real" ? kdeOverlay(i, g.particles, weights) : null,
          title:
            g === prior
              ? `${param.name}: prior`
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
  <SidebarLayout>
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
          <Button v-if="supportsFilePicker()" @click="openLive"
            >Watch a run (live)</Button
          >
          <label class="file-label">
            <span class="file-label__text">{{
              supportsFilePicker()
                ? "Or load a finished run:"
                : "Load a finished run:"
            }}</span>
            <input
              type="file"
              accept=".jsonl,application/jsonl"
              aria-label="Load run.jsonl"
              @change="onFileInput"
            />
          </label>
        </div>
        <p v-if="error" class="error">{{ error }}</p>
      </section>

      <section class="side-section">
        <h3>Run</h3>
        <dl class="facts">
          <dt v-if="source">File</dt>
          <dd v-if="source" class="source-name">
            {{ source }}<span v-if="live"> (watching)</span>
          </dd>
          <dt>Status</dt>
          <dd data-testid="status">{{ statusLabel }}</dd>
          <template v-if="run.id">
            <dt>Id</dt>
            <dd data-testid="run-id">{{ run.id }}</dd>
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
                g.status === 'completed' && (selectedGeneration = g.generation)
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
      <Toggle v-model="logScale" label="Log scale" class="log-toggle" />

      <section v-if="!run.id" class="empty">
        <p>
          Pick the <code>run.jsonl</code> a calibration is writing. "Watch a
          run" re-reads the file as the run appends to it, so generations fill
          in as they complete. "Load a finished run" (or dropping the file here)
          reads it once.
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
              :categories="chart.categories"
              :series="[{ data: chart.data, color: palette.posterior }]"
              :height="160"
              :filename="chart.filename"
              :data-export-name="chart.filename"
              :title="chart.title"
              :csv="chart.csv"
              tooltip-trigger="hover"
            />
          </div>
        </div>
      </section>

      <section
        v-if="shownGeneration && trajectorySeries.length"
        data-testid="trajectories"
      >
        <h2>
          Trajectories
          <small
            >{{ generationLabel(shownGeneration) }},
            {{ shownGeneration.trajectories.length }} of
            {{ shownGeneration.particles.length }} particles</small
          >
        </h2>
        <LineChart
          :series="trajectorySeries"
          :height="260"
          x-label="Index"
          y-label="Value"
          :y-scale-type="yScaleType"
          :filename="trajectoryDownload.filename"
          :data-export-name="trajectoryDownload.filename"
          :title="trajectoryDownload.title"
          :csv="trajectoryDownload.csv"
        />
      </section>

      <section v-if="projectionCharts.length" data-testid="projections">
        <h2>Projections</h2>
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
            />
          </div>
        </div>
      </section>

      <section v-if="completed.length" data-testid="overlays">
        <h2>Posterior across generations</h2>
        <p class="muted">
          Prior in grey, later generations {{ isDark ? "lighter" : "darker" }}.
        </p>
        <div class="grid">
          <div v-for="o in overlays" :key="o.name" class="cell">
            <LineChart
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
              tooltip-value-format="%.3f"
            />
            <BarChart
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
              tooltip-value-format="%.3f"
            />
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
            @click="selectedGeneration = cell.generation.generation"
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
                  tooltip-value-format="%.3f"
                />
              </div>
            </div>
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
