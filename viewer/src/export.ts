// "Export outputs": the run's particles, trajectories, and posterior
// summaries as CSV and every rendered chart as a PNG, bundled into one zip.

import { strToU8, zipSync } from "fflate";
import { columnsToCsv, fileStem } from "./csv";
import { completedGenerations, type RunState } from "./run";
import { parameterSummaries, type ParamSummary } from "./stats";

// One row per particle of every completed generation, parameters as columns.
export function particlesCsv(run: RunState): string {
  const generation: number[] = [];
  const particle: number[] = [];
  const weight: number[] = [];
  const distance: number[] = [];
  const seed: string[] = [];
  const params: number[][] = run.params.map(() => []);
  for (const g of completedGenerations(run)) {
    g.particles.forEach((p, i) => {
      generation.push(g.generation);
      particle.push(i);
      weight.push(p.weight);
      distance.push(p.distance);
      seed.push(p.seed);
      params.forEach((column, k) => column.push(p.params[k]!));
    });
  }
  return columnsToCsv([
    { header: "generation", values: generation },
    { header: "particle", values: particle },
    { header: "weight", values: weight },
    { header: "distance", values: distance },
    { header: "seed", values: seed },
    ...run.params.map((p, k) => ({ header: p.name, values: params[k]! })),
  ]);
}

// Long format: one row per value of every logged trajectory.
export function trajectoriesCsv(run: RunState): string {
  const generation: number[] = [];
  const particle: number[] = [];
  const index: number[] = [];
  const value: (number | null)[] = [];
  for (const g of completedGenerations(run)) {
    for (const t of g.trajectories) {
      t.values.forEach((v, i) => {
        generation.push(g.generation);
        particle.push(t.particle);
        index.push(i);
        value.push(v);
      });
    }
  }
  return columnsToCsv([
    { header: "generation", values: generation },
    { header: "particle", values: particle },
    { header: "index", values: index },
    { header: "value", values: value },
  ]);
}

export const SUMMARY_COLUMNS = [
  "mean",
  "sd",
  "q05",
  "q25",
  "median",
  "q75",
  "q95",
] as const satisfies readonly (keyof ParamSummary)[];

// Long format: one row per parameter of every completed generation.
export function posteriorsCsv(run: RunState): string {
  const generation: number[] = [];
  const parameter: string[] = [];
  const rows: ParamSummary[] = [];
  for (const g of completedGenerations(run)) {
    parameterSummaries(run.params.length, g.particles).forEach((s, i) => {
      generation.push(g.generation);
      parameter.push(run.params[i]!.name);
      rows.push(s);
    });
  }
  return columnsToCsv([
    { header: "generation", values: generation },
    { header: "parameter", values: parameter },
    ...SUMMARY_COLUMNS.map((key) => ({
      header: key,
      values: rows.map((s) => s[key]),
    })),
  ]);
}

export interface ExportFile {
  path: string;
  data: string | Uint8Array;
}

// Paths that repeat get a numeric suffix so no entry is dropped.
export function buildZip(files: ExportFile[]): Uint8Array {
  const entries: Record<string, Uint8Array> = {};
  for (const file of files) {
    let path = file.path;
    for (let n = 2; path in entries; n++) {
      path = file.path.replace(/(\.[^./]*)?$/, `-${n}$1`);
    }
    entries[path] =
      typeof file.data === "string" ? strToU8(file.data) : file.data;
  }
  return zipSync(entries);
}

const INHERITED_PROPS = ["color", "font-family", "font-size", "font-weight"];

// A standalone copy of a chart's svg: inherited text styles and theme
// variables are inlined, and the page background is drawn behind the marks so
// blend modes and light-on-dark text come out as they look on the page.
function standaloneSvg(svg: SVGSVGElement, background: string): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const computed = getComputedStyle(svg);
  const style = INHERITED_PROPS.map(
    (prop) => `${prop}: ${computed.getPropertyValue(prop)}`,
  );
  clone.setAttribute(
    "style",
    [clone.getAttribute("style"), ...style].filter(Boolean).join("; "),
  );
  // Theme-dependent paints are replaced by what the live element resolved to.
  const live = svg.querySelectorAll("*");
  clone.querySelectorAll("*").forEach((el, i) => {
    for (const attr of ["fill", "stroke"] as const) {
      const raw = el.getAttribute(attr);
      if (!raw || !/var\(|light-dark\(/.test(raw)) continue;
      el.setAttribute(attr, getComputedStyle(live[i]!)[attr]);
    }
  });
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("width", "100%");
  rect.setAttribute("height", "100%");
  rect.setAttribute("fill", background);
  clone.insertBefore(rect, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
}

async function svgToPng(svg: SVGSVGElement, background: string) {
  const width = svg.width.baseVal.value || svg.clientWidth;
  const height = svg.height.baseVal.value || svg.clientHeight;
  const url = URL.createObjectURL(
    new Blob([standaloneSvg(svg, background)], {
      type: "image/svg+xml;charset=utf-8",
    }),
  );
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("could not render chart"));
      img.src = url;
    });
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
    if (!blob) throw new Error("could not encode chart");
    return new Uint8Array(await blob.arrayBuffer());
  } finally {
    URL.revokeObjectURL(url);
  }
}

// The nearest opaque background behind `el`, as the page shows it.
function backgroundBehind(el: Element): string {
  for (let node: Element | null = el; node; node = node.parentElement) {
    const color = getComputedStyle(node).backgroundColor;
    if (color && color !== "transparent" && !/,\s*0\)$/.test(color))
      return color;
  }
  return getComputedStyle(document.documentElement).colorScheme === "dark"
    ? "#1a1a1a"
    : "#ffffff";
}

// Every chart under `root` whose wrapper carries `data-export-name`.
export async function chartImages(root: ParentNode): Promise<ExportFile[]> {
  const files: ExportFile[] = [];
  for (const wrapper of root.querySelectorAll<HTMLElement>(
    "[data-export-name]",
  )) {
    const svg = wrapper.querySelector<SVGSVGElement>(":scope > svg");
    if (!svg) continue;
    const background = backgroundBehind(wrapper);
    files.push({
      path: `plots/${wrapper.dataset.exportName}.png`,
      data: await svgToPng(svg, background),
    });
  }
  return files;
}

export async function exportOutputs(
  run: RunState,
  root: ParentNode,
): Promise<{ name: string; zip: Uint8Array }> {
  const zip = buildZip([
    { path: "particles.csv", data: particlesCsv(run) },
    { path: "trajectories.csv", data: trajectoriesCsv(run) },
    { path: "posteriors.csv", data: posteriorsCsv(run) },
    ...(await chartImages(root)),
  ]);
  return { name: `${fileStem(run.id, "outputs") || "outputs"}.zip`, zip };
}

export function downloadBytes(bytes: Uint8Array, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
