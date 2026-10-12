import { weightedCorrelation } from "./stats";

export interface ParticleLike {
  params: number[];
  weight: number;
}

export interface CorrelationPair {
  // Parameter indices; `x` is the focused parameter when there is one,
  // otherwise the one declared first.
  x: number;
  y: number;
  r: number;
  // The same pair's correlation in the reference generation, NaN when there
  // is none.
  referenceR: number;
}

export type CorrelationSort = "strongest" | "weakest" | "name";

function column(particles: ParticleLike[], i: number): number[] {
  return particles.map((p) => p.params[i]!);
}

// Every pair of parameters, or only those with `focus` when it is given.
export function correlationPairs(
  paramCount: number,
  particles: ParticleLike[],
  reference: ParticleLike[] | null,
  focus: number | null = null,
): CorrelationPair[] {
  const weights = particles.map((p) => p.weight);
  const referenceWeights = reference?.map((p) => p.weight) ?? [];
  const columns = Array.from({ length: paramCount }, (_, i) =>
    column(particles, i),
  );
  const referenceColumns = reference
    ? Array.from({ length: paramCount }, (_, i) => column(reference, i))
    : null;
  const pair = (x: number, y: number): CorrelationPair => ({
    x,
    y,
    r: weightedCorrelation(columns[x]!, columns[y]!, weights),
    referenceR: referenceColumns
      ? weightedCorrelation(
          referenceColumns[x]!,
          referenceColumns[y]!,
          referenceWeights,
        )
      : NaN,
  });
  const pairs: CorrelationPair[] = [];
  if (focus !== null) {
    for (let j = 0; j < paramCount; j++)
      if (j !== focus) pairs.push(pair(focus, j));
  } else {
    for (let i = 0; i < paramCount; i++)
      for (let j = i + 1; j < paramCount; j++) pairs.push(pair(i, j));
  }
  return pairs;
}

// An undefined correlation (a parameter that does not vary) counts as none.
export function strength(r: number): number {
  return Number.isFinite(r) ? Math.abs(r) : 0;
}

export function sortPairs(
  pairs: CorrelationPair[],
  sort: CorrelationSort,
  names: string[],
): CorrelationPair[] {
  const label = (p: CorrelationPair) => `${names[p.x]} ~ ${names[p.y]}`;
  const byName = (a: CorrelationPair, b: CorrelationPair) =>
    label(a).localeCompare(label(b));
  const sorted = [...pairs];
  if (sort === "strongest")
    sorted.sort((a, b) => strength(b.r) - strength(a.r) || byName(a, b));
  else if (sort === "weakest")
    sorted.sort((a, b) => strength(a.r) - strength(b.r) || byName(a, b));
  else sorted.sort(byName);
  return sorted;
}

export function pageCount(total: number, perPage: number): number {
  return Math.max(1, Math.ceil(total / perPage));
}

export function pageOf<T>(items: T[], page: number, perPage: number): T[] {
  return items.slice(page * perPage, (page + 1) * perPage);
}
