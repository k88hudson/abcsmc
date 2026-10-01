// Weighted summaries of a particle population for plotting.

export interface HistogramBin {
  center: number;
  weight: number;
}

// Evenly spaced bins on [lo, hi]; the last bin includes hi. Out-of-range
// values are dropped.
export function weightedHistogram(
  values: number[],
  weights: number[],
  binCount: number,
  lo: number,
  hi: number,
): HistogramBin[] {
  if (binCount <= 0 || hi <= lo) return [];
  const width = (hi - lo) / binCount;
  const bins: HistogramBin[] = Array.from({ length: binCount }, (_, i) => ({
    center: lo + width * (i + 0.5),
    weight: 0,
  }));
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v) || v < lo || v > hi) continue;
    const idx = Math.min(Math.floor((v - lo) / width), binCount - 1);
    bins[idx]!.weight += weights[i] ?? 0;
  }
  return bins;
}

export interface KdeResult {
  x: number[];
  y: number[];
}

// Weighted Gaussian KDE on a regular grid over [lo, hi] with the robust
// Silverman bandwidth 0.9 * min(sd, IQR / 1.34) * n_eff^(-1/5), where
// n_eff = 1 / sum(w^2) for normalized weights. Weighted variance uses the
// 1 / (1 - sum(w^2)) reliability-weights correction.
export function weightedKde(
  values: number[],
  weights: number[],
  lo: number,
  hi: number,
  nGrid: number,
): KdeResult {
  if (values.length === 0 || hi <= lo || nGrid < 2) return { x: [], y: [] };
  let mean = 0;
  for (let i = 0; i < values.length; i++) mean += weights[i]! * values[i]!;
  let biasedVar = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i]! - mean;
    biasedVar += weights[i]! * d * d;
  }
  let sumW2 = 0;
  for (const w of weights) sumW2 += w * w;
  const nEff = sumW2 > 0 ? 1 / sumW2 : values.length;
  const denom = 1 - sumW2;
  const variance = denom > 1e-12 ? biasedVar / denom : biasedVar;
  const sd = Math.sqrt(Math.max(variance, 1e-12));

  const idx = values.map((_, i) => i).sort((a, b) => values[a]! - values[b]!);
  const cum: number[] = new Array(idx.length);
  let acc = 0;
  for (let k = 0; k < idx.length; k++) {
    acc += weights[idx[k]!]!;
    cum[k] = acc;
  }
  const totalW = cum.length > 0 ? cum[cum.length - 1]! : 0;
  const quantile = (q: number): number => {
    const target = q * totalW;
    for (let k = 0; k < idx.length; k++) {
      if (cum[k]! >= target) {
        if (k === 0) return values[idx[0]!]!;
        const w0 = cum[k - 1]!;
        const w1 = cum[k]!;
        const v0 = values[idx[k - 1]!]!;
        const v1 = values[idx[k]!]!;
        const t = (target - w0) / Math.max(w1 - w0, 1e-12);
        return v0 + t * (v1 - v0);
      }
    }
    return values[idx[idx.length - 1]!]!;
  };
  const iqr = quantile(0.75) - quantile(0.25);
  const a = iqr > 0 ? Math.min(sd, iqr / 1.34) : sd;
  const h = Math.max(0.9 * a * Math.pow(nEff, -1 / 5), 1e-9);
  const norm = 1 / (h * Math.sqrt(2 * Math.PI));
  const step = (hi - lo) / (nGrid - 1);
  const x: number[] = new Array(nGrid);
  const y: number[] = new Array(nGrid);
  for (let i = 0; i < nGrid; i++) {
    const xi = lo + i * step;
    let dens = 0;
    for (let j = 0; j < values.length; j++) {
      const z = (xi - values[j]!) / h;
      dens += weights[j]! * Math.exp(-0.5 * z * z);
    }
    x[i] = xi;
    y[i] = dens * norm;
  }
  return { x, y };
}

export function normalizedWeights(weights: number[]): number[] {
  let total = 0;
  for (const w of weights) total += w;
  return total > 0
    ? weights.map((w) => w / total)
    : weights.map(() => 1 / weights.length);
}

export function extent(values: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : [0, 1];
}

// The values at cumulative weight fractions `qs` (each in [0, 1]), taking the
// first value whose running weight reaches the fraction. Non-finite values
// are skipped. Returns NaN for every fraction when nothing is left.
export function weightedQuantiles(
  values: (number | null)[],
  weights: number[],
  qs: number[],
): number[] {
  const pairs: [number, number][] = [];
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const w = weights[i] ?? 0;
    if (v === null || v === undefined || !Number.isFinite(v) || !(w > 0))
      continue;
    pairs.push([v, w]);
    total += w;
  }
  if (pairs.length === 0) return qs.map(() => NaN);
  pairs.sort((a, b) => a[0] - b[0]);
  return qs.map((q) => {
    const target = q * total;
    let acc = 0;
    for (const [v, w] of pairs) {
      acc += w;
      if (acc >= target) return v;
    }
    return pairs[pairs.length - 1]![0];
  });
}

export interface QuantileBands {
  x: number[];
  // One series per requested fraction, in the order given.
  bands: number[][];
}

// Weighted quantiles across trajectories at every index any of them reaches.
export function quantileBands(
  trajectories: { weight: number; values: (number | null)[] }[],
  qs: number[],
): QuantileBands {
  let length = 0;
  for (const t of trajectories) length = Math.max(length, t.values.length);
  const weights = trajectories.map((t) => t.weight);
  const bands: number[][] = qs.map(() => new Array<number>(length));
  for (let i = 0; i < length; i++) {
    const at = weightedQuantiles(
      trajectories.map((t) => t.values[i] ?? null),
      weights,
      qs,
    );
    for (let k = 0; k < qs.length; k++) bands[k]![i] = at[k]!;
  }
  return { x: Array.from({ length }, (_, i) => i), bands };
}

export interface ParamSummary {
  mean: number;
  // Weighted standard deviation with the same reliability-weights correction
  // as the KDE bandwidth.
  sd: number;
  q05: number;
  q25: number;
  median: number;
  q75: number;
  q95: number;
}

// Weighted moments and quantiles of one parameter. Weights need not be
// normalized. Every field is NaN for an empty population.
export function weightedSummary(
  values: number[],
  weights: number[],
): ParamSummary {
  const w = normalizedWeights(weights);
  let mean = values.length > 0 ? 0 : NaN;
  for (let i = 0; i < values.length; i++) mean += w[i]! * values[i]!;
  let biasedVar = 0;
  let sumW2 = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i]! - mean;
    biasedVar += w[i]! * d * d;
    sumW2 += w[i]! * w[i]!;
  }
  const denom = 1 - sumW2;
  const variance = denom > 1e-12 ? biasedVar / denom : biasedVar;
  const [q05, q25, median, q75, q95] = weightedQuantiles(
    values,
    w,
    [0.05, 0.25, 0.5, 0.75, 0.95],
  ) as [number, number, number, number, number];
  return {
    mean,
    sd: values.length > 0 ? Math.sqrt(variance) : NaN,
    q05,
    q25,
    median,
    q75,
    q95,
  };
}

// One summary per parameter, in declaration order.
export function parameterSummaries(
  paramCount: number,
  particles: { params: number[]; weight: number }[],
): ParamSummary[] {
  const weights = particles.map((p) => p.weight);
  return Array.from({ length: paramCount }, (_, i) =>
    weightedSummary(
      particles.map((p) => p.params[i]!),
      weights,
    ),
  );
}
