import { describe, expect, it } from "vitest";
import {
  axisScale,
  finitePoints,
  parameterSummaries,
  quantileBands,
  weightedCorrelation,
  weightedQuantiles,
  weightedSummary,
} from "./stats";

describe("weightedQuantiles", () => {
  it("matches plain quantiles under equal weights", () => {
    const values = [5, 1, 3, 2, 4];
    const weights = values.map(() => 0.2);
    expect(weightedQuantiles(values, weights, [0.2, 0.5, 1])).toEqual([
      1, 3, 5,
    ]);
  });

  it("follows the weights", () => {
    expect(weightedQuantiles([1, 2, 3], [0.1, 0.1, 0.8], [0.5])).toEqual([3]);
    expect(weightedQuantiles([1, 2, 3], [0.8, 0.1, 0.1], [0.5])).toEqual([1]);
  });

  it("skips nulls and zero weights, and is NaN when nothing is left", () => {
    expect(weightedQuantiles([null, 7, 9], [1, 1, 0], [0.5])).toEqual([7]);
    expect(weightedQuantiles([null], [1], [0.5])).toEqual([NaN]);
  });
});

describe("quantileBands", () => {
  it("computes each band per index across trajectories of unequal length", () => {
    const { x, bands } = quantileBands(
      [
        { weight: 0.5, values: [0, 10, 20] },
        { weight: 0.5, values: [2, 12] },
      ],
      [0.5, 1],
    );
    expect(x).toEqual([0, 1, 2]);
    expect(bands[0]).toEqual([0, 10, 20]);
    expect(bands[1]).toEqual([2, 12, 20]);
  });
});

describe("weightedSummary", () => {
  it("matches the sample mean and sd under equal weights", () => {
    const s = weightedSummary([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]);
    expect(s.mean).toBeCloseTo(3);
    expect(s.sd).toBeCloseTo(Math.sqrt(2.5));
    expect(s.median).toBe(3);
    expect(s.q05).toBe(1);
    expect(s.q95).toBe(5);
  });

  it("follows unnormalized weights", () => {
    const s = weightedSummary([0, 10], [1, 3]);
    expect(s.mean).toBeCloseTo(7.5);
    expect(s.median).toBe(10);
    expect(s.q25).toBe(0);
  });

  it("is NaN for an empty population", () => {
    const s = weightedSummary([], []);
    expect(s.mean).toBeNaN();
    expect(s.sd).toBeNaN();
    expect(s.median).toBeNaN();
  });
});

describe("parameterSummaries", () => {
  it("summarizes each parameter by position", () => {
    const [a, b] = parameterSummaries(2, [
      { params: [1, 10], weight: 0.5 },
      { params: [3, 30], weight: 0.5 },
    ]);
    expect(a!.mean).toBeCloseTo(2);
    expect(b!.mean).toBeCloseTo(20);
  });
});

describe("weightedCorrelation", () => {
  it("is 1 and -1 on a line", () => {
    expect(weightedCorrelation([1, 2, 3], [2, 4, 6], [1, 1, 1])).toBeCloseTo(1);
    expect(weightedCorrelation([1, 2, 3], [6, 4, 2], [1, 1, 1])).toBeCloseTo(
      -1,
    );
  });

  it("follows the weights", () => {
    // The off-line point carries almost no weight.
    const r = weightedCorrelation([1, 2, 3, 2], [1, 2, 3, 9], [1, 1, 1, 1e-9]);
    expect(r).toBeCloseTo(1, 5);
    expect(
      weightedCorrelation([1, 2, 3, 2], [1, 2, 3, 9], [1, 1, 1, 1]),
    ).toBeLessThan(0.5);
  });

  it("is NaN when a parameter does not vary", () => {
    expect(weightedCorrelation([1, 1, 1], [1, 2, 3], [1, 1, 1])).toBeNaN();
    expect(weightedCorrelation([1], [1], [1])).toBeNaN();
  });
});

describe("axisScale", () => {
  it("leaves readable magnitudes alone", () => {
    expect(axisScale("r0", 3.2)).toEqual({ factor: 1, label: "r0" });
    expect(axisScale("n", 0)).toEqual({ factor: 1, label: "n" });
  });

  it("moves an extreme exponent into the label", () => {
    const small = axisScale("decline", 3e-5);
    expect(small.label).toBe("decline (×10⁻⁵)");
    expect(3e-5 * small.factor).toBeCloseTo(3);
    const large = axisScale("population", 2.5e6);
    expect(large.label).toBe("population (×10⁶)");
    expect(2.5e6 * large.factor).toBeCloseTo(2.5);
  });
});

describe("finitePoints", () => {
  it("keeps each value at its index and leaves out the missing ones", () => {
    expect(finitePoints([null, null, 3, NaN, 5])).toEqual({
      x: [2, 4],
      data: [3, 5],
    });
  });
});
