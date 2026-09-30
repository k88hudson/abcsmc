import { describe, expect, it } from "vitest";
import { quantileBands, weightedQuantiles } from "./stats";

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
