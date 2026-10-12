import { describe, expect, it } from "vitest";
import { correlationPairs, pageCount, pageOf, sortPairs } from "./correlations";

// b follows a exactly, c runs against a loosely, d is constant.
const particles = Array.from({ length: 20 }, (_, i) => ({
  params: [i, 2 * i, 20 - i + (i % 3) * 4, 5],
  weight: 1,
}));
const names = ["a", "b", "c", "d"];

describe("correlationPairs", () => {
  it("builds every pair once, the earlier parameter on x", () => {
    const pairs = correlationPairs(4, particles, null);
    expect(pairs.map((p) => [p.x, p.y])).toEqual([
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [1, 3],
      [2, 3],
    ]);
    expect(pairs[0]!.r).toBeCloseTo(1);
    expect(pairs[0]!.referenceR).toBeNaN();
  });

  it("builds only the focused parameter's pairs, it on x", () => {
    const pairs = correlationPairs(4, particles, null, 2);
    expect(pairs.map((p) => [p.x, p.y])).toEqual([
      [2, 0],
      [2, 1],
      [2, 3],
    ]);
  });

  it("reports the reference generation's correlation", () => {
    const flipped = particles.map((p) => ({
      ...p,
      params: [p.params[0]!, -p.params[1]!, p.params[2]!, p.params[3]!],
    }));
    const [ab] = correlationPairs(4, particles, flipped);
    expect(ab!.referenceR).toBeCloseTo(-1);
  });
});

describe("sortPairs", () => {
  const pairs = correlationPairs(4, particles, null);
  const label = (p: { x: number; y: number }) => `${names[p.x]}~${names[p.y]}`;

  it("puts the strongest first, undefined ones last, ties by name", () => {
    const sorted = sortPairs(pairs, "strongest", names).map(label);
    expect(sorted.slice(0, 3)).toEqual(["a~b", "a~c", "b~c"]);
    expect(sorted.slice(3)).toEqual(["a~d", "b~d", "c~d"]);
  });

  it("reverses for weakest and orders by name", () => {
    expect(sortPairs(pairs, "weakest", names).map(label).at(-1)).toBe("a~b");
    expect(sortPairs(pairs, "name", names).map(label)).toEqual([
      "a~b",
      "a~c",
      "a~d",
      "b~c",
      "b~d",
      "c~d",
    ]);
  });
});

describe("pagination", () => {
  it("counts pages, at least one", () => {
    expect(pageCount(0, 6)).toBe(1);
    expect(pageCount(6, 6)).toBe(1);
    expect(pageCount(15, 6)).toBe(3);
  });

  it("slices a page", () => {
    const items = Array.from({ length: 15 }, (_, i) => i);
    expect(pageOf(items, 2, 6)).toEqual([12, 13, 14]);
  });
});
