import { strFromU8, unzipSync } from "fflate";
import { describe, expect, it } from "vitest";
import {
  buildZip,
  particlesCsv,
  posteriorsCsv,
  trajectoriesCsv,
} from "./export";
import { applyLine, emptyRun } from "./run";

const stats = {
  tolerance: 1,
  accepted: 2,
  attempts: 4,
  acceptance_ratio: 0.5,
  ess: 2,
  perplexity: 2,
  duration_seconds: 0.1,
};

function finishedRun() {
  const run = emptyRun();
  const lines = [
    {
      type: "run_started",
      version: 1,
      id: "t",
      n_particles: 2,
      n_generations: 2,
      quantiles: [0.5],
      params: [
        { name: "a", kind: "real" },
        { name: "k", kind: "int" },
      ],
      observed: [1, 2],
    },
    { type: "generation_started", generation: 0, tolerance: null },
    {
      type: "generation_completed",
      generation: 0,
      stats: { ...stats, tolerance: null },
      particles: [
        { params: [0.5, 1], weight: 0.5, distance: 0.1, seed: "11" },
        { params: [0.6, 2], weight: 0.5, distance: 0.2, seed: "12" },
      ],
      trajectories: [{ particle: 1, values: [3, 4] }],
    },
    { type: "generation_started", generation: 1, tolerance: 1 },
    {
      type: "generation_completed",
      generation: 1,
      stats,
      particles: [
        { params: [0.7, 2], weight: 0.25, distance: 0.05, seed: "21" },
        { params: [0.8, 3], weight: 0.75, distance: 0.01, seed: "22" },
      ],
      trajectories: [{ particle: 0, values: [5] }],
    },
    // Still running: must not appear in the export.
    { type: "generation_started", generation: 2, tolerance: 0.5 },
  ];
  for (const line of lines) applyLine(run, JSON.stringify(line));
  return run;
}

describe("particlesCsv", () => {
  it("writes one row per particle of each completed generation", () => {
    expect(particlesCsv(finishedRun()).split("\n")).toEqual([
      "generation,particle,weight,distance,seed,a,k",
      "0,0,0.5,0.1,11,0.5,1",
      "0,1,0.5,0.2,12,0.6,2",
      "1,0,0.25,0.05,21,0.7,2",
      "1,1,0.75,0.01,22,0.8,3",
    ]);
  });
});

describe("trajectoriesCsv", () => {
  it("writes one row per trajectory value", () => {
    expect(trajectoriesCsv(finishedRun()).split("\n")).toEqual([
      "generation,particle,index,value",
      "0,1,0,3",
      "0,1,1,4",
      "1,0,0,5",
    ]);
  });
});

describe("posteriorsCsv", () => {
  it("writes one row per parameter of each completed generation", () => {
    const rows = posteriorsCsv(finishedRun())
      .split("\n")
      .map((r) => r.split(","));
    expect(rows[0]).toEqual([
      "generation",
      "parameter",
      "mean",
      "sd",
      "q05",
      "q25",
      "median",
      "q75",
      "q95",
    ]);
    expect(rows.slice(1).map((r) => r.slice(0, 2))).toEqual([
      ["0", "a"],
      ["0", "k"],
      ["1", "a"],
      ["1", "k"],
    ]);
    // Generation 1 of "a": 0.7 at weight 0.25, 0.8 at weight 0.75.
    expect(Number(rows[3]![2])).toBeCloseTo(0.775);
    expect(rows[3]!.slice(4)).toEqual(["0.7", "0.7", "0.8", "0.8", "0.8"]);
  });
});

describe("buildZip", () => {
  it("round-trips text and binary entries and keeps repeated paths", () => {
    const run = finishedRun();
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    const entries = unzipSync(
      buildZip([
        { path: "particles.csv", data: particlesCsv(run) },
        { path: "plots/a.png", data: png },
        { path: "plots/a.png", data: png },
      ]),
    );
    expect(Object.keys(entries).sort()).toEqual([
      "particles.csv",
      "plots/a-2.png",
      "plots/a.png",
    ]);
    expect(strFromU8(entries["particles.csv"]!)).toBe(particlesCsv(run));
    expect(entries["plots/a-2.png"]).toEqual(png);
  });
});
