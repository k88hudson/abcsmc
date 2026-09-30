import { describe, expect, it } from "vitest";
import {
  LineSplitter,
  applyLine,
  completedGenerations,
  currentGeneration,
  emptyRun,
} from "./run";

const started = JSON.stringify({
  type: "run_started",
  version: 1,
  id: "t",
  n_particles: 3,
  n_generations: 2,
  quantiles: [0.5],
  params: [
    { name: "a", kind: "real" },
    { name: "k", kind: "int" },
  ],
  observed: [1, 2],
});
const stats = {
  tolerance: 1,
  accepted: 3,
  attempts: 6,
  acceptance_ratio: 0.5,
  ess: 2.5,
  perplexity: 2.4,
  duration_seconds: 0.1,
};
const particles = [
  { params: [0.5, 1], weight: 0.5, distance: 0.1, seed: "1" },
  { params: [0.6, 2], weight: 0.25, distance: 0.2, seed: "2" },
  { params: [0.7, 2], weight: 0.25, distance: 0.3, seed: "3" },
];

describe("applyLine", () => {
  it("folds a run into generations", () => {
    const s = emptyRun();
    applyLine(s, started);
    expect(s.status).toBe("running");
    expect(s.params.map((p) => p.name)).toEqual(["a", "k"]);

    applyLine(
      s,
      JSON.stringify({
        type: "generation_started",
        generation: 0,
        tolerance: null,
      }),
    );
    applyLine(
      s,
      JSON.stringify({
        type: "progress",
        generation: 0,
        accepted: 2,
        attempts: 4,
        batch: [
          { params: [0.5, 1], distance: 0.1 },
          { params: [0.6, 2], distance: 0.2 },
        ],
      }),
    );
    const live = currentGeneration(s)!;
    expect(live.generation).toBe(0);
    expect(live.liveAccepted).toBe(2);
    expect(live.live).toHaveLength(2);

    applyLine(
      s,
      JSON.stringify({
        type: "generation_completed",
        generation: 0,
        stats: { ...stats, tolerance: null },
        particles,
        trajectories: [{ particle: 0, values: [1, 2] }],
      }),
    );
    expect(currentGeneration(s)).toBeNull();
    expect(completedGenerations(s)).toHaveLength(1);
    expect(s.generations[0]!.live).toEqual([]);
    expect(s.generations[0]!.particles).toHaveLength(3);

    applyLine(s, JSON.stringify({ type: "run_finished", generations: 1 }));
    expect(s.status).toBe("finished");
    expect(s.lines).toBe(5);
    expect(s.badLines).toBe(0);
  });

  it("counts unparseable lines and skips blanks", () => {
    const s = emptyRun();
    applyLine(s, "");
    applyLine(s, '{"type":"run_started"');
    expect(s.lines).toBe(1);
    expect(s.badLines).toBe(1);
  });

  it("marks an abandoned generation and finishes the run", () => {
    const s = emptyRun();
    applyLine(s, started);
    applyLine(
      s,
      JSON.stringify({
        type: "generation_started",
        generation: 0,
        tolerance: 1,
      }),
    );
    applyLine(
      s,
      JSON.stringify({ type: "generation_abandoned", generation: 0 }),
    );
    expect(s.generations[0]!.status).toBe("abandoned");
    expect(s.status).toBe("finished");
  });
});

describe("LineSplitter", () => {
  it("joins partial lines across chunks", () => {
    const sp = new LineSplitter();
    expect(sp.push("a\nb")).toEqual(["a"]);
    expect(sp.push("c\n\nd")).toEqual(["bc", ""]);
    expect(sp.flush()).toEqual(["d"]);
    expect(sp.flush()).toEqual([]);
  });

  it("decodes multi-byte characters split across chunks", () => {
    const sp = new LineSplitter();
    const bytes = new TextEncoder().encode("é\n");
    expect(sp.push(bytes.slice(0, 1))).toEqual([]);
    expect(sp.push(bytes.slice(1))).toEqual(["é"]);
  });
});

describe("progress batches", () => {
  it("appends very large batches without spreading them as arguments", () => {
    const s = emptyRun();
    applyLine(s, started);
    applyLine(
      s,
      JSON.stringify({
        type: "generation_started",
        generation: 0,
        tolerance: null,
      }),
    );
    const batch = Array.from({ length: 200_000 }, (_, i) => ({
      params: [i, 1],
      distance: i,
    }));
    applyLine(
      s,
      JSON.stringify({
        type: "progress",
        generation: 0,
        accepted: batch.length,
        attempts: batch.length,
        batch,
      }),
    );
    expect(currentGeneration(s)!.live).toHaveLength(200_000);
  });
});
