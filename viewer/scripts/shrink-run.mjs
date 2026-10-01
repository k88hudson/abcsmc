// Subsamples a run.jsonl so it can live in the repo as a test fixture.
// Usage: node scripts/shrink-run.mjs <in> <out> [particles=100] [trajectories=20]
import { readFileSync, writeFileSync } from "node:fs";

const [input, output, particlesArg = "100", trajectoriesArg = "20"] =
  process.argv.slice(2);
if (!input || !output) {
  console.error("usage: shrink-run.mjs <in> <out> [particles] [trajectories]");
  process.exit(1);
}
const keepParticles = Number(particlesArg);
const keepTrajectories = Number(trajectoriesArg);

const every = (arr, n) => {
  if (arr.length <= n) return arr;
  const stride = arr.length / n;
  return Array.from({ length: n }, (_, i) => arr[Math.floor(i * stride)]);
};

const lines = readFileSync(input, "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l))
  .map((e) => {
    if (e.type === "progress") e.batch = every(e.batch, keepParticles);
    if (e.type === "generation_completed") {
      e.particles = every(e.particles, keepParticles);
      const total = e.particles.reduce((s, p) => s + p.weight, 0);
      for (const p of e.particles) p.weight /= total;
      e.trajectories = every(e.trajectories, keepTrajectories);
      if (e.rejected) e.rejected = every(e.rejected, keepTrajectories);
    }
    return e;
  });
writeFileSync(output, lines.map((e) => JSON.stringify(e)).join("\n") + "\n");
