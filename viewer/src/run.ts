// The run.jsonl event model and a reducer that folds events into a RunState.
// Every source (dropped file, tailed file handle, server stream) produces
// lines; this module is the only place that knows what they mean.

export type ParamKind = "real" | "int";

export interface ParamMeta {
  name: string;
  kind: ParamKind;
}

export interface GenerationStats {
  tolerance: number | null;
  accepted: number;
  attempts: number;
  acceptance_ratio: number;
  ess: number;
  perplexity: number;
  duration_seconds: number;
}

export interface Particle {
  params: number[];
  weight: number;
  distance: number;
  seed: string;
}

export interface Trajectory {
  particle: number;
  values: number[];
}

export interface Generation {
  generation: number;
  tolerance: number | null;
  status: "running" | "completed" | "abandoned";
  stats: GenerationStats | null;
  particles: Particle[];
  trajectories: Trajectory[];
  // Accepted so far while running: params and distance only, no weights yet.
  live: { params: number[]; distance: number }[];
  liveAccepted: number;
  liveAttempts: number;
}

export interface ProjectedTrajectory {
  weight: number;
  // Non-finite values are written as null.
  values: (number | null)[];
}

// Posterior particles simulated under a scenario, appended to the log after
// the run. A later projection with the same label replaces the earlier one.
export interface Projection {
  label: string;
  trajectories: ProjectedTrajectory[];
}

export interface RunState {
  id: string | null;
  description: string | null;
  distanceDescription: string | null;
  version: number | null;
  nParticles: number;
  nGenerations: number;
  quantiles: number[] | null;
  params: ParamMeta[];
  observed: number[] | null;
  generations: Generation[];
  projections: Projection[];
  status: "empty" | "running" | "finished";
  lines: number;
  badLines: number;
}

export type RunEvent =
  | {
      type: "run_started";
      version: number;
      id: string;
      // Absent in logs written before descriptions existed.
      description?: string | null;
      distance_description?: string | null;
      n_particles: number;
      n_generations: number;
      quantiles: number[] | null;
      params: ParamMeta[];
      observed: number[] | null;
    }
  | { type: "generation_started"; generation: number; tolerance: number | null }
  | {
      type: "progress";
      generation: number;
      accepted: number;
      attempts: number;
      batch: { params: number[]; distance: number }[];
    }
  | {
      type: "generation_completed";
      generation: number;
      stats: GenerationStats;
      particles: Particle[];
      trajectories: Trajectory[];
    }
  | { type: "generation_abandoned"; generation: number }
  | { type: "run_finished"; generations: number }
  | { type: "projection"; label: string; trajectories: ProjectedTrajectory[] };

export function emptyRun(): RunState {
  return {
    id: null,
    description: null,
    distanceDescription: null,
    version: null,
    nParticles: 0,
    nGenerations: 0,
    quantiles: null,
    params: [],
    observed: null,
    generations: [],
    projections: [],
    status: "empty",
    lines: 0,
    badLines: 0,
  };
}

function generationAt(state: RunState, index: number): Generation {
  while (state.generations.length <= index) {
    state.generations.push({
      generation: state.generations.length,
      tolerance: null,
      status: "running",
      stats: null,
      particles: [],
      trajectories: [],
      live: [],
      liveAccepted: 0,
      liveAttempts: 0,
    });
  }
  return state.generations[index]!;
}

// Mutates `state` in place and returns it, so a Vue `reactive` state updates
// without re-cloning particle arrays on every event.
export function applyEvent(state: RunState, event: RunEvent): RunState {
  switch (event.type) {
    case "run_started": {
      Object.assign(state, emptyRun(), {
        lines: state.lines,
        badLines: state.badLines,
        id: event.id,
        description: event.description ?? null,
        distanceDescription: event.distance_description ?? null,
        version: event.version,
        nParticles: event.n_particles,
        nGenerations: event.n_generations,
        quantiles: event.quantiles,
        params: event.params,
        observed: event.observed,
        status: "running",
      });
      break;
    }
    case "generation_started": {
      const g = generationAt(state, event.generation);
      g.tolerance = event.tolerance;
      g.status = "running";
      break;
    }
    case "progress": {
      const g = generationAt(state, event.generation);
      g.liveAccepted = event.accepted;
      g.liveAttempts = event.attempts;
      for (const accepted of event.batch) g.live.push(accepted);
      break;
    }
    case "generation_completed": {
      const g = generationAt(state, event.generation);
      g.status = "completed";
      g.tolerance = event.stats.tolerance;
      g.stats = event.stats;
      g.particles = event.particles;
      g.trajectories = event.trajectories;
      g.live = [];
      g.liveAccepted = event.stats.accepted;
      g.liveAttempts = event.stats.attempts;
      break;
    }
    case "generation_abandoned": {
      generationAt(state, event.generation).status = "abandoned";
      state.status = "finished";
      break;
    }
    case "run_finished": {
      state.status = "finished";
      break;
    }
    case "projection": {
      const projection = {
        label: event.label,
        trajectories: event.trajectories,
      };
      const existing = state.projections.findIndex(
        (p) => p.label === event.label,
      );
      if (existing >= 0) state.projections[existing] = projection;
      else state.projections.push(projection);
      break;
    }
  }
  return state;
}

// Parses one line and applies it. Blank lines are skipped; unparseable ones
// (a line still being written when a tail read it) are counted, not applied.
export function applyLine(state: RunState, line: string): RunState {
  const trimmed = line.trim();
  if (!trimmed) return state;
  state.lines += 1;
  let event: RunEvent;
  try {
    event = JSON.parse(trimmed) as RunEvent;
  } catch {
    state.badLines += 1;
    return state;
  }
  return applyEvent(state, event);
}

// Splits a byte stream into complete lines; a trailing partial line waits for
// the next chunk.
export class LineSplitter {
  private decoder = new TextDecoder();
  private carry = "";

  push(chunk: Uint8Array | string): string[] {
    const text =
      typeof chunk === "string"
        ? chunk
        : this.decoder.decode(chunk, { stream: true });
    const parts = (this.carry + text).split("\n");
    this.carry = parts.pop() ?? "";
    return parts;
  }

  // Returns the trailing partial line, if any, and resets.
  flush(): string[] {
    const rest = this.carry + this.decoder.decode();
    this.carry = "";
    return rest ? [rest] : [];
  }
}

export function currentGeneration(state: RunState): Generation | null {
  for (let i = state.generations.length - 1; i >= 0; i--) {
    const g = state.generations[i]!;
    if (g.status === "running") return g;
  }
  return null;
}

export function completedGenerations(state: RunState): Generation[] {
  return state.generations.filter((g) => g.status === "completed");
}
