//! Running generation by generation, and resuming a run from its log.
#![cfg(feature = "serde")]

use std::path::{Path, PathBuf};

use abcsmc::{
    CalibrationModel, Generation, IntPrior, JsonlObserver, Priors, RealPrior, RunLog, Sampler,
    Silent, define_priors, resume_quantiles_with, resume_with, run_quantiles_with, run_with,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

define_priors! {
    pub struct Draw / DrawPriors {
        rate: Real,
        count: Int,
    }
}

/// Output is `rate * count` plus seed-driven noise on `[0, 1)`, targeting 12.
struct NoisyProduct {
    widest_rate: f64,
}

const MODEL: NoisyProduct = NoisyProduct { widest_rate: 10.0 };

impl CalibrationModel for NoisyProduct {
    type Draw = Draw;
    type Output = f64;

    fn priors(&self) -> Priors {
        DrawPriors {
            rate: RealPrior::uniform(0.0, self.widest_rate).unwrap(),
            count: IntPrior::discrete_uniform(1, 5).unwrap(),
        }
        .into()
    }

    fn simulate(&self, draw: &Draw, seed: u64) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        draw.rate * draw.count as f64 + noise
    }
}

fn to_twelve(output: &f64) -> f64 {
    (output - 12.0).abs()
}

const N: usize = 100;
const TOLERANCES: [f64; 4] = [8.0, 4.0, 2.0, 1.0];
const QUANTILES: [f64; 3] = [0.5, 0.2, 0.1];

/// A directory of run logs, removed when dropped.
struct Runs(PathBuf);

impl Runs {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("abcsmc-resume-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Runs(dir)
    }

    fn create(&self, id: &str) -> JsonlObserver {
        JsonlObserver::create(&self.0, id).unwrap()
    }

    fn append(&self, id: &str) -> JsonlObserver {
        JsonlObserver::append(&self.0, id).unwrap()
    }

    fn path(&self, id: &str) -> PathBuf {
        self.0.join(id).join("run.jsonl")
    }

    fn read(&self, id: &str) -> RunLog {
        RunLog::read(self.path(id)).unwrap()
    }
}

impl Drop for Runs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Parameters, weight, distance, and seed of every particle.
fn particles(generation: &Generation<NoisyProduct>) -> Vec<(f64, i64, f64, f64, u64)> {
    generation
        .particles
        .iter()
        .map(|p| {
            let draw = p.draw();
            (draw.rate, draw.count, p.weight, p.distance, p.seed)
        })
        .collect()
}

fn logged_particles(log: &RunLog, generation: usize) -> Vec<(f64, i64, f64, f64, u64)> {
    log.generations[generation]
        .particles
        .iter()
        .map(|p| {
            let draw: Draw = p.draw();
            (draw.rate, draw.count, p.weight, p.distance, p.seed)
        })
        .collect()
}

fn assert_every_line_is_an_event(path: &Path) {
    for line in std::fs::read_to_string(path).unwrap().lines() {
        serde_json::from_str::<serde_json::Value>(line).unwrap();
    }
}

/// The log of `whole`, cut off partway through `generation` and in the middle
/// of a line, as a killed run leaves it.
fn cut_during(log: &Path, generation: usize) -> String {
    let text = std::fs::read_to_string(log).unwrap();
    let started = format!(r#"{{"type":"generation_started","generation":{generation},"#);
    let from = text.find(&started).unwrap();
    let line_end = from + text[from..].find('\n').unwrap() + 1;
    let next_line_end = line_end + text[line_end..].find('\n').unwrap();
    text[..next_line_end - 20].to_string()
}

#[test]
fn a_resumed_run_matches_the_uninterrupted_one() {
    let runs = Runs::new("fixed");
    let whole = run_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES,
        N,
        "",
        &runs.create("whole"),
    );

    run_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES[..2],
        N,
        "",
        &runs.create("stopped"),
    );
    let resumed = resume_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES,
        &runs.read("stopped"),
        "",
        &runs.append("stopped"),
    )
    .unwrap();

    assert_eq!(resumed.len(), 2);
    for (resumed, whole) in resumed.iter().zip(&whole[2..]) {
        assert_eq!(resumed.stats.generation, whole.stats.generation);
        assert_eq!(resumed.stats.attempts, whole.stats.attempts);
        assert_eq!(particles(resumed), particles(whole));
    }

    let log = runs.read("stopped");
    assert!(log.finished);
    assert_eq!(log.generations.len(), 4);
    for (generation, whole) in whole.iter().enumerate() {
        assert_eq!(logged_particles(&log, generation), particles(whole));
    }
}

#[test]
fn a_run_killed_mid_generation_resumes_from_the_last_completed_one() {
    let runs = Runs::new("killed");
    let whole = run_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES,
        N,
        "",
        &runs.create("whole"),
    );

    std::fs::create_dir_all(runs.path("killed").parent().unwrap()).unwrap();
    std::fs::write(runs.path("killed"), cut_during(&runs.path("whole"), 2)).unwrap();
    let log = runs.read("killed");
    assert_eq!(log.generations.len(), 2);
    assert!(!log.finished);

    let resumed = resume_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES,
        &log,
        "",
        &runs.append("killed"),
    )
    .unwrap();
    assert_eq!(resumed.len(), 2);
    assert_eq!(particles(&resumed[1]), particles(&whole[3]));

    assert_every_line_is_an_event(&runs.path("killed"));
    let log = runs.read("killed");
    assert!(log.finished);
    assert_eq!(log.generations.len(), 4);
    assert_eq!(logged_particles(&log, 2), particles(&whole[2]));
}

#[test]
fn a_quantile_run_resumes_with_the_tolerances_of_its_prior() {
    let runs = Runs::new("quantiles");
    let whole = run_quantiles_with(&MODEL, &to_twelve, &QUANTILES, N, "", &runs.create("whole"));

    std::fs::create_dir_all(runs.path("killed").parent().unwrap()).unwrap();
    std::fs::write(runs.path("killed"), cut_during(&runs.path("whole"), 2)).unwrap();
    let log = runs.read("killed");
    assert_eq!(log.quantiles.as_deref(), Some(&QUANTILES[..]));

    let resumed = resume_quantiles_with(
        &MODEL,
        &to_twelve,
        &QUANTILES,
        &log,
        "",
        &runs.append("killed"),
    )
    .unwrap();
    assert_eq!(resumed.len(), 2);
    for (resumed, whole) in resumed.iter().zip(&whole[2..]) {
        assert_eq!(resumed.stats.tolerance, whole.stats.tolerance);
        assert_eq!(particles(resumed), particles(whole));
    }
}

#[test]
fn a_run_killed_before_its_first_generation_starts_over() {
    let runs = Runs::new("restart");
    let whole = run_quantiles_with(&MODEL, &to_twelve, &QUANTILES, N, "", &runs.create("whole"));

    std::fs::create_dir_all(runs.path("killed").parent().unwrap()).unwrap();
    std::fs::write(runs.path("killed"), cut_during(&runs.path("whole"), 0)).unwrap();
    let log = runs.read("killed");
    assert!(log.generations.is_empty());

    let resumed = resume_quantiles_with(
        &MODEL,
        &to_twelve,
        &QUANTILES,
        &log,
        "",
        &runs.append("killed"),
    )
    .unwrap();
    assert_eq!(resumed.len(), whole.len());
    assert_eq!(particles(&resumed[3]), particles(&whole[3]));
    assert_eq!(runs.read("killed").generations.len(), whole.len());
}

#[test]
fn a_finished_run_is_extended_by_a_longer_schedule() {
    let runs = Runs::new("extended");
    let whole = run_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES,
        N,
        "",
        &runs.create("whole"),
    );
    run_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES[..3],
        N,
        "",
        &runs.create("short"),
    );
    assert!(runs.read("short").finished);

    // Nothing is left of the schedule the run finished.
    let nothing = resume_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES[..3],
        &runs.read("short"),
        "",
        &Silent,
    )
    .unwrap();
    assert!(nothing.is_empty());

    let extension = resume_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES,
        &runs.read("short"),
        "",
        &runs.append("short"),
    )
    .unwrap();
    assert_eq!(extension.len(), 1);
    assert_eq!(particles(&extension[0]), particles(&whole[3]));

    let text = std::fs::read_to_string(runs.path("short")).unwrap();
    let events: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let resumed = events.iter().find(|e| e["type"] == "run_resumed").unwrap();
    assert_eq!(resumed["completed"], 3);
    assert_eq!(resumed["n_generations"], 4);
    assert_eq!(events.last().unwrap()["type"], "run_finished");
    assert_eq!(events.last().unwrap()["generations"], 4);
}

#[test]
fn a_log_of_another_run_is_refused() {
    let runs = Runs::new("refused");
    run_with(
        &MODEL,
        &to_twelve,
        &TOLERANCES[..2],
        N,
        "",
        &runs.create("run"),
    );
    let log = runs.read("run");

    let other_prior = NoisyProduct { widest_rate: 20.0 };
    let error = resume_with(&other_prior, &to_twelve, &TOLERANCES, &log, "", &Silent)
        .err()
        .unwrap();
    assert!(error.to_string().contains("rate"), "{error}");

    let other_schedule = [8.0, 5.0, 2.0];
    let error = resume_with(&MODEL, &to_twelve, &other_schedule, &log, "", &Silent)
        .err()
        .unwrap();
    assert!(error.to_string().contains("generation 1"), "{error}");
}

#[test]
fn a_sampler_runs_one_generation_at_a_time() {
    let whole = run_with(&MODEL, &to_twelve, &TOLERANCES, N, "", &Silent);

    let mut sampler = Sampler::start(&MODEL, &to_twelve, N, "", TOLERANCES.len(), &Silent);
    for (index, &tolerance) in TOLERANCES.iter().enumerate() {
        assert_eq!(sampler.completed(), index);
        let generation = sampler.next_generation(tolerance).unwrap();
        assert_eq!(generation.stats.generation, index);
        assert_eq!(particles(generation), particles(&whole[index]));
    }
    assert_eq!(sampler.generations().len(), TOLERANCES.len());
    assert_eq!(sampler.finish().len(), TOLERANCES.len());
}

#[test]
fn a_sampler_resumes_from_a_population() {
    let whole = run_with(&MODEL, &to_twelve, &TOLERANCES, N, "", &Silent);

    let population = (&whole[1]).into();
    let mut sampler = Sampler::resume(&MODEL, &to_twelve, population, 2, "", 4, &Silent);
    assert_eq!(sampler.completed(), 2);
    let generation = sampler.next_generation(TOLERANCES[2]).unwrap();
    assert_eq!(generation.stats.generation, 2);
    assert_eq!(particles(generation), particles(&whole[2]));
}

/// Too few particles to estimate a perturbation kernel from, which only a
/// generation after this one would need.
#[test]
fn the_last_generation_is_not_adapted_to() {
    let generations = run_with(&MODEL, &to_twelve, &[f64::INFINITY], 1, "", &Silent);
    assert_eq!(generations.len(), 1);
    assert_eq!(generations[0].particles.len(), 1);
}
