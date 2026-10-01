//! Integration tests for the drivers on 1- and 2-parameter toy models where the
//! posterior is known.

use std::collections::HashSet;

use abcsmc::{
    Distance, Draw, Generation, IntPrior, JsonlObserver, Model, Params, Priors, RealPrior, Silent,
    define_priors, distance_to, run, run_quantiles, run_quantiles_with, run_with,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn to_five(output: &f64) -> f64 {
    (output - 5.0).abs()
}

/// Estimate `x` under `Uniform(0, 10)` with `simulate(x) = x`, targeting 5.
struct ToyModel;

impl Model for ToyModel {
    type Draw = Params;
    type Output = f64;

    fn priors(&self) -> Priors {
        Priors::new().push(RealPrior::uniform(0.0, 10.0).unwrap())
    }

    fn simulate(&self, params: &Params, _seed: u64) -> f64 {
        params.real(0)
    }
}

/// `ToyModel` plus seed-driven uniform noise on `[0, 1)`.
struct NoisyModel;

impl Model for NoisyModel {
    type Draw = Params;
    type Output = f64;

    fn priors(&self) -> Priors {
        Priors::new().push(RealPrior::uniform(0.0, 10.0).unwrap())
    }

    fn simulate(&self, params: &Params, seed: u64) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        params.real(0) + noise
    }
}

const N: usize = 200;
const QUANTILES: [f64; 3] = [0.5, 0.2, 0.05];
const TOLERANCES: [f64; 3] = [2.5, 1.0, 0.25];

fn weighted_mean<M: Model<Draw = Params>>(population: &[abcsmc::Particle<M>]) -> f64 {
    population.iter().map(|p| p.weight * p.params.real(0)).sum()
}

fn max_distance<M: Model>(population: &[abcsmc::Particle<M>]) -> f64 {
    population.iter().map(|p| p.distance).fold(0.0, f64::max)
}

fn assert_tightens_and_converges<M: Model<Draw = Params>>(generations: &[Generation<M>]) {
    for k in 1..generations.len() {
        assert!(
            max_distance(&generations[k].particles) <= max_distance(&generations[k - 1].particles),
            "generation {k} did not tighten"
        );
    }
    let mean = weighted_mean(&generations.last().unwrap().particles);
    assert!((mean - 5.0).abs() < 0.5, "expected ~5, got {mean}");
}

#[test]
fn run_returns_one_generation_per_tolerance() {
    let generations = run(&ToyModel, &to_five, &TOLERANCES, N);
    assert_eq!(generations.len(), TOLERANCES.len());
    for (generation, tolerance) in generations.iter().zip(TOLERANCES) {
        assert_eq!(generation.particles.len(), N);
        assert_eq!(generation.stats.tolerance, tolerance);
        assert!(generation.particles.iter().all(|p| p.distance <= tolerance));
    }
    assert!(
        generations[0]
            .particles
            .iter()
            .all(|p| p.weight == 1.0 / N as f64)
    );
    assert_tightens_and_converges(&generations);
}

#[test]
fn run_quantiles_returns_prior_plus_one_generation_per_quantile() {
    let generations = run_quantiles(&ToyModel, &to_five, &QUANTILES, N);
    assert_eq!(generations.len(), QUANTILES.len() + 1);
    assert!(generations.iter().all(|g| g.particles.len() == N));
    assert_eq!(generations[0].stats.tolerance, f64::INFINITY);

    // Each quantile threshold is read off generation 0's distances.
    let mut prior_distances: Vec<f64> = generations[0]
        .particles
        .iter()
        .map(|p| p.distance)
        .collect();
    prior_distances.sort_by(f64::total_cmp);
    for (generation, quantile) in generations[1..].iter().zip(QUANTILES) {
        let expected = prior_distances[(quantile * N as f64) as usize - 1];
        assert_eq!(generation.stats.tolerance, expected);
    }
    assert_tightens_and_converges(&generations);
}

#[test]
fn weights_are_normalized() {
    for generation in run_quantiles(&ToyModel, &to_five, &QUANTILES, N) {
        let total: f64 = generation.particles.iter().map(|p| p.weight).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}

#[test]
fn stored_distance_matches_output() {
    let model = NoisyModel;
    for generation in run_quantiles(&model, &to_five, &QUANTILES, N) {
        for particle in &generation.particles {
            assert_eq!(particle.distance, to_five(&particle.output));
        }
    }
}

#[test]
fn particle_seed_replays_its_simulation() {
    let model = NoisyModel;
    let generations = run_quantiles(&model, &to_five, &QUANTILES, N);
    for generation in &generations {
        for particle in &generation.particles {
            assert_eq!(
                model.simulate(&particle.params, particle.seed),
                particle.output
            );
        }
    }
    let last = generations.last().unwrap();
    let distinct: HashSet<u64> = last.particles.iter().map(|p| p.seed).collect();
    assert_eq!(distinct.len(), N, "expected a distinct seed per particle");
    let particle = &last.particles[0];
    assert_ne!(
        model.simulate(&particle.params, particle.seed ^ 1),
        particle.output
    );
}

/// An unreachable tolerance exhausts the attempt cap and the run stops early
/// instead of hanging.
struct UnreachableModel;

impl Model for UnreachableModel {
    type Draw = Params;
    type Output = f64;

    fn priors(&self) -> Priors {
        Priors::new().push(RealPrior::uniform(0.0, 10.0).unwrap())
    }

    fn simulate(&self, params: &Params, _seed: u64) -> f64 {
        params.real(0)
    }

    fn max_attempts_per_proposal(&self) -> u64 {
        50
    }
}

#[test]
fn exhausted_attempts_stop_the_run_early() {
    let generations = run(&UnreachableModel, &to_five, &[5.0, 1.0, -1.0], 20);
    assert_eq!(generations.len(), 2);
}

define_priors! {
    pub struct MacroDraw / MacroPriors {
        x: Real,
        k: Int,
    }
}

struct MacroModel;

fn macro_distance(output: &(f64, i64)) -> f64 {
    (output.0 - 5.0).abs() + ((output.1 - 3).abs() as f64)
}

impl Model for MacroModel {
    type Draw = MacroDraw;
    type Output = (f64, i64);

    fn priors(&self) -> Priors {
        MacroPriors {
            x: RealPrior::uniform(0.0, 10.0).unwrap(),
            k: IntPrior::discrete_uniform(0, 10).unwrap(),
        }
        .into()
    }

    fn simulate(&self, draw: &MacroDraw, _seed: u64) -> (f64, i64) {
        (draw.x, draw.k)
    }
}

#[test]
fn macro_generated_draw_maps_priors_to_named_fields() {
    assert_eq!(MacroDraw::NAMES, &["x", "k"]);

    let generations = run_quantiles(&MacroModel, &macro_distance, &QUANTILES, N);
    assert_eq!(generations.len(), QUANTILES.len() + 1);
    for generation in &generations {
        for particle in &generation.particles {
            let draw = particle.draw();
            assert_eq!(draw.x, particle.params.real(0));
            assert_eq!(draw.k, particle.params.int(1));
            assert_eq!((draw.x, draw.k), particle.output);
        }
    }

    let last = generations.last().unwrap();
    let mean_x: f64 = last.particles.iter().map(|p| p.weight * p.draw().x).sum();
    let mean_k: f64 = last
        .particles
        .iter()
        .map(|p| p.weight * p.draw().k as f64)
        .sum();
    assert!(
        (mean_x - 5.0).abs() < 0.5,
        "x should approach 5, got {mean_x}"
    );
    assert!(
        (mean_k - 3.0).abs() < 0.5,
        "k should approach 3, got {mean_k}"
    );
}

/// `NoisyModel` that also exposes trajectories.
struct TracedModel;

impl Model for TracedModel {
    type Draw = Params;
    type Output = f64;

    fn priors(&self) -> Priors {
        NoisyModel.priors()
    }

    fn simulate(&self, params: &Params, seed: u64) -> f64 {
        NoisyModel.simulate(params, seed)
    }

    fn trajectory(&self, output: &f64) -> Option<Vec<f64>> {
        Some(vec![*output, *output])
    }
}

#[test]
fn a_distance_chosen_at_run_time_matches_the_same_distance_passed_directly() {
    let boxed: Box<dyn Distance<f64>> = Box::new(to_five);
    let chosen = run_with(&NoisyModel, boxed.as_ref(), &TOLERANCES, N, "", &Silent);
    let direct = run_with(&NoisyModel, &to_five, &TOLERANCES, N, "", &Silent);
    for (a, b) in chosen.iter().zip(&direct) {
        assert_eq!(a.stats.attempts, b.stats.attempts);
        for (p, q) in a.particles.iter().zip(&b.particles) {
            assert_eq!(p.params.real(0), q.params.real(0));
            assert_eq!(p.distance, q.distance);
        }
    }
}

#[test]
fn observers_do_not_change_the_result() {
    let silent = run_with(&NoisyModel, &to_five, &TOLERANCES, N, "", &Silent);
    let printed = run(&NoisyModel, &to_five, &TOLERANCES, N);
    for (a, b) in silent.iter().zip(&printed) {
        assert_eq!(a.stats.attempts, b.stats.attempts);
        for (p, q) in a.particles.iter().zip(&b.particles) {
            assert_eq!(p.params.real(0), q.params.real(0));
            assert_eq!(p.weight, q.weight);
        }
    }
}

#[test]
fn jsonl_log_records_the_whole_run() {
    let dir = std::env::temp_dir().join(format!("abcsmc-jsonl-{}", std::process::id()));
    let log = JsonlObserver::create(&dir, "test")
        .unwrap()
        .max_trajectories(10);
    let path = log.path().to_path_buf();
    let distance = distance_to(
        "distance to 5",
        &[5.0, 5.0],
        |output: &f64, observed: &[f64]| (output - observed[0]).abs(),
    );
    let generations = run_quantiles_with(
        &TracedModel,
        &distance,
        &QUANTILES,
        N,
        "a \"traced\" run",
        &log,
    );
    drop(log);

    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let events: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let of_type =
        |t: &str| -> Vec<&serde_json::Value> { events.iter().filter(|e| e["type"] == t).collect() };

    let started = of_type("run_started");
    assert_eq!(started.len(), 1);
    assert_eq!(started[0]["id"], "test");
    assert_eq!(started[0]["description"], "a \"traced\" run");
    assert_eq!(started[0]["distance_description"], "distance to 5");
    assert_eq!(started[0]["n_particles"], N);
    assert_eq!(started[0]["n_generations"], QUANTILES.len() + 1);
    assert_eq!(
        started[0]["quantiles"].as_array().unwrap().len(),
        QUANTILES.len()
    );
    assert_eq!(started[0]["params"][0]["name"], "param_0");
    assert_eq!(started[0]["params"][0]["kind"], "real");
    assert_eq!(started[0]["observed"], serde_json::json!([5.0, 5.0]));

    let gen_started = of_type("generation_started");
    assert_eq!(gen_started.len(), generations.len());
    assert!(
        gen_started[0]["tolerance"].is_null(),
        "infinite tolerance is null"
    );
    assert!(gen_started[1]["tolerance"].is_number());

    let completed = of_type("generation_completed");
    assert_eq!(completed.len(), generations.len());
    for (event, generation) in completed.iter().zip(&generations) {
        assert_eq!(event["generation"], generation.stats.generation);
        assert_eq!(event["stats"]["attempts"], generation.stats.attempts);
        let particles = event["particles"].as_array().unwrap();
        assert_eq!(particles.len(), N);
        assert_eq!(
            particles[0]["params"][0],
            generation.particles[0].params.real(0)
        );
        assert_eq!(
            particles[0]["seed"],
            generation.particles[0].seed.to_string()
        );
        let trajectories = event["trajectories"].as_array().unwrap();
        assert_eq!(trajectories.len(), 10);
        assert_eq!(trajectories[1]["particle"], N / 10);
        assert_eq!(trajectories[9]["particle"], 9 * N / 10);
        assert_eq!(trajectories[0]["values"].as_array().unwrap().len(), 2);
    }

    let progress = of_type("progress");
    assert!(!progress.is_empty());
    let batched: usize = progress
        .iter()
        .filter(|e| e["generation"] == 0)
        .map(|e| e["batch"].as_array().unwrap().len())
        .sum();
    assert_eq!(
        batched, N,
        "every accepted particle is batched exactly once"
    );
    assert_eq!(progress.last().unwrap()["accepted"], N);
    for generation in 0..generations.len() {
        let last_of_generation = events
            .iter()
            .rposition(|e| e["generation"] == generation && e["type"] == "progress")
            .expect("each generation writes progress");
        assert_eq!(
            events[last_of_generation + 1]["type"],
            "generation_completed",
            "the pending batch is flushed right before the generation completes"
        );
    }

    assert_eq!(events.last().unwrap()["type"], "run_finished");
    assert_eq!(events.last().unwrap()["generations"], generations.len());
}

#[test]
fn trajectory_sample_spans_a_population_not_divisible_by_the_cap() {
    use abcsmc::diagnostics::trajectory_rows;
    let generations = run_with(&TracedModel, &to_five, &TOLERANCES[..1], 399, "", &Silent);
    let rows = trajectory_rows(&generations, 200, |output| vec![*output]);
    let particles: Vec<usize> = rows.iter().map(|r| r.particle).collect();
    assert_eq!(particles.len(), 200);
    assert_eq!(particles[0], 0);
    assert_eq!(particles[199], 398);
}
