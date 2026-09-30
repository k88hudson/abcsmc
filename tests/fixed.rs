//! Integration tests for the drivers on 1- and 2-parameter toy models where the
//! posterior is known.

use std::collections::HashSet;

use abcsmc::{
    Draw, Generation, IntPrior, Model, Params, Priors, RealPrior, define_priors, run, run_quantiles,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

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

    fn distance(&self, output: &f64) -> f64 {
        (output - 5.0).abs()
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

    fn distance(&self, output: &f64) -> f64 {
        (output - 5.0).abs()
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
    let generations = run(&ToyModel, &TOLERANCES, N);
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
    let generations = run_quantiles(&ToyModel, &QUANTILES, N);
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
    for generation in run_quantiles(&ToyModel, &QUANTILES, N) {
        let total: f64 = generation.particles.iter().map(|p| p.weight).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }
}

#[test]
fn stored_distance_matches_output() {
    let model = NoisyModel;
    for generation in run_quantiles(&model, &QUANTILES, N) {
        for particle in &generation.particles {
            assert_eq!(particle.distance, model.distance(&particle.output));
        }
    }
}

#[test]
fn particle_seed_replays_its_simulation() {
    let model = NoisyModel;
    let generations = run_quantiles(&model, &QUANTILES, N);
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

    fn distance(&self, output: &f64) -> f64 {
        (output - 5.0).abs()
    }

    fn max_attempts_per_proposal(&self) -> u64 {
        50
    }
}

#[test]
fn exhausted_attempts_stop_the_run_early() {
    let generations = run(&UnreachableModel, &[5.0, 1.0, -1.0], 20);
    assert_eq!(generations.len(), 2);
}

define_priors! {
    pub struct MacroDraw / MacroPriors {
        x: Real,
        k: Int,
    }
}

struct MacroModel;

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

    fn distance(&self, output: &(f64, i64)) -> f64 {
        (output.0 - 5.0).abs() + ((output.1 - 3).abs() as f64)
    }
}

#[test]
fn macro_generated_draw_maps_priors_to_named_fields() {
    assert_eq!(MacroDraw::NAMES, &["x", "k"]);

    let generations = run_quantiles(&MacroModel, &QUANTILES, N);
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
