//! The adaptive schedule: tolerances set from each generation, stopping at
//! the floor, at a low acceptance ratio, or at the generation limit, and
//! resuming from a log.

use abcsmc::{
    Adaptive, CalibrationModel, Generation, IntPrior, Priors, RealPrior, Silent, define_priors,
    run_adaptive_with,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

define_priors! {
    pub struct Draw / DrawPriors {
        rate: Real,
        count: Int,
    }
}

/// Output is `rate * count` plus seed-driven noise on `[0, 1)`, targeting 12.
struct NoisyProduct;

impl CalibrationModel for NoisyProduct {
    type Draw = Draw;
    type Output = f64;

    fn priors(&self) -> Priors {
        DrawPriors {
            rate: RealPrior::uniform(0.0, 10.0).unwrap(),
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

const SCHEDULE: Adaptive = Adaptive {
    quantile: 0.5,
    floor: 1.0,
    min_acceptance: 0.001,
    max_generations: 20,
};

fn tolerances(generations: &[Generation<NoisyProduct>]) -> Vec<f64> {
    generations.iter().map(|g| g.stats.tolerance).collect()
}

#[test]
fn tolerances_fall_by_the_quantile_and_stop_at_the_floor() {
    let generations = run_adaptive_with(&NoisyProduct, &to_twelve, SCHEDULE, N, "", &Silent);
    let tolerances = tolerances(&generations);
    assert!(tolerances[0].is_infinite());
    for (previous, generation) in generations.iter().zip(&generations[1..]) {
        let mut distances: Vec<f64> = previous.particles.iter().map(|p| p.distance).collect();
        distances.sort_by(f64::total_cmp);
        // The `q * n`th smallest of the previous generation, down to the floor.
        let n = distances.len();
        let expected = distances[((n as f64 * SCHEDULE.quantile) as usize).clamp(1, n) - 1]
            .max(SCHEDULE.floor);
        assert_eq!(generation.stats.tolerance, expected);
        assert!(generation.stats.tolerance < previous.stats.tolerance);
    }
    assert_eq!(*tolerances.last().unwrap(), SCHEDULE.floor);
    assert!(tolerances.len() > 2 && tolerances.len() <= SCHEDULE.max_generations);
    assert!(
        generations
            .last()
            .unwrap()
            .particles
            .iter()
            .all(|p| p.distance <= SCHEDULE.floor)
    );
}

#[test]
fn the_generation_limit_counts_the_prior() {
    let schedule = Adaptive {
        max_generations: 3,
        ..SCHEDULE
    };
    let generations = run_adaptive_with(&NoisyProduct, &to_twelve, schedule, N, "", &Silent);
    assert_eq!(generations.len(), 3);
    assert!(generations[2].stats.tolerance > SCHEDULE.floor);
}

#[test]
fn a_generation_accepting_too_little_is_the_last() {
    // Every generation accepts at most all of its proposals, so this stops
    // after the prior.
    let schedule = Adaptive {
        min_acceptance: 1.5,
        ..SCHEDULE
    };
    let generations = run_adaptive_with(&NoisyProduct, &to_twelve, schedule, N, "", &Silent);
    assert_eq!(generations.len(), 1);
}

#[test]
fn next_tolerance_follows_the_rules() {
    let distances = || [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_iter();
    assert_eq!(
        SCHEDULE.next_tolerance(f64::INFINITY, 1.0, distances(), 1),
        Some(5.0)
    );
    // Never below the floor.
    let small = || distances().map(|d| d / 10.0);
    assert_eq!(
        SCHEDULE.next_tolerance(3.0, 0.5, small(), 1),
        Some(SCHEDULE.floor)
    );
    // Stops: at the floor, too few accepted, the limit, or no fall.
    assert_eq!(SCHEDULE.next_tolerance(1.0, 0.5, distances(), 1), None);
    assert_eq!(SCHEDULE.next_tolerance(8.0, 0.0001, distances(), 1), None);
    assert_eq!(SCHEDULE.next_tolerance(8.0, 0.5, distances(), 20), None);
    assert_eq!(SCHEDULE.next_tolerance(5.0, 0.5, distances(), 1), None);
}

#[cfg(feature = "serde")]
mod resume {
    use super::*;
    use abcsmc::{JsonlObserver, RunLog, resume_adaptive_with};
    use std::path::PathBuf;

    struct Runs(PathBuf);

    impl Runs {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("abcsmc-adaptive-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            Runs(dir)
        }

        fn path(&self, id: &str) -> PathBuf {
            self.0.join(id).join("run.jsonl")
        }
    }

    impl Drop for Runs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

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

    /// The log of `whole` cut off partway through `generation`.
    fn cut_during(log: &std::path::Path, generation: usize) -> String {
        let text = std::fs::read_to_string(log).unwrap();
        let started = format!(r#"{{"type":"generation_started","generation":{generation},"#);
        let from = text.find(&started).unwrap();
        let next = text[from..].find("\n").map_or(text.len(), |i| from + i + 1);
        let after = text[next..].find('\n').map_or(text.len(), |i| next + i / 2);
        text[..after].to_string()
    }

    #[test]
    fn an_adaptive_run_resumes_with_the_tolerances_of_its_log() {
        let runs = Runs::new();
        let whole = run_adaptive_with(
            &NoisyProduct,
            &to_twelve,
            SCHEDULE,
            N,
            "",
            &JsonlObserver::create(&runs.0, "whole").unwrap(),
        );
        let log = RunLog::read(runs.path("whole")).unwrap();
        assert_eq!(log.adaptive, Some(SCHEDULE));
        assert_eq!(log.generations.len(), whole.len());

        std::fs::create_dir_all(runs.path("killed").parent().unwrap()).unwrap();
        std::fs::write(runs.path("killed"), cut_during(&runs.path("whole"), 2)).unwrap();
        let killed = RunLog::read(runs.path("killed")).unwrap();
        assert_eq!(killed.generations.len(), 2);

        let resumed = resume_adaptive_with(
            &NoisyProduct,
            &to_twelve,
            SCHEDULE,
            &killed,
            "",
            &JsonlObserver::append(&runs.0, "killed").unwrap(),
        )
        .unwrap();
        assert_eq!(resumed.len(), whole.len() - 2);
        for (resumed, whole) in resumed.iter().zip(&whole[2..]) {
            assert_eq!(resumed.stats.tolerance, whole.stats.tolerance);
            assert_eq!(particles(resumed), particles(whole));
        }
        // A finished run has nothing left to do.
        let again =
            resume_adaptive_with(&NoisyProduct, &to_twelve, SCHEDULE, &log, "", &Silent).unwrap();
        assert!(again.is_empty());
        // A log of another schedule is refused.
        let other = Adaptive {
            quantile: 0.3,
            ..SCHEDULE
        };
        assert!(
            resume_adaptive_with(&NoisyProduct, &to_twelve, other, &killed, "", &Silent).is_err()
        );
    }
}
