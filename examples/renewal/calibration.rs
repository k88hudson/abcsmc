//! Calibrating the renewal model with ABC-SMC. The [`renewal`] module is a plain
//! simulator; everything inference-specific lives here.
//!
//! [`renewal`]: crate::renewal

use std::path::Path;

use abcsmc::{IntPrior, Model, Priors, RealPrior, define_priors, run_quantiles};
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;

use crate::renewal::{Parameters, Population, RenewalModel, RenewalOutput};

define_priors! {
    pub struct RenewalDraw / RenewalPriors {
        r0: Real,
        initial_infections: Int,
    }
}

struct RenewalFit {
    observed: Vec<u64>,
}

impl Model for RenewalFit {
    type Draw = RenewalDraw;
    type Output = RenewalOutput;

    fn priors(&self) -> Priors {
        RenewalPriors {
            r0: RealPrior::exponential(1.0).unwrap(),
            initial_infections: IntPrior::discrete_uniform(1, 4).unwrap(),
        }
        .into()
    }

    fn simulate(&self, draw: &RenewalDraw, seed: u64) -> RenewalOutput {
        let parameters = Parameters {
            population: Population::Finite(10_000),
            r0: draw.r0,
            generation_interval_pmf: vec![0., 0., 0.25, 0.5, 0.25],
            symptom_onset_pmf: vec![0., 0.5, 0.5],
            initial_infections: vec![draw.initial_infections as u64],
            sim_length: 7 * 24,
        };
        RenewalModel::simulate(&parameters, &mut StdRng::seed_from_u64(seed))
    }

    /// Absolute difference in total symptomatic incidence over the fitting window.
    fn distance(&self, output: &RenewalOutput) -> f64 {
        let simulated: u64 = output
            .symptomatic_incidence
            .iter()
            .take(self.observed.len())
            .sum();
        let observed: u64 = self.observed.iter().sum();
        u64::abs_diff(simulated, observed) as f64
    }
}

#[derive(Serialize)]
struct ParticleRow {
    generation: usize,
    particle_number: usize,
    weight: f64,
    r0: f64,
    initial_infections: u64,
    distance: f64,
    seed: u64,
}

#[derive(Serialize)]
struct TrajectoryRow {
    generation: usize,
    particle_number: usize,
    day: usize,
    symptomatic_incidence: u64,
}

fn write_csv<R: Serialize>(rows: &[R], path: &Path) {
    let mut writer = csv::Writer::from_path(path).unwrap();
    for row in rows {
        writer.serialize(row).unwrap();
    }
    writer.flush().unwrap();
}

pub fn fit() {
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let incidence_path = crate_path.join("examples/input/sample_incidence.csv");
    let mut reader = csv::Reader::from_path(incidence_path).unwrap();
    let incidence: Vec<u64> = reader
        .records()
        .map(|record| record.unwrap().get(1).unwrap().parse().unwrap())
        .collect();

    // Fit the first 6 weeks; later observations are held out.
    let model = RenewalFit {
        observed: incidence.into_iter().take(7 * 6).collect(),
    };
    let generations = run_quantiles(&model, &[0.1, 0.05, 0.01, 0.005], 2_000);

    let mut particle_rows = Vec::new();
    let mut trajectory_rows = Vec::new();
    for population in &generations {
        let generation = population.stats.generation;
        for (particle_number, particle) in population.particles.iter().enumerate() {
            let draw = particle.draw();
            particle_rows.push(ParticleRow {
                generation,
                particle_number,
                weight: particle.weight,
                r0: draw.r0,
                initial_infections: draw.initial_infections as u64,
                distance: particle.distance,
                seed: particle.seed,
            });
            for (day, incidence) in particle.output.symptomatic_incidence.iter().enumerate() {
                trajectory_rows.push(TrajectoryRow {
                    generation,
                    particle_number,
                    day,
                    symptomatic_incidence: *incidence,
                });
            }
        }
    }

    let output_dir = crate_path.join("examples/output");
    write_csv(&particle_rows, &output_dir.join("particles.csv"));
    write_csv(&trajectory_rows, &output_dir.join("trajectories.csv"));
}
