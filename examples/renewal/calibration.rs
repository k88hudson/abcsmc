//! Calibrating the renewal model with ABC-SMC. The [`renewal`] module is a plain
//! simulator; everything inference-specific lives here.
//!
//! [`renewal`]: crate::renewal

use std::path::Path;

use abcsmc::{
    CalibrationModel, Distance, Generation, JsonlObserver, Particle, Priors, StdoutObserver,
    define_priors, discrete_uniform_prior, distance, exponential_prior, run_quantiles_with,
};
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;

use crate::projection;
use crate::renewal::{Parameters, Population, RenewalModel, RenewalOutput, TransmissionChange};

/// Fit the first weeks of the data; later observations are held out.
const FITTED_WEEKS: usize = 6;
const FITTED_DAYS: usize = 7 * FITTED_WEEKS;
/// Which distance to fit with: change this to try another.
const METRIC: Metric = Metric::DailyL1;
const N_PARTICLES: usize = 2_000;
/// Each generation's tolerance, as a quantile of the prior's distances.
const QUANTILES: [f64; 4] = [0.1, 0.05, 0.01, 0.005];

define_priors! {
    pub struct RenewalDraw / RenewalPriors {
        r0: Real,
        initial_infections: Int,
    }
}

/// Fixed settings live here rather than in the priors, so a scenario is this
/// model with a field changed.
#[derive(Clone)]
pub struct RenewalFit {
    pub transmission_change: Option<TransmissionChange>,
}

impl CalibrationModel for RenewalFit {
    type Draw = RenewalDraw;
    type Output = RenewalOutput;

    fn priors(&self) -> Priors {
        RenewalPriors {
            r0: exponential_prior!(rate = 1.0).unwrap(),
            initial_infections: discrete_uniform_prior!(a = 1, b = 4).unwrap(),
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
            transmission_change: self.transmission_change,
        };
        RenewalModel::simulate(&parameters, &mut StdRng::seed_from_u64(seed))
    }

    fn trajectory(&self, output: &RenewalOutput) -> Option<Vec<f64>> {
        Some(as_f64s(&output.symptomatic_incidence))
    }
}

fn as_f64s(counts: &[u64]) -> Vec<f64> {
    counts.iter().map(|&x| x as f64).collect()
}

/// Ways to score simulated symptomatic incidence against the observed series.
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum Metric {
    /// Day-by-day L1: follows the shape of the curve, noise included.
    DailyL1,
    /// L1 on weekly totals: follows the shape but smooths daily noise.
    WeeklyL1,
    /// Difference in total cases: outbreak size only, ignoring timing.
    TotalCases,
}

/// The run's distance: one [`Metric`] over the fitted days.
struct IncidenceDistance {
    metric: Metric,
    observed: Vec<u64>,
}

impl Distance<RenewalOutput> for IncidenceDistance {
    fn distance(&self, output: &RenewalOutput) -> f64 {
        let observed = &self.observed;
        let simulated = &output.symptomatic_incidence[..observed.len()];
        match self.metric {
            Metric::DailyL1 => distance::l1(simulated, observed),
            Metric::WeeklyL1 => distance::l1(&weekly(simulated), &weekly(observed)),
            Metric::TotalCases => distance::total_difference(simulated, observed),
        }
    }

    fn description(&self) -> Option<String> {
        let metric = match self.metric {
            Metric::DailyL1 => "L1 distance between daily symptomatic incidence",
            Metric::WeeklyL1 => "L1 distance between weekly symptomatic incidence",
            Metric::TotalCases => "Absolute difference in total symptomatic incidence",
        };
        Some(format!(
            "{metric} over the first {} days",
            self.observed.len()
        ))
    }

    fn observed(&self) -> Option<Vec<f64>> {
        Some(as_f64s(&self.observed))
    }
}

fn weekly(daily: &[u64]) -> Vec<u64> {
    daily.chunks(7).map(|week| week.iter().sum()).collect()
}

pub fn fit() {
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output_dir = crate_path.join("examples/output");

    let incidence = read_incidence(&crate_path.join("examples/input/sample_incidence.csv"));
    let observed = &incidence[..FITTED_DAYS];

    let model = RenewalFit {
        transmission_change: None,
    };
    let distance = IncidenceDistance {
        metric: METRIC,
        observed: observed.to_vec(),
    };

    let log = JsonlObserver::create(output_dir.join("runs"), "renewal").unwrap();
    let log_path = log.path().to_path_buf();
    println!("Writing {}", log_path.display());

    let generations = run_quantiles_with(
        &model,
        &distance,
        &QUANTILES,
        N_PARTICLES,
        &format!("Renewal model fitted to the first {FITTED_WEEKS} weeks of sample incidence"),
        &(StdoutObserver::new(), log),
    );

    write_csv(
        &output_dir.join("particles.csv"),
        particle_rows(&generations),
    );
    write_csv(
        &output_dir.join("trajectories.csv"),
        trajectory_rows(&generations),
    );
    if let Some(posterior) = generations.last() {
        projection::write_scenarios(&model, posterior, FITTED_DAYS, &log_path);
    }
}

/// The `symptomatic_incidence` column of a `day,symptomatic_incidence` file.
fn read_incidence(path: &Path) -> Vec<u64> {
    csv::Reader::from_path(path)
        .unwrap()
        .records()
        .map(|record| record.unwrap().get(1).unwrap().parse().unwrap())
        .collect()
}

fn write_csv(path: &Path, rows: impl Iterator<Item = impl Serialize>) {
    let mut writer = csv::Writer::from_path(path).unwrap();
    for row in rows {
        writer.serialize(row).unwrap();
    }
    writer.flush().unwrap();
}

/// Every particle of every generation, with its index within the generation.
fn numbered_particles(
    generations: &[Generation<RenewalFit>],
) -> impl Iterator<Item = (usize, usize, &Particle<RenewalFit>)> {
    generations.iter().flat_map(|generation| {
        generation
            .particles
            .iter()
            .enumerate()
            .map(|(number, particle)| (generation.stats.generation, number, particle))
    })
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

fn particle_rows(generations: &[Generation<RenewalFit>]) -> impl Iterator<Item = ParticleRow> {
    numbered_particles(generations).map(|(generation, particle_number, particle)| {
        let draw = particle.draw();
        ParticleRow {
            generation,
            particle_number,
            weight: particle.weight,
            r0: draw.r0,
            initial_infections: draw.initial_infections as u64,
            distance: particle.distance,
            seed: particle.seed,
        }
    })
}

#[derive(Serialize)]
struct TrajectoryRow {
    generation: usize,
    particle_number: usize,
    day: usize,
    symptomatic_incidence: u64,
}

fn trajectory_rows(generations: &[Generation<RenewalFit>]) -> impl Iterator<Item = TrajectoryRow> {
    numbered_particles(generations).flat_map(|(generation, particle_number, particle)| {
        particle
            .output
            .symptomatic_incidence
            .iter()
            .enumerate()
            .map(move |(day, &symptomatic_incidence)| TrajectoryRow {
                generation,
                particle_number,
                day,
                symptomatic_incidence,
            })
    })
}
