//! Calibrating the renewal model with ABC-SMC. The [`renewal`] module is a plain
//! simulator; everything inference-specific lives here.
//!
//! [`renewal`]: crate::renewal

use std::path::Path;

use abcsmc::{
    CalibrationModel, Distance, Generation, JsonlObserver, Marker, Particle, Priors, Series,
    StdoutObserver, TargetRange, define_priors, discrete_uniform_prior, distance,
    exponential_prior, run_quantiles_with,
};
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;

use crate::projection;
use crate::renewal::{Parameters, Population, RenewalModel, RenewalOutput, TransmissionChange};

/// Fit the first weeks of the data; later observations are held out.
const FITTED_WEEKS: usize = 6;
const FITTED_DAYS: usize = 7 * FITTED_WEEKS;
/// Which distance to fit with: change this to try another.
const METRIC: Metric = Metric::DailyRange;
/// How far above the observed count the target range of [`Metric::DailyRange`]
/// reaches, as a fraction of it.
const RANGE_ABOVE: f64 = 0.25;
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

    /// The two scored series, see [`IncidenceDistance`].
    fn trajectories(&self, output: &RenewalOutput) -> Vec<Series> {
        vec![
            Series::named(DAILY, as_f64s(&output.symptomatic_incidence)),
            Series::named(
                CUMULATIVE,
                as_f64s(&cumulative(&output.symptomatic_incidence)),
            ),
        ]
    }
}

/// Series names, shared by the model's trajectories and the distance's
/// observed series so viewers pair them.
const DAILY: &str = "Symptomatic incidence";
const CUMULATIVE: &str = "Cumulative symptomatic incidence";

fn as_f64s(counts: &[u64]) -> Vec<f64> {
    counts.iter().map(|&x| x as f64).collect()
}

fn cumulative(counts: &[u64]) -> Vec<u64> {
    counts
        .iter()
        .scan(0, |total, &x| {
            *total += x;
            Some(*total)
        })
        .collect()
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
    /// Day-by-day distance outside a range from the observed count up to
    /// [`RANGE_ABOVE`] more: a floor the simulation should stay at or above,
    /// with some room over it.
    DailyRange,
}

/// The run's distance over two series of the fitted days: one [`Metric`] on
/// daily symptomatic incidence, plus the mean absolute error of cumulative
/// incidence, which holds the outbreak's size to the data as well as its
/// shape.
struct IncidenceDistance {
    metric: Metric,
    observed: Vec<u64>,
}

impl Distance<RenewalOutput> for IncidenceDistance {
    fn distance(&self, output: &RenewalOutput) -> f64 {
        let observed = &self.observed;
        let simulated = &output.symptomatic_incidence[..observed.len()];
        let daily = match self.metric {
            Metric::DailyL1 => distance::l1(simulated, observed),
            Metric::WeeklyL1 => distance::l1(&weekly(simulated), &weekly(observed)),
            Metric::TotalCases => distance::total_difference(simulated, observed),
            Metric::DailyRange => distance::l1_outside(simulated, observed, &self.range_ceiling()),
        };
        daily + distance::mean_absolute_error(&cumulative(simulated), &cumulative(observed))
    }

    fn description(&self) -> Option<String> {
        let metric = match self.metric {
            Metric::DailyL1 => "L1 distance between daily symptomatic incidence",
            Metric::WeeklyL1 => "L1 distance between weekly symptomatic incidence",
            Metric::TotalCases => "Absolute difference in total symptomatic incidence",
            Metric::DailyRange => {
                "Distance of daily symptomatic incidence outside the range from the observed count to 25% above it"
            }
        };
        Some(format!(
            "{metric} over the first {} days, plus the mean absolute error of cumulative incidence",
            self.observed.len()
        ))
    }

    fn observed_series(&self) -> Vec<Series> {
        vec![
            Series::named(DAILY, as_f64s(&self.observed)),
            Series::named(CUMULATIVE, as_f64s(&cumulative(&self.observed))),
        ]
    }

    /// The last fitted day, and under the weekly metric each week's first day.
    fn markers(&self) -> Vec<Marker> {
        let days = self.observed.len();
        let mut markers = vec![Marker::labelled((days - 1) as f64, "Last fitted day")];
        if let Metric::WeeklyL1 = self.metric {
            markers.extend((7..days).step_by(7).map(|day| Marker::at(day as f64)));
        }
        markers
    }

    /// Under the range metric, each fitted day's target range.
    fn ranges(&self) -> Vec<TargetRange> {
        let Metric::DailyRange = self.metric else {
            return Vec::new();
        };
        self.observed
            .iter()
            .zip(self.range_ceiling())
            .enumerate()
            .map(|(day, (&floor, ceiling))| {
                TargetRange::on(DAILY, day as f64, floor as f64, ceiling)
            })
            .collect()
    }
}

impl IncidenceDistance {
    /// The top of each fitted day's target range under [`Metric::DailyRange`].
    fn range_ceiling(&self) -> Vec<f64> {
        self.observed
            .iter()
            .map(|&count| count as f64 * (1.0 + RANGE_ABOVE))
            .collect()
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
