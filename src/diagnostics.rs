//! Per-generation diagnostic data for plotting elsewhere.
//!
//! Everything here computes plain long-format rows from a finished run's
//! [`Generation`]s: the per-generation statistics table, every parameter draw across
//! generations, weighted histograms and Gaussian KDEs (bins and grids are shared
//! across generations per parameter, so prior and posterior are directly
//! comparable), and per-particle output trajectories. With the `serde` feature
//! the row types derive `Serialize`, so a host can hand them straight to a CSV
//! or JSON writer.

use crate::{CalibrationModel, Draw, Generation, GenerationStats, Value};

fn param_name<M: CalibrationModel>(index: usize) -> String {
    <M::Draw as Draw>::NAMES
        .get(index)
        .map(|name| (*name).to_string())
        .unwrap_or_else(|| format!("param_{index}"))
}

fn n_params<M: CalibrationModel>(generations: &[Generation<M>]) -> usize {
    generations
        .first()
        .and_then(|s| s.particles.first())
        .map(|p| p.params.len())
        .unwrap_or(0)
}

/// `(value, weight)` pairs for one parameter of one generation, weights normalized
fn generation_values<M: CalibrationModel>(
    generation: &Generation<M>,
    param: usize,
) -> Vec<(f64, f64)> {
    let total: f64 = generation.particles.iter().map(|p| p.weight).sum();
    generation
        .particles
        .iter()
        .map(|p| (p.params.values()[param].as_f64(), p.weight / total))
        .collect()
}

/// The per-generation statistics as rows (the particles/acceptance table).
pub fn generation_rows<M: CalibrationModel>(generations: &[Generation<M>]) -> Vec<GenerationStats> {
    generations.iter().map(|s| s.stats).collect()
}

/// One parameter value from one particle: the full posterior across generations in
/// long format.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ParamRow {
    pub generation: usize,
    pub param: String,
    pub particle: usize,
    pub value: f64,
    pub weight: f64,
}

pub fn param_rows<M: CalibrationModel>(generations: &[Generation<M>]) -> Vec<ParamRow> {
    let mut rows = Vec::new();
    for generation in generations {
        for (particle_number, particle) in generation.particles.iter().enumerate() {
            for (index, value) in particle.params.values().iter().enumerate() {
                rows.push(ParamRow {
                    generation: generation.stats.generation,
                    param: param_name::<M>(index),
                    particle: particle_number,
                    value: value.as_f64(),
                    weight: particle.weight,
                });
            }
        }
    }
    rows
}

/// One weighted histogram bar. Weights are normalized within each generation, and
/// bin edges are shared across generations per parameter, so the prior (generation 0)
/// and each posterior generation are directly comparable. Integer parameters get
/// one bin per value, centered on it.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct HistogramRow {
    pub generation: usize,
    pub param: String,
    pub bin_start: f64,
    pub bin_end: f64,
    pub weight: f64,
}

pub fn histogram_rows<M: CalibrationModel>(
    generations: &[Generation<M>],
    bins: usize,
) -> Vec<HistogramRow> {
    let mut rows = Vec::new();
    for param in 0..n_params::<M>(generations) {
        let is_int = matches!(
            generations[0].particles[0].params.values()[param],
            Value::Int(_)
        );
        let all: Vec<f64> = generations
            .iter()
            .flat_map(|s| s.particles.iter())
            .map(|p| p.params.values()[param].as_f64())
            .collect();
        let min = all.iter().copied().fold(f64::INFINITY, f64::min);
        let max = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for generation in generations {
            let values = generation_values(generation, param);
            if is_int {
                let (lo, hi) = (min as i64, max as i64);
                let mut weights = vec![0.0; (hi - lo + 1) as usize];
                for (value, weight) in &values {
                    weights[(*value as i64 - lo) as usize] += weight;
                }
                for (offset, weight) in weights.iter().enumerate() {
                    let center = (lo + offset as i64) as f64;
                    rows.push(HistogramRow {
                        generation: generation.stats.generation,
                        param: param_name::<M>(param),
                        bin_start: center - 0.5,
                        bin_end: center + 0.5,
                        weight: *weight,
                    });
                }
            } else {
                let width = ((max - min) / bins as f64).max(f64::MIN_POSITIVE);
                let mut weights = vec![0.0; bins];
                for (value, weight) in &values {
                    let bin = (((value - min) / width) as usize).min(bins - 1);
                    weights[bin] += weight;
                }
                for (bin, weight) in weights.iter().enumerate() {
                    rows.push(HistogramRow {
                        generation: generation.stats.generation,
                        param: param_name::<M>(param),
                        bin_start: min + bin as f64 * width,
                        bin_end: min + (bin + 1) as f64 * width,
                        weight: *weight,
                    });
                }
            }
        }
    }
    rows
}

/// One point of a weighted Gaussian KDE curve, evaluated on a grid shared
/// across generations per parameter (real-valued parameters only).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct KdeRow {
    pub generation: usize,
    pub param: String,
    pub x: f64,
    pub density: f64,
}

pub fn kde_rows<M: CalibrationModel>(
    generations: &[Generation<M>],
    grid_points: usize,
) -> Vec<KdeRow> {
    let mut rows = Vec::new();
    for param in 0..n_params::<M>(generations) {
        if matches!(
            generations[0].particles[0].params.values()[param],
            Value::Int(_)
        ) {
            continue;
        }
        let all: Vec<f64> = generations
            .iter()
            .flat_map(|s| s.particles.iter())
            .map(|p| p.params.values()[param].as_f64())
            .collect();
        let min = all.iter().copied().fold(f64::INFINITY, f64::min);
        let max = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pad = 0.1 * (max - min);

        for generation in generations {
            let values = generation_values(generation, param);
            let mean: f64 = values.iter().map(|(v, w)| v * w).sum();
            let sd = values
                .iter()
                .map(|(v, w)| w * (v - mean).powi(2))
                .sum::<f64>()
                .sqrt();
            // Silverman's rule on the weighted sd and effective sample size
            let bandwidth = 1.06 * sd * generation.stats.ess.powf(-0.2);
            if !bandwidth.is_finite() || bandwidth <= 0.0 {
                continue;
            }
            for point in 0..grid_points {
                let x =
                    (min - pad) + (max - min + 2.0 * pad) * point as f64 / (grid_points - 1) as f64;
                let density: f64 = values
                    .iter()
                    .map(|(v, w)| {
                        let z = (x - v) / bandwidth;
                        w * (-0.5 * z * z).exp()
                    })
                    .sum::<f64>()
                    / (bandwidth * (2.0 * std::f64::consts::PI).sqrt());
                rows.push(KdeRow {
                    generation: generation.stats.generation,
                    param: param_name::<M>(param),
                    x,
                    density,
                });
            }
        }
    }
    rows
}

/// One point of one particle's output series, for overlaying simulated
/// trajectories on the target data per generation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TrajectoryRow {
    pub generation: usize,
    pub particle: usize,
    pub index: usize,
    pub value: f64,
}

/// Extract each particle's series with `extract`, keeping at most
/// `max_per_generation` particles per generation (evenly strided, so the sample spans
/// the population).
pub fn trajectory_rows<M: CalibrationModel>(
    generations: &[Generation<M>],
    max_per_generation: usize,
    extract: impl Fn(&M::Output) -> Vec<f64>,
) -> Vec<TrajectoryRow> {
    let mut rows = Vec::new();
    for generation in generations {
        let stride = generation
            .particles
            .len()
            .div_ceil(max_per_generation.max(1))
            .max(1);
        for (particle_number, particle) in generation
            .particles
            .iter()
            .enumerate()
            .step_by(stride)
            .take(max_per_generation)
        {
            for (index, value) in extract(&particle.output).into_iter().enumerate() {
                rows.push(TrajectoryRow {
                    generation: generation.stats.generation,
                    particle: particle_number,
                    index,
                    value,
                });
            }
        }
    }
    rows
}
