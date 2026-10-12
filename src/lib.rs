pub mod diagnostics;
pub mod distance;
pub use distance::{Distance, DistanceFn, DistanceTo, distance_fn, distance_to};
mod kernel;
pub use kernel::*;
mod variance_adapter;
pub use variance_adapter::*;
#[macro_use]
mod macros;
mod param;
pub use param::*;
mod draw;
pub use draw::*;
mod distribution;
pub use distribution::*;
mod prior;
pub use prior::*;
mod model;
pub use model::*;
mod step;
pub use step::*;
mod particle;
pub use particle::*;
mod marker;
pub use marker::*;
mod range;
pub use range::*;
mod series;
pub use series::*;
mod observer;
pub use observer::*;
mod platform;
mod sampler;
pub use sampler::*;
mod run_log;
pub use run_log::*;
#[cfg(feature = "serde")]
mod posterior;
#[cfg(feature = "serde")]
pub use posterior::*;

/// One completed generation: its accepted particles and run statistics.
pub struct Generation<M: CalibrationModel> {
    pub particles: Vec<Particle<M>>,
    pub stats: GenerationStats,
}

impl<M: CalibrationModel> Generation<M> {
    /// `n` particles drawn with replacement in proportion to their weights:
    /// an equally weighted sample of this generation's posterior.
    pub fn resample(&self, n: usize, rng: &mut impl rand::Rng) -> Vec<&Particle<M>> {
        let weights = self.particles.iter().map(|p| p.weight);
        resample_indices(weights, n, rng)
            .into_iter()
            .map(|i| &self.particles[i])
            .collect()
    }
}

/// `n` indices drawn with replacement in proportion to `weights`.
pub(crate) fn resample_indices(
    weights: impl Iterator<Item = f64>,
    n: usize,
    rng: &mut impl rand::Rng,
) -> Vec<usize> {
    use rand_distr::{Distribution, weighted::WeightedIndex};
    let sampler =
        WeightedIndex::new(weights).expect("weights must be non-negative and not all zero");
    (0..n).map(|_| sampler.sample(rng)).collect()
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct GenerationStats {
    pub generation: usize,
    pub tolerance: f64,
    pub accepted: usize,
    /// Simulations run, accepted or not
    pub attempts: u64,
    pub acceptance_ratio: f64,
    pub ess: f64,
    pub perplexity: f64,
    pub duration_seconds: f64,
}

/// Runs the generations of `schedule` that `sampler` has not completed yet,
/// stopping early if one cannot be filled.
fn run_schedule<M, D, O>(sampler: &mut Sampler<'_, M, D, O>, schedule: &[f64])
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
    D: Distance<M::Output> + ?Sized,
    O: RunObserver<M>,
{
    for &tolerance in schedule.iter().skip(sampler.completed()) {
        if sampler.next_generation(tolerance).is_none() {
            break;
        }
    }
}

/// Run ABC-SMC with one generation per tolerance: generation 0 is rejection
/// sampling from the prior at `tolerances[0]`, and each later generation
/// resamples, perturbs, and reweights the previous one. A simulation is
/// accepted when `distance` puts it within the generation's tolerance.
/// Progress is printed to stdout; see [`run_with`] to observe the run.
pub fn run<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    tolerances: &[f64],
    n_particles: usize,
) -> Vec<Generation<M>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    run_with(
        model,
        distance,
        tolerances,
        n_particles,
        "",
        &StdoutObserver::new(),
    )
}

/// [`run`] reporting to `observer` instead of stdout. Pair observers with a
/// tuple: `&(StdoutObserver::new(), JsonlObserver::create(dir, id)?)`.
/// `description` says what this run is, for observers to show and log; pass
/// `""` for none.
pub fn run_with<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    tolerances: &[f64],
    n_particles: usize,
    description: &str,
    observer: &impl RunObserver<M>,
) -> Vec<Generation<M>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    let mut sampler = Sampler::start(
        model,
        distance,
        n_particles,
        description,
        tolerances.len(),
        observer,
    );
    run_schedule(&mut sampler, tolerances);
    sampler.finish()
}

/// Like [`run`], but with tolerances given as quantiles of the prior's distance
/// distribution. Generation 0 is an unconstrained prior sample (infinite
/// tolerance); generation `g >= 1` accepts distances up to `quantiles[g - 1]`
/// of generation 0's distances. Returns `quantiles.len() + 1` generations.
pub fn run_quantiles<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    quantiles: &[f64],
    n_particles: usize,
) -> Vec<Generation<M>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    run_quantiles_with(
        model,
        distance,
        quantiles,
        n_particles,
        "",
        &StdoutObserver::new(),
    )
}

/// [`run_quantiles`] reporting to `observer` instead of stdout, with a
/// `description` of the run as in [`run_with`].
pub fn run_quantiles_with<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    quantiles: &[f64],
    n_particles: usize,
    description: &str,
    observer: &impl RunObserver<M>,
) -> Vec<Generation<M>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    let mut sampler = Sampler::start_with_quantiles(
        model,
        distance,
        n_particles,
        description,
        quantiles.len() + 1,
        Some(quantiles),
        observer,
    );
    if let Some(prior) = sampler.next_generation(f64::INFINITY) {
        let schedule = quantile_schedule(prior.particles.iter().map(|p| p.distance), quantiles);
        run_schedule(&mut sampler, &schedule);
    }
    sampler.finish()
}

/// The tolerance of every generation of a quantile run: infinite for
/// generation 0, then each quantile of generation 0's `distances`.
fn quantile_schedule(distances: impl Iterator<Item = f64>, quantiles: &[f64]) -> Vec<f64> {
    let mut distances: Vec<f64> = distances.collect();
    distances.sort_by(|a, b| a.total_cmp(b));
    std::iter::once(f64::INFINITY)
        .chain(quantiles.iter().map(|&q| quantile_threshold(&distances, q)))
        .collect()
}

#[cfg(feature = "serde")]
fn invalid(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

/// Checks that `log` is a run of `model`, so its particles mean what the
/// model's priors say they mean.
#[cfg(feature = "serde")]
fn check_same_model<M: CalibrationModel>(model: &M, log: &RunLog) -> std::io::Result<()> {
    let priors = model.priors();
    if priors.len() != log.params.len() {
        return Err(invalid(format!(
            "the model has {} parameters but the log has {}",
            priors.len(),
            log.params.len()
        )));
    }
    let names = <M::Draw as Draw>::NAMES;
    for (i, (prior, logged)) in priors.iter().zip(&log.params).enumerate() {
        if names.get(i).is_some_and(|name| *name != logged.name) {
            return Err(invalid(format!(
                "parameter {i} is {} in the model but {} in the log",
                names[i], logged.name
            )));
        }
        let same_prior = logged
            .prior
            .as_ref()
            .is_none_or(|p| p.to_string() == prior.to_string());
        if priors.is_real(i) != logged.real || !same_prior {
            return Err(invalid(format!(
                "the prior of {} is {prior} in the model but not in the log",
                logged.name
            )));
        }
    }
    Ok(())
}

/// A sampler placed after the last generation `log` completed, or at the
/// start of a new run if it completed none.
#[cfg(feature = "serde")]
fn sampler_after<'a, M, D, O>(
    model: &'a M,
    distance: &'a D,
    log: &RunLog,
    description: &str,
    n_generations: usize,
    quantiles: Option<&[f64]>,
    observer: &'a O,
) -> std::io::Result<Sampler<'a, M, D, O>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
    D: Distance<M::Output> + ?Sized,
    O: RunObserver<M>,
{
    check_same_model(model, log)?;
    let Some(last) = log.generations.last() else {
        return Ok(Sampler::start_with_quantiles(
            model,
            distance,
            log.n_particles,
            description,
            n_generations,
            quantiles,
            observer,
        ));
    };
    if last.particles.len() != log.n_particles {
        return Err(invalid(format!(
            "generation {} has {} particles but the run declared {}",
            last.stats.generation,
            last.particles.len(),
            log.n_particles
        )));
    }
    Ok(Sampler::resume_with_quantiles(
        model,
        distance,
        last.population(),
        log.generations.len(),
        description,
        n_generations,
        quantiles,
        observer,
    ))
}

/// Continues the [`run_with`] run recorded in `log` through `tolerances`, the
/// run's whole schedule: generations the log completed are not run again, and
/// a generation that was under way starts over. Returns the generations run
/// now, which is none if the log already completed the schedule.
///
/// The result is the one the uninterrupted run would have given. Give
/// `observer` a [`JsonlObserver::append`] on the same log to keep recording
/// to it. A longer schedule than the run started with extends it.
///
/// Fails if the log is not a run of `model`, or completed a generation at a
/// different tolerance than `tolerances` gives it.
#[cfg(feature = "serde")]
pub fn resume_with<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    tolerances: &[f64],
    log: &RunLog,
    description: &str,
    observer: &impl RunObserver<M>,
) -> std::io::Result<Vec<Generation<M>>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    resume_schedule(
        model,
        distance,
        tolerances,
        None,
        log,
        description,
        observer,
    )
}

/// Continues the [`run_quantiles_with`] run recorded in `log`, as
/// [`resume_with`] does for a run with fixed tolerances. The tolerances come
/// from the log's generation 0 if it completed one.
#[cfg(feature = "serde")]
pub fn resume_quantiles_with<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    quantiles: &[f64],
    log: &RunLog,
    description: &str,
    observer: &impl RunObserver<M>,
) -> std::io::Result<Vec<Generation<M>>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    let Some(prior) = log.generations.first() else {
        check_same_model(model, log)?;
        return Ok(run_quantiles_with(
            model,
            distance,
            quantiles,
            log.n_particles,
            description,
            observer,
        ));
    };
    let schedule = quantile_schedule(prior.particles.iter().map(|p| p.distance), quantiles);
    resume_schedule(
        model,
        distance,
        &schedule,
        Some(quantiles),
        log,
        description,
        observer,
    )
}

#[cfg(feature = "serde")]
fn resume_schedule<M>(
    model: &M,
    distance: &(impl Distance<M::Output> + ?Sized),
    schedule: &[f64],
    quantiles: Option<&[f64]>,
    log: &RunLog,
    description: &str,
    observer: &impl RunObserver<M>,
) -> std::io::Result<Vec<Generation<M>>>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
{
    for (logged, &tolerance) in log.generations.iter().zip(schedule) {
        if logged.stats.tolerance != tolerance {
            return Err(invalid(format!(
                "generation {} ran at tolerance {} but the schedule gives it {tolerance}",
                logged.stats.generation, logged.stats.tolerance
            )));
        }
    }
    if log.generations.len() >= schedule.len() {
        check_same_model(model, log)?;
        return Ok(Vec::new());
    }
    let mut sampler = sampler_after(
        model,
        distance,
        log,
        description,
        schedule.len(),
        quantiles,
        observer,
    )?;
    run_schedule(&mut sampler, schedule);
    Ok(sampler.finish())
}

/// The `q * n`th smallest of an ascending-sorted sample (truncating, so the
/// threshold is never looser than asked). Falls back to the single best
/// distance when `q * n < 1`, with a warning: the generation is then close to
/// degenerate and needs more particles.
fn quantile_threshold(sorted_distances: &[f64], quantile: f64) -> f64 {
    let n = sorted_distances.len();
    assert!(
        n > 0,
        "cannot calibrate a quantile threshold from an empty sample"
    );
    assert!(
        quantile > 0.0 && quantile <= 1.0,
        "quantile must be in (0, 1], got {quantile}"
    );
    let selected = quantile * n as f64;
    if selected < 1.0 {
        println!(
            "WARNING: quantile {quantile} of {n} particles selects fewer than one draw; \
             using the single best distance. Use at least {} particles.",
            (1.0 / quantile).ceil() as usize
        );
    }
    sorted_distances[(selected as usize).clamp(1, n) - 1]
}

#[cfg(test)]
mod test {
    use super::quantile_threshold;

    fn sample() -> Vec<f64> {
        (1..=10).map(f64::from).collect()
    }

    #[test]
    fn quantile_picks_the_q_times_n_th_smallest() {
        let s = sample();
        assert_eq!(quantile_threshold(&s, 0.5), 5.0);
        assert_eq!(quantile_threshold(&s, 0.2), 2.0);
        assert_eq!(quantile_threshold(&s, 0.1), 1.0);
        assert_eq!(quantile_threshold(&s, 0.25), 2.0);
    }

    #[test]
    fn quantile_finer_than_one_particle_keeps_the_best_draw() {
        let s = sample();
        assert_eq!(quantile_threshold(&s, 0.01), 1.0);
        assert_eq!(quantile_threshold(&s, 1e-9), 1.0);
    }

    #[test]
    fn quantile_of_one_keeps_the_whole_sample() {
        assert_eq!(quantile_threshold(&sample(), 1.0), 10.0);
        assert_eq!(quantile_threshold(&[42.0], 0.5), 42.0);
    }
}
