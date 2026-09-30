use rand::{SeedableRng, rngs::StdRng};

pub mod diagnostics;
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
mod observer;
pub use observer::*;
mod platform;

/// One completed generation: its accepted particles and run statistics.
pub struct Generation<M: Model> {
    pub particles: Vec<Particle<M>>,
    pub stats: GenerationStats,
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

/// Run one generation: prior rejection when there is no previous population,
/// otherwise one SMC step from it.
fn run_generation<M>(
    model: &M,
    generation: usize,
    tolerance: f64,
    n_particles: usize,
    previous: Option<(&[Particle<M>], &dyn PerturbationKernel)>,
    rng: &mut StdRng,
    observer: &impl RunObserver<M>,
) -> Option<Generation<M>>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    observer.generation_started(generation, tolerance);
    let started = platform::Instant::now();
    let on_accept = |params: &Params, distance: f64, attempts: u64| {
        observer.particle_accepted(generation, params, distance, attempts)
    };
    let output = match previous {
        None => initialize(model, tolerance, n_particles, rng, on_accept),
        Some((previous, kernel)) => step(
            model,
            tolerance,
            n_particles,
            previous,
            kernel,
            rng,
            on_accept,
        ),
    };
    if !output.complete {
        observer.generation_abandoned(generation);
        return None;
    }
    let stats = GenerationStats {
        generation,
        tolerance,
        accepted: output.particles.len(),
        attempts: output.n_attempts,
        acceptance_ratio: output.particles.len() as f64 / output.n_attempts as f64,
        ess: get_ess(&output),
        perplexity: get_perplexity(&output),
        duration_seconds: started.elapsed().as_secs_f64(),
    };
    let completed = Generation {
        particles: output.particles,
        stats,
    };
    observer.generation_completed(model, &completed);
    Some(completed)
}

/// Continue an ABC-SMC run through `tolerances`, one generation per tolerance,
/// starting from whatever generations have already been completed. The kernel
/// is adapted to each completed population. Stops early if a generation cannot
/// be filled.
fn run_from<M>(
    model: &M,
    tolerances: &[f64],
    n_particles: usize,
    mut generations: Vec<Generation<M>>,
    rng: &mut StdRng,
    observer: &impl RunObserver<M>,
) -> Vec<Generation<M>>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    let mut kernel = model.perturbation_kernel();
    let adapter = model.variance_adapter();
    if let Some(last) = generations.last() {
        adapter.adapt(&params_of(last), kernel.as_mut());
    }
    for &tolerance in tolerances {
        let generation = generations.len();
        let previous = generations
            .last()
            .map(|g| (g.particles.as_slice(), kernel.as_ref()));
        match run_generation(
            model,
            generation,
            tolerance,
            n_particles,
            previous,
            rng,
            observer,
        ) {
            Some(completed) => {
                adapter.adapt(&params_of(&completed), kernel.as_mut());
                generations.push(completed);
            }
            None => break,
        }
    }
    observer.run_finished(model, &generations);
    generations
}

fn params_of<M: Model>(generation: &Generation<M>) -> Vec<Params> {
    generation
        .particles
        .iter()
        .map(|p| p.params.clone())
        .collect()
}

/// Run ABC-SMC with one generation per tolerance: generation 0 is rejection sampling from the prior at `tolerances[0]`, and
/// each later generation resamples, perturbs, and reweights the previous one.
/// Progress is printed to stdout; see [`run_with`] to observe the run.
pub fn run<M>(model: &M, tolerances: &[f64], n_particles: usize) -> Vec<Generation<M>>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    run_with(model, tolerances, n_particles, &StdoutObserver::new())
}

/// [`run`] reporting to `observer` instead of stdout. Pair observers with a
/// tuple: `&(StdoutObserver::new(), JsonlObserver::create(dir, id)?)`.
pub fn run_with<M>(
    model: &M,
    tolerances: &[f64],
    n_particles: usize,
    observer: &impl RunObserver<M>,
) -> Vec<Generation<M>>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(model.rng_seed());
    let meta = RunMeta::new(model, n_particles, tolerances.len(), None);
    observer.run_started(model, &meta);
    run_from(
        model,
        tolerances,
        n_particles,
        Vec::new(),
        &mut rng,
        observer,
    )
}

/// Like [`run`], but with tolerances given as quantiles of the prior's distance
/// distribution. Generation 0 is an unconstrained prior sample (infinite
/// tolerance); generation `g >= 1` accepts distances up to `quantiles[g - 1]`
/// of generation 0's distances. Returns `quantiles.len() + 1` generations.
pub fn run_quantiles<M>(model: &M, quantiles: &[f64], n_particles: usize) -> Vec<Generation<M>>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    run_quantiles_with(model, quantiles, n_particles, &StdoutObserver::new())
}

/// [`run_quantiles`] reporting to `observer` instead of stdout.
pub fn run_quantiles_with<M>(
    model: &M,
    quantiles: &[f64],
    n_particles: usize,
    observer: &impl RunObserver<M>,
) -> Vec<Generation<M>>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(model.rng_seed());
    let meta = RunMeta::new(model, n_particles, quantiles.len() + 1, Some(quantiles));
    observer.run_started(model, &meta);
    let Some(prior) = run_generation(
        model,
        0,
        f64::INFINITY,
        n_particles,
        None,
        &mut rng,
        observer,
    ) else {
        observer.run_finished(model, &[]);
        return Vec::new();
    };
    let mut distances: Vec<f64> = prior.particles.iter().map(|p| p.distance).collect();
    distances.sort_by(|a, b| a.total_cmp(b));
    let tolerances: Vec<f64> = quantiles
        .iter()
        .map(|&q| quantile_threshold(&distances, q))
        .collect();
    run_from(
        model,
        &tolerances,
        n_particles,
        vec![prior],
        &mut rng,
        observer,
    )
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
