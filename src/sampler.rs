//! The generation-by-generation driver behind [`run`](crate::run) and
//! [`run_quantiles`](crate::run_quantiles). Use it directly to decide each
//! tolerance after seeing the generation before it, or to continue a run that
//! stopped.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::platform::Instant;
use crate::{
    Adaptive, CalibrationModel, Distance, Generation, GenerationStats, Params, PerturbationKernel,
    RunMeta, RunObserver, VarianceAdapter, get_ess, get_perplexity, initialize, step_from,
};

/// The part of a completed generation that the next one proposes from.
#[derive(Clone, Debug)]
pub struct Population {
    pub params: Vec<Params>,
    /// Normalized weights, one per entry of `params`.
    pub weights: Vec<f64>,
}

impl<M: CalibrationModel> From<&Generation<M>> for Population {
    fn from(generation: &Generation<M>) -> Self {
        Population {
            params: generation
                .particles
                .iter()
                .map(|p| p.params.clone())
                .collect(),
            weights: generation.particles.iter().map(|p| p.weight).collect(),
        }
    }
}

/// An ABC-SMC run advanced one generation at a time.
///
/// ```
/// use abcsmc::{CalibrationModel, Params, Priors, RealPrior, Sampler, Silent};
///
/// struct Identity;
///
/// impl CalibrationModel for Identity {
///     type Draw = Params;
///     type Output = f64;
///     fn priors(&self) -> Priors {
///         Priors::new().push(RealPrior::uniform(0.0, 10.0).unwrap())
///     }
///     fn simulate(&self, draw: &Params, _seed: u64) -> f64 {
///         draw.real(0)
///     }
/// }
///
/// let distance = |output: &f64| (output - 5.0).abs();
/// let mut sampler = Sampler::start(&Identity, &distance, 50, "", 2, &Silent);
/// let prior = sampler.next_generation(f64::INFINITY).unwrap();
/// // Choose the next tolerance from what the last generation reached.
/// let widest = prior.particles.iter().map(|p| p.distance).fold(0.0, f64::max);
/// sampler.next_generation(widest / 2.0).unwrap();
/// assert_eq!(sampler.finish().len(), 2);
/// ```
///
/// Generation `g` of a run draws the same random numbers however the run got
/// there, so a run resumed after generation `g - 1` continues exactly as the
/// uninterrupted run would have. That holds as long as the model, the
/// distance, and the particle count are unchanged, and the variance adapter
/// depends only on the population it is given.
pub struct Sampler<'a, M: CalibrationModel, D: ?Sized, O> {
    model: &'a M,
    distance: &'a D,
    observer: &'a O,
    n_particles: usize,
    rng: StdRng,
    kernel: Box<dyn PerturbationKernel>,
    adapter: Box<dyn VarianceAdapter>,
    /// The last completed generation, which the next one proposes from.
    population: Option<Population>,
    /// Whether `kernel` has been adapted to `population`.
    adapted: bool,
    /// Generations completed in this run, including any before a resume.
    completed: usize,
    /// Generations completed by this sampler.
    generations: Vec<Generation<M>>,
}

impl<'a, M, D, O> Sampler<'a, M, D, O>
where
    M: CalibrationModel + Sync,
    M::Output: Send + Sync,
    D: Distance<M::Output> + ?Sized,
    O: RunObserver<M>,
{
    /// Begins a run. `description` says what the run is (`""` for none) and
    /// `n_generations` how many generations it plans, both for observers; the
    /// sampler itself runs as many generations as it is asked for.
    pub fn start(
        model: &'a M,
        distance: &'a D,
        n_particles: usize,
        description: &str,
        n_generations: usize,
        observer: &'a O,
    ) -> Self {
        Self::start_with_schedule(
            model,
            distance,
            n_particles,
            description,
            n_generations,
            None,
            None,
            observer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_with_schedule(
        model: &'a M,
        distance: &'a D,
        n_particles: usize,
        description: &str,
        n_generations: usize,
        quantiles: Option<&[f64]>,
        adaptive: Option<Adaptive>,
        observer: &'a O,
    ) -> Self {
        let meta = RunMeta::new(
            model,
            distance,
            description,
            n_particles,
            n_generations,
            quantiles,
            adaptive,
        );
        observer.run_started(model, &meta);
        Self::new(model, distance, n_particles, None, 0, observer)
    }

    /// Continues a run whose first `completed` generations are done, the last
    /// of them being `population`. `n_generations` is the total the run now
    /// plans, including those.
    ///
    /// Panics if `completed` is zero or `population` is empty.
    pub fn resume(
        model: &'a M,
        distance: &'a D,
        population: Population,
        completed: usize,
        description: &str,
        n_generations: usize,
        observer: &'a O,
    ) -> Self {
        Self::resume_with_schedule(
            model,
            distance,
            population,
            completed,
            description,
            n_generations,
            None,
            None,
            observer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resume_with_schedule(
        model: &'a M,
        distance: &'a D,
        population: Population,
        completed: usize,
        description: &str,
        n_generations: usize,
        quantiles: Option<&[f64]>,
        adaptive: Option<Adaptive>,
        observer: &'a O,
    ) -> Self {
        assert!(completed > 0, "a run with no completed generation starts");
        assert_eq!(
            population.params.len(),
            population.weights.len(),
            "a population has one weight per particle"
        );
        let n_particles = population.params.len();
        assert!(n_particles > 0, "cannot resume from an empty population");
        let meta = RunMeta::new(
            model,
            distance,
            description,
            n_particles,
            n_generations,
            quantiles,
            adaptive,
        );
        observer.run_resumed(model, &meta, completed);
        Self::new(
            model,
            distance,
            n_particles,
            Some(population),
            completed,
            observer,
        )
    }

    fn new(
        model: &'a M,
        distance: &'a D,
        n_particles: usize,
        population: Option<Population>,
        completed: usize,
        observer: &'a O,
    ) -> Self {
        // Every generation draws one seed per particle, so skipping those of
        // the completed generations puts the stream where they left it.
        let mut rng = StdRng::seed_from_u64(model.rng_seed());
        for _ in 0..completed * n_particles {
            rng.next_u64();
        }
        Sampler {
            model,
            distance,
            observer,
            n_particles,
            rng,
            kernel: model.perturbation_kernel(),
            adapter: model.variance_adapter(),
            population,
            adapted: false,
            completed,
            generations: Vec::new(),
        }
    }

    /// Generations completed in this run, including any before a resume.
    pub fn completed(&self) -> usize {
        self.completed
    }

    /// The generations this sampler has completed, oldest first. After a
    /// resume these do not include the generations completed before it.
    pub fn generations(&self) -> &[Generation<M>] {
        &self.generations
    }

    /// Runs the next generation, accepting simulations within `tolerance`.
    /// The first generation of a run samples the prior; each later one
    /// resamples, perturbs, and reweights the one before it.
    ///
    /// Returns `None` if the generation could not be filled within
    /// `max_attempts_per_proposal`, leaving the run where it was.
    pub fn next_generation(&mut self, tolerance: f64) -> Option<&Generation<M>> {
        let (model, observer) = (self.model, self.observer);
        let generation = self.completed;
        observer.generation_started(generation, tolerance);
        let started = Instant::now();
        let on_accept = |params: &Params, distance: f64, attempts: u64| {
            observer.particle_accepted(generation, params, distance, attempts)
        };
        let on_reject = |params: &Params, output: &M::Output, distance: f64| {
            observer.particle_rejected(model, generation, params, output, distance)
        };
        let output = match &self.population {
            None => initialize(
                model,
                self.distance,
                tolerance,
                self.n_particles,
                &mut self.rng,
                on_accept,
                on_reject,
            ),
            Some(population) => {
                // Adapting here, not when a generation completes, leaves the
                // last generation of a run unadapted to.
                if !self.adapted {
                    self.adapter.adapt(&population.params, self.kernel.as_mut());
                    self.adapted = true;
                }
                step_from(
                    model,
                    self.distance,
                    tolerance,
                    self.n_particles,
                    population,
                    self.kernel.as_ref(),
                    &mut self.rng,
                    on_accept,
                    on_reject,
                )
            }
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
        self.population = Some(Population::from(&completed));
        self.adapted = false;
        self.completed += 1;
        self.generations.push(completed);
        self.generations.last()
    }

    /// Ends the run and returns the generations this sampler completed.
    pub fn finish(self) -> Vec<Generation<M>> {
        self.observer.run_finished(self.model, &self.generations);
        self.generations
    }
}
