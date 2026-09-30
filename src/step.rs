use crate::{Draw, Model, Params, Particle, PerturbationKernel, Priors};
use core::f64;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, weighted::WeightedIndex};
use rayon::prelude::*;

/// Skipped for unnamed draws (`type Draw = Params`), which have nothing to check against.
fn check_arity<M: Model>(priors: &Priors) {
    let names = <M::Draw as Draw>::NAMES;
    assert!(
        names.is_empty() || names.len() == priors.len(),
        "model declares {} parameters {names:?} but priors() supplied {}",
        names.len(),
        priors.len(),
    );
}

pub struct StepOutput<M: Model> {
    pub particles: Vec<Particle<M>>,
    /// Simulations run, accepted or not. Out-of-support proposals never reach
    /// the model and are not counted.
    pub n_attempts: u64,
    /// `false` if some slot exhausted `max_attempts_per_proposal`; `particles`
    /// is then partial and unnormalized.
    pub complete: bool,
}

/// Rejection sampling from the prior at `error_threshold`; equal weights.
///
/// Each particle runs on its own `StdRng` seeded from `rng`, in parallel, so
/// results do not depend on the thread count.
pub fn initialize<M>(
    model: &M,
    error_threshold: f64,
    n_particles: usize,
    rng: &mut impl Rng,
    on_accept: impl Fn(&Params, f64, u64) + Sync,
) -> StepOutput<M>
where
    M: Model + Sync,
    M::Output: Send,
{
    let priors = model.priors();
    check_arity::<M>(&priors);
    let max_attempts = model.max_attempts_per_proposal();
    let seeds: Vec<u64> = (0..n_particles).map(|_| rng.next_u64()).collect();

    let results: Vec<(Option<Particle<M>>, u64)> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            for n_attempts in 1..=max_attempts {
                let params = priors.sample(&mut rng);
                let sim_seed = rng.next_u64();
                let output = model.simulate(&M::Draw::from_values(params.values()), sim_seed);
                let distance = model.distance(&output);
                if distance <= error_threshold {
                    on_accept(&params, distance, n_attempts);
                    return (
                        Some(Particle {
                            params,
                            output,
                            distance,
                            weight: 1.0 / n_particles as f64,
                            seed: sim_seed,
                        }),
                        n_attempts,
                    );
                }
            }
            (None, max_attempts)
        })
        .collect();

    let n_attempts = results.iter().map(|(_, attempts)| attempts).sum();
    let complete = results.iter().all(|(particle, _)| particle.is_some());
    let particles = results
        .into_iter()
        .filter_map(|(particle, _)| particle)
        .collect();
    StepOutput {
        particles,
        n_attempts,
        complete,
    }
}

/// One SMC generation: resample by weight, perturb with `kernel`, accept within
/// `error_threshold`, and weight by `prior(θ*) / Σ wⱼ K(θⱼ → θ*)`.
///
/// Out-of-support proposals resample a new ancestor before perturbing again;
/// re-perturbing the same ancestor would bias proposals near the support
/// boundary.
pub fn step<M>(
    model: &M,
    error_threshold: f64,
    n_particles: usize,
    previous_particles: &[Particle<M>],
    kernel: &dyn PerturbationKernel,
    rng: &mut impl Rng,
    on_accept: impl Fn(&Params, f64, u64) + Sync,
) -> StepOutput<M>
where
    M: Model + Sync,
    M::Output: Send + Sync,
{
    let priors = model.priors();
    check_arity::<M>(&priors);
    let max_attempts = model.max_attempts_per_proposal();
    let weights: Vec<f64> = previous_particles.iter().map(|x| x.weight).collect();
    let particle_sampler = WeightedIndex::new(weights).unwrap();

    let seeds: Vec<u64> = (0..n_particles).map(|_| rng.next_u64()).collect();

    let results: Vec<(Option<Particle<M>>, u64)> = seeds
        .into_par_iter()
        .map(|seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            for n_attempts in 1..=max_attempts {
                let Some((proposed, proposed_prior_density)) = (0..max_attempts).find_map(|_| {
                    let base = &previous_particles[particle_sampler.sample(&mut rng)].params;
                    let proposed = kernel.perturb(base, &mut rng);
                    let density = priors.density(&proposed);
                    (density > 0.0).then_some((proposed, density))
                }) else {
                    return (None, n_attempts - 1);
                };

                let sim_seed = rng.next_u64();
                let output = model.simulate(&M::Draw::from_values(proposed.values()), sim_seed);
                let distance = model.distance(&output);
                if distance <= error_threshold {
                    let proposal_weight: f64 = previous_particles
                        .iter()
                        .map(|particle| {
                            particle.weight
                                * kernel.transition_probability(&proposed, &particle.params)
                        })
                        .sum();
                    let weight = if proposal_weight > 0.0 {
                        proposed_prior_density / proposal_weight
                    } else {
                        0.0
                    };

                    on_accept(&proposed, distance, n_attempts);
                    return (
                        Some(Particle {
                            params: proposed,
                            output,
                            distance,
                            weight,
                            seed: sim_seed,
                        }),
                        n_attempts,
                    );
                }
            }
            (None, max_attempts)
        })
        .collect();

    let n_attempts = results.iter().map(|(_, attempts)| attempts).sum();
    let complete = results.iter().all(|(particle, _)| particle.is_some());
    let mut particles: Vec<Particle<M>> = results
        .into_iter()
        .filter_map(|(particle, _)| particle)
        .collect();

    if complete {
        let total_weight: f64 = particles.iter().map(|x| x.weight).sum();
        particles.iter_mut().for_each(|x| x.weight /= total_weight);
    }

    StepOutput {
        particles,
        n_attempts,
        complete,
    }
}

pub fn get_ess<M: Model>(step_output: &StepOutput<M>) -> f64 {
    1.0 / step_output
        .particles
        .iter()
        .map(|x| x.weight * x.weight)
        .sum::<f64>()
}

pub fn get_perplexity<M: Model>(step_output: &StepOutput<M>) -> f64 {
    f64::exp(
        step_output
            .particles
            .iter()
            .map(|x| entropy(x.weight))
            .sum(),
    )
}

fn entropy(p: f64) -> f64 {
    if p == 0. { 0. } else { -p * f64::ln(p) }
}
