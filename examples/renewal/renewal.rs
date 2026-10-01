//! A stochastic renewal-equation epidemic simulator.
//!
//! Deliberately independent of the calibration next door: no priors, no
//! distances, no particles, no dependency on `abcsmc` at all. It takes
//! [`Parameters`] and an `Rng` and returns a [`RenewalOutput`].

use rand::Rng;
use rand::distr::Distribution;
use rand_distr::{Binomial, Poisson};

// `Infinite` is only exercised by the model's own tests, not the fit.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Population {
    Finite(u64),
    Infinite,
}

/// Scales transmission by `multiplier` from step `start` onward.
#[derive(Clone, Copy)]
pub struct TransmissionChange {
    pub start: usize,
    pub multiplier: f64,
}

pub struct Parameters {
    pub population: Population,
    pub r0: f64,
    pub generation_interval_pmf: Vec<f64>,
    pub symptom_onset_pmf: Vec<f64>,
    pub initial_infections: Vec<u64>,
    pub sim_length: usize,
    pub transmission_change: Option<TransmissionChange>,
}

#[derive(Default)]
pub struct RenewalOutput {
    pub infection_incidence: Vec<u64>,
    pub symptomatic_incidence: Vec<u64>,
}

impl RenewalOutput {
    pub fn new(len: usize) -> RenewalOutput {
        RenewalOutput {
            infection_incidence: vec![0; len],
            symptomatic_incidence: vec![0; len],
        }
    }
}

pub struct RenewalModel {}

impl RenewalModel {
    pub fn simulate(parameters: &Parameters, rng: &mut impl Rng) -> RenewalOutput {
        let mut output = RenewalOutput::new(parameters.sim_length);
        let mut rt = vec![parameters.r0; parameters.sim_length];
        let mut cum_infected = 0;
        for step in 0..parameters.sim_length {
            // Determine infections
            let infections: u64;
            if step < parameters.initial_infections.len() {
                // Use initial infections at first
                infections = parameters.initial_infections[step];
            } else {
                // Use renewal equation calculation
                let mut current_infectious = 0.0;
                for lag in 0..usize::min(step, parameters.generation_interval_pmf.len()) {
                    current_infectious += output.infection_incidence[step - lag - 1] as f64
                        * parameters.generation_interval_pmf[lag];
                }
                let scale = match parameters.transmission_change {
                    Some(change) if step >= change.start => change.multiplier,
                    _ => 1.0,
                };
                let transmission_rate = rt[step] * scale * current_infectious;

                match parameters.population {
                    Population::Finite(population) => {
                        let susceptible = population - cum_infected;
                        infections = if susceptible > 0 {
                            Binomial::new(
                                susceptible,
                                f64::min(transmission_rate / susceptible as f64, 1.0),
                            )
                            .unwrap()
                            .sample(rng)
                        } else {
                            0
                        };
                    }
                    Population::Infinite => {
                        infections = if transmission_rate > 0. {
                            // Poisson requires non-zero rate
                            Poisson::new(transmission_rate).unwrap().sample(rng) as u64
                        } else {
                            0
                        }
                    }
                }
            }
            output.infection_incidence[step] = infections;
            cum_infected += infections;
            // Update rt if needed
            if let Population::Finite(population) = parameters.population
                && step < parameters.sim_length - 1
            {
                rt[step + 1] =
                    parameters.r0 * (population - cum_infected) as f64 / population as f64
            }

            // Distribute symptom onset times
            if infections > 0 {
                let mut residual_mass = 1.;
                let mut cum_onsets = 0;
                for (mass, output_onsets) in parameters
                    .symptom_onset_pmf
                    .iter()
                    .zip(output.symptomatic_incidence.iter_mut().skip(step + 1))
                {
                    let onsets = Binomial::new(infections - cum_onsets, *mass / residual_mass)
                        .unwrap()
                        .sample(rng);
                    *output_onsets += onsets;
                    cum_onsets += onsets;
                    residual_mass -= *mass;
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod test {
    use super::{Parameters, Population, RenewalModel, TransmissionChange};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_transmission_change_keeps_earlier_history() {
        let start = 30;
        let simulate = |transmission_change| {
            let parameters = Parameters {
                population: Population::Finite(10_000),
                r0: 2.0,
                generation_interval_pmf: vec![0., 0., 0.25, 0.5, 0.25],
                symptom_onset_pmf: vec![0., 0.5, 0.5],
                initial_infections: vec![5],
                sim_length: 100,
                transmission_change,
            };
            RenewalModel::simulate(&parameters, &mut StdRng::seed_from_u64(42))
        };
        let baseline = simulate(None);
        let reduced = simulate(Some(TransmissionChange {
            start,
            multiplier: 0.5,
        }));
        assert_eq!(
            baseline.infection_incidence[..start],
            reduced.infection_incidence[..start]
        );
        let total = |incidence: &[u64]| incidence.iter().sum::<u64>();
        assert!(total(&reduced.infection_incidence) < total(&baseline.infection_incidence));
    }

    #[test]
    fn test_final_size() {
        let population = 100_000;
        let parameters = Parameters {
            population: Population::Finite(population),
            r0: 2.0,
            generation_interval_pmf: vec![0., 0., 0.25, 0.5, 0.25],
            symptom_onset_pmf: vec![1.],
            initial_infections: vec![1],
            sim_length: 200,
            transmission_change: None,
        };
        let mut rng = StdRng::seed_from_u64(8675308);
        let output = RenewalModel::simulate(&parameters, &mut rng);
        let cum_infected: u64 = output.infection_incidence.iter().sum();
        let fraction_infected = cum_infected as f64 / population as f64;
        // Final size for r0: 2. is ~0.796811
        assert!(f64::abs(fraction_infected - 0.796811) < 0.1);
    }

    #[test]
    fn test_generation_interval() {
        let n_samples = 10000;
        let initial_infections = 100;
        let generation_interval_pmf = vec![0., 0., 0.25, 0.5, 0.25];

        let mut cumulative_output = vec![0u64; generation_interval_pmf.len() + 1];
        let mut total = 0;
        for seed in 0..n_samples {
            let parameters = Parameters {
                population: Population::Infinite,
                r0: 1.,
                generation_interval_pmf: generation_interval_pmf.clone(),
                symptom_onset_pmf: vec![1.],
                initial_infections: vec![initial_infections],
                sim_length: generation_interval_pmf.len() + 1,
                transmission_change: None,
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let output = RenewalModel::simulate(&parameters, &mut rng);
            for (i, entry) in cumulative_output.iter_mut().enumerate() {
                let incidence = output.infection_incidence[i];
                *entry += incidence;
                if i > 0 {
                    // Accumulate secondary infections
                    total += incidence;
                }
            }
        }
        for (step, mass) in generation_interval_pmf.iter().enumerate() {
            let fraction = cumulative_output[step + 1] as f64 / total as f64;
            assert!(f64::abs(fraction - mass) < 1e-3);
        }
    }

    #[test]
    fn test_symptom_onset() {
        let initial_infections = 1000000;
        let symptom_onset_pmf = vec![0., 0., 0.25, 0.5, 0.25];
        let parameters = Parameters {
            population: Population::Infinite,
            r0: 0.,
            generation_interval_pmf: vec![1.],
            symptom_onset_pmf: symptom_onset_pmf.clone(),
            initial_infections: vec![initial_infections],
            sim_length: symptom_onset_pmf.len() + 1,
            transmission_change: None,
        };
        let mut rng = StdRng::seed_from_u64(8675309);
        let output = RenewalModel::simulate(&parameters, &mut rng);
        let total: u64 = output.symptomatic_incidence.iter().skip(1).sum();
        for (step, mass) in symptom_onset_pmf.iter().enumerate() {
            let fraction = output.symptomatic_incidence[step + 1] as f64 / total as f64;
            assert!(f64::abs(fraction - mass) < 1e-3);
        }
    }
}
