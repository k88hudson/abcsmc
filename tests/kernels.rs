//! Kernel transition probabilities and variance adapters against closed forms,
//! plus kernel selection through `Model`.

use abcsmc::{
    AdaptDiscreteUniformVariance, AdaptIdentityVariance, AdaptMultivariateNormalVariance,
    AdaptNormalVariance, AdaptUniformVariance, DiscreteUniformKernel, IndependentKernels, IntPrior,
    Model, MultivariateNormalKernel, NormalKernel, Params, PerturbationKernel, Priors, RealPrior,
    UniformKernel, Value, VarianceAdapter, run,
};
use rand::{SeedableRng, rngs::StdRng};
use std::f64::consts::PI;

fn reals(values: &[f64]) -> Params {
    Params::new(values.iter().map(|&x| Value::Real(x)).collect())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

#[test]
fn normal_kernel_density_matches_closed_form() {
    let kernel = NormalKernel::new(0, 2.0);
    let pdf = kernel.transition_probability(&reals(&[1.5]), &reals(&[0.5]));
    assert!(close(pdf, (-0.125f64).exp() / (2.0 * (2.0 * PI).sqrt())));
}

#[test]
fn uniform_kernel_density_is_flat_and_inclusive() {
    let kernel = UniformKernel::new(0, 4.0);
    let from = reals(&[1.0]);
    assert!(close(
        kernel.transition_probability(&reals(&[3.0]), &from),
        0.25
    ));
    assert!(close(
        kernel.transition_probability(&reals(&[-1.0]), &from),
        0.25
    ));
    assert_eq!(kernel.transition_probability(&reals(&[3.1]), &from), 0.0);
}

#[test]
fn discrete_uniform_kernel_mass_includes_zero_step() {
    let kernel = DiscreteUniformKernel::new(0, 2);
    let from = Params::new(vec![Value::Int(5)]);
    assert!(close(kernel.transition_probability(&from, &from), 0.2));
    assert!(close(
        kernel.transition_probability(&Params::new(vec![Value::Int(7)]), &from),
        0.2
    ));
    assert_eq!(
        kernel.transition_probability(&Params::new(vec![Value::Int(8)]), &from),
        0.0
    );
}

#[test]
fn multivariate_normal_density_matches_closed_form() {
    let cov = vec![vec![2.0, 0.5], vec![0.5, 1.0]];
    let kernel = MultivariateNormalKernel::with_covariance(vec![0, 1], cov);
    let pdf = kernel.transition_probability(&reals(&[1.0, 1.0]), &reals(&[0.0, 0.0]));
    // det = 1.75, inverse = [[1, -0.5], [-0.5, 2]] / 1.75, quadratic form = 2 / 1.75
    let expected = (-0.5 * 2.0 / 1.75f64).exp() / (2.0 * PI * 1.75f64.sqrt());
    assert!(close(pdf, expected), "{pdf} vs {expected}");
}

#[test]
fn multivariate_normal_samples_have_the_requested_covariance() {
    let cov = vec![vec![2.0, 0.5], vec![0.5, 1.0]];
    let kernel = MultivariateNormalKernel::with_covariance(vec![0, 1], cov);
    let mut rng = StdRng::seed_from_u64(1);
    let from = reals(&[0.0, 0.0]);
    let draws: Vec<Params> = (0..50_000)
        .map(|_| kernel.perturb(&from, &mut rng))
        .collect();
    let mut sum = [[0.0; 2]; 2];
    for d in &draws {
        for i in 0..2 {
            for j in 0..2 {
                sum[i][j] += d.real(i) * d.real(j);
            }
        }
    }
    let n = draws.len() as f64;
    assert!((sum[0][0] / n - 2.0).abs() < 0.05);
    assert!((sum[1][1] / n - 1.0).abs() < 0.05);
    assert!((sum[0][1] / n - 0.5).abs() < 0.05);
}

#[test]
fn independent_kernels_multiply_probabilities() {
    let kernel = IndependentKernels::new(vec![
        Box::new(NormalKernel::new(0, 1.0)),
        Box::new(UniformKernel::new(1, 2.0)),
    ]);
    let pdf = kernel.transition_probability(&reals(&[0.0, 0.5]), &reals(&[0.0, 0.0]));
    assert!(close(pdf, 0.5 / (2.0 * PI).sqrt()));
    let mut rng = StdRng::seed_from_u64(3);
    let perturbed = kernel.perturb(&reals(&[0.0, 0.0]), &mut rng);
    assert_ne!(perturbed.real(0), 0.0);
    assert_ne!(perturbed.real(1), 0.0);
}

fn population() -> Vec<Params> {
    vec![
        reals(&[1.0, 10.0]),
        reals(&[2.0, 20.0]),
        reals(&[4.0, 30.0]),
    ]
}

#[test]
fn normal_and_uniform_adapters_use_population_variance() {
    // values 1, 2, 4: population variance = 14/9
    let var: f64 = 14.0 / 9.0;
    let mut normal: Box<dyn PerturbationKernel> = Box::new(NormalKernel::new(0, 1.0));
    AdaptNormalVariance.adapt(&population(), normal.as_mut());
    let pdf = normal.transition_probability(&reals(&[0.0, 0.0]), &reals(&[0.0, 0.0]));
    assert!(close(pdf, 1.0 / ((2.0 * var).sqrt() * (2.0 * PI).sqrt())));

    let mut uniform: Box<dyn PerturbationKernel> = Box::new(UniformKernel::new(0, 1.0));
    AdaptUniformVariance.adapt(&population(), uniform.as_mut());
    let pdf = uniform.transition_probability(&reals(&[0.0, 0.0]), &reals(&[0.0, 0.0]));
    assert!(close(pdf, 1.0 / (2.0 * (2.0 * var).sqrt())));
}

#[test]
fn multivariate_adapter_uses_twice_the_sample_covariance() {
    // columns (1, 2, 4) and (10, 20, 30): sample variances 7/3 and 100, covariance 15
    let mut kernel: Box<dyn PerturbationKernel> =
        Box::new(MultivariateNormalKernel::new(vec![0, 1]));
    AdaptMultivariateNormalVariance.adapt(&population(), kernel.as_mut());
    let (a, b, c): (f64, f64, f64) = (2.0 * 7.0 / 3.0, 2.0 * 100.0, 2.0 * 15.0);
    let det = a * b - c * c;
    let pdf = kernel.transition_probability(&reals(&[0.0, 0.0]), &reals(&[0.0, 0.0]));
    assert!(close(pdf, 1.0 / (2.0 * PI * det.sqrt())), "{pdf}");
}

#[test]
fn adapters_reach_the_first_matching_child_only() {
    let mut kernel: Box<dyn PerturbationKernel> = Box::new(IndependentKernels::new(vec![
        Box::new(UniformKernel::new(1, 1.0)),
        Box::new(NormalKernel::new(0, 1.0)),
        Box::new(NormalKernel::new(1, 1.0)),
    ]));
    AdaptNormalVariance.adapt(&population(), kernel.as_mut());
    AdaptIdentityVariance.adapt(&population(), kernel.as_mut());
    let composite = kernel.as_ref() as &dyn std::any::Any;
    let composite = composite.downcast_ref::<IndependentKernels>().unwrap();
    let sd = |i: usize| {
        (composite.kernels[i].as_ref() as &dyn std::any::Any)
            .downcast_ref::<NormalKernel>()
            .unwrap()
            .std_dev
    };
    assert!(close(sd(1), (2.0 * 14.0 / 9.0f64).sqrt()));
    assert_eq!(sd(2), 1.0);
}

#[test]
fn discrete_adapter_rescales_every_discrete_walk() {
    let ints = |values: &[i64]| Params::new(values.iter().map(|&k| Value::Int(k)).collect());
    // values 0, 10, 20: population variance 200/3, sqrt(400/3) = 11.5 -> 12
    let population = vec![ints(&[0, 5]), ints(&[10, 5]), ints(&[20, 5])];
    let mut kernel: Box<dyn PerturbationKernel> = Box::new(IndependentKernels::new(vec![
        Box::new(DiscreteUniformKernel::new(0, 1)),
        Box::new(DiscreteUniformKernel::new(1, 3)),
    ]));
    AdaptDiscreteUniformVariance.adapt(&population, kernel.as_mut());
    let from = ints(&[0, 5]);
    assert!(close(
        kernel.transition_probability(&ints(&[12, 5]), &from),
        1.0 / 25.0 / 3.0
    ));
    assert_eq!(kernel.transition_probability(&ints(&[13, 5]), &from), 0.0);
    // Collapsed column keeps the minimum half-width of 1.
    assert!(close(
        kernel.transition_probability(&ints(&[0, 6]), &from),
        1.0 / 3.0 / 25.0
    ));
}

/// A wide integer prior targeting 500 must still converge under the default kernel.
struct WideDiscreteModel;

impl Model for WideDiscreteModel {
    type Draw = Params;
    type Output = i64;

    fn priors(&self) -> Priors {
        Priors::new().push(IntPrior::discrete_uniform(0, 1000).unwrap())
    }

    fn simulate(&self, params: &Params, _seed: u64) -> i64 {
        params.int(0)
    }

    fn distance(&self, output: &i64) -> f64 {
        (output - 500).abs() as f64
    }
}

#[test]
fn default_kernel_adapts_discrete_walk_width() {
    let generations = run(&WideDiscreteModel, &[400.0, 100.0, 20.0, 5.0], 200);
    assert_eq!(generations.len(), 4);
    let mean: f64 = generations[3]
        .particles
        .iter()
        .map(|p| p.weight * p.params.int(0) as f64)
        .sum();
    assert!((mean - 500.0).abs() < 3.0, "{mean}");
}

/// Estimate `x` under `Uniform(0, 10)` targeting 5, with a fixed uniform kernel.
struct FixedKernelModel;

impl Model for FixedKernelModel {
    type Draw = Params;
    type Output = f64;

    fn priors(&self) -> Priors {
        Priors::new().push(RealPrior::uniform(0.0, 10.0).unwrap())
    }

    fn simulate(&self, params: &Params, _seed: u64) -> f64 {
        params.real(0)
    }

    fn distance(&self, output: &f64) -> f64 {
        (output - 5.0).abs()
    }

    fn perturbation_kernel(&self) -> Box<dyn PerturbationKernel> {
        Box::new(UniformKernel::new(0, 1.0))
    }

    fn variance_adapter(&self) -> Box<dyn VarianceAdapter> {
        Box::new(AdaptIdentityVariance)
    }
}

#[test]
fn model_selected_kernel_drives_the_run() {
    let generations = run(&FixedKernelModel, &[2.5, 1.0, 0.25], 200);
    assert_eq!(generations.len(), 3);
    let mean: f64 = generations[2]
        .particles
        .iter()
        .map(|p| p.weight * p.params.real(0))
        .sum();
    assert!((mean - 5.0).abs() < 0.3, "{mean}");
    // Every accepted proposal lies within the fixed half-width of some ancestor.
    for (prev, next) in generations.iter().zip(&generations[1..]) {
        for p in &next.particles {
            assert!(
                prev.particles
                    .iter()
                    .any(|q| (q.params.real(0) - p.params.real(0)).abs() <= 0.5)
            );
        }
    }
}
