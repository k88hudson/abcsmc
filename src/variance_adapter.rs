//! Variance adapters. Each adapter rescales one kernel type: a standalone kernel of that type, or
//! the first such child of an `IndependentKernels`.

use std::any::Any;

use crate::Params;
use crate::kernel::{
    DiscreteUniformKernel, IndependentKernels, MultivariateNormalKernel, NormalKernel,
    PerturbationKernel, UniformKernel,
};

pub trait VarianceAdapter {
    fn adapt(&self, population: &[Params], kernel: &mut dyn PerturbationKernel);
}

/// Matches [`default_kernel`](crate::default_kernel): rescales the multivariate
/// normal and every discrete walk.
pub fn default_adapter() -> Box<dyn VarianceAdapter> {
    Box::new(IndependentAdapters::new(vec![
        Box::new(AdaptMultivariateNormalVariance),
        Box::new(AdaptDiscreteUniformVariance),
    ]))
}

pub struct IndependentAdapters {
    pub adapters: Vec<Box<dyn VarianceAdapter>>,
}

impl IndependentAdapters {
    pub fn new(adapters: Vec<Box<dyn VarianceAdapter>>) -> Self {
        IndependentAdapters { adapters }
    }
}

impl VarianceAdapter for IndependentAdapters {
    fn adapt(&self, population: &[Params], kernel: &mut dyn PerturbationKernel) {
        for adapter in &self.adapters {
            adapter.adapt(population, kernel);
        }
    }
}

pub struct AdaptIdentityVariance;

impl VarianceAdapter for AdaptIdentityVariance {
    fn adapt(&self, _population: &[Params], _kernel: &mut dyn PerturbationKernel) {}
}

/// `std_dev = sqrt(2 * var)`, population variance.
pub struct AdaptNormalVariance;

impl VarianceAdapter for AdaptNormalVariance {
    fn adapt(&self, population: &[Params], kernel: &mut dyn PerturbationKernel) {
        if let Some(kernel) = find_kernels::<NormalKernel>(kernel).pop() {
            kernel.std_dev = (2.0 * variance(&column(population, kernel.param))).sqrt();
        }
    }
}

/// `width = 2 * sqrt(2 * var)`, population variance.
pub struct AdaptUniformVariance;

impl VarianceAdapter for AdaptUniformVariance {
    fn adapt(&self, population: &[Params], kernel: &mut dyn PerturbationKernel) {
        if let Some(kernel) = find_kernels::<UniformKernel>(kernel).pop() {
            kernel.width = 2.0 * (2.0 * variance(&column(population, kernel.param))).sqrt();
        }
    }
}

/// `cov = 2 * sample covariance`.
pub struct AdaptMultivariateNormalVariance;

impl VarianceAdapter for AdaptMultivariateNormalVariance {
    fn adapt(&self, population: &[Params], kernel: &mut dyn PerturbationKernel) {
        if let Some(kernel) = find_kernels::<MultivariateNormalKernel>(kernel).pop() {
            let columns: Vec<Vec<f64>> = kernel
                .params
                .iter()
                .map(|&param| column(population, param))
                .collect();
            let mut cov = sample_covariance(&columns);
            cov.iter_mut().flatten().for_each(|c| *c *= 2.0);
            kernel.set_covariance(cov);
        }
    }
}

/// `half_width = round(sqrt(2 * var))`, at least 1, for every discrete walk.
pub struct AdaptDiscreteUniformVariance;

impl VarianceAdapter for AdaptDiscreteUniformVariance {
    fn adapt(&self, population: &[Params], kernel: &mut dyn PerturbationKernel) {
        for kernel in find_kernels::<DiscreteUniformKernel>(kernel) {
            let var = variance(&column(population, kernel.param));
            kernel.half_width = (2.0 * var).sqrt().round().max(1.0) as i64;
        }
    }
}

/// The kernel itself if it is a `K`, else the `K` children of an
/// `IndependentKernels`, in **reverse** order so `pop()` yields the first.
fn find_kernels<K: PerturbationKernel>(kernel: &mut dyn PerturbationKernel) -> Vec<&mut K> {
    if (kernel as &dyn Any).is::<K>() {
        return (kernel as &mut dyn Any)
            .downcast_mut::<K>()
            .into_iter()
            .collect();
    }
    match (kernel as &mut dyn Any).downcast_mut::<IndependentKernels>() {
        Some(composite) => composite
            .kernels
            .iter_mut()
            .rev()
            .filter_map(|k| (k.as_mut() as &mut dyn Any).downcast_mut::<K>())
            .collect(),
        None => Vec::new(),
    }
}

fn column(population: &[Params], param: usize) -> Vec<f64> {
    population
        .iter()
        .map(|p| p.values()[param].as_f64())
        .collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

fn sample_covariance(columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = columns.first().map_or(0, Vec::len);
    assert!(n >= 2, "covariance needs at least two particles, got {n}");
    let means: Vec<f64> = columns.iter().map(|c| mean(c)).collect();
    columns
        .iter()
        .enumerate()
        .map(|(i, a)| {
            columns
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let s: f64 = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - means[i]) * (y - means[j]))
                        .sum();
                    s / (n - 1) as f64
                })
                .collect()
        })
        .collect()
}
