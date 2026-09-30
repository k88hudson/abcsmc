//! Perturbation kernels. Parameters are identified by prior-declaration index.

use std::any::Any;

use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

use crate::{Params, Priors, Value};

pub trait PerturbationKernel: Any + Send + Sync {
    fn perturb(&self, from: &Params, rng: &mut StdRng) -> Params;
    fn transition_probability(&self, to: &Params, from: &Params) -> f64;
}

/// One multivariate normal over the real parameters, plus a fixed ±1 walk for
/// each integer parameter.
pub fn default_kernel(priors: &Priors) -> Box<dyn PerturbationKernel> {
    let (reals, ints): (Vec<usize>, Vec<usize>) =
        (0..priors.len()).partition(|&i| priors.is_real(i));
    let mut kernels: Vec<Box<dyn PerturbationKernel>> = Vec::new();
    if !reals.is_empty() {
        kernels.push(Box::new(MultivariateNormalKernel::new(reals)));
    }
    for param in ints {
        kernels.push(Box::new(DiscreteUniformKernel::new(param, 1)));
    }
    Box::new(IndependentKernels::new(kernels))
}

pub struct NormalKernel {
    pub param: usize,
    pub std_dev: f64,
}

impl NormalKernel {
    pub fn new(param: usize, std_dev: f64) -> Self {
        assert!(
            std_dev > 0.0,
            "Standard deviation must be positive for NormalKernel."
        );
        NormalKernel { param, std_dev }
    }
}

impl PerturbationKernel for NormalKernel {
    fn perturb(&self, from: &Params, rng: &mut StdRng) -> Params {
        let z: f64 = StandardNormal.sample(rng);
        let mut to = from.clone();
        to.set(
            self.param,
            Value::Real(from.real(self.param) + self.std_dev * z),
        );
        to
    }

    fn transition_probability(&self, to: &Params, from: &Params) -> f64 {
        normal_pdf(to.real(self.param) - from.real(self.param), self.std_dev)
    }
}

pub struct UniformKernel {
    pub param: usize,
    pub width: f64,
}

impl UniformKernel {
    pub fn new(param: usize, width: f64) -> Self {
        assert!(width > 0.0, "Width must be positive for UniformKernel.");
        UniformKernel { param, width }
    }
}

impl PerturbationKernel for UniformKernel {
    fn perturb(&self, from: &Params, rng: &mut StdRng) -> Params {
        let u: f64 = rand_distr::StandardUniform.sample(rng);
        let mut to = from.clone();
        to.set(
            self.param,
            Value::Real(from.real(self.param) + (u - 0.5) * self.width),
        );
        to
    }

    fn transition_probability(&self, to: &Params, from: &Params) -> f64 {
        let half = 0.5 * self.width;
        let from = from.real(self.param);
        if (from - half..=from + half).contains(&to.real(self.param)) {
            1.0 / self.width
        } else {
            0.0
        }
    }
}

/// Integer random walk: uniform on `[-half_width, half_width]`, including zero.
pub struct DiscreteUniformKernel {
    pub param: usize,
    pub half_width: i64,
}

impl DiscreteUniformKernel {
    pub fn new(param: usize, half_width: i64) -> Self {
        assert!(
            half_width > 0,
            "Half-width must be positive for DiscreteUniformKernel."
        );
        DiscreteUniformKernel { param, half_width }
    }
}

impl PerturbationKernel for DiscreteUniformKernel {
    fn perturb(&self, from: &Params, rng: &mut StdRng) -> Params {
        let step = rng.random_range(-self.half_width..=self.half_width);
        let mut to = from.clone();
        to.set(self.param, Value::Int(from.int(self.param) + step));
        to
    }

    fn transition_probability(&self, to: &Params, from: &Params) -> f64 {
        if (to.int(self.param) - from.int(self.param)).abs() <= self.half_width {
            1.0 / (2 * self.half_width + 1) as f64
        } else {
            0.0
        }
    }
}

/// The covariance must be set (normally by `AdaptMultivariateNormalVariance`
/// after generation 0) before perturbing.
pub struct MultivariateNormalKernel {
    pub params: Vec<usize>,
    cov: Option<Covariance>,
}

struct Covariance {
    cholesky: Vec<Vec<f64>>,
    precision: Vec<Vec<f64>>,
    normalizer: f64,
}

impl MultivariateNormalKernel {
    pub fn new(params: Vec<usize>) -> Self {
        MultivariateNormalKernel { params, cov: None }
    }

    pub fn with_covariance(params: Vec<usize>, cov: Vec<Vec<f64>>) -> Self {
        let mut kernel = Self::new(params);
        kernel.set_covariance(cov);
        kernel
    }

    pub fn set_covariance(&mut self, cov: Vec<Vec<f64>>) {
        let d = self.params.len();
        assert_eq!(cov.len(), d, "covariance dimension mismatch");
        let cholesky = cholesky(&cov).expect(
            "covariance must be positive definite (has the population collapsed onto a single value?)",
        );
        // Columns of L⁻¹, so Σ⁻¹ = L⁻ᵀ L⁻¹ has entries Σₖ inverse[i][k] inverse[j][k].
        let inverse: Vec<Vec<f64>> = (0..d)
            .map(|j| {
                let mut e = vec![0.0; d];
                e[j] = 1.0;
                forward_solve(&cholesky, &e)
            })
            .collect();
        let precision = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| (0..d).map(|k| inverse[i][k] * inverse[j][k]).sum())
                    .collect()
            })
            .collect();
        let sqrt_det: f64 = (0..d).map(|i| cholesky[i][i]).product();
        let normalizer = 1.0 / ((2.0 * std::f64::consts::PI).powf(d as f64 / 2.0) * sqrt_det);
        self.cov = Some(Covariance {
            cholesky,
            precision,
            normalizer,
        });
    }

    fn cov(&self) -> &Covariance {
        self.cov.as_ref().expect(
            "MultivariateNormalKernel covariance is unset: adapt it with AdaptMultivariateNormalVariance or call set_covariance",
        )
    }
}

impl PerturbationKernel for MultivariateNormalKernel {
    fn perturb(&self, from: &Params, rng: &mut StdRng) -> Params {
        let l = &self.cov().cholesky;
        let z: Vec<f64> = (0..l.len()).map(|_| StandardNormal.sample(rng)).collect();
        let mut to = from.clone();
        for (i, &param) in self.params.iter().enumerate() {
            let step: f64 = (0..=i).map(|k| l[i][k] * z[k]).sum();
            to.set(param, Value::Real(from.real(param) + step));
        }
        to
    }

    fn transition_probability(&self, to: &Params, from: &Params) -> f64 {
        let cov = self.cov();
        let diff = |&p: &usize| to.real(p) - from.real(p);
        let quad: f64 = self
            .params
            .iter()
            .enumerate()
            .map(|(i, pi)| {
                let row: f64 = self
                    .params
                    .iter()
                    .enumerate()
                    .map(|(j, pj)| cov.precision[i][j] * diff(pj))
                    .sum();
                diff(pi) * row
            })
            .sum();
        cov.normalizer * (-0.5 * quad).exp()
    }
}

/// Applies its kernels in sequence; the transition probability is their product.
pub struct IndependentKernels {
    pub kernels: Vec<Box<dyn PerturbationKernel>>,
}

impl IndependentKernels {
    pub fn new(kernels: Vec<Box<dyn PerturbationKernel>>) -> Self {
        IndependentKernels { kernels }
    }
}

impl PerturbationKernel for IndependentKernels {
    fn perturb(&self, from: &Params, rng: &mut StdRng) -> Params {
        self.kernels
            .iter()
            .fold(from.clone(), |params, kernel| kernel.perturb(&params, rng))
    }

    fn transition_probability(&self, to: &Params, from: &Params) -> f64 {
        self.kernels
            .iter()
            .map(|kernel| kernel.transition_probability(to, from))
            .product()
    }
}

fn normal_pdf(x: f64, std_dev: f64) -> f64 {
    let z = x / std_dev;
    (-0.5 * z * z).exp() / (std_dev * (2.0 * std::f64::consts::PI).sqrt())
}

/// Lower-triangular `L` with `L Lᵀ = a`, or `None` if `a` is not positive definite.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d.is_nan() || d <= 0.0 {
                    return None;
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` for lower-triangular `L`.
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; b.len()];
    for i in 0..b.len() {
        let s: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i][i];
    }
    y
}
