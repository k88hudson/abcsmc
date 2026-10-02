//! Prior distribution families.
//!
//! Sampling is delegated to `rand_distr` and densities to `statrs`; each
//! variant carries one object from each crate, built from a single set of
//! parameters so the two cannot disagree, converted where a backing crate
//! uses a different parameterization.
//!
//! Where a family has more than one common parameterization, each has its own
//! constructor named for its parameters and there is no unqualified one: the
//! exponential by rate or scale, the gamma by shape and scale, shape and
//! rate, or mean and shape, and the negative binomial by `r` and `p` or mean and dispersion. A
//! prior prints and serializes with the parameters it was given.

use std::fmt;

use rand::Rng;
use rand_distr::Distribution as _;
use statrs::distribution::{Continuous, Discrete};

use crate::param::Value;

/// A prior's parameters were outside the family's valid range.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorError {
    family: &'static str,
    message: String,
}

impl PriorError {
    fn new(family: &'static str, err: impl fmt::Display) -> Self {
        PriorError {
            family,
            message: err.to_string(),
        }
    }
}

impl fmt::Display for PriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} prior: {}", self.family, self.message)
    }
}

impl std::error::Error for PriorError {}

/// The family and parameters a prior was declared with, kept alongside the
/// backing distributions so a prior can be printed and serialized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Declared {
    family: &'static str,
    fields: [Option<(&'static str, Value)>; 4],
}

impl Declared {
    fn new<const N: usize>(family: &'static str, given: [(&'static str, Value); N]) -> Self {
        let mut fields = [None; 4];
        for (slot, field) in fields.iter_mut().zip(given) {
            *slot = Some(field);
        }
        Declared { family, fields }
    }

    pub(crate) fn family(&self) -> &'static str {
        self.family
    }

    pub(crate) fn parameters(&self) -> impl Iterator<Item = (&'static str, Value)> + '_ {
        self.fields.iter().flatten().copied()
    }
}

/// `Family(name = value, ...)`, such as `Uniform(a = 0.5, b = 3)`.
impl fmt::Display for Declared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.family)?;
        for (i, (name, value)) in self.parameters().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match value {
                Value::Real(x) => write!(f, "{name} = {x}")?,
                Value::Int(k) => write!(f, "{name} = {k}")?,
            }
        }
        write!(f, ")")
    }
}

macro_rules! declared_accessors {
    ($prior:ident) => {
        impl $prior {
            /// The family name, such as `"Uniform"`.
            pub fn family(&self) -> &'static str {
                self.1.family()
            }

            /// The parameters the prior was declared with, by name.
            pub fn parameters(&self) -> impl Iterator<Item = (&'static str, Value)> + '_ {
                self.1.parameters()
            }

            pub(crate) fn declared(&self) -> &Declared {
                &self.1
            }
        }

        impl fmt::Display for $prior {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.1.fmt(f)
            }
        }
    };
}

declared_accessors!(RealPrior);
declared_accessors!(IntPrior);

#[derive(Clone, Copy, Debug)]
enum Real {
    Uniform(rand_distr::Uniform<f64>, statrs::distribution::Uniform),
    Normal(rand_distr::Normal<f64>, statrs::distribution::Normal),
    Exponential(rand_distr::Exp<f64>, statrs::distribution::Exp),
    LogNormal(rand_distr::LogNormal<f64>, statrs::distribution::LogNormal),
    Gamma(rand_distr::Gamma<f64>, statrs::distribution::Gamma),
    Weibull(rand_distr::Weibull<f64>, statrs::distribution::Weibull),
    /// Shifted and scaled onto `[min, min + width]`.
    Beta(rand_distr::Beta<f64>, statrs::distribution::Beta, f64, f64),
}

/// A prior over a real-valued parameter.
///
/// Kept distinct from [`IntPrior`] so that [`define_priors!`] can require a
/// `Real` field to be given a continuous prior; a mismatch is a type error at
/// the call site rather than a wrong answer at run time.
///
/// [`define_priors!`]: crate::define_priors
#[derive(Clone, Copy, Debug)]
pub struct RealPrior(Real, Declared);

impl RealPrior {
    /// Uniform on `[a, b)`.
    pub fn uniform(a: f64, b: f64) -> Result<Self, PriorError> {
        let sampler = rand_distr::Uniform::new(a, b).map_err(|e| PriorError::new("Uniform", e))?;
        let density =
            statrs::distribution::Uniform::new(a, b).map_err(|e| PriorError::new("Uniform", e))?;
        Ok(RealPrior(
            Real::Uniform(sampler, density),
            Declared::new("Uniform", [("a", a.into()), ("b", b.into())]),
        ))
    }

    pub fn normal(mean: f64, std_dev: f64) -> Result<Self, PriorError> {
        let sampler =
            rand_distr::Normal::new(mean, std_dev).map_err(|e| PriorError::new("Normal", e))?;
        let density = statrs::distribution::Normal::new(mean, std_dev)
            .map_err(|e| PriorError::new("Normal", e))?;
        Ok(RealPrior(
            Real::Normal(sampler, density),
            Declared::new(
                "Normal",
                [("mean", mean.into()), ("std_dev", std_dev.into())],
            ),
        ))
    }

    /// Exponential with the given rate (mean `1 / rate`).
    pub fn exponential_rate(rate: f64) -> Result<Self, PriorError> {
        Self::exponential_declared(rate, [("rate", rate.into())])
    }

    /// Exponential with the given scale, which is its mean (rate `1 / scale`).
    pub fn exponential_scale(scale: f64) -> Result<Self, PriorError> {
        let rate = reciprocal("Exponential", "scale", scale)?;
        Self::exponential_declared(rate, [("scale", scale.into())])
    }

    fn exponential_declared(
        rate: f64,
        declared: [(&'static str, Value); 1],
    ) -> Result<Self, PriorError> {
        let sampler = rand_distr::Exp::new(rate).map_err(|e| PriorError::new("Exponential", e))?;
        let density =
            statrs::distribution::Exp::new(rate).map_err(|e| PriorError::new("Exponential", e))?;
        Ok(RealPrior(
            Real::Exponential(sampler, density),
            Declared::new("Exponential", declared),
        ))
    }

    /// Log-normal: `ln(x) ~ Normal(mu, sigma)`.
    pub fn log_normal(mu: f64, sigma: f64) -> Result<Self, PriorError> {
        let sampler =
            rand_distr::LogNormal::new(mu, sigma).map_err(|e| PriorError::new("LogNormal", e))?;
        let density = statrs::distribution::LogNormal::new(mu, sigma)
            .map_err(|e| PriorError::new("LogNormal", e))?;
        Ok(RealPrior(
            Real::LogNormal(sampler, density),
            Declared::new("LogNormal", [("mu", mu.into()), ("sigma", sigma.into())]),
        ))
    }

    /// Gamma with the given shape and scale (mean `shape * scale`).
    pub fn gamma_shape_scale(shape: f64, scale: f64) -> Result<Self, PriorError> {
        let rate = reciprocal("Gamma", "scale", scale)?;
        Self::gamma_declared(
            shape,
            scale,
            rate,
            [("shape", shape.into()), ("scale", scale.into())],
        )
    }

    /// Gamma with the given shape and rate (mean `shape / rate`).
    pub fn gamma_shape_rate(shape: f64, rate: f64) -> Result<Self, PriorError> {
        let scale = reciprocal("Gamma", "rate", rate)?;
        Self::gamma_declared(
            shape,
            scale,
            rate,
            [("shape", shape.into()), ("rate", rate.into())],
        )
    }

    /// Gamma with the given mean and shape (scale `mean / shape`).
    pub fn gamma_mean_shape(mean: f64, shape: f64) -> Result<Self, PriorError> {
        positive("Gamma", "mean", mean)?;
        positive("Gamma", "shape", shape)?;
        let (scale, rate) = (mean / shape, shape / mean);
        if !(scale.is_finite() && scale > 0.0 && rate.is_finite() && rate > 0.0) {
            return Err(PriorError::new(
                "Gamma",
                format!("mean {mean} and shape {shape} give a scale of {scale}, out of range"),
            ));
        }
        Self::gamma_declared(
            shape,
            scale,
            rate,
            [("mean", mean.into()), ("shape", shape.into())],
        )
    }

    /// The sampler takes the scale and the density the rate, so each
    /// constructor passes the one it was given unconverted.
    fn gamma_declared(
        shape: f64,
        scale: f64,
        rate: f64,
        declared: [(&'static str, Value); 2],
    ) -> Result<Self, PriorError> {
        let sampler =
            rand_distr::Gamma::new(shape, scale).map_err(|e| PriorError::new("Gamma", e))?;
        let density = statrs::distribution::Gamma::new(shape, rate)
            .map_err(|e| PriorError::new("Gamma", e))?;
        Ok(RealPrior(
            Real::Gamma(sampler, density),
            Declared::new("Gamma", declared),
        ))
    }

    pub fn weibull(shape: f64, scale: f64) -> Result<Self, PriorError> {
        // rand_distr takes (scale, shape).
        let sampler =
            rand_distr::Weibull::new(scale, shape).map_err(|e| PriorError::new("Weibull", e))?;
        let density = statrs::distribution::Weibull::new(shape, scale)
            .map_err(|e| PriorError::new("Weibull", e))?;
        Ok(RealPrior(
            Real::Weibull(sampler, density),
            Declared::new(
                "Weibull",
                [("shape", shape.into()), ("scale", scale.into())],
            ),
        ))
    }

    /// Beta on `[0, 1]`.
    pub fn beta(alpha: f64, beta: f64) -> Result<Self, PriorError> {
        Self::scaled_beta(alpha, beta, 0.0, 1.0)
    }

    /// Beta shifted and scaled onto `[min, max]`.
    pub fn scaled_beta(alpha: f64, beta: f64, min: f64, max: f64) -> Result<Self, PriorError> {
        if min >= max || !(min.is_finite() && max.is_finite()) {
            return Err(PriorError::new(
                "Beta",
                format!("min must be less than max, got [{min}, {max}]"),
            ));
        }
        let sampler = rand_distr::Beta::new(alpha, beta).map_err(|e| PriorError::new("Beta", e))?;
        let density =
            statrs::distribution::Beta::new(alpha, beta).map_err(|e| PriorError::new("Beta", e))?;
        Ok(RealPrior(
            Real::Beta(sampler, density, min, max - min),
            Declared::new(
                "Beta",
                [
                    ("alpha", alpha.into()),
                    ("beta", beta.into()),
                    ("min", min.into()),
                    ("max", max.into()),
                ],
            ),
        ))
    }

    pub(crate) fn sample(&self, rng: &mut impl Rng) -> f64 {
        match &self.0 {
            Real::Uniform(d, _) => d.sample(rng),
            Real::Normal(d, _) => d.sample(rng),
            Real::Exponential(d, _) => d.sample(rng),
            Real::LogNormal(d, _) => d.sample(rng),
            Real::Gamma(d, _) => d.sample(rng),
            Real::Weibull(d, _) => d.sample(rng),
            Real::Beta(d, _, min, width) => min + width * d.sample(rng),
        }
    }

    /// Probability density at `x`; zero outside the support.
    pub(crate) fn density(&self, x: f64) -> f64 {
        match &self.0 {
            Real::Uniform(_, d) => d.pdf(x),
            Real::Normal(_, d) => d.pdf(x),
            Real::Exponential(_, d) => d.pdf(x),
            Real::LogNormal(_, d) => d.pdf(x),
            Real::Gamma(_, d) => d.pdf(x),
            Real::Weibull(_, d) => d.pdf(x),
            Real::Beta(_, d, min, width) => d.pdf((x - min) / width) / width,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Int {
    DiscreteUniform(
        rand_distr::Uniform<i64>,
        statrs::distribution::DiscreteUniform,
    ),
    Poisson(rand_distr::Poisson<f64>, statrs::distribution::Poisson),
    Binomial(rand_distr::Binomial, statrs::distribution::Binomial),
    /// Sampled as a gamma-Poisson mixture: `lambda ~ Gamma(r, (1 - p) / p)`,
    /// then `Poisson(lambda)`.
    NegativeBinomial(
        rand_distr::Gamma<f64>,
        statrs::distribution::NegativeBinomial,
    ),
}

/// A prior over an integer-valued parameter.
#[derive(Clone, Copy, Debug)]
pub struct IntPrior(Int, Declared);

impl IntPrior {
    /// Uniform on the **inclusive** range `[a, b]`.
    pub fn discrete_uniform(a: i64, b: i64) -> Result<Self, PriorError> {
        let sampler = rand_distr::Uniform::new_inclusive(a, b)
            .map_err(|e| PriorError::new("DiscreteUniform", e))?;
        let density = statrs::distribution::DiscreteUniform::new(a, b)
            .map_err(|e| PriorError::new("DiscreteUniform", e))?;
        Ok(IntPrior(
            Int::DiscreteUniform(sampler, density),
            Declared::new("DiscreteUniform", [("a", a.into()), ("b", b.into())]),
        ))
    }

    pub fn poisson(lambda: f64) -> Result<Self, PriorError> {
        let sampler =
            rand_distr::Poisson::new(lambda).map_err(|e| PriorError::new("Poisson", e))?;
        let density = statrs::distribution::Poisson::new(lambda)
            .map_err(|e| PriorError::new("Poisson", e))?;
        Ok(IntPrior(
            Int::Poisson(sampler, density),
            Declared::new("Poisson", [("lambda", lambda.into())]),
        ))
    }

    /// Binomial with `n` trials of success probability `p`.
    pub fn binomial(n: u64, p: f64) -> Result<Self, PriorError> {
        let sampler =
            rand_distr::Binomial::new(n, p).map_err(|e| PriorError::new("Binomial", e))?;
        // statrs takes (p, n).
        let density = statrs::distribution::Binomial::new(p, n)
            .map_err(|e| PriorError::new("Binomial", e))?;
        Ok(IntPrior(
            Int::Binomial(sampler, density),
            Declared::new("Binomial", [("n", Value::Int(n as i64)), ("p", p.into())]),
        ))
    }

    /// Negative binomial counting failures before `r` successes of probability
    /// `p`, so `P(0) = p^r` and the mean is `r (1 - p) / p`. `p` must lie in
    /// `(0, 1)`: `p = 1` is a point mass at zero, which cannot be sampled.
    pub fn negative_binomial_r_p(r: f64, p: f64) -> Result<Self, PriorError> {
        if !(p > 0.0 && p < 1.0) {
            return Err(PriorError::new(
                "NegativeBinomial",
                format!("p must be in (0, 1), got {p}"),
            ));
        }
        Self::negative_binomial_declared(r, p, (1.0 - p) / p, [("r", r.into()), ("p", p.into())])
    }

    /// Negative binomial with the given mean and dispersion `k`, so the
    /// variance is `mean + mean^2 / k`: smaller `k` is more dispersed, and it
    /// approaches the Poisson as `k` grows. Equivalent to `r = k`,
    /// `p = k / (k + mean)`.
    pub fn negative_binomial_mean_dispersion(mean: f64, k: f64) -> Result<Self, PriorError> {
        positive("NegativeBinomial", "mean", mean)?;
        positive("NegativeBinomial", "k", k)?;
        let p = k / (k + mean);
        if !(p > 0.0 && p < 1.0) {
            return Err(PriorError::new(
                "NegativeBinomial",
                format!("mean {mean} and k {k} give p = {p}, which must be in (0, 1)"),
            ));
        }
        Self::negative_binomial_declared(k, p, mean / k, [("mean", mean.into()), ("k", k.into())])
    }

    fn negative_binomial_declared(
        r: f64,
        p: f64,
        gamma_scale: f64,
        declared: [(&'static str, Value); 2],
    ) -> Result<Self, PriorError> {
        let sampler = rand_distr::Gamma::new(r, gamma_scale)
            .map_err(|e| PriorError::new("NegativeBinomial", e))?;
        let density = statrs::distribution::NegativeBinomial::new(r, p)
            .map_err(|e| PriorError::new("NegativeBinomial", e))?;
        Ok(IntPrior(
            Int::NegativeBinomial(sampler, density),
            Declared::new("NegativeBinomial", declared),
        ))
    }

    pub(crate) fn sample(&self, rng: &mut impl Rng) -> i64 {
        match &self.0 {
            Int::DiscreteUniform(d, _) => d.sample(rng),
            Int::Poisson(d, _) => d.sample(rng) as i64,
            Int::Binomial(d, _) => d.sample(rng) as i64,
            Int::NegativeBinomial(gamma, _) => {
                let lambda = gamma.sample(rng);
                if lambda <= 0.0 {
                    return 0;
                }
                rand_distr::Poisson::new(lambda)
                    .expect("gamma draw is finite and positive")
                    .sample(rng) as i64
            }
        }
    }

    /// Probability mass at `k`; zero outside the support.
    pub(crate) fn density(&self, k: i64) -> f64 {
        match &self.0 {
            Int::DiscreteUniform(_, d) => d.pmf(k),
            Int::Poisson(_, d) => non_negative(k).map_or(0.0, |k| d.pmf(k)),
            Int::Binomial(_, d) => non_negative(k).map_or(0.0, |k| d.pmf(k)),
            Int::NegativeBinomial(_, d) => non_negative(k).map_or(0.0, |k| d.pmf(k)),
        }
    }
}

/// `1 / value` for a parameter given in the other parameterization, which
/// must be positive and small or large enough that the reciprocal is too.
fn reciprocal(family: &'static str, name: &str, value: f64) -> Result<f64, PriorError> {
    positive(family, name, value)?;
    let inverse = 1.0 / value;
    if inverse.is_finite() && inverse > 0.0 {
        Ok(inverse)
    } else {
        Err(PriorError::new(
            family,
            format!("{name} {value} is out of range: its reciprocal is {inverse}"),
        ))
    }
}

fn non_negative(k: i64) -> Option<u64> {
    u64::try_from(k).ok()
}

fn positive(family: &'static str, name: &str, value: f64) -> Result<(), PriorError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PriorError::new(
            family,
            format!("{name} must be finite and positive, got {value}"),
        ))
    }
}

/// With the `serde` feature, priors serialize to and deserialize from a
/// type-tagged document such as `{ "type": "Uniform", "a": 1.0, "b": 3.5 }`,
/// so a config file can declare them without any code-level wrapper. Where a
/// family has two parameterizations a document gives exactly one of them, and
/// a prior is written with the one it was declared with, so a round trip is
/// exact (`min` and `max` are always written for beta).
/// Field names per family:
///
/// | `type` | fields |
/// |---|---|
/// | `Uniform` | `a`, `b` |
/// | `Normal` | `mean`, `std_dev` |
/// | `Exponential` | `rate` or `scale` |
/// | `LogNormal` | `mu`, `sigma` |
/// | `Gamma` | `shape`, then `scale`, `rate`, or `mean` |
/// | `Weibull` | `shape`, `scale` |
/// | `Beta` | `alpha`, `beta`, optional `min`, `max` (default `[0, 1]`) |
/// | `DiscreteUniform` | `a`, `b` (inclusive) |
/// | `Poisson` | `lambda` |
/// | `Binomial` | `n`, `p` |
/// | `NegativeBinomial` | `r`, `p` or `mean`, `k` |
#[cfg(feature = "serde")]
pub(crate) mod wire {
    use serde::ser::SerializeMap;

    use super::{Declared, IntPrior, PriorError, RealPrior};
    use crate::param::Value;

    impl serde::Serialize for Declared {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("type", self.family())?;
            for (name, value) in self.parameters() {
                match value {
                    Value::Real(x) => map.serialize_entry(name, &x)?,
                    Value::Int(k) => map.serialize_entry(name, &k)?,
                }
            }
            map.end()
        }
    }

    impl serde::Serialize for RealPrior {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.declared().serialize(serializer)
        }
    }

    impl serde::Serialize for IntPrior {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.declared().serialize(serializer)
        }
    }

    #[derive(serde::Deserialize)]
    #[serde(tag = "type")]
    pub(crate) enum RealRepr {
        Uniform {
            a: f64,
            b: f64,
        },
        Normal {
            mean: f64,
            std_dev: f64,
        },
        Exponential {
            rate: Option<f64>,
            scale: Option<f64>,
        },
        LogNormal {
            mu: f64,
            sigma: f64,
        },
        Gamma {
            shape: f64,
            scale: Option<f64>,
            rate: Option<f64>,
            mean: Option<f64>,
        },
        Weibull {
            shape: f64,
            scale: f64,
        },
        Beta {
            alpha: f64,
            beta: f64,
            #[serde(default)]
            min: Option<f64>,
            #[serde(default)]
            max: Option<f64>,
        },
    }

    #[derive(serde::Deserialize)]
    #[serde(tag = "type")]
    pub(crate) enum IntRepr {
        DiscreteUniform {
            a: i64,
            b: i64,
        },
        Poisson {
            lambda: f64,
        },
        Binomial {
            n: u64,
            p: f64,
        },
        NegativeBinomial {
            r: Option<f64>,
            p: Option<f64>,
            mean: Option<f64>,
            k: Option<f64>,
        },
    }

    fn one_of(family: &'static str, forms: &str) -> PriorError {
        PriorError::new(family, format!("give exactly one of {forms}"))
    }

    impl TryFrom<RealRepr> for RealPrior {
        type Error = PriorError;

        fn try_from(repr: RealRepr) -> Result<Self, PriorError> {
            match repr {
                RealRepr::Uniform { a, b } => RealPrior::uniform(a, b),
                RealRepr::Normal { mean, std_dev } => RealPrior::normal(mean, std_dev),
                RealRepr::Exponential { rate, scale } => match (rate, scale) {
                    (Some(rate), None) => RealPrior::exponential_rate(rate),
                    (None, Some(scale)) => RealPrior::exponential_scale(scale),
                    _ => Err(one_of("Exponential", "`rate` or `scale`")),
                },
                RealRepr::LogNormal { mu, sigma } => RealPrior::log_normal(mu, sigma),
                RealRepr::Gamma {
                    shape,
                    scale,
                    rate,
                    mean,
                } => match (scale, rate, mean) {
                    (Some(scale), None, None) => RealPrior::gamma_shape_scale(shape, scale),
                    (None, Some(rate), None) => RealPrior::gamma_shape_rate(shape, rate),
                    (None, None, Some(mean)) => RealPrior::gamma_mean_shape(mean, shape),
                    _ => Err(one_of("Gamma", "`scale`, `rate`, or `mean`")),
                },
                RealRepr::Weibull { shape, scale } => RealPrior::weibull(shape, scale),
                RealRepr::Beta {
                    alpha,
                    beta,
                    min,
                    max,
                } => RealPrior::scaled_beta(alpha, beta, min.unwrap_or(0.0), max.unwrap_or(1.0)),
            }
        }
    }

    impl TryFrom<IntRepr> for IntPrior {
        type Error = PriorError;

        fn try_from(repr: IntRepr) -> Result<Self, PriorError> {
            match repr {
                IntRepr::DiscreteUniform { a, b } => IntPrior::discrete_uniform(a, b),
                IntRepr::Poisson { lambda } => IntPrior::poisson(lambda),
                IntRepr::Binomial { n, p } => IntPrior::binomial(n, p),
                IntRepr::NegativeBinomial { r, p, mean, k } => match (r, p, mean, k) {
                    (Some(r), Some(p), None, None) => IntPrior::negative_binomial_r_p(r, p),
                    (None, None, Some(mean), Some(k)) => {
                        IntPrior::negative_binomial_mean_dispersion(mean, k)
                    }
                    _ => Err(one_of("NegativeBinomial", "`r`, `p` or `mean`, `k`")),
                },
            }
        }
    }

    impl<'de> serde::Deserialize<'de> for RealPrior {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            RealRepr::deserialize(deserializer)?
                .try_into()
                .map_err(serde::de::Error::custom)
        }
    }

    impl<'de> serde::Deserialize<'de> for IntPrior {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            IntRepr::deserialize(deserializer)?
                .try_into()
                .map_err(serde::de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn continuous_densities_match_closed_forms() {
        assert!(close(
            RealPrior::uniform(0.0, 10.0).unwrap().density(3.0),
            0.1
        ));
        assert_eq!(RealPrior::uniform(0.0, 10.0).unwrap().density(-1.0), 0.0);
        assert_eq!(RealPrior::uniform(0.0, 10.0).unwrap().density(11.0), 0.0);

        let normal = RealPrior::normal(1.0, 2.0).unwrap();
        assert!(close(
            normal.density(1.0),
            1.0 / (2.0 * (2.0 * std::f64::consts::PI).sqrt())
        ));

        let exp = RealPrior::exponential_rate(2.0).unwrap();
        assert!(close(exp.density(0.5), 2.0 * (-1.0f64).exp()));
        assert_eq!(exp.density(-0.5), 0.0);

        let gamma = RealPrior::gamma_shape_scale(2.0, 3.0).unwrap();
        // shape 2: pdf = x e^{-x/scale} / scale^2
        assert!(close(gamma.density(1.5), 1.5 * (-0.5f64).exp() / 9.0));

        let weibull = RealPrior::weibull(2.0, 3.0).unwrap();
        // (k/l)(x/l)^{k-1} e^{-(x/l)^k}
        let x: f64 = 1.5;
        assert!(close(
            weibull.density(x),
            (2.0 / 3.0) * (x / 3.0) * (-(x / 3.0).powi(2)).exp()
        ));

        // Beta(2, 2): 6 x (1 - x)
        assert!(close(
            RealPrior::beta(2.0, 2.0).unwrap().density(0.25),
            6.0 * 0.25 * 0.75
        ));
        assert_eq!(RealPrior::beta(2.0, 2.0).unwrap().density(1.5), 0.0);

        // Beta(2, 2) on [1, 3]: 6 u (1 - u) / 2 at u = (x - 1) / 2
        let scaled = RealPrior::scaled_beta(2.0, 2.0, 1.0, 3.0).unwrap();
        assert!(close(scaled.density(1.5), 6.0 * 0.25 * 0.75 / 2.0));
        assert_eq!(scaled.density(0.5), 0.0);
        assert_eq!(scaled.density(3.5), 0.0);

        let log_normal = RealPrior::log_normal(0.0, 1.0).unwrap();
        // x = 1: 1 / sqrt(2 pi)
        assert!(close(
            log_normal.density(1.0),
            1.0 / (2.0 * std::f64::consts::PI).sqrt()
        ));
    }

    #[test]
    fn discrete_masses_match_closed_forms() {
        let du = IntPrior::discrete_uniform(1, 4).unwrap();
        assert!(close(du.density(1), 0.25));
        assert!(close(du.density(4), 0.25));
        assert_eq!(du.density(0), 0.0);
        assert_eq!(du.density(5), 0.0);

        let poisson = IntPrior::poisson(3.0).unwrap();
        assert!(close(poisson.density(2), 4.5 * (-3.0f64).exp()));
        assert_eq!(poisson.density(-1), 0.0);

        let binomial = IntPrior::binomial(4, 0.5).unwrap();
        assert!(close(binomial.density(2), 6.0 / 16.0));
        assert_eq!(binomial.density(5), 0.0);

        // P(0) = p^r
        let nb = IntPrior::negative_binomial_r_p(3.0, 0.4).unwrap();
        assert!(close(nb.density(0), 0.4f64.powi(3)));
        assert!(close(nb.density(1), 3.0 * 0.4f64.powi(3) * 0.6));
    }

    #[test]
    fn samples_stay_in_support() {
        let mut rng = StdRng::seed_from_u64(7);
        let uniform = RealPrior::uniform(2.0, 3.0).unwrap();
        let du = IntPrior::discrete_uniform(-2, 2).unwrap();
        let beta = RealPrior::scaled_beta(2.0, 5.0, 1.0, 3.0).unwrap();
        let nb = IntPrior::negative_binomial_r_p(2.0, 0.3).unwrap();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2000 {
            let x = uniform.sample(&mut rng);
            assert!((2.0..3.0).contains(&x));
            let k = du.sample(&mut rng);
            assert!((-2..=2).contains(&k));
            seen.insert(k);
            let b = beta.sample(&mut rng);
            assert!((1.0..=3.0).contains(&b));
            assert!(nb.sample(&mut rng) >= 0);
        }
        assert_eq!(seen.len(), 5, "inclusive bounds should all be drawn");
    }

    #[test]
    fn negative_binomial_sample_mean_matches() {
        let mut rng = StdRng::seed_from_u64(11);
        let (r, p) = (4.0, 0.25);
        let nb = IntPrior::negative_binomial_r_p(r, p).unwrap();
        let n = 20_000;
        let mean = (0..n).map(|_| nb.sample(&mut rng) as f64).sum::<f64>() / n as f64;
        let expected = r * (1.0 - p) / p;
        assert!((mean - expected).abs() < 0.2, "mean {mean} vs {expected}");
    }

    #[test]
    fn alternative_parameterizations_agree() {
        let by_rate = RealPrior::exponential_rate(4.0).unwrap();
        let by_scale = RealPrior::exponential_scale(0.25).unwrap();
        assert_eq!(by_rate.density(0.3), by_scale.density(0.3));
        assert_eq!(by_scale.to_string(), "Exponential(scale = 0.25)");

        let by_scale = RealPrior::gamma_shape_scale(2.0, 4.0).unwrap();
        let by_rate = RealPrior::gamma_shape_rate(2.0, 0.25).unwrap();
        assert_eq!(by_scale.density(1.5), by_rate.density(1.5));
        assert_eq!(by_rate.to_string(), "Gamma(shape = 2, rate = 0.25)");
        let by_mean = RealPrior::gamma_mean_shape(8.0, 2.0).unwrap();
        assert_eq!(by_scale.density(1.5), by_mean.density(1.5));
        assert_eq!(by_mean.to_string(), "Gamma(mean = 8, shape = 2)");

        // mean 6, k 3: r = 3, p = 1/3
        let by_p = IntPrior::negative_binomial_r_p(3.0, 1.0 / 3.0).unwrap();
        let by_mean = IntPrior::negative_binomial_mean_dispersion(6.0, 3.0).unwrap();
        for k in 0..20 {
            assert!(close(by_p.density(k), by_mean.density(k)));
        }
        assert_eq!(by_mean.to_string(), "NegativeBinomial(mean = 6, k = 3)");
    }

    #[test]
    fn alternative_parameterizations_sample_the_same_distribution() {
        let n = 40_000;
        let mean =
            |mut draw: Box<dyn FnMut() -> f64>| (0..n).map(|_| draw()).sum::<f64>() / n as f64;

        let mut rng = StdRng::seed_from_u64(3);
        let exp = RealPrior::exponential_scale(2.5).unwrap();
        let m = mean(Box::new(|| exp.sample(&mut rng)));
        assert!((m - 2.5).abs() < 0.05, "exponential mean {m}");

        let mut rng = StdRng::seed_from_u64(4);
        let gamma = RealPrior::gamma_shape_rate(3.0, 2.0).unwrap();
        let m = mean(Box::new(|| gamma.sample(&mut rng)));
        assert!((m - 1.5).abs() < 0.03, "gamma mean {m}");

        let mut rng = StdRng::seed_from_u64(6);
        let gamma = RealPrior::gamma_mean_shape(1.5, 3.0).unwrap();
        let m = mean(Box::new(|| gamma.sample(&mut rng)));
        assert!((m - 1.5).abs() < 0.03, "gamma mean {m}");

        let mut rng = StdRng::seed_from_u64(5);
        let nb = IntPrior::negative_binomial_mean_dispersion(6.0, 3.0).unwrap();
        let draws: Vec<f64> = (0..n).map(|_| nb.sample(&mut rng) as f64).collect();
        let m = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        assert!((m - 6.0).abs() < 0.1, "negative binomial mean {m}");
        // mean + mean^2 / k
        assert!((var - 18.0).abs() < 1.0, "negative binomial variance {var}");
    }

    #[test]
    fn macros_expand_to_the_named_constructors() {
        use crate::*;
        let same = |a: Result<RealPrior, PriorError>, b: Result<RealPrior, PriorError>| {
            assert_eq!(a.unwrap().declared(), b.unwrap().declared());
        };
        same(
            uniform_prior!(a = 0.5, b = 3.0),
            RealPrior::uniform(0.5, 3.0),
        );
        same(
            normal_prior!(mean = 1.0, std_dev = 2.0),
            RealPrior::normal(1.0, 2.0),
        );
        same(
            exponential_prior!(rate = 2.0),
            RealPrior::exponential_rate(2.0),
        );
        same(
            exponential_prior!(scale = 2.0),
            RealPrior::exponential_scale(2.0),
        );
        same(
            log_normal_prior!(mu = 0.0, sigma = 1.0),
            RealPrior::log_normal(0.0, 1.0),
        );
        same(
            gamma_prior!(shape = 2.0, scale = 3.0),
            RealPrior::gamma_shape_scale(2.0, 3.0),
        );
        same(
            gamma_prior!(shape = 2.0, rate = 3.0,),
            RealPrior::gamma_shape_rate(2.0, 3.0),
        );
        same(
            gamma_prior!(mean = 6.0, shape = 2.0),
            RealPrior::gamma_mean_shape(6.0, 2.0),
        );
        same(
            weibull_prior!(shape = 2.0, scale = 3.0),
            RealPrior::weibull(2.0, 3.0),
        );
        same(
            beta_prior!(alpha = 2.0, beta = 5.0),
            RealPrior::beta(2.0, 5.0),
        );
        same(
            beta_prior!(alpha = 2.0, beta = 5.0, min = 1.0, max = 3.0),
            RealPrior::scaled_beta(2.0, 5.0, 1.0, 3.0),
        );

        let same = |a: Result<IntPrior, PriorError>, b: Result<IntPrior, PriorError>| {
            assert_eq!(a.unwrap().declared(), b.unwrap().declared());
        };
        same(
            discrete_uniform_prior!(a = 1, b = 4),
            IntPrior::discrete_uniform(1, 4),
        );
        same(poisson_prior!(lambda = 3.0), IntPrior::poisson(3.0));
        same(binomial_prior!(n = 4, p = 0.5), IntPrior::binomial(4, 0.5));
        same(
            negative_binomial_prior!(r = 3.0, p = 0.4),
            IntPrior::negative_binomial_r_p(3.0, 0.4),
        );
        same(
            negative_binomial_prior!(mean = 6.0, k = 3.0),
            IntPrior::negative_binomial_mean_dispersion(6.0, 3.0),
        );
        assert!(exponential_prior!(rate = 0.0).is_err());
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        for bad in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(RealPrior::exponential_scale(bad).is_err());
            assert!(RealPrior::gamma_shape_scale(2.0, bad).is_err());
            assert!(RealPrior::gamma_shape_rate(2.0, bad).is_err());
            assert!(RealPrior::gamma_mean_shape(bad, 2.0).is_err());
            assert!(RealPrior::gamma_mean_shape(2.0, bad).is_err());
            assert!(IntPrior::negative_binomial_mean_dispersion(bad, 1.0).is_err());
            assert!(IntPrior::negative_binomial_mean_dispersion(1.0, bad).is_err());
        }
        assert!(RealPrior::exponential_scale(1e-320).is_err());
        assert!(RealPrior::gamma_shape_scale(2.0, 1e-320).is_err());
        assert!(RealPrior::gamma_shape_rate(2.0, 1e-320).is_err());
        assert!(RealPrior::gamma_mean_shape(1e300, 1e-300).is_err());
        // p rounds to 1
        assert!(IntPrior::negative_binomial_mean_dispersion(1e-300, 1.0).is_err());
        assert!(RealPrior::uniform(1.0, 1.0).is_err());
        assert!(RealPrior::normal(0.0, -1.0).is_err());
        assert!(RealPrior::exponential_rate(0.0).is_err());
        assert!(IntPrior::discrete_uniform(3, 2).is_err());
        assert!(IntPrior::negative_binomial_r_p(2.0, 1.0).is_err());
        assert!(RealPrior::scaled_beta(2.0, 2.0, 3.0, 1.0).is_err());
        let err = RealPrior::uniform(1.0, 1.0).unwrap_err().to_string();
        assert!(err.starts_with("Uniform prior:"), "{err}");
    }

    #[test]
    fn priors_print_their_family_and_parameters() {
        assert_eq!(
            RealPrior::uniform(0.5, 3.0).unwrap().to_string(),
            "Uniform(a = 0.5, b = 3)"
        );
        assert_eq!(
            IntPrior::binomial(4, 0.5).unwrap().to_string(),
            "Binomial(n = 4, p = 0.5)"
        );
        let beta = RealPrior::beta(2.0, 5.0).unwrap();
        assert_eq!(beta.family(), "Beta");
        assert_eq!(beta.parameters().count(), 4);
    }

    #[cfg(feature = "serde")]
    mod serde {
        use super::*;

        #[test]
        fn deserializes_tagged_documents() {
            let u: RealPrior =
                serde_json::from_str(r#"{"type":"Uniform","a":0.02,"b":2.2}"#).unwrap();
            assert!(close(u.density(1.0), 1.0 / 2.18));

            let ln: RealPrior =
                serde_json::from_str(r#"{"type":"LogNormal","mu":0.0,"sigma":1.0}"#).unwrap();
            assert!(close(
                ln.density(1.0),
                1.0 / (2.0 * std::f64::consts::PI).sqrt()
            ));

            let b: RealPrior =
                serde_json::from_str(r#"{"type":"Beta","alpha":2.0,"beta":2.0}"#).unwrap();
            assert!(close(b.density(0.25), 6.0 * 0.25 * 0.75));
            let b: RealPrior = serde_json::from_str(
                r#"{"type":"Beta","alpha":2.0,"beta":2.0,"min":1.0,"max":3.0}"#,
            )
            .unwrap();
            assert!(close(b.density(1.5), 6.0 * 0.25 * 0.75 / 2.0));

            let e_rate: RealPrior =
                serde_json::from_str(r#"{"type":"Exponential","rate":2.0}"#).unwrap();
            let e_scale: RealPrior =
                serde_json::from_str(r#"{"type":"Exponential","scale":0.5}"#).unwrap();
            assert!(close(e_rate.density(0.5), e_scale.density(0.5)));

            let g_scale: RealPrior =
                serde_json::from_str(r#"{"type":"Gamma","shape":2.0,"scale":3.0}"#).unwrap();
            let g_rate: RealPrior =
                serde_json::from_str(r#"{"type":"Gamma","shape":2.0,"rate":0.3333333333333333}"#)
                    .unwrap();
            assert!(close(g_scale.density(1.5), g_rate.density(1.5)));

            let du: IntPrior =
                serde_json::from_str(r#"{"type":"DiscreteUniform","a":0,"b":215}"#).unwrap();
            assert!(close(du.density(215), 1.0 / 216.0));

            let nb: IntPrior =
                serde_json::from_str(r#"{"type":"NegativeBinomial","r":3.0,"p":0.4}"#).unwrap();
            assert!(close(nb.density(0), 0.4f64.powi(3)));
        }

        #[test]
        fn serializes_to_the_documents_it_reads() {
            let priors = [
                RealPrior::uniform(0.02, 2.2).unwrap(),
                RealPrior::normal(1.0, 2.0).unwrap(),
                RealPrior::exponential_rate(2.0).unwrap(),
                RealPrior::exponential_scale(0.3).unwrap(),
                RealPrior::log_normal(0.0, 1.0).unwrap(),
                RealPrior::gamma_shape_scale(2.0, 3.0).unwrap(),
                RealPrior::gamma_shape_rate(2.0, 0.3).unwrap(),
                RealPrior::gamma_mean_shape(0.7, 0.3).unwrap(),
                RealPrior::weibull(2.0, 3.0).unwrap(),
                RealPrior::scaled_beta(2.0, 5.0, 1.0, 3.0).unwrap(),
            ];
            for prior in priors {
                let json = serde_json::to_string(&prior).unwrap();
                let back: RealPrior = serde_json::from_str(&json).unwrap();
                assert_eq!(back.declared(), prior.declared(), "{json}");
            }
            let priors = [
                IntPrior::discrete_uniform(0, 215).unwrap(),
                IntPrior::poisson(3.0).unwrap(),
                IntPrior::binomial(4, 0.5).unwrap(),
                IntPrior::negative_binomial_r_p(3.0, 0.4).unwrap(),
                IntPrior::negative_binomial_mean_dispersion(6.0, 0.3).unwrap(),
            ];
            for prior in priors {
                let json = serde_json::to_string(&prior).unwrap();
                let back: IntPrior = serde_json::from_str(&json).unwrap();
                assert_eq!(back.declared(), prior.declared(), "{json}");
            }
            assert_eq!(
                serde_json::to_string(&RealPrior::uniform(0.5, 3.0).unwrap()).unwrap(),
                r#"{"type":"Uniform","a":0.5,"b":3.0}"#
            );
            assert_eq!(
                serde_json::to_string(&IntPrior::discrete_uniform(1, 4).unwrap()).unwrap(),
                r#"{"type":"DiscreteUniform","a":1,"b":4}"#
            );
            assert_eq!(
                serde_json::to_string(&RealPrior::gamma_shape_rate(2.0, 0.3).unwrap()).unwrap(),
                r#"{"type":"Gamma","shape":2.0,"rate":0.3}"#
            );
            assert_eq!(
                serde_json::to_string(
                    &IntPrior::negative_binomial_mean_dispersion(6.0, 0.3).unwrap()
                )
                .unwrap(),
                r#"{"type":"NegativeBinomial","mean":6.0,"k":0.3}"#
            );
        }

        #[test]
        fn rejects_bad_documents() {
            assert!(serde_json::from_str::<RealPrior>(r#"{"type":"Exponential"}"#).is_err());
            assert!(
                serde_json::from_str::<RealPrior>(
                    r#"{"type":"Exponential","rate":1.0,"scale":1.0}"#
                )
                .is_err()
            );
            assert!(
                serde_json::from_str::<RealPrior>(r#"{"type":"Uniform","a":5.0,"b":1.0}"#).is_err()
            );
            assert!(serde_json::from_str::<RealPrior>(r#"{"type":"Cauchy","x0":0.0}"#).is_err());
            assert!(
                serde_json::from_str::<RealPrior>(
                    r#"{"type":"Gamma","shape":2.0,"scale":1.0,"rate":1.0}"#
                )
                .is_err()
            );
            assert!(
                serde_json::from_str::<RealPrior>(
                    r#"{"type":"Gamma","shape":2.0,"scale":1.0,"mean":2.0}"#
                )
                .is_err()
            );
            assert!(serde_json::from_str::<RealPrior>(r#"{"type":"Gamma","shape":2.0}"#).is_err());
            for doc in [
                r#"{"type":"NegativeBinomial","r":3.0,"k":0.4}"#,
                r#"{"type":"NegativeBinomial","mean":3.0}"#,
                r#"{"type":"NegativeBinomial","r":3.0,"p":0.4,"mean":4.5,"k":3.0}"#,
            ] {
                assert!(serde_json::from_str::<IntPrior>(doc).is_err(), "{doc}");
            }
            assert!(
                serde_json::from_str::<IntPrior>(r#"{"type":"Uniform","a":0.0,"b":1.0}"#).is_err()
            );
        }
    }
}
