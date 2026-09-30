use rand::Rng;

use crate::distribution::{IntPrior, RealPrior};
use crate::param::{Params, Value};

pub enum ParamPrior {
    Continuous(RealPrior),
    Discrete(IntPrior),
}

impl From<RealPrior> for ParamPrior {
    fn from(prior: RealPrior) -> Self {
        ParamPrior::Continuous(prior)
    }
}

impl From<IntPrior> for ParamPrior {
    fn from(prior: IntPrior) -> Self {
        ParamPrior::Discrete(prior)
    }
}

impl ParamPrior {
    fn sample(&self, rng: &mut impl Rng) -> Value {
        match self {
            ParamPrior::Continuous(d) => Value::Real(d.sample(rng)),
            ParamPrior::Discrete(d) => Value::Int(d.sample(rng)),
        }
    }

    fn density(&self, value: Value) -> f64 {
        match (self, value) {
            (ParamPrior::Continuous(d), Value::Real(x)) => d.density(x),
            (ParamPrior::Discrete(d), Value::Int(k)) => d.density(k),
            _ => 0.0,
        }
    }
}

/// Independent per-parameter priors, in declaration order.
#[derive(Default)]
pub struct Priors {
    priors: Vec<ParamPrior>,
}

impl Priors {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, prior: impl Into<ParamPrior>) -> Self {
        self.priors.push(prior.into());
        self
    }

    pub fn len(&self) -> usize {
        self.priors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.priors.is_empty()
    }

    pub fn is_real(&self, i: usize) -> bool {
        matches!(self.priors[i], ParamPrior::Continuous(_))
    }

    pub fn sample(&self, rng: &mut impl Rng) -> Params {
        Params(self.priors.iter().map(|p| p.sample(rng)).collect())
    }

    pub fn density(&self, params: &Params) -> f64 {
        self.priors
            .iter()
            .zip(&params.0)
            .map(|(p, v)| p.density(*v))
            .product()
    }
}
