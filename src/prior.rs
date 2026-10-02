use std::fmt;

use rand::Rng;

use crate::distribution::{Declared, IntPrior, RealPrior};
use crate::param::{Params, Value};

#[derive(Clone, Copy, Debug)]
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
    /// The family name, such as `"Uniform"`.
    pub fn family(&self) -> &'static str {
        self.declared().family()
    }

    /// The parameters the prior was declared with, by name.
    pub fn parameters(&self) -> impl Iterator<Item = (&'static str, Value)> + '_ {
        self.declared().parameters()
    }

    fn declared(&self) -> &Declared {
        match self {
            ParamPrior::Continuous(d) => d.declared(),
            ParamPrior::Discrete(d) => d.declared(),
        }
    }

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

impl fmt::Display for ParamPrior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.declared().fmt(f)
    }
}

/// Independent per-parameter priors, in declaration order.
///
/// With the `serde` feature this is a sequence of the type-tagged documents
/// described on the prior types.
#[derive(Clone, Debug, Default)]
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

    pub fn iter(&self) -> impl Iterator<Item = &ParamPrior> {
        self.priors.iter()
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

#[cfg(feature = "serde")]
mod wire {
    use serde::{Deserialize, Serialize};

    use super::{ParamPrior, Priors};
    use crate::distribution::wire::{IntRepr, RealRepr};
    use crate::distribution::{IntPrior, RealPrior};

    impl Serialize for ParamPrior {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.declared().serialize(serializer)
        }
    }

    // Family names are distinct across the two kinds, so the tag picks one.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Real(RealRepr),
        Int(IntRepr),
    }

    impl<'de> Deserialize<'de> for ParamPrior {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            match Repr::deserialize(deserializer)? {
                Repr::Real(repr) => RealPrior::try_from(repr).map(ParamPrior::Continuous),
                Repr::Int(repr) => IntPrior::try_from(repr).map(ParamPrior::Discrete),
            }
            .map_err(serde::de::Error::custom)
        }
    }

    impl Serialize for Priors {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.priors.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Priors {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(Priors {
                priors: Vec::deserialize(deserializer)?,
            })
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;

    #[test]
    fn priors_round_trip_through_json() {
        let priors = Priors::new()
            .push(RealPrior::exponential_rate(1.0).unwrap())
            .push(IntPrior::discrete_uniform(1, 4).unwrap());
        let json = serde_json::to_string(&priors).unwrap();
        assert_eq!(
            json,
            r#"[{"type":"Exponential","rate":1.0},{"type":"DiscreteUniform","a":1,"b":4}]"#
        );
        let back: Priors = serde_json::from_str(&json).unwrap();
        assert!(back.is_real(0) && !back.is_real(1));
        assert_eq!(
            back.iter().map(ToString::to_string).collect::<Vec<_>>(),
            ["Exponential(rate = 1)", "DiscreteUniform(a = 1, b = 4)"]
        );
        assert!(serde_json::from_str::<Priors>(r#"[{"type":"Cauchy"}]"#).is_err());
    }
}
