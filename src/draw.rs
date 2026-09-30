use crate::{Params, Value};

/// A model's calibrated parameters as a typed struct.
///
/// The engine works in a positional vector of [`Value`]s — that is what the
/// priors, the prior density, and the perturbation kernel all operate on — but a
/// model never has to. Implementations of this trait rebuild a named struct from
/// that vector, so [`Model::simulate`] reads fields rather than indices.
///
/// Write implementations with [`define_priors!`], which generates the parameter
/// struct, a matching struct of priors over it, and this impl from one
/// declaration. [`Params`] also implements it, as an escape hatch for models that
/// want the raw positional vector.
///
/// [`Model::simulate`]: crate::Model::simulate
/// [`define_priors!`]: crate::define_priors
pub trait Draw: Sized {
    /// Parameter names, in the order their priors are declared.
    ///
    /// Empty for types that do not name their parameters (the [`Params`] impl).
    /// The engine treats an empty list as "unnamed" and skips the arity check it
    /// would otherwise run against [`crate::Priors`].
    const NAMES: &'static [&'static str];

    /// Rebuild from the engine's positional values.
    ///
    /// Called once per accepted-or-rejected proposal, so keep it cheap. Panics if
    /// `values` does not match the declaration — that is an engine/model
    /// mismatch, not bad input, and the arity check in `initialize`/`step` should
    /// have caught it first.
    fn from_values(values: &[Value]) -> Self;
}

/// The positional escape hatch: a model can set `type Draw = Params` and keep
/// reading parameters by declaration index with [`Params::real`] /
/// [`Params::int`].
///
/// Costs a `Vec` clone per proposal, which the generated structs avoid.
impl Draw for Params {
    const NAMES: &'static [&'static str] = &[];

    fn from_values(values: &[Value]) -> Self {
        Params(values.to_vec())
    }
}
