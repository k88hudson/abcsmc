use crate::{Draw, PerturbationKernel, Priors, VarianceAdapter, default_adapter, default_kernel};

/// A model to be fit with ABC-SMC: it declares priors and simulates. How a
/// simulation is scored against data is a separate [`Distance`](crate::Distance)
/// passed to the run. The engine derives sampling, prior density, and the
/// perturbation kernel from the priors.
pub trait CalibrationModel {
    /// Calibrated parameters as a typed struct, usually from [`define_priors!`].
    /// `Params` reads them positionally instead.
    ///
    /// [`define_priors!`]: crate::define_priors
    type Draw: Draw;

    /// Simulated output retained on each accepted particle.
    type Output;

    /// Priors in the same order as [`CalibrationModel::Draw`] declares its fields.
    fn priors(&self) -> Priors;

    /// Simulate under `seed`. `(draw, seed)` fully determines the output, so a
    /// particle can be replayed from the seed retained on it.
    fn simulate(&self, draw: &Self::Draw, seed: u64) -> Self::Output;

    fn perturbation_kernel(&self) -> Box<dyn PerturbationKernel> {
        default_kernel(&self.priors())
    }

    fn variance_adapter(&self) -> Box<dyn VarianceAdapter> {
        default_adapter()
    }

    /// Bounds both the search for an in-support perturbed proposal and the search
    /// for an accepted simulation.
    fn max_attempts_per_proposal(&self) -> u64 {
        i32::MAX as u64
    }

    fn rng_seed(&self) -> u64 {
        8675309
    }

    /// An output's series on the same axis as [`Distance::observed`], for
    /// viewers. Not used by the engine.
    ///
    /// [`Distance::observed`]: crate::Distance::observed
    fn trajectory(&self, _output: &Self::Output) -> Option<Vec<f64>> {
        None
    }
}
