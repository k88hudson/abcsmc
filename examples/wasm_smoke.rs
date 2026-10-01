//! A calibration exported as a plain C function, built for
//! `wasm32-unknown-unknown` and called from `tests/wasm_smoke.mjs` to check
//! that the engine runs there, not only that it compiles. `plz wasm` does both.

use abcsmc::{CalibrationModel, Params, Priors, RealPrior, Silent, run, run_quantiles_with};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Estimate `x` under `Uniform(0, 10)` from `x` plus uniform noise, targeting 5.
struct Toy;

impl CalibrationModel for Toy {
    type Draw = Params;
    type Output = f64;

    fn priors(&self) -> Priors {
        Priors::new().push(RealPrior::uniform(0.0, 10.0).unwrap())
    }

    fn simulate(&self, params: &Params, seed: u64) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let noise = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        params.real(0) + noise
    }
}

fn to_target(output: &f64) -> f64 {
    (output - 5.5).abs()
}

/// Posterior mean after a quantile-scheduled run with no observer output.
#[unsafe(no_mangle)]
pub extern "C" fn posterior_mean() -> f64 {
    let generations = run_quantiles_with(&Toy, &to_target, &[0.5, 0.2, 0.05], 200, "", &Silent);
    if generations.len() != 4 {
        return f64::NAN;
    }
    let last = generations.last().unwrap();
    last.particles
        .iter()
        .map(|p| p.weight * p.params.real(0))
        .sum()
}

/// Generations completed by the default driver, which reports to stdout.
#[unsafe(no_mangle)]
pub extern "C" fn default_run_generations() -> u32 {
    run(&Toy, &to_target, &[2.5, 1.0], 100).len() as u32
}
