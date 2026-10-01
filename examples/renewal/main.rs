//! Fit a renewal epidemic model to observed symptomatic incidence with ABC-SMC.
//!
//! - [`renewal`] is the simulator alone; it knows nothing about ABC-SMC.
//! - [`calibration`] holds the priors, distance, tolerance schedule, and output.
//! - [`projection`] simulates the posterior under scenarios and adds them to
//!   the run log for the viewer.
//!
//! Run with `cargo run --example renewal`. The model's own statistical tests
//! run under `cargo test --example renewal`.

mod calibration;
mod projection;
mod renewal;

fn main() {
    calibration::fit();
}
