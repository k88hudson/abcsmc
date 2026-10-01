//! Scenario projections: posterior particles simulated again with something
//! changed, appended to the run log so the viewer draws them next to the fit.
//!
//! A scenario can change either kind of parameter:
//!
//! - a fixed setting, by simulating with a modified copy of the model;
//! - a calibrated parameter, by editing the particle's draw before simulating.
//!
//! Every scenario reuses each particle's seed, so scenarios share their random
//! numbers and differ only through what was changed.

use std::path::Path;

use abcsmc::{CalibrationModel, Generation, ProjectedTrajectory, Projection};
use rand::{SeedableRng, rngs::StdRng};

use crate::calibration::{RenewalDraw, RenewalFit};
use crate::renewal::TransmissionChange;

const N_PROJECTED: usize = 200;

struct Scenario {
    label: &'static str,
    model: RenewalFit,
    adjust_draw: fn(&mut RenewalDraw),
}

pub fn write_scenarios(
    model: &RenewalFit,
    posterior: &Generation<RenewalFit>,
    fitted_days: usize,
    log_path: &Path,
) {
    let scenarios = [
        Scenario {
            label: "Baseline",
            model: model.clone(),
            adjust_draw: |_| {},
        },
        // A fixed setting: the change starts where the data ends, so the
        // fitted window is simulated exactly as it was calibrated.
        Scenario {
            label: "Transmission -30% after the fitted window",
            model: RenewalFit {
                transmission_change: Some(TransmissionChange {
                    start: fitted_days,
                    multiplier: 0.7,
                }),
            },
            adjust_draw: |_| {},
        },
        // A calibrated parameter: this applies from day 0, so it is a
        // counterfactual history and no longer matches the observations.
        Scenario {
            label: "Counterfactual: r0 -10% from the start",
            model: model.clone(),
            adjust_draw: |draw| draw.r0 *= 0.9,
        },
    ];

    // An equally weighted posterior sample, shared by every scenario.
    let mut rng = StdRng::seed_from_u64(model.rng_seed());
    let particles = posterior.resample(N_PROJECTED, &mut rng);
    let weight = 1.0 / particles.len() as f64;

    println!(
        "Projections: {} scenarios, {N_PROJECTED} particles each",
        scenarios.len()
    );
    for scenario in scenarios {
        let trajectories = particles
            .iter()
            .map(|particle| {
                let mut draw = particle.draw();
                (scenario.adjust_draw)(&mut draw);
                let output = scenario.model.simulate(&draw, particle.seed);
                ProjectedTrajectory {
                    weight,
                    values: scenario.model.trajectory(&output).unwrap(),
                }
            })
            .collect();
        Projection {
            label: scenario.label.into(),
            trajectories,
        }
        .append_to(log_path)
        .unwrap();
    }
}
