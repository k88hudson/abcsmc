use crate::{CalibrationModel, Draw, Params};

pub struct Particle<M: CalibrationModel> {
    pub params: Params,
    pub output: M::Output,
    pub distance: f64,
    pub weight: f64,
    /// The seed this particle's accepted proposal was simulated under.
    /// `CalibrationModel::simulate(&draw, seed)` reproduces `output` exactly.
    pub seed: u64,
}

impl<M: CalibrationModel> Particle<M> {
    /// This particle's parameters as the model's typed [`CalibrationModel::Draw`], for
    /// reading them by name at analysis time.
    pub fn draw(&self) -> M::Draw {
        M::Draw::from_values(self.params.values())
    }
}
