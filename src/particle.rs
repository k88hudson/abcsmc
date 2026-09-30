use crate::{Draw, Model, Params};

pub struct Particle<M: Model> {
    pub params: Params,
    pub output: M::Output,
    pub distance: f64,
    pub weight: f64,
    /// The seed this particle's accepted proposal was simulated under.
    /// `Model::simulate(&draw, seed)` reproduces `output` exactly.
    pub seed: u64,
}

impl<M: Model> Particle<M> {
    /// This particle's parameters as the model's typed [`Model::Draw`], for
    /// reading them by name at analysis time.
    pub fn draw(&self) -> M::Draw {
        M::Draw::from_values(self.params.values())
    }
}
