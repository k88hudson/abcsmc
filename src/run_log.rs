//! Working with a finished (or still running) `run.jsonl` after the fact:
//! appending projections to it, and, with the `serde` feature, reading its
//! particles back so a projection does not have to rerun the calibration.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

use crate::observer::{push_f64, push_f64s, push_str};

/// Simulated series from posterior particles under some scenario, stored in
/// the run log as a `projection` event:
/// `{"type":"projection","label":...,"trajectories":[{"weight":...,"values":[...]}]}`.
///
/// Values are on the same axis as the model's observed series and start at the
/// same index, so a viewer can draw them past the end of the data. A later
/// projection with the same label replaces the earlier one.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub label: String,
    pub trajectories: Vec<ProjectedTrajectory>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedTrajectory {
    /// The posterior weight of the particle this was simulated from.
    pub weight: f64,
    pub values: Vec<f64>,
}

impl Projection {
    /// Appends this projection to the run log at `path` as one line.
    pub fn append_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut line = String::from(r#"{"type":"projection","label":"#);
        push_str(&mut line, &self.label);
        line.push_str(r#","trajectories":["#);
        for (i, trajectory) in self.trajectories.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(r#"{"weight":"#);
            push_f64(&mut line, trajectory.weight);
            line.push_str(r#","values":"#);
            push_f64s(&mut line, &trajectory.values);
            line.push('}');
        }
        line.push_str("]}");
        let mut file = OpenOptions::new().append(true).open(path)?;
        writeln!(file, "{line}")
    }
}

#[cfg(feature = "serde")]
pub use reader::*;

#[cfg(feature = "serde")]
mod reader {
    use std::fs;
    use std::io;
    use std::path::Path;

    use serde::Deserialize;

    use super::{ProjectedTrajectory, Projection};
    use crate::{Draw, GenerationStats, ParamMeta, ParamPrior, Params, Value, resample_indices};

    /// A run log read back from disk: what the run declared, every completed
    /// generation's weighted particles, and any projections appended since.
    #[derive(Clone, Debug)]
    pub struct RunLog {
        pub id: String,
        /// The run's description, if it was given one.
        pub description: Option<String>,
        /// What the run's distance measures, if it said.
        pub distance_description: Option<String>,
        pub n_particles: usize,
        pub params: Vec<ParamMeta>,
        pub observed: Option<Vec<f64>>,
        pub generations: Vec<LoggedGeneration>,
        /// Whether the log records the run's end.
        pub finished: bool,
        pub projections: Vec<Projection>,
    }

    #[derive(Clone, Debug)]
    pub struct LoggedGeneration {
        pub stats: GenerationStats,
        pub particles: Vec<LoggedParticle>,
    }

    /// A particle without its simulation output. `(draw, seed)` replays the
    /// output through `CalibrationModel::simulate`.
    #[derive(Clone, Debug)]
    pub struct LoggedParticle {
        pub params: Params,
        pub weight: f64,
        pub distance: f64,
        pub seed: u64,
    }

    impl LoggedParticle {
        /// The parameters as a model's typed draw.
        pub fn draw<D: Draw>(&self) -> D {
            D::from_values(self.params.values())
        }
    }

    impl LoggedGeneration {
        /// `n` particles drawn with replacement in proportion to their weights.
        pub fn resample(&self, n: usize, rng: &mut impl rand::Rng) -> Vec<&LoggedParticle> {
            resample_indices(self.particles.iter().map(|p| p.weight), n, rng)
                .into_iter()
                .map(|i| &self.particles[i])
                .collect()
        }
    }

    #[derive(Deserialize)]
    struct ParamSpec {
        name: String,
        kind: String,
        // Absent in logs written before priors were logged.
        prior: Option<ParamPrior>,
    }

    #[derive(Deserialize)]
    struct StatsSpec {
        tolerance: Option<f64>,
        accepted: usize,
        attempts: u64,
        acceptance_ratio: f64,
        ess: f64,
        perplexity: f64,
        duration_seconds: f64,
    }

    #[derive(Deserialize)]
    struct ParticleSpec {
        params: Vec<f64>,
        weight: f64,
        /// `null` when the distance was not finite.
        distance: Option<f64>,
        seed: String,
    }

    /// Non-finite values are written as `null` and read back as NaN.
    #[derive(Deserialize)]
    struct TrajectorySpec {
        weight: f64,
        values: Vec<Option<f64>>,
    }

    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Event {
        RunStarted {
            id: String,
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            distance_description: Option<String>,
            n_particles: usize,
            params: Vec<ParamSpec>,
            observed: Option<Vec<Option<f64>>>,
        },
        GenerationCompleted {
            generation: usize,
            stats: StatsSpec,
            particles: Vec<ParticleSpec>,
        },
        RunFinished {},
        Projection {
            label: String,
            trajectories: Vec<TrajectorySpec>,
        },
        #[serde(other)]
        Other,
    }

    fn invalid(message: impl Into<String>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, message.into())
    }

    impl RunLog {
        pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
            Self::parse(&fs::read_to_string(path)?)
        }

        /// Parses the text of a run log. A final line that does not parse is
        /// taken to be one still being written and is ignored.
        pub fn parse(text: &str) -> io::Result<Self> {
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            let mut log: Option<RunLog> = None;
            for (index, line) in lines.iter().enumerate() {
                let event: Event = match serde_json::from_str(line) {
                    Ok(event) => event,
                    Err(_) if index + 1 == lines.len() => break,
                    Err(error) => return Err(invalid(format!("line {}: {error}", index + 1))),
                };
                match event {
                    Event::RunStarted {
                        id,
                        description,
                        distance_description,
                        n_particles,
                        params,
                        observed,
                    } => {
                        let params = params
                            .into_iter()
                            .map(|p| match p.kind.as_str() {
                                "real" => Ok(ParamMeta {
                                    name: p.name,
                                    real: true,
                                    prior: p.prior,
                                }),
                                "int" => Ok(ParamMeta {
                                    name: p.name,
                                    real: false,
                                    prior: p.prior,
                                }),
                                other => Err(invalid(format!("unknown parameter kind {other}"))),
                            })
                            .collect::<io::Result<Vec<_>>>()?;
                        log = Some(RunLog {
                            id,
                            description,
                            distance_description,
                            n_particles,
                            params,
                            observed: observed.map(|series| {
                                series.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect()
                            }),
                            generations: Vec::new(),
                            finished: false,
                            projections: Vec::new(),
                        });
                    }
                    Event::Other => {}
                    event => {
                        let log = log
                            .as_mut()
                            .ok_or_else(|| invalid("the log does not start with run_started"))?;
                        match event {
                            Event::GenerationCompleted {
                                generation,
                                stats,
                                particles,
                            } => {
                                let particles = particles
                                    .into_iter()
                                    .map(|p| logged_particle(p, &log.params))
                                    .collect::<io::Result<Vec<_>>>()?;
                                log.generations.push(LoggedGeneration {
                                    stats: GenerationStats {
                                        generation,
                                        tolerance: stats.tolerance.unwrap_or(f64::INFINITY),
                                        accepted: stats.accepted,
                                        attempts: stats.attempts,
                                        acceptance_ratio: stats.acceptance_ratio,
                                        ess: stats.ess,
                                        perplexity: stats.perplexity,
                                        duration_seconds: stats.duration_seconds,
                                    },
                                    particles,
                                });
                            }
                            Event::RunFinished {} => log.finished = true,
                            Event::Projection {
                                label,
                                trajectories,
                            } => {
                                log.projections.retain(|p| p.label != label);
                                log.projections.push(Projection {
                                    label,
                                    trajectories: trajectories
                                        .into_iter()
                                        .map(|t| ProjectedTrajectory {
                                            weight: t.weight,
                                            values: t
                                                .values
                                                .into_iter()
                                                .map(|v| v.unwrap_or(f64::NAN))
                                                .collect(),
                                        })
                                        .collect(),
                                });
                            }
                            Event::RunStarted { .. } | Event::Other => unreachable!(),
                        }
                    }
                }
            }
            log.ok_or_else(|| invalid("the log has no run_started event"))
        }

        /// The last completed generation.
        pub fn posterior(&self) -> Option<&LoggedGeneration> {
            self.generations.last()
        }
    }

    fn logged_particle(spec: ParticleSpec, params: &[ParamMeta]) -> io::Result<LoggedParticle> {
        if spec.params.len() != params.len() {
            return Err(invalid(format!(
                "a particle has {} parameters but the run declared {}",
                spec.params.len(),
                params.len()
            )));
        }
        let values = spec
            .params
            .iter()
            .zip(params)
            .map(|(&x, meta)| {
                if meta.real {
                    Value::Real(x)
                } else {
                    Value::Int(x as i64)
                }
            })
            .collect();
        Ok(LoggedParticle {
            params: Params::new(values),
            weight: spec.weight,
            distance: spec.distance.unwrap_or(f64::INFINITY),
            seed: spec
                .seed
                .parse()
                .map_err(|_| invalid(format!("seed {} is not a u64", spec.seed)))?,
        })
    }
}
