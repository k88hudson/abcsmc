//! A run's posterior as a file, `posterior.json` beside `run.jsonl`, for a
//! viewer or a model's front end to take up without the engine.
//!
//! The file has two parts. The first needs no model: the parameter names
//! and every particle's draw, weight, distance and seed. The second is the
//! model's view of the same particles, for a front end that runs them: a
//! reference parameter document (one particle's, in the model's own JSON),
//! the paths in that document whose numbers differ between particles, and
//! each particle's values at those paths. A consumer takes the document,
//! puts particle `i`'s `values[i]` at `paths`, and runs; a value it has
//! changed from `reference` it keeps for every particle.
//!
//! A path is the document's keys joined by `.`; an array element is
//! `[i]` by index or `[name]` by its `name` field when that is unique in
//! the array and reads back as a name (not empty, not a number, no `.`,
//! `[` or `]`): `policies[Fitted modifier].start`, `importations[0].day`.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::{Params, RunLog, Value};

/// The posterior's file name, beside the run log.
pub const POSTERIOR_FILE: &str = "posterior.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Posterior {
    /// The run's id.
    pub run: String,
    /// The generation the particles are from: the last the run completed.
    pub generation: usize,
    pub tolerance: f64,
    /// The parameter names, in the order of a draw.
    pub params: Vec<String>,
    /// Whether each parameter is real; the others are integers.
    pub real: Vec<bool>,
    /// One draw per particle, in the order of `params`; an integer
    /// parameter is a whole number.
    pub draws: Vec<Vec<f64>>,
    /// Normalized posterior weights.
    pub weights: Vec<f64>,
    pub distances: Vec<f64>,
    pub seeds: Vec<u64>,
    /// The model's view of the particles, when the model gave one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<Documents>,
}

/// The particles as values in a model's parameter document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Documents {
    /// The particle `parameters` is the document of.
    pub representative: usize,
    /// That particle's parameter document.
    pub parameters: Json,
    /// Where the particles differ: paths into `parameters`.
    pub paths: Vec<String>,
    /// `parameters`' value at each path.
    pub reference: Vec<f64>,
    /// Each particle's values at `paths`.
    pub values: Vec<Vec<f64>>,
}

impl Posterior {
    /// The last completed generation of `log`, without a model's view.
    pub fn of(log: &RunLog) -> Option<Self> {
        let last = log.posterior()?;
        Some(Self {
            run: log.id.clone(),
            generation: last.stats.generation,
            tolerance: last.stats.tolerance,
            params: log.params.iter().map(|p| p.name.clone()).collect(),
            real: log.params.iter().map(|p| p.real).collect(),
            draws: last
                .particles
                .iter()
                .map(|p| p.params.values().iter().map(|v| v.as_f64()).collect())
                .collect(),
            weights: last.particles.iter().map(|p| p.weight).collect(),
            distances: last.particles.iter().map(|p| p.distance).collect(),
            seeds: last.particles.iter().map(|p| p.seed).collect(),
            document: None,
        })
    }

    /// The number of particles.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// The index of the parameter called `name`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p == name)
    }

    /// The draws of the parameter called `name`, one per particle.
    pub fn column(&self, name: &str) -> Option<Vec<f64>> {
        let i = self.index_of(name)?;
        Some(self.draws.iter().map(|draw| draw[i]).collect())
    }

    /// A draw as the engine's values, integers where the parameter is one.
    pub fn params_of(&self, draw: &[f64]) -> Params {
        Params::new(
            draw.iter()
                .zip(&self.real)
                .map(|(&x, &real)| {
                    if real {
                        Value::Real(x)
                    } else {
                        Value::Int(x as i64)
                    }
                })
                .collect(),
        )
    }

    /// Adds the model's view: `document` gives each particle's draw as the
    /// model's parameter document, and `representative` is the particle
    /// whose document is kept whole. Every number that differs between the
    /// documents (beyond floating-point noise) becomes a path; the documents
    /// must have the same shape, so a number one of them lacks is a panic.
    pub fn with_documents(
        mut self,
        representative: usize,
        document: impl Fn(&Params) -> Json,
    ) -> Self {
        assert!(representative < self.len(), "no particle {representative}");
        let documents: Vec<Json> = self
            .draws
            .iter()
            .map(|draw| document(&self.params_of(draw)))
            .collect();
        self.document = Some(Documents::diff(documents, representative));
        self
    }

    /// Writes the posterior as `posterior.json` beside `run_log` and returns
    /// where.
    pub fn write_beside(&self, run_log: &Path) -> io::Result<PathBuf> {
        let path = run_log.with_file_name(POSTERIOR_FILE);
        self.write(&path)?;
        Ok(path)
    }

    pub fn write(&self, path: &Path) -> io::Result<()> {
        if let Some(folder) = path.parent() {
            fs::create_dir_all(folder)?;
        }
        let text = serde_json::to_string(self)?;
        fs::write(path, text)
    }

    pub fn read(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(io::Error::other)
    }
}

impl Documents {
    /// `documents[reference]` whole, with every number that differs between
    /// the documents as a path.
    fn diff(documents: Vec<Json>, reference: usize) -> Self {
        let flat: Vec<HashMap<String, f64>> = documents.iter().map(leaves).collect();
        let at = |i: usize, path: &str| {
            *flat[i].get(path).unwrap_or_else(|| {
                panic!("particle {i} has no value at {path}: the documents differ in shape")
            })
        };
        // Beyond floating-point noise: a time moved onto a calendar and
        // back need not come back exactly.
        let differs = |a: f64, b: f64| (a - b).abs() > 1e-9 * b.abs().max(1.0);
        let mut paths: Vec<String> = flat[reference]
            .iter()
            .filter(|(path, value)| (0..flat.len()).any(|i| differs(at(i, path), **value)))
            .map(|(path, _)| path.clone())
            .collect();
        paths.sort();
        let values_at = |i: usize| -> Vec<f64> { paths.iter().map(|p| at(i, p)).collect() };
        Self {
            representative: reference,
            reference: values_at(reference),
            values: (0..flat.len()).map(values_at).collect(),
            parameters: documents.into_iter().nth(reference).expect("checked"),
            paths,
        }
    }
}

/// An array element's step: its name when that is unique in the array and
/// reads back as one, else its index.
fn element_step(items: &[Json], i: usize) -> String {
    fn name_of(item: &Json) -> Option<&str> {
        item.get("name").and_then(Json::as_str)
    }
    match name_of(&items[i]) {
        Some(n)
            if !n.is_empty()
                && !n.contains(['.', '[', ']'])
                && n.parse::<usize>().is_err()
                && items.iter().filter(|item| name_of(item) == Some(n)).count() == 1 =>
        {
            n.to_owned()
        }
        _ => i.to_string(),
    }
}

/// Every number in `document` by its path.
fn leaves(document: &Json) -> HashMap<String, f64> {
    fn walk(value: &Json, path: &str, out: &mut HashMap<String, f64>) {
        match value {
            Json::Number(n) => {
                out.insert(path.to_owned(), n.as_f64().unwrap_or(f64::NAN));
            }
            Json::Object(map) => {
                for (key, v) in map {
                    let p = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    walk(v, &p, out);
                }
            }
            Json::Array(items) => {
                for (i, v) in items.iter().enumerate() {
                    walk(v, &format!("{path}[{}]", element_step(items, i)), out);
                }
            }
            _ => {}
        }
    }
    let mut out = HashMap::new();
    walk(document, "", &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn documents_differ_at_paths_named_by_key_index_or_name() {
        let doc = |rate: f64, start: f64, day: f64| {
            json!({
                "rate": rate,
                "fixed": 1,
                "policies": [{"name": "Fitted", "start": start}, {"name": "Fitted", "start": 0}],
                "importations": [{"day": day}],
                "label": "text"
            })
        };
        let docs = Documents::diff(vec![doc(1.0, 10.0, -2.0), doc(2.0, 15.0, -5.0)], 1);
        // Two policies share a name, so they go by index.
        assert_eq!(
            docs.paths,
            ["importations[0].day", "policies[0].start", "rate"]
        );
        assert_eq!(docs.reference, [-5.0, 15.0, 2.0]);
        assert_eq!(docs.values, [vec![-2.0, 10.0, 1.0], vec![-5.0, 15.0, 2.0]]);
        assert_eq!(docs.representative, 1);
        assert_eq!(docs.parameters["rate"], 2.0);

        let unique = json!({"policies": [{"name": "Fitted modifier", "start": 1.0}]});
        assert!(leaves(&unique).contains_key("policies[Fitted modifier].start"));
        let numeric = json!({"policies": [{"name": "3", "start": 1.0}]});
        assert!(leaves(&numeric).contains_key("policies[0].start"));
    }

    #[test]
    #[should_panic(expected = "particle 1 has no value at extra")]
    fn documents_of_different_shapes_are_a_bug() {
        Documents::diff(
            vec![json!({"rate": 1.0, "extra": 2.0}), json!({"rate": 2.0})],
            0,
        );
    }
}
