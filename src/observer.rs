//! Hooks into a run as it progresses: what the driver reports, and two
//! observers that consume it. [`StdoutObserver`] is the terminal progress
//! output; [`JsonlObserver`] appends the run to a `run.jsonl` event log that a
//! viewer can tail while the run is still going.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::platform::{Bar, Instant};
use crate::{CalibrationModel, Distance, Draw, Generation, Params, Value};

/// What a run declares before its first generation.
#[derive(Clone, Debug)]
pub struct RunMeta {
    /// What this run is, as given to the driver.
    pub description: Option<String>,
    /// What the run's distance measures, if it says.
    pub distance_description: Option<String>,
    pub n_particles: usize,
    /// Generations the schedule asks for; the run may stop short.
    pub n_generations: usize,
    /// The quantile schedule, when tolerances come from the prior's distances.
    pub quantiles: Option<Vec<f64>>,
    pub params: Vec<ParamMeta>,
    /// The observed series the distance fits, if it exposes one.
    pub observed: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct ParamMeta {
    pub name: String,
    pub real: bool,
}

impl RunMeta {
    pub(crate) fn new<M: CalibrationModel>(
        model: &M,
        distance: &(impl Distance<M::Output> + ?Sized),
        description: &str,
        n_particles: usize,
        n_generations: usize,
        quantiles: Option<&[f64]>,
    ) -> Self {
        let priors = model.priors();
        let params = (0..priors.len())
            .map(|i| ParamMeta {
                name: <M::Draw as Draw>::NAMES
                    .get(i)
                    .map(|name| (*name).to_string())
                    .unwrap_or_else(|| format!("param_{i}")),
                real: priors.is_real(i),
            })
            .collect();
        RunMeta {
            description: (!description.is_empty()).then(|| description.to_string()),
            distance_description: distance.description(),
            n_particles,
            n_generations,
            quantiles: quantiles.map(<[f64]>::to_vec),
            params,
            observed: distance.observed(),
        }
    }
}

/// Receives a run's events. Every method has a no-op default.
///
/// `particle_accepted` is called from worker threads while a generation is
/// filling, before weights exist; the complete weighted population arrives
/// with `generation_completed`.
pub trait RunObserver<M: CalibrationModel>: Sync {
    fn run_started(&self, _model: &M, _meta: &RunMeta) {}
    fn generation_started(&self, _generation: usize, _tolerance: f64) {}
    /// `attempts` counts this particle's own simulations, accepted or not.
    fn particle_accepted(
        &self,
        _generation: usize,
        _params: &Params,
        _distance: f64,
        _attempts: u64,
    ) {
    }
    fn generation_completed(&self, _model: &M, _generation: &Generation<M>) {}
    /// The generation exhausted `max_attempts_per_proposal`; the run stops.
    fn generation_abandoned(&self, _generation: usize) {}
    fn run_finished(&self, _model: &M, _generations: &[Generation<M>]) {}
}

impl<M: CalibrationModel, A: RunObserver<M>, B: RunObserver<M>> RunObserver<M> for (A, B) {
    fn run_started(&self, model: &M, meta: &RunMeta) {
        self.0.run_started(model, meta);
        self.1.run_started(model, meta);
    }
    fn generation_started(&self, generation: usize, tolerance: f64) {
        self.0.generation_started(generation, tolerance);
        self.1.generation_started(generation, tolerance);
    }
    fn particle_accepted(&self, generation: usize, params: &Params, distance: f64, attempts: u64) {
        self.0
            .particle_accepted(generation, params, distance, attempts);
        self.1
            .particle_accepted(generation, params, distance, attempts);
    }
    fn generation_completed(&self, model: &M, generation: &Generation<M>) {
        self.0.generation_completed(model, generation);
        self.1.generation_completed(model, generation);
    }
    fn generation_abandoned(&self, generation: usize) {
        self.0.generation_abandoned(generation);
        self.1.generation_abandoned(generation);
    }
    fn run_finished(&self, model: &M, generations: &[Generation<M>]) {
        self.0.run_finished(model, generations);
        self.1.run_finished(model, generations);
    }
}

/// Observes nothing.
pub struct Silent;

impl<M: CalibrationModel> RunObserver<M> for Silent {}

/// Terminal output: a progress bar per generation and a summary line after
/// each one.
#[derive(Default)]
pub struct StdoutObserver {
    bar: Bar,
    started: Mutex<Option<Instant>>,
}

impl StdoutObserver {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<M: CalibrationModel> RunObserver<M> for StdoutObserver {
    fn run_started(&self, _model: &M, meta: &RunMeta) {
        if let Some(description) = &meta.description {
            println!("{description}");
        }
        if let Some(distance) = &meta.distance_description {
            println!("Distance: {distance}");
        }
        *self.started.lock().unwrap() = Some(Instant::now());
        self.bar.start(meta.n_particles as u64);
    }

    fn generation_started(&self, generation: usize, tolerance: f64) {
        println!("Generation {generation}, tolerance {tolerance}");
        self.bar.reset();
    }

    fn particle_accepted(
        &self,
        _generation: usize,
        _params: &Params,
        _distance: f64,
        _attempts: u64,
    ) {
        self.bar.inc();
    }

    fn generation_completed(&self, _model: &M, generation: &Generation<M>) {
        self.bar.finish();
        let stats = &generation.stats;
        println!(
            "Acceptance ratio: {:.3}, ESS: {:.1}, Perplexity: {:.1}, Duration: {:.1}s",
            stats.acceptance_ratio, stats.ess, stats.perplexity, stats.duration_seconds
        );
    }

    fn generation_abandoned(&self, generation: usize) {
        self.bar.abandon();
        println!(
            "Generation {generation} could not be filled within max_attempts_per_proposal; stopping."
        );
    }

    fn run_finished(&self, _model: &M, generations: &[Generation<M>]) {
        let attempts: u64 = generations.iter().map(|g| g.stats.attempts).sum();
        let elapsed = self
            .started
            .lock()
            .unwrap()
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        println!(
            "Calibration: {} generations, {attempts} simulations, {elapsed:.1}s",
            generations.len()
        );
    }
}

/// Appends the run to `<runs_dir>/<run_id>/run.jsonl`, one JSON object per
/// line, flushed after every line so a reader can follow along.
///
/// Event `type`s, in order of appearance:
///
/// - `run_started`: `version`, `id`, `description` and
///   `distance_description` (each a string or `null`), `n_particles`, `n_generations`,
///   `quantiles` (or `null`), `params` (`[{name, kind}]`, kind `real` or
///   `int`), `observed` (or `null`).
/// - `generation_started`: `generation`, `tolerance` (`null` when infinite).
/// - `progress`: `generation`, running `accepted` and `attempts` counts, and
///   `batch`, the `{params, distance}` of every particle accepted since the
///   previous progress line. Written at most every `flush_every`.
/// - `generation_completed`: `generation`, `stats`, `particles`
///   (`{params, weight, distance, seed}`, the seed as a string), and
///   `trajectories` (`{particle, values}` for an even sample of at most
///   `max_trajectories` particles, empty when the model exposes none).
///
/// Every accepted particle appears in exactly one `progress` batch: the batch
/// pending when a generation completes is written before its
/// `generation_completed` line.
/// - `generation_abandoned`: `generation`.
/// - `run_finished`: `generations`.
///
/// Integer parameters are written as JSON integers, real ones as numbers.
pub struct JsonlObserver {
    path: PathBuf,
    id: String,
    writer: Mutex<BufWriter<File>>,
    live: Mutex<Live>,
    max_trajectories: usize,
    flush_every: Duration,
}

struct Live {
    generation: usize,
    accepted: u64,
    attempts: u64,
    pending: Vec<(Params, f64)>,
    last_flush: Instant,
}

impl JsonlObserver {
    /// Creates `<runs_dir>/<run_id>/run.jsonl`, truncating an existing log.
    pub fn create(runs_dir: impl AsRef<Path>, run_id: &str) -> io::Result<Self> {
        let dir = runs_dir.as_ref().join(run_id);
        fs::create_dir_all(&dir)?;
        let path = dir.join("run.jsonl");
        let file = File::create(&path)?;
        Ok(JsonlObserver {
            path,
            id: run_id.to_string(),
            writer: Mutex::new(BufWriter::new(file)),
            live: Mutex::new(Live {
                generation: 0,
                accepted: 0,
                attempts: 0,
                pending: Vec::new(),
                last_flush: Instant::now(),
            }),
            max_trajectories: 200,
            flush_every: Duration::from_millis(100),
        })
    }

    /// Like [`create`](Self::create) with a UTC timestamp (`YYYYMMDD-HHMMSS`)
    /// as the run id.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn timestamped(runs_dir: impl AsRef<Path>) -> io::Result<Self> {
        Self::create(runs_dir, &utc_timestamp())
    }

    /// Particles per generation whose trajectory is written. Default 200.
    pub fn max_trajectories(mut self, n: usize) -> Self {
        self.max_trajectories = n;
        self
    }

    /// Minimum interval between `progress` lines. Default 100 ms.
    pub fn flush_every(mut self, interval: Duration) -> Self {
        self.flush_every = interval;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn write_line(&self, line: &str) {
        let mut writer = self.writer.lock().unwrap();
        // Errors here are not worth aborting a calibration over.
        let _ = writeln!(writer, "{line}");
        let _ = writer.flush();
    }

    fn write_progress(&self, live: &mut Live) {
        let mut line = String::new();
        line.push_str(&format!(
            r#"{{"type":"progress","generation":{},"accepted":{},"attempts":{},"batch":["#,
            live.generation, live.accepted, live.attempts
        ));
        for (i, (params, distance)) in live.pending.drain(..).enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(r#"{"params":"#);
            push_params(&mut line, &params);
            line.push_str(r#","distance":"#);
            push_f64(&mut line, distance);
            line.push('}');
        }
        line.push_str("]}");
        live.last_flush = Instant::now();
        self.write_line(&line);
    }
}

impl<M: CalibrationModel> RunObserver<M> for JsonlObserver {
    fn run_started(&self, _model: &M, meta: &RunMeta) {
        let mut line = String::new();
        line.push_str(r#"{"type":"run_started","version":1,"id":"#);
        push_str(&mut line, &self.id);
        line.push_str(r#","description":"#);
        push_opt_str(&mut line, meta.description.as_deref());
        line.push_str(r#","distance_description":"#);
        push_opt_str(&mut line, meta.distance_description.as_deref());
        line.push_str(&format!(
            r#","n_particles":{},"n_generations":{},"quantiles":"#,
            meta.n_particles, meta.n_generations
        ));
        push_opt_f64s(&mut line, meta.quantiles.as_deref());
        line.push_str(r#","params":["#);
        for (i, param) in meta.params.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(r#"{"name":"#);
            push_str(&mut line, &param.name);
            line.push_str(r#","kind":"#);
            push_str(&mut line, if param.real { "real" } else { "int" });
            line.push('}');
        }
        line.push_str(r#"],"observed":"#);
        push_opt_f64s(&mut line, meta.observed.as_deref());
        line.push('}');
        self.write_line(&line);
    }

    fn generation_started(&self, generation: usize, tolerance: f64) {
        {
            let mut live = self.live.lock().unwrap();
            *live = Live {
                generation,
                accepted: 0,
                attempts: 0,
                pending: Vec::new(),
                last_flush: Instant::now(),
            };
        }
        let mut line =
            format!(r#"{{"type":"generation_started","generation":{generation},"tolerance":"#);
        push_f64(&mut line, tolerance);
        line.push('}');
        self.write_line(&line);
    }

    fn particle_accepted(&self, generation: usize, params: &Params, distance: f64, attempts: u64) {
        let mut live = self.live.lock().unwrap();
        if live.generation != generation {
            return;
        }
        live.accepted += 1;
        live.attempts += attempts;
        live.pending.push((params.clone(), distance));
        if live.last_flush.elapsed() >= self.flush_every {
            self.write_progress(&mut live);
        }
    }

    fn generation_completed(&self, model: &M, generation: &Generation<M>) {
        {
            let mut live = self.live.lock().unwrap();
            if !live.pending.is_empty() {
                self.write_progress(&mut live);
            }
        }
        let stats = &generation.stats;
        let mut line = String::new();
        line.push_str(&format!(
            r#"{{"type":"generation_completed","generation":{},"stats":{{"tolerance":"#,
            stats.generation
        ));
        push_f64(&mut line, stats.tolerance);
        line.push_str(&format!(
            r#","accepted":{},"attempts":{},"acceptance_ratio":"#,
            stats.accepted, stats.attempts
        ));
        push_f64(&mut line, stats.acceptance_ratio);
        line.push_str(r#","ess":"#);
        push_f64(&mut line, stats.ess);
        line.push_str(r#","perplexity":"#);
        push_f64(&mut line, stats.perplexity);
        line.push_str(r#","duration_seconds":"#);
        push_f64(&mut line, stats.duration_seconds);
        line.push_str(r#"},"particles":["#);
        for (i, particle) in generation.particles.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(r#"{"params":"#);
            push_params(&mut line, &particle.params);
            line.push_str(r#","weight":"#);
            push_f64(&mut line, particle.weight);
            line.push_str(r#","distance":"#);
            push_f64(&mut line, particle.distance);
            line.push_str(&format!(r#","seed":"{}"}}"#, particle.seed));
        }
        line.push_str(r#"],"trajectories":["#);
        let stride = generation
            .particles
            .len()
            .div_ceil(self.max_trajectories.max(1))
            .max(1);
        let sampled = generation
            .particles
            .iter()
            .enumerate()
            .step_by(stride)
            .take(self.max_trajectories);
        for (written, (index, particle)) in sampled.enumerate() {
            let Some(values) = model.trajectory(&particle.output) else {
                break;
            };
            if written > 0 {
                line.push(',');
            }
            line.push_str(&format!(r#"{{"particle":{index},"values":"#));
            push_f64s(&mut line, &values);
            line.push('}');
        }
        line.push_str("]}");
        self.write_line(&line);
    }

    fn generation_abandoned(&self, generation: usize) {
        self.write_line(&format!(
            r#"{{"type":"generation_abandoned","generation":{generation}}}"#
        ));
    }

    fn run_finished(&self, _model: &M, generations: &[Generation<M>]) {
        self.write_line(&format!(
            r#"{{"type":"run_finished","generations":{}}}"#,
            generations.len()
        ));
    }
}

pub(crate) fn push_f64(out: &mut String, x: f64) {
    if x.is_finite() {
        out.push_str(&format!("{x:?}"));
    } else {
        out.push_str("null");
    }
}

pub(crate) fn push_f64s(out: &mut String, xs: &[f64]) {
    out.push('[');
    for (i, &x) in xs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_f64(out, x);
    }
    out.push(']');
}

fn push_opt_f64s(out: &mut String, xs: Option<&[f64]>) {
    match xs {
        Some(xs) => push_f64s(out, xs),
        None => out.push_str("null"),
    }
}

fn push_opt_str(out: &mut String, s: Option<&str>) {
    match s {
        Some(s) => push_str(out, s),
        None => out.push_str("null"),
    }
}

fn push_params(out: &mut String, params: &Params) {
    out.push('[');
    for (i, value) in params.values().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match *value {
            Value::Real(x) => push_f64(out, x),
            Value::Int(k) => out.push_str(&k.to_string()),
        }
    }
    out.push(']');
}

pub(crate) fn push_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `YYYYMMDD-HHMMSS` in UTC, from the proleptic Gregorian civil-from-days
/// conversion.
#[cfg(not(target_arch = "wasm32"))]
fn utc_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn json_numbers_round_trip_and_non_finite_become_null() {
        let mut s = String::new();
        push_f64(&mut s, 0.1);
        push_f64(&mut s, f64::INFINITY);
        push_f64(&mut s, f64::NAN);
        push_f64(&mut s, 1e-7);
        assert_eq!(s, "0.1nullnull1e-7");
    }

    #[test]
    fn strings_are_escaped() {
        let mut s = String::new();
        push_str(&mut s, "a\"b\\c\n");
        assert_eq!(s, r#""a\"b\\c\n""#);
    }
}
