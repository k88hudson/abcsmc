//! Reading a run log back and appending projections to it.
#![cfg(feature = "serde")]

use abcsmc::{
    CalibrationModel, Draw as _, IntPrior, JsonlObserver, Marker, Posterior, Priors,
    ProjectedTrajectory, Projection, RealPrior, RunLog, append_markers, define_priors, distance_to,
    run_quantiles_with,
};
use rand::{SeedableRng, rngs::StdRng};

define_priors! {
    pub struct Draw / DrawPriors {
        rate: Real,
        count: Int,
    }
}

/// Output is `rate * count`, targeting 12.
struct Product;

impl CalibrationModel for Product {
    type Draw = Draw;
    type Output = f64;

    fn markers(&self) -> Vec<Marker> {
        vec![Marker::labelled(3.0, "model")]
    }

    fn priors(&self) -> Priors {
        DrawPriors {
            rate: RealPrior::uniform(0.0, 10.0).unwrap(),
            count: IntPrior::discrete_uniform(1, 5).unwrap(),
        }
        .into()
    }

    fn simulate(&self, draw: &Draw, _seed: u64) -> f64 {
        draw.rate * draw.count as f64
    }
}

/// A finished run in its own temp dir; the closure gets the log's path.
fn with_run<T>(
    name: &str,
    check: impl FnOnce(&std::path::Path, &[abcsmc::Generation<Product>]) -> T,
) -> T {
    let dir = std::env::temp_dir().join(format!("abcsmc-{name}-{}", std::process::id()));
    let log = JsonlObserver::create(&dir, name).unwrap();
    let path = log.path().to_path_buf();
    let distance = distance_to(
        "distance to 12",
        &[12.0],
        |output: &f64, observed: &[f64]| (output - observed[0]).abs(),
    )
    .with_markers([Marker::labelled(0.0, "scored")]);
    let generations =
        run_quantiles_with(&Product, &distance, &[0.5, 0.2], 100, "product fit", &log);
    drop(log);
    let result = check(&path, &generations);
    std::fs::remove_dir_all(&dir).unwrap();
    result
}

#[test]
fn a_log_reads_back_as_the_run_that_wrote_it() {
    with_run("readback", |path, generations| {
        let log = RunLog::read(path).unwrap();
        assert_eq!(log.id, "readback");
        assert_eq!(log.description.as_deref(), Some("product fit"));
        assert_eq!(log.distance_description.as_deref(), Some("distance to 12"));
        assert_eq!(log.n_particles, 100);
        assert!(log.finished);
        assert_eq!(log.observed, Some(vec![12.0]));
        assert_eq!(
            log.markers,
            vec![
                Marker::labelled(0.0, "scored"),
                Marker::labelled(3.0, "model")
            ],
            "the distance's markers, then the model's"
        );
        let names: Vec<(&str, bool)> = log
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.real))
            .collect();
        assert_eq!(names, [("rate", true), ("count", false)]);

        assert_eq!(log.generations.len(), generations.len());
        for (logged, generation) in log.generations.iter().zip(generations) {
            assert_eq!(logged.stats.generation, generation.stats.generation);
            assert_eq!(logged.stats.tolerance, generation.stats.tolerance);
            assert_eq!(logged.stats.attempts, generation.stats.attempts);
            assert_eq!(logged.particles.len(), generation.particles.len());
            for (p, q) in logged.particles.iter().zip(&generation.particles) {
                let (a, b): (Draw, Draw) = (p.draw(), q.draw());
                assert_eq!(a.rate, b.rate);
                assert_eq!(a.count, b.count);
                assert_eq!(p.weight, q.weight);
                assert_eq!(p.distance, q.distance);
                assert_eq!(p.seed, q.seed);
            }
        }
        assert_eq!(
            log.posterior().unwrap().stats.generation,
            generations.len() - 1
        );
    });
}

#[test]
fn logged_particles_replay_their_simulation() {
    with_run("replay", |path, generations| {
        let log = RunLog::read(path).unwrap();
        let posterior = log.posterior().unwrap();
        for (p, q) in posterior
            .particles
            .iter()
            .zip(&generations.last().unwrap().particles)
        {
            assert_eq!(Product.simulate(&p.draw(), p.seed), q.output);
        }
    });
}

#[test]
fn projections_append_and_the_latest_label_wins() {
    with_run("projection", |path, _| {
        let first = Projection {
            label: "baseline".into(),
            trajectories: vec![ProjectedTrajectory {
                weight: 1.0,
                values: vec![1.0, 2.0],
            }],
        };
        let other = Projection {
            label: "control \"measures\"".into(),
            trajectories: vec![],
        };
        let replacement = Projection {
            label: "baseline".into(),
            trajectories: vec![
                ProjectedTrajectory {
                    weight: 0.25,
                    values: vec![3.0, f64::INFINITY],
                },
                ProjectedTrajectory {
                    weight: 0.75,
                    values: vec![4.0, 5.0],
                },
            ],
        };
        first.append_to(path).unwrap();
        other.append_to(path).unwrap();
        replacement.append_to(path).unwrap();

        let log = RunLog::read(path).unwrap();
        assert!(log.finished);
        assert_eq!(log.projections.len(), 2);
        assert_eq!(log.projections[0], other);
        assert_eq!(log.projections[1].label, "baseline");
        assert_eq!(
            log.projections[1].trajectories[1],
            replacement.trajectories[1]
        );
    });
}

#[test]
fn appended_markers_replace_by_label_and_add_the_rest() {
    with_run("markers", |path, _| {
        append_markers(path, &[Marker::labelled(1.0, "scored"), Marker::at(2.0)]).unwrap();
        append_markers(path, &[Marker::labelled(4.0, "change")]).unwrap();
        let log = RunLog::read(path).unwrap();
        assert_eq!(
            log.markers,
            vec![
                Marker::labelled(1.0, "scored"),
                Marker::labelled(3.0, "model"),
                Marker::at(2.0),
                Marker::labelled(4.0, "change"),
            ]
        );
    });
}

#[test]
fn a_half_written_last_line_is_ignored_but_a_bad_middle_line_is_an_error() {
    with_run("partial", |path, generations| {
        let text = std::fs::read_to_string(path).unwrap();
        let cut = text.trim_end().rfind('\n').unwrap();
        let partial = format!("{}\n{{\"type\":\"run_fin", &text[..cut]);
        let log = RunLog::parse(&partial).unwrap();
        assert!(!log.finished);
        assert_eq!(log.generations.len(), generations.len());

        let broken = text.replacen("\n{", "\n{oops\n{", 1);
        assert!(RunLog::parse(&broken).is_err());
        assert!(RunLog::parse("").is_err());
    });
}

#[test]
fn resampling_follows_the_weights() {
    with_run("resample", |path, generations| {
        let log = RunLog::read(path).unwrap();
        let posterior = log.posterior().unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let draws = posterior.resample(20_000, &mut rng);
        assert_eq!(draws.len(), 20_000);
        let resampled_mean: f64 =
            draws.iter().map(|p| p.draw::<Draw>().rate).sum::<f64>() / draws.len() as f64;
        let weighted_mean: f64 = posterior
            .particles
            .iter()
            .map(|p| p.weight * p.draw::<Draw>().rate)
            .sum();
        assert!(
            (resampled_mean - weighted_mean).abs() < 0.1,
            "{resampled_mean} vs {weighted_mean}"
        );

        let mut rng = StdRng::seed_from_u64(1);
        let in_memory = generations.last().unwrap().resample(20_000, &mut rng);
        for (a, b) in draws.iter().zip(&in_memory) {
            assert_eq!(a.seed, b.seed);
        }
    });
}

#[test]
fn the_posterior_is_written_beside_the_log_with_the_models_documents() {
    with_run("posterior", |path, generations| {
        let log = RunLog::read(path).unwrap();
        let posterior = Posterior::of(&log).unwrap();
        assert_eq!(posterior.run, "posterior");
        assert_eq!(posterior.generation, generations.len() - 1);
        assert_eq!(posterior.params, ["rate", "count"]);
        assert_eq!(posterior.real, [true, false]);
        assert_eq!(posterior.len(), 100);
        let last = generations.last().unwrap();
        assert_eq!(posterior.draws[3][0], last.particles[3].params.real(0));
        assert_eq!(posterior.weights[3], last.particles[3].weight);
        assert_eq!(posterior.seeds[3], last.particles[3].seed);
        assert_eq!(
            posterior.column("count").unwrap()[3],
            last.particles[3].params.int(1) as f64
        );
        assert_eq!(posterior.document, None);

        // The model's view: the draw spliced into a parameter document.
        let posterior = posterior.with_documents(3, |params| {
            let draw = Draw::from_values(params.values());
            serde_json::json!({
                "rate": draw.rate,
                "fixed": 7,
                "items": [{"name": "count", "value": draw.count}],
            })
        });
        let documents = posterior.document.as_ref().unwrap();
        assert_eq!(documents.representative, 3);
        assert_eq!(documents.paths, ["items[count].value", "rate"]);
        assert_eq!(documents.parameters["fixed"], 7);
        assert_eq!(
            documents.reference,
            [posterior.draws[3][1], posterior.draws[3][0]]
        );
        assert_eq!(documents.values.len(), 100);

        posterior.write_beside(path).unwrap();
        let file = path.with_file_name("posterior.json");
        assert_eq!(Posterior::read(&file).unwrap(), posterior);
    });
}
