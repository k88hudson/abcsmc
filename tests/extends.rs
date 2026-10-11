//! `define_priors!` with `extends`: a draw of everything another draws plus
//! its own fields, laid out after the base's.

use abcsmc::{
    CalibrationModel, Draw, IntPrior, Priors, RealPrior, Silent, Value, define_priors, run_with,
};

define_priors! {
    pub struct Base / BasePriors {
        rate: Real,
        count: Int,
    }
}

define_priors! {
    /// The base plus a bonus.
    pub struct Extended / ExtendedPriors extends Base / BasePriors {
        bonus: Real,
    }
}

define_priors! {
    pub struct Twice / TwicePriors extends Extended / ExtendedPriors {
        again: Int,
    }
}

fn priors() -> ExtendedPriors {
    ExtendedPriors {
        base: BasePriors {
            rate: RealPrior::uniform(0.0, 10.0).unwrap(),
            count: IntPrior::discrete_uniform(1, 5).unwrap(),
        },
        bonus: RealPrior::uniform(0.0, 1.0).unwrap(),
    }
}

#[test]
fn an_extended_draw_lays_its_fields_out_after_the_base() {
    assert_eq!(Extended::NAMES, ["rate", "count", "bonus"]);
    assert_eq!(Twice::NAMES, ["rate", "count", "bonus", "again"]);
    let priors: Priors = priors().into();
    assert_eq!(priors.len(), 3);
    assert!(priors.is_real(0) && !priors.is_real(1) && priors.is_real(2));

    let draw = Twice::from_values(&[
        Value::Real(2.0),
        Value::Int(3),
        Value::Real(0.5),
        Value::Int(7),
    ]);
    assert_eq!(draw.base.base.rate, 2.0);
    assert_eq!(draw.base.base.count, 3);
    assert_eq!(draw.base.bonus, 0.5);
    assert_eq!(draw.again, 7);
}

#[test]
#[should_panic(expected = "no value drawn for parameter `bonus`")]
fn a_short_vector_names_the_missing_field() {
    Extended::from_values(&[Value::Real(2.0), Value::Int(3)]);
}

/// Output is `rate * count + bonus`, targeting 12.
struct Product;

impl CalibrationModel for Product {
    type Draw = Extended;
    type Output = f64;

    fn priors(&self) -> Priors {
        priors().into()
    }

    fn simulate(&self, draw: &Extended, _seed: u64) -> f64 {
        draw.base.rate * draw.base.count as f64 + draw.bonus
    }
}

#[test]
fn an_extended_draw_fits() {
    let generations = run_with(
        &Product,
        &|output: &f64| (output - 12.0).abs(),
        &[5.0, 1.0],
        50,
        "",
        &Silent,
    );
    let last = generations.last().unwrap();
    assert_eq!(last.particles.len(), 50);
    assert!(last.particles.iter().all(|p| p.distance <= 1.0));
}
