//! The [`Distance`] a run accepts particles by, and common distances between
//! a simulated and an observed series to build one from.
//!
//! A run takes any [`Distance`] over the model's output: a plain closure, a
//! [`distance_fn`] that adds a description, a [`distance_to`] an observed
//! series (which logs and viewers then show), or a type of your own.
//!
//! The series helpers take two equal-length slices of any built-in numeric
//! type (counts and reals can be mixed) and panic if the lengths differ: slice
//! a longer simulation down to the fitted window first, e.g.
//! `l1(&simulated[..observed.len()], &observed)`.
//!
//! - [`l1`], [`l2`], [`linf`]: the usual norms of the pointwise difference.
//! - [`mean_absolute_error`], [`root_mean_squared_error`]: `l1` and `l2`
//!   scaled by length, so tolerances are comparable across window sizes.
//! - [`total_difference`]: compares totals only, ignoring timing.
//! - [`relative_l1`]: `l1` as a fraction of the observed total.

/// How far a simulated output is from the observed data. A particle is
/// accepted when this is within the generation's tolerance.
///
/// The value must not depend on the generation. Evaluated from several
/// threads at once, hence `Sync`. The run functions also accept a
/// `&dyn Distance<O>`, for a distance chosen at run time.
pub trait Distance<O>: Sync {
    fn distance(&self, output: &O) -> f64;

    /// What the distance measures, in a sentence, for run logs and viewers.
    fn description(&self) -> Option<String> {
        None
    }

    /// The observed series being fitted, for viewers to draw under simulated
    /// trajectories. Not used by the engine.
    fn observed(&self) -> Option<Vec<f64>> {
        None
    }
}

impl<O, F: Fn(&O) -> f64 + Sync> Distance<O> for F {
    fn distance(&self, output: &O) -> f64 {
        self(output)
    }
}

/// A closure with a description. Build one with [`distance_fn`].
pub struct DistanceFn<F> {
    description: String,
    f: F,
}

/// A [`Distance`] from a closure and a description of what it measures.
pub fn distance_fn<O, F>(description: impl Into<String>, f: F) -> DistanceFn<F>
where
    F: Fn(&O) -> f64 + Sync,
{
    DistanceFn {
        description: description.into(),
        f,
    }
}

impl<O, F: Fn(&O) -> f64 + Sync> Distance<O> for DistanceFn<F> {
    fn distance(&self, output: &O) -> f64 {
        (self.f)(output)
    }

    fn description(&self) -> Option<String> {
        Some(self.description.clone())
    }
}

/// A distance to an observed series that it owns. Build one with
/// [`distance_to`].
pub struct DistanceTo<B, F> {
    description: String,
    observed: Vec<B>,
    f: F,
}

/// A [`Distance`] to `observed`: the closure is handed each output together
/// with the observed series, and the same series is what logs and viewers
/// show, so the two cannot disagree.
///
/// ```
/// use abcsmc::{distance, distance_to};
///
/// let observed = [0u64, 1, 3, 4];
/// let l1 = distance_to(
///     "L1 over the observed days",
///     &observed,
///     |simulated: &Vec<u64>, observed: &[u64]| {
///         distance::l1(&simulated[..observed.len()], observed)
///     },
/// );
/// ```
pub fn distance_to<O, B, F>(
    description: impl Into<String>,
    observed: &[B],
    f: F,
) -> DistanceTo<B, F>
where
    B: Numeric,
    F: Fn(&O, &[B]) -> f64 + Sync,
{
    DistanceTo {
        description: description.into(),
        observed: observed.to_vec(),
        f,
    }
}

impl<O, B, F> Distance<O> for DistanceTo<B, F>
where
    B: Numeric + Sync,
    F: Fn(&O, &[B]) -> f64 + Sync,
{
    fn distance(&self, output: &O) -> f64 {
        (self.f)(output, &self.observed)
    }

    fn description(&self) -> Option<String> {
        Some(self.description.clone())
    }

    fn observed(&self) -> Option<Vec<f64>> {
        Some(self.observed.iter().map(|x| x.to_f64()).collect())
    }
}

/// A numeric type the distance helpers accept.
pub trait Numeric: Copy {
    fn to_f64(self) -> f64;
}

macro_rules! impl_numeric {
    ($($t:ty),*) => {
        $(impl Numeric for $t {
            fn to_f64(self) -> f64 {
                self as f64
            }
        })*
    };
}

impl_numeric!(f64, f32, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

fn differences<'a, A: Numeric, B: Numeric>(
    simulated: &'a [A],
    observed: &'a [B],
) -> impl Iterator<Item = f64> + 'a {
    assert_eq!(
        simulated.len(),
        observed.len(),
        "distance between series of different lengths"
    );
    simulated
        .iter()
        .zip(observed)
        .map(|(a, b)| a.to_f64() - b.to_f64())
}

fn total<A: Numeric>(xs: &[A]) -> f64 {
    xs.iter().map(|x| x.to_f64()).sum()
}

/// Sum of absolute differences.
pub fn l1<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    differences(simulated, observed).map(f64::abs).sum()
}

/// Euclidean distance: the square root of the sum of squared differences.
pub fn l2<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    sum_of_squares(simulated, observed).sqrt()
}

/// Largest absolute difference. Zero for empty series.
pub fn linf<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    differences(simulated, observed)
        .map(f64::abs)
        .fold(0.0, f64::max)
}

/// Sum of squared differences.
pub fn sum_of_squares<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    differences(simulated, observed).map(|d| d * d).sum()
}

/// [`l1`] divided by the length. Zero for empty series.
pub fn mean_absolute_error<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    match observed.len() {
        0 => 0.0,
        n => l1(simulated, observed) / n as f64,
    }
}

/// Square root of the mean squared difference. Zero for empty series.
pub fn root_mean_squared_error<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    match observed.len() {
        0 => 0.0,
        n => (sum_of_squares(simulated, observed) / n as f64).sqrt(),
    }
}

/// Absolute difference between the two totals: how far apart the overall
/// sizes are, regardless of when the values occur.
pub fn total_difference<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    assert_eq!(
        simulated.len(),
        observed.len(),
        "distance between series of different lengths"
    );
    (total(simulated) - total(observed)).abs()
}

/// [`l1`] as a fraction of the observed total, so it is unitless. Infinite
/// when the observed total is zero and the series differ.
pub fn relative_l1<A: Numeric, B: Numeric>(simulated: &[A], observed: &[B]) -> f64 {
    let distance = l1(simulated, observed);
    if distance == 0.0 {
        return 0.0;
    }
    distance / total(observed).abs()
}

#[cfg(test)]
mod test {
    use super::*;

    const SIMULATED: [u64; 4] = [1, 5, 2, 8];
    const OBSERVED: [f64; 4] = [2.0, 3.0, 2.0, 4.0];
    // Differences: -1, 2, 0, 4.

    #[test]
    fn closures_and_described_closures_are_distances() {
        fn evaluate(distance: &impl Distance<f64>) -> (f64, Option<String>, Option<Vec<f64>>) {
            (
                distance.distance(&3.0),
                distance.description(),
                distance.observed(),
            )
        }
        assert_eq!(evaluate(&|x: &f64| (x - 5.0).abs()), (2.0, None, None));
        assert_eq!(
            evaluate(&distance_fn("distance to 5", |x: &f64| (x - 5.0).abs())),
            (2.0, Some("distance to 5".into()), None)
        );
    }

    #[test]
    fn distance_to_hands_the_closure_the_series_it_reports() {
        let distance = distance_to(
            "L1 to the observed counts",
            &[5u64, 1],
            |output: &Vec<f64>, observed: &[u64]| l1(output, observed),
        );
        assert_eq!(distance.distance(&vec![3.0, 2.0]), 3.0);
        assert_eq!(
            Distance::<Vec<f64>>::observed(&distance),
            Some(vec![5.0, 1.0])
        );
        assert_eq!(
            Distance::<Vec<f64>>::description(&distance).as_deref(),
            Some("L1 to the observed counts")
        );
    }

    #[test]
    fn norms_match_closed_forms() {
        assert_eq!(l1(&SIMULATED, &OBSERVED), 7.0);
        assert_eq!(sum_of_squares(&SIMULATED, &OBSERVED), 21.0);
        assert_eq!(l2(&SIMULATED, &OBSERVED), 21f64.sqrt());
        assert_eq!(linf(&SIMULATED, &OBSERVED), 4.0);
    }

    #[test]
    fn scaled_and_total_distances_match_closed_forms() {
        assert_eq!(mean_absolute_error(&SIMULATED, &OBSERVED), 7.0 / 4.0);
        assert_eq!(
            root_mean_squared_error(&SIMULATED, &OBSERVED),
            (21.0f64 / 4.0).sqrt()
        );
        assert_eq!(total_difference(&SIMULATED, &OBSERVED), 5.0);
        assert_eq!(relative_l1(&SIMULATED, &OBSERVED), 7.0 / 11.0);
    }

    #[test]
    fn identical_series_are_at_distance_zero() {
        let zeros = [0u64; 3];
        for distance in [
            l1(&zeros, &zeros),
            l2(&zeros, &zeros),
            linf(&zeros, &zeros),
            mean_absolute_error(&zeros, &zeros),
            root_mean_squared_error(&zeros, &zeros),
            total_difference(&zeros, &zeros),
            relative_l1(&zeros, &zeros),
        ] {
            assert_eq!(distance, 0.0);
        }
    }

    #[test]
    fn empty_series_are_at_distance_zero() {
        let empty: [f64; 0] = [];
        assert_eq!(linf(&empty, &empty), 0.0);
        assert_eq!(mean_absolute_error(&empty, &empty), 0.0);
        assert_eq!(root_mean_squared_error(&empty, &empty), 0.0);
    }

    #[test]
    fn relative_l1_is_infinite_against_an_all_zero_observation() {
        assert_eq!(relative_l1(&[1.0, 2.0], &[0.0, 0.0]), f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "different lengths")]
    fn mismatched_lengths_panic() {
        l1(&[1.0, 2.0], &[1.0]);
    }
}
