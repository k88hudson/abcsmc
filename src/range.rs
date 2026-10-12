//! Ranges a series is fitted to, for viewers to draw as whiskers over the
//! trajectories: a floor and ceiling the simulated value should fall between
//! at a position, when the distance scores against a range rather than a
//! point.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

use crate::observer::{push_f64, push_opt_str};

/// A target range at an x position of one series, in the same index space
/// as [`Distance::observed`] and the trajectories. Not used by the engine.
///
/// A run collects them from [`Distance::ranges`] into its log;
/// [`append_ranges`] adds more after the fact. A viewer draws each as a
/// whisker from `lower` to `upper` on the charts of the series named
/// `series` (`None` for the unnamed series), with a dot at `value` when the
/// range has a point estimate.
///
/// [`Distance::observed`]: crate::Distance::observed
/// [`Distance::ranges`]: crate::Distance::ranges
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRange {
    pub x: f64,
    pub lower: f64,
    pub upper: f64,
    /// A point estimate within the range, if there is one to show.
    pub value: Option<f64>,
    /// The series the range is on; `None` for the unnamed series.
    pub series: Option<String>,
}

impl TargetRange {
    /// A range on the unnamed series.
    pub fn new(x: f64, lower: f64, upper: f64) -> Self {
        Self {
            x,
            lower,
            upper,
            value: None,
            series: None,
        }
    }

    /// A range on the series called `series`.
    pub fn on(series: impl Into<String>, x: f64, lower: f64, upper: f64) -> Self {
        Self {
            series: Some(series.into()),
            ..Self::new(x, lower, upper)
        }
    }

    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }
}

/// Appends `ranges` to the run log at `path` as a `ranges` event:
/// `{"type":"ranges","ranges":[{"x":...,"lower":...,"upper":...,"value":...,"series":...}]}`.
/// Readers merge it into the run's ranges: one on the same series at the same
/// x replaces an earlier range, any other is added.
pub fn append_ranges(path: impl AsRef<Path>, ranges: &[TargetRange]) -> io::Result<()> {
    let mut line = String::from(r#"{"type":"ranges","ranges":"#);
    push_ranges(&mut line, ranges);
    line.push('}');
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(file, "{line}")
}

/// Merges `added` into `ranges` as readers of a `ranges` event do: one on
/// the same series at the same x replaces an earlier range, any other is
/// added.
pub fn merge_ranges(ranges: &mut Vec<TargetRange>, added: Vec<TargetRange>) {
    for range in added {
        match ranges
            .iter()
            .position(|r| r.series == range.series && r.x == range.x)
        {
            Some(i) => ranges[i] = range,
            None => ranges.push(range),
        }
    }
}

pub(crate) fn push_ranges(out: &mut String, ranges: &[TargetRange]) {
    out.push('[');
    for (i, range) in ranges.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(r#"{"x":"#);
        push_f64(out, range.x);
        out.push_str(r#","lower":"#);
        push_f64(out, range.lower);
        out.push_str(r#","upper":"#);
        push_f64(out, range.upper);
        out.push_str(r#","value":"#);
        match range.value {
            Some(value) => push_f64(out, value),
            None => out.push_str("null"),
        }
        out.push_str(r#","series":"#);
        push_opt_str(out, range.series.as_deref());
        out.push('}');
    }
    out.push(']');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_replaces_by_series_and_x_and_appends_the_rest() {
        let mut ranges = vec![
            TargetRange::on("a", 1.0, 0.0, 1.0),
            TargetRange::new(1.0, 0.0, 1.0),
        ];
        merge_ranges(
            &mut ranges,
            vec![
                TargetRange::on("a", 1.0, 5.0, 6.0),
                TargetRange::new(2.0, 0.0, 1.0),
                TargetRange::on("b", 1.0, 0.0, 1.0),
            ],
        );
        assert_eq!(
            ranges,
            vec![
                TargetRange::on("a", 1.0, 5.0, 6.0),
                TargetRange::new(1.0, 0.0, 1.0),
                TargetRange::new(2.0, 0.0, 1.0),
                TargetRange::on("b", 1.0, 0.0, 1.0),
            ]
        );
    }

    #[test]
    fn writes_values_and_series_or_null() {
        let mut out = String::new();
        push_ranges(
            &mut out,
            &[
                TargetRange::on("Cases", 7.0, 10.0, 12.5).with_value(11.0),
                TargetRange::new(1.5, 0.0, 1.0),
            ],
        );
        assert_eq!(
            out,
            concat!(
                r#"[{"x":7.0,"lower":10.0,"upper":12.5,"value":11.0,"series":"Cases"},"#,
                r#"{"x":1.5,"lower":0.0,"upper":1.0,"value":null,"series":null}]"#
            )
        );
    }
}
