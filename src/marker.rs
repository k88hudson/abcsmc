//! Positions on the series' axis that viewers draw as vertical lines over the
//! trajectories: a day the distance scores on, when a change takes effect,
//! where the fitted window ends.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

use crate::observer::{push_f64, push_str};

/// A labelled x position, in the same index space as [`Distance::observed`]
/// and the trajectories (every [`Series`](crate::Series) of a run shares
/// it). Not used by the engine.
///
/// A run collects them from [`Distance::markers`] and
/// [`CalibrationModel::markers`] into its log; [`append_markers`] adds more
/// after the fact. A viewer draws them as vertical lines with their labels.
///
/// [`Distance::observed`]: crate::Distance::observed
/// [`Distance::markers`]: crate::Distance::markers
/// [`CalibrationModel::markers`]: crate::CalibrationModel::markers
#[derive(Clone, Debug, PartialEq)]
pub struct Marker {
    pub x: f64,
    pub label: Option<String>,
}

impl Marker {
    pub fn at(x: f64) -> Self {
        Self { x, label: None }
    }

    pub fn labelled(x: f64, label: impl Into<String>) -> Self {
        Self {
            x,
            label: Some(label.into()),
        }
    }
}

/// Appends `markers` to the run log at `path` as a `markers` event:
/// `{"type":"markers","markers":[{"x":...,"label":...}]}`. Readers merge it
/// into the run's markers: one with a label replaces an earlier marker of the
/// same label, one without is added.
pub fn append_markers(path: impl AsRef<Path>, markers: &[Marker]) -> io::Result<()> {
    let mut line = String::from(r#"{"type":"markers","markers":"#);
    push_markers(&mut line, markers);
    line.push('}');
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(file, "{line}")
}

/// Merges `added` into `markers` as readers of a `markers` event do: one
/// with a label replaces an earlier marker of the same label, one without is
/// added.
pub fn merge_markers(markers: &mut Vec<Marker>, added: Vec<Marker>) {
    for marker in added {
        match marker
            .label
            .as_ref()
            .and_then(|label| markers.iter().position(|m| m.label.as_ref() == Some(label)))
        {
            Some(i) => markers[i] = marker,
            None => markers.push(marker),
        }
    }
}

pub(crate) fn push_markers(out: &mut String, markers: &[Marker]) {
    out.push('[');
    for (i, marker) in markers.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(r#"{"x":"#);
        push_f64(out, marker.x);
        out.push_str(r#","label":"#);
        match &marker.label {
            Some(label) => push_str(out, label),
            None => out.push_str("null"),
        }
        out.push('}');
    }
    out.push(']');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_replaces_by_label_and_appends_the_rest() {
        let mut markers = vec![Marker::labelled(1.0, "a"), Marker::at(2.0)];
        merge_markers(
            &mut markers,
            vec![
                Marker::labelled(5.0, "a"),
                Marker::at(3.0),
                Marker::labelled(4.0, "b"),
            ],
        );
        assert_eq!(
            markers,
            vec![
                Marker::labelled(5.0, "a"),
                Marker::at(2.0),
                Marker::at(3.0),
                Marker::labelled(4.0, "b"),
            ]
        );
    }

    #[test]
    fn writes_labels_or_null() {
        let mut out = String::new();
        push_markers(&mut out, &[Marker::labelled(60.0, "End"), Marker::at(1.5)]);
        assert_eq!(out, r#"[{"x":60.0,"label":"End"},{"x":1.5,"label":null}]"#);
    }
}
