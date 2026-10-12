//! A named series on the trajectories' axis, for logs and viewers.

/// One series of a simulation or of the observed data: a model's output may
/// have several (cases and hospitalizations, say), each scored and drawn
/// against the observed series of the same name. Series share one index axis,
/// so [`Marker`](crate::Marker)s apply to all of them. Not used by the engine.
///
/// A single unnamed series is what [`CalibrationModel::trajectory`] and
/// [`Distance::observed`] give; `name` is `None` for it.
///
/// [`CalibrationModel::trajectory`]: crate::CalibrationModel::trajectory
/// [`Distance::observed`]: crate::Distance::observed
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    pub name: Option<String>,
    pub values: Vec<f64>,
}

impl Series {
    pub fn named(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: Some(name.into()),
            values,
        }
    }

    pub fn unnamed(values: Vec<f64>) -> Self {
        Self { name: None, values }
    }
}
