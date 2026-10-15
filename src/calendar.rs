//! The dates a run's series' axis stands for, so viewers can draw it in
//! calendar time.

use std::fmt;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

use crate::observer::push_str;

/// Index `i` of every [`Series`](crate::Series) of a run (and every
/// [`Marker`](crate::Marker) and [`TargetRange`](crate::TargetRange) x) is
/// the date `start + i * step_days` days. Not used by the engine.
///
/// A run takes it from [`Distance::calendar`], or failing that
/// [`CalibrationModel::calendar`], into its log; [`append_calendar`] sets one
/// after the fact. A viewer offers it as the charts' calendar time.
///
/// [`Distance::calendar`]: crate::Distance::calendar
/// [`CalibrationModel::calendar`]: crate::CalibrationModel::calendar
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Calendar {
    start: String,
    step_days: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalendarError {
    /// The start is not a `yyyy-mm-dd` date that exists.
    InvalidDate(String),
    ZeroStep,
}

impl fmt::Display for CalendarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalendarError::InvalidDate(s) => write!(f, "{s:?} is not a yyyy-mm-dd date"),
            CalendarError::ZeroStep => write!(f, "a calendar step must be at least one day"),
        }
    }
}

impl std::error::Error for CalendarError {}

impl Calendar {
    /// Index 0 on `start` (`yyyy-mm-dd`), each index `step_days` after the
    /// one before.
    pub fn new(start: &str, step_days: u32) -> Result<Self, CalendarError> {
        if step_days == 0 {
            return Err(CalendarError::ZeroStep);
        }
        if !is_date(start) {
            return Err(CalendarError::InvalidDate(start.to_string()));
        }
        Ok(Self {
            start: start.to_string(),
            step_days,
        })
    }

    /// One index per day from `start`.
    pub fn daily(start: &str) -> Result<Self, CalendarError> {
        Self::new(start, 1)
    }

    /// One index per week from `start`.
    pub fn weekly(start: &str) -> Result<Self, CalendarError> {
        Self::new(start, 7)
    }

    /// The date of index 0, `yyyy-mm-dd`.
    pub fn start(&self) -> &str {
        &self.start
    }

    pub fn step_days(&self) -> u32 {
        self.step_days
    }
}

fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let digits = |r: std::ops::Range<usize>| -> Option<u32> {
        let part = &s[r];
        part.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| part.parse().ok())
            .flatten()
    };
    let (Some(year), Some(month), Some(day)) = (digits(0..4), digits(5..7), digits(8..10)) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days).contains(&day)
}

/// Appends `calendar` to the run log at `path` as a `calendar` event:
/// `{"type":"calendar","calendar":{"start":...,"step_days":...}}`. Readers
/// take the latest one over the run's.
pub fn append_calendar(path: impl AsRef<Path>, calendar: &Calendar) -> io::Result<()> {
    let mut line = String::from(r#"{"type":"calendar","calendar":"#);
    push_calendar(&mut line, Some(calendar));
    line.push('}');
    let mut file = OpenOptions::new().append(true).open(path)?;
    writeln!(file, "{line}")
}

pub(crate) fn push_calendar(out: &mut String, calendar: Option<&Calendar>) {
    match calendar {
        Some(c) => {
            out.push_str(r#"{"start":"#);
            push_str(out, &c.start);
            out.push_str(&format!(r#","step_days":{}}}"#, c.step_days));
        }
        None => out.push_str("null"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_the_start_and_step() {
        assert!(Calendar::daily("2024-02-29").is_ok());
        assert_eq!(
            Calendar::daily("2023-02-29"),
            Err(CalendarError::InvalidDate("2023-02-29".into()))
        );
        assert!(Calendar::daily("2024-13-01").is_err());
        assert!(Calendar::daily("2024-1-01").is_err());
        assert!(Calendar::daily("+024-01-01").is_err());
        assert_eq!(Calendar::new("2024-01-01", 0), Err(CalendarError::ZeroStep));
        assert_eq!(Calendar::weekly("2024-01-01").unwrap().step_days(), 7);
    }

    #[test]
    fn writes_the_log_document() {
        let mut out = String::new();
        push_calendar(&mut out, Some(&Calendar::weekly("2026-01-05").unwrap()));
        assert_eq!(out, r#"{"start":"2026-01-05","step_days":7}"#);
    }
}
