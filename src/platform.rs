//! The two things the engine needs from its platform that
//! `wasm32-unknown-unknown` does not have: a clock and a terminal. There the
//! clock reads zero, so durations are reported as `0.0`, and the progress bar
//! does nothing.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
pub(crate) struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub(crate) fn now() -> Self {
        Instant
    }

    pub(crate) fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

/// A terminal progress bar counting accepted particles.
#[derive(Default)]
pub(crate) struct Bar {
    #[cfg(not(target_arch = "wasm32"))]
    bar: std::sync::Mutex<Option<indicatif::ProgressBar>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Bar {
    pub(crate) fn start(&self, length: u64) {
        *self.bar.lock().unwrap() = Some(indicatif::ProgressBar::new(length));
    }

    fn with(&self, f: impl FnOnce(&indicatif::ProgressBar)) {
        if let Some(bar) = self.bar.lock().unwrap().as_ref() {
            f(bar);
        }
    }

    pub(crate) fn reset(&self) {
        self.with(|bar| bar.reset());
    }

    pub(crate) fn inc(&self) {
        self.with(|bar| bar.inc(1));
    }

    pub(crate) fn finish(&self) {
        self.with(|bar| bar.finish());
    }

    pub(crate) fn abandon(&self) {
        self.with(|bar| bar.abandon());
    }
}

#[cfg(target_arch = "wasm32")]
impl Bar {
    pub(crate) fn start(&self, _length: u64) {}
    pub(crate) fn reset(&self) {}
    pub(crate) fn inc(&self) {}
    pub(crate) fn finish(&self) {}
    pub(crate) fn abandon(&self) {}
}
