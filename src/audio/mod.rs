//! Audio post-processing (design §6): clip type, WAV I/O, loudness, edits, resampling,
//! true-peak limiter, peaks and the pipeline that chains them.
//!
//! Everything works on interleaved `f32`. Quantisation happens only in [`wav::write_wav`].

pub mod clip;
pub mod edit;
pub mod limiter;
pub mod loudness;
pub mod peaks;
pub mod pipeline;
pub mod resample;
pub mod wav;

pub use clip::Clip;
pub use edit::Seam;
pub use limiter::LimiterReport;
pub use loudness::Measurements;
pub use pipeline::{LoopMode, PostOptions, PostReport, process};
pub use wav::BitDepth;

/// Everything that can go wrong in the audio code.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("audio i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("WAV error: {0}")]
    Wav(#[from] hound::Error),
    /// Raw PCM whose byte length is not a whole number of frames.
    #[error(
        "raw PCM length {len} bytes is not a whole number of frames ({bytes_per_frame} bytes per frame)"
    )]
    BadPcmLength { len: usize, bytes_per_frame: usize },
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),
    #[error("invalid audio: {0}")]
    Invalid(String),
    #[error("the clip is empty")]
    Empty,
    #[error("resampling failed: {0}")]
    Resample(String),
    #[error("loudness measurement failed: {0}")]
    Loudness(String),
    /// Caller asked for options that cannot be combined (e.g. a loop with fades).
    #[error("conflicting options: {0}")]
    OptionConflict(String),
}

/// Dezibel helpers shared by the modules.
pub(crate) fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Linear amplitude to dB, floored at -200 dB so reports stay finite (JSON has no -inf).
pub(crate) fn lin_to_db(lin: f64) -> f64 {
    if lin <= 1e-10 {
        -200.0
    } else {
        20.0 * lin.log10()
    }
}
