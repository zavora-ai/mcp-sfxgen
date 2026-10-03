//! The post-processing chain (design §6 steps 2-5).
//!
//! Order: loop crossfade, trim, mono, resample, loudness normalisation, limiter, fades.
//! Quantisation to 16/24 bit happens later, in [`super::wav::write_wav`].

use serde::{Deserialize, Serialize};

use super::edit::{self, Seam};
use super::limiter::{LimiterReport, limit_true_peak};
use super::loudness::{Measurements, measure};
use super::resample::resample;
use super::wav::BitDepth;
use super::{AudioError, Clip};

/// Gain reduction above this is reported as a warning.
const LIMITER_WARN_DB: f64 = 2.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostOptions {
    pub trim_silence: bool,
    pub fade_in_ms: Option<u32>,
    pub fade_out_ms: Option<u32>,
    /// Integrated loudness target. One gain, measured before the limiter.
    pub loudness_lufs: Option<f64>,
    /// Limiter ceiling in dBTP; `None` turns the limiter off.
    pub true_peak_dbtp: Option<f64>,
    pub mono: bool,
    pub sample_rate: Option<u32>,
    pub bit_depth: BitDepth,
}

impl Default for PostOptions {
    /// The R7 default: the limiter at -1 dBTP and nothing else.
    fn default() -> Self {
        PostOptions {
            trim_silence: false,
            fade_in_ms: None,
            fade_out_ms: None,
            loudness_lufs: None,
            true_peak_dbtp: Some(-1.0),
            mono: false,
            sample_rate: None,
            bit_depth: BitDepth::Pcm16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum LoopMode {
    /// Not a loop.
    None,
    /// Local loop: `raw` was generated at `target_s + loop_extra_s(target_s)` and is crossfaded
    /// into a loop of exactly `target_s`.
    Crossfade { target_s: f64 },
    /// The backend made the loop (ElevenLabs `loop`); kept as delivered, only measured.
    Native,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PostReport {
    /// Steps that ran, in order, e.g. `"loop_crossfade"`, `"trim_silence"`, `"resample:48000"`.
    pub applied: Vec<String>,
    /// The loudness normalisation gain, when one was applied.
    pub loudness_gain_db: Option<f64>,
    pub limiter: Option<LimiterReport>,
    /// Measured on the processed clip (before quantisation).
    pub measurements: Measurements,
    /// Set for loops.
    pub seam: Option<Seam>,
    pub warnings: Vec<String>,
}

pub fn process(
    raw: &Clip,
    opts: &PostOptions,
    loop_mode: LoopMode,
) -> Result<(Clip, PostReport), AudioError> {
    if raw.is_empty() {
        return Err(AudioError::Empty);
    }
    let fades = opts.fade_in_ms.unwrap_or(0) > 0 || opts.fade_out_ms.unwrap_or(0) > 0;
    if loop_mode != LoopMode::None && (opts.trim_silence || fades) {
        return Err(AudioError::OptionConflict(
            "a loop cannot be combined with trim_silence or fades: both would break the seam"
                .into(),
        ));
    }

    let mut applied = Vec::new();
    let mut warnings = Vec::new();
    let mut loudness_gain_db = None;
    let mut limiter = None;

    // 2. Loop.
    let mut clip = match loop_mode {
        LoopMode::Crossfade { target_s } => {
            applied.push("loop_crossfade".to_string());
            edit::loop_crossfade(raw, target_s)?
        }
        LoopMode::Native => {
            applied.push("loop_native".to_string());
            raw.clone()
        }
        LoopMode::None => raw.clone(),
    };

    // 3. Trim, mono, resample, loudness.
    if opts.trim_silence {
        let trimmed = edit::trim_silence(&clip, edit::TRIM_THRESHOLD_DB, edit::TRIM_PAD_MS);
        if trimmed.frames() == clip.frames() && measure(&clip)?.sample_peak_dbfs <= -200.0 {
            warnings.push("trim_silence: the clip is silent, nothing to trim".into());
        } else {
            applied.push("trim_silence".into());
        }
        clip = trimmed;
    }
    if opts.mono && clip.channels > 1 {
        clip = edit::to_mono(&clip);
        applied.push("mono".into());
    }
    if let Some(sr) = opts.sample_rate
        && sr != clip.sample_rate
    {
        clip = resample(&clip, sr)?;
        applied.push(format!("resample:{sr}"));
    }
    if let Some(target) = opts.loudness_lufs {
        match measure(&clip)?.lufs {
            Some(current) => {
                let gain = target - current;
                edit::apply_gain_db(&mut clip, gain);
                loudness_gain_db = Some(gain);
                applied.push(format!("loudness:{target:.1}LUFS"));
            }
            None => warnings.push(
                "loudness normalisation skipped: the clip is silent or shorter than 0.4 s, so it has no integrated loudness"
                    .into(),
            ),
        }
    }

    // 4. Limiter.
    if let Some(ceiling) = opts.true_peak_dbtp {
        let (limited, report) = limit_true_peak(&clip, ceiling)?;
        clip = limited;
        applied.push(format!("limiter:{ceiling:.1}dBTP"));
        if report.max_gain_reduction_db > LIMITER_WARN_DB {
            warnings.push(format!(
                "limiter gain reduction reached {:.1} dB (over {LIMITER_WARN_DB} dB): the sound may be squashed or the loudness target out of reach",
                report.max_gain_reduction_db
            ));
        }
        if let Some(db) = report.safety_rescale_db {
            warnings.push(format!("limiter needed a safety rescale of {db:.2} dB"));
        }
        limiter = Some(report);
    }

    // 5. Fades (they only lower the level, so the ceiling holds).
    if let Some(ms) = opts.fade_in_ms.filter(|&m| m > 0) {
        edit::fade_in(&mut clip, ms as f64);
        applied.push(format!("fade_in:{ms}ms"));
    }
    if let Some(ms) = opts.fade_out_ms.filter(|&m| m > 0) {
        edit::fade_out(&mut clip, ms as f64);
        applied.push(format!("fade_out:{ms}ms"));
    }

    let measurements = measure(&clip)?;
    let seam = (loop_mode != LoopMode::None).then(|| edit::seam(&clip));
    Ok((
        clip,
        PostReport {
            applied,
            loudness_gain_db,
            limiter,
            measurements,
            seam,
            warnings,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::{noise, sine};
    use super::*;

    #[test]
    fn defaults_only_limit() {
        let c = sine(1000.0, 3.0, 0.5, 44100, 2, 0.0);
        let (out, rep) = process(&c, &PostOptions::default(), LoopMode::None).unwrap();
        assert_eq!(out.sample_rate, 44100);
        assert_eq!(out.frames(), c.frames());
        assert!(rep.measurements.true_peak_dbtp <= -1.0);
        assert_eq!(rep.applied, vec!["limiter:-1.0dBTP"]);
        assert!(rep.limiter.unwrap().max_gain_reduction_db > 3.9);
        assert!(rep.warnings.iter().any(|w| w.contains("gain reduction")));
        assert!(rep.seam.is_none());
    }

    #[test]
    fn all_steps_in_order() {
        let sr = 44100;
        let body = sine(1000.0, -20.0, 1.0, sr, 2, 0.0);
        let mut samples = Clip::silence(sr as usize / 4, sr, 2).samples;
        samples.extend(&body.samples);
        samples.extend(&Clip::silence(sr as usize / 4, sr, 2).samples);
        let raw = Clip::new(samples, sr, 2).unwrap();
        let opts = PostOptions {
            trim_silence: true,
            fade_in_ms: Some(5),
            fade_out_ms: Some(20),
            loudness_lufs: Some(-16.0),
            true_peak_dbtp: Some(-1.0),
            mono: true,
            sample_rate: Some(48000),
            bit_depth: BitDepth::Pcm24,
        };
        let (out, rep) = process(&raw, &opts, LoopMode::None).unwrap();
        assert_eq!(out.channels, 1);
        assert_eq!(out.sample_rate, 48000);
        assert!(out.duration_s() < 1.05 && out.duration_s() > 1.0);
        let names: Vec<_> = rep
            .applied
            .iter()
            .map(|s| s.split(':').next().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "trim_silence",
                "mono",
                "resample",
                "loudness",
                "limiter",
                "fade_in",
                "fade_out"
            ]
        );
        // Mono of a -20 dBFS sine reads -23 LUFS, so -16 needs about +7 dB.
        assert!((rep.loudness_gain_db.unwrap() - 7.0).abs() < 1.0);
        assert!(rep.measurements.true_peak_dbtp <= -1.0);
        assert_eq!(out.samples[0], 0.0);
    }

    #[test]
    fn loudness_target_is_hit_when_the_limiter_stays_out_of_the_way() {
        let c = sine(1000.0, -20.0, 3.0, 48000, 2, 0.0); // -20 LUFS (both channels)
        let opts = PostOptions {
            loudness_lufs: Some(-18.0),
            ..PostOptions::default()
        };
        let (_, rep) = process(&c, &opts, LoopMode::None).unwrap();
        let l = rep.measurements.lufs.unwrap();
        assert!((l - -18.0).abs() < 0.2, "{l}");
        assert_eq!(rep.limiter.unwrap().max_gain_reduction_db, 0.0);
    }

    #[test]
    fn loop_conflicts_with_trim_and_fades() {
        let c = sine(1000.0, -6.0, 2.0, 44100, 2, 0.0);
        let mut o = PostOptions {
            trim_silence: true,
            ..PostOptions::default()
        };
        assert!(matches!(
            process(&c, &o, LoopMode::Crossfade { target_s: 1.0 }),
            Err(AudioError::OptionConflict(_))
        ));
        o.trim_silence = false;
        o.fade_out_ms = Some(10);
        assert!(matches!(
            process(&c, &o, LoopMode::Native),
            Err(AudioError::OptionConflict(_))
        ));
        // Zero-length fades are not fades.
        o.fade_out_ms = Some(0);
        assert!(process(&c, &o, LoopMode::Native).is_ok());
    }

    #[test]
    fn crossfade_loop_has_exact_length_and_a_seam_report() {
        let sr = 44100;
        let d = 1.5;
        let raw = noise(
            ((d + edit::loop_extra_s(d)) * sr as f64) as usize,
            sr,
            2,
            0.5,
            9,
        );
        let (out, rep) = process(
            &raw,
            &PostOptions::default(),
            LoopMode::Crossfade { target_s: d },
        )
        .unwrap();
        assert_eq!(out.frames(), (d * sr as f64).round() as usize);
        let seam = rep.seam.unwrap();
        assert!(seam.wrap_ratio < 3.0, "{}", seam.wrap_ratio);
        assert_eq!(rep.applied[0], "loop_crossfade");
    }

    #[test]
    fn short_clip_skips_loudness_with_a_warning() {
        let c = sine(1000.0, -20.0, 0.2, 44100, 1, 0.0);
        let opts = PostOptions {
            loudness_lufs: Some(-16.0),
            ..PostOptions::default()
        };
        let (_, rep) = process(&c, &opts, LoopMode::None).unwrap();
        assert!(rep.loudness_gain_db.is_none());
        assert!(rep.warnings.iter().any(|w| w.contains("loudness")));
    }

    #[test]
    fn empty_input_is_an_error() {
        let c = Clip::silence(0, 44100, 2);
        assert!(matches!(
            process(&c, &PostOptions::default(), LoopMode::None),
            Err(AudioError::Empty)
        ));
    }

    #[test]
    fn report_serialises() {
        let c = sine(1000.0, -6.0, 0.5, 44100, 2, 0.0);
        let (_, rep) = process(&c, &PostOptions::default(), LoopMode::None).unwrap();
        let json = serde_json::to_string(&rep).unwrap();
        assert!(json.contains("true_peak_dbtp"));
    }
}
