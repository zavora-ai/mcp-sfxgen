//! Loudness and peak measurement (`ebur128`).

use ebur128::{EbuR128, Mode};
use serde::Serialize;

use super::{AudioError, Clip, lin_to_db};

/// Integrated loudness needs at least one 400 ms gating block.
const MIN_INTEGRATED_S: f64 = 0.4;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measurements {
    pub duration_s: f64,
    /// Integrated loudness (BS.1770-4 / EBU R128 gated). `None` under 0.4 s or for silence.
    pub lufs: Option<f64>,
    /// Highest momentary (400 ms) loudness. Under 0.4 s the window is scaled to the clip
    /// length, so short clips still get a usable number. `None` for silence.
    pub momentary_max_lufs: Option<f64>,
    /// True peak over all channels, dBTP. Floored at -200 for digital silence.
    pub true_peak_dbtp: f64,
    /// Sample peak over all channels, dBFS. Floored at -200 for digital silence.
    pub sample_peak_dbfs: f64,
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

fn loud_err(e: ebur128::Error) -> AudioError {
    AudioError::Loudness(e.to_string())
}

/// Measure a clip. Never fails for a valid clip; an empty clip gives the silent values.
pub fn measure(clip: &Clip) -> Result<Measurements, AudioError> {
    let duration_s = clip.duration_s();
    if clip.is_empty() {
        return Ok(Measurements {
            duration_s,
            lufs: None,
            momentary_max_lufs: None,
            true_peak_dbtp: -200.0,
            sample_peak_dbfs: -200.0,
        });
    }
    let ch = clip.channels as usize;
    let mut meter = EbuR128::new(
        clip.channels as u32,
        clip.sample_rate,
        Mode::I | Mode::M | Mode::TRUE_PEAK,
    )
    .map_err(loud_err)?;

    // Feed in 100 ms hops and read the momentary loudness after each one.
    let hop = ((clip.sample_rate as usize) / 10).max(1) * ch;
    let mut momentary_max = f64::NEG_INFINITY;
    for chunk in clip.samples.chunks(hop) {
        meter.add_frames_f32(chunk).map_err(loud_err)?;
        let m = meter.loudness_momentary().map_err(loud_err)?;
        if m > momentary_max {
            momentary_max = m;
        }
    }
    // The ring buffer is zero-filled, so a clip shorter than the 400 ms window reads diluted
    // by the zeros. Scale back to the clip's own mean square.
    if duration_s < MIN_INTEGRATED_S && momentary_max.is_finite() {
        momentary_max += 10.0 * (MIN_INTEGRATED_S / duration_s).log10();
    }

    let lufs = if duration_s < MIN_INTEGRATED_S {
        None
    } else {
        finite(meter.loudness_global().map_err(loud_err)?)
    };

    let mut tp = 0f64;
    let mut sp = 0f64;
    for c in 0..clip.channels as u32 {
        tp = tp.max(meter.true_peak(c).map_err(loud_err)?);
        sp = sp.max(meter.sample_peak(c).map_err(loud_err)?);
    }
    Ok(Measurements {
        duration_s,
        lufs,
        momentary_max_lufs: finite(momentary_max),
        true_peak_dbtp: lin_to_db(tp),
        sample_peak_dbfs: lin_to_db(sp),
    })
}

/// True peak of a clip in linear amplitude, via `ebur128`.
pub fn true_peak_linear(clip: &Clip) -> Result<f64, AudioError> {
    if clip.is_empty() {
        return Ok(0.0);
    }
    let mut meter =
        EbuR128::new(clip.channels as u32, clip.sample_rate, Mode::TRUE_PEAK).map_err(loud_err)?;
    meter.add_frames_f32(&clip.samples).map_err(loud_err)?;
    let mut tp = 0f64;
    for c in 0..clip.channels as u32 {
        tp = tp.max(meter.true_peak(c).map_err(loud_err)?);
    }
    Ok(tp)
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::sine;
    use super::*;

    /// A 1 kHz sine of `dbfs` peak in the left channel only, the right channel silent.
    fn left_only(dbfs: f64, secs: f64, sr: u32) -> Clip {
        let mut c = sine(1000.0, dbfs, secs, sr, 2, 0.0);
        for f in c.samples.chunks_exact_mut(2) {
            f[1] = 0.0;
        }
        c
    }

    #[test]
    fn minus20_sine_reads_minus23_lufs() {
        // BS.1770: a 1 kHz sine of -20 dBFS peak is -23 dBFS RMS and, in one channel, -23 LUFS
        // (the K-weighting gain at 1 kHz cancels the -0.691 offset). The same sine in both
        // channels adds the two powers: -20 LUFS.
        let m = measure(&left_only(-20.0, 5.0, 48000)).unwrap();
        let l = m.lufs.unwrap();
        assert!((l - -23.0).abs() < 0.5, "one channel: {l}");
        assert!((m.sample_peak_dbfs - -20.0).abs() < 0.05);
        assert!((m.true_peak_dbtp - -20.0).abs() < 0.1);
        assert!((m.momentary_max_lufs.unwrap() - -23.0).abs() < 0.5);
        assert!((m.duration_s - 5.0).abs() < 1e-9);
        let both = measure(&sine(1000.0, -20.0, 5.0, 48000, 2, 0.0)).unwrap();
        assert!((both.lufs.unwrap() - -20.0).abs() < 0.5, "{:?}", both.lufs);
    }

    #[test]
    fn short_clip_has_no_integrated_but_has_momentary() {
        let m = measure(&left_only(-20.0, 0.25, 48000)).unwrap();
        assert!(m.lufs.is_none());
        let mom = m.momentary_max_lufs.unwrap();
        assert!((mom - -23.0).abs() < 1.5, "got {mom}");
    }

    #[test]
    fn silence_is_none() {
        let c = Clip::silence(48000, 48000, 2);
        let m = measure(&c).unwrap();
        assert!(m.lufs.is_none() && m.momentary_max_lufs.is_none());
        assert_eq!(m.true_peak_dbtp, -200.0);
        assert_eq!(m.sample_peak_dbfs, -200.0);
    }

    #[test]
    fn empty_clip_is_ok() {
        let c = Clip::silence(0, 48000, 2);
        assert!(measure(&c).unwrap().lufs.is_none());
    }
}
