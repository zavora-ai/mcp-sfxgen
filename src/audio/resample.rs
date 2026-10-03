//! Fixed-ratio sample rate conversion with rubato's FFT resampler.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use super::{AudioError, Clip};

fn rs_err(e: impl std::fmt::Display) -> AudioError {
    AudioError::Resample(e.to_string())
}

/// Convert to `target_sr`. The output has exactly `round(frames * target / source)` frames and
/// the content is not shifted: rubato's start-up delay is trimmed by `process_all`, and any
/// off-by-one in the tail is trimmed or zero-padded here.
pub fn resample(clip: &Clip, target_sr: u32) -> Result<Clip, AudioError> {
    if target_sr == 0 {
        return Err(AudioError::Invalid(
            "target sample rate must be above zero".into(),
        ));
    }
    if target_sr == clip.sample_rate || clip.is_empty() {
        return Ok(Clip {
            samples: clip.samples.clone(),
            sample_rate: target_sr,
            channels: clip.channels,
        });
    }
    let ch = clip.channels as usize;
    let frames = clip.frames();
    let want = (frames as f64 * target_sr as f64 / clip.sample_rate as f64).round() as usize;

    let mut resampler = Fft::<f32>::new(
        clip.sample_rate as usize,
        target_sr as usize,
        1024,
        ch,
        FixedSync::Both,
    )
    .map_err(rs_err)?;
    let input = InterleavedSlice::new(&clip.samples, ch, frames).map_err(rs_err)?;
    let out = resampler
        .process_all(&input, frames, None)
        .map_err(rs_err)?;
    let mut samples = out.take_data();
    samples.resize(want * ch, 0.0);
    Ok(Clip {
        samples,
        sample_rate: target_sr,
        channels: clip.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::sine;
    use super::*;

    /// Positions (in seconds) of rising zero crossings, by linear interpolation.
    fn rising_crossings(c: &Clip) -> Vec<f64> {
        let ch = c.channels as usize;
        let mut v = Vec::new();
        for f in 1..c.frames() {
            let (a, b) = (c.samples[(f - 1) * ch], c.samples[f * ch]);
            if a < 0.0 && b >= 0.0 {
                v.push((f as f64 - 1.0 + (-a / (b - a)) as f64) / c.sample_rate as f64);
            }
        }
        v
    }

    #[test]
    fn upsample_44k1_to_48k_keeps_length_and_pitch() {
        let src = sine(1000.0, -6.0, 1.0, 44100, 2, 0.0);
        let out = resample(&src, 48000).unwrap();
        assert_eq!(out.sample_rate, 48000);
        assert_eq!(out.channels, 2);
        assert!(out.frames().abs_diff(48000) <= 1, "{}", out.frames());
        let z = rising_crossings(&out);
        // 1 kHz for ~1 s: about 1000 rising crossings, spaced 1 ms apart.
        assert!((z.len() as i64 - 1000).abs() <= 1, "{}", z.len());
        let span = (z.last().unwrap() - z[0]) / (z.len() - 1) as f64;
        assert!((span - 0.001).abs() < 1e-6, "{span}");
        // Not shifted: the sine started at phase 0, so crossings sit on whole milliseconds.
        // The sine started at phase 0, so crossings should sit on whole milliseconds. The FFT
        // resampler leaves up to half a sample of the lower rate (11 us here) of offset.
        let mid = z[z.len() / 2] * 1000.0;
        assert!((mid - mid.round()).abs() < 0.02, "{mid}");
    }

    #[test]
    fn downsample_and_odd_lengths() {
        let src = sine(500.0, -6.0, 0.37, 48000, 1, 0.0);
        let out = resample(&src, 44100).unwrap();
        let want = (src.frames() as f64 * 44100.0 / 48000.0).round() as usize;
        assert!(out.frames().abs_diff(want) <= 1);
        // A clip shorter than one FFT chunk still works.
        let tiny = sine(500.0, -6.0, 0.005, 44100, 1, 0.0);
        let want = (tiny.frames() as f64 * 48000.0 / 44100.0).round() as usize;
        assert_eq!(resample(&tiny, 48000).unwrap().frames(), want);
    }

    #[test]
    fn same_rate_is_identity() {
        let src = sine(500.0, -6.0, 0.1, 48000, 2, 0.0);
        assert_eq!(resample(&src, 48000).unwrap(), src);
    }
}
