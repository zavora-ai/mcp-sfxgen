//! Clip edits: trim, fades, mono, gain, the local loop crossfade and the seam metric.

use std::f64::consts::FRAC_PI_2;

use serde::Serialize;

use super::{AudioError, Clip, db_to_lin, lin_to_db};

/// Default silence threshold for [`trim_silence`] (design §6 step 3).
pub const TRIM_THRESHOLD_DB: f64 = -60.0;
/// Default pad kept around the content by [`trim_silence`].
pub const TRIM_PAD_MS: f64 = 5.0;
/// Silence is also anything quieter than the clip's peak by this much, so a generated clip's room tone (often louder
/// than the absolute threshold) is trimmed too.
pub const TRIM_RELATIVE_DB: f64 = -45.0;
/// An event starts where the level first comes within this of the peak ...
pub const EVENT_ONSET_DB: f64 = -30.0;
/// ... and has died away once it stays below this.
pub const EVENT_TAIL_DB: f64 = -45.0;

fn ms_to_frames(ms: f64, sr: u32) -> usize {
    (ms.max(0.0) * sr as f64 / 1000.0).round() as usize
}

/// Cut leading and trailing frames whose every channel is below `threshold_db`, keeping `pad_ms`
/// of margin. An all-silent clip is returned unchanged (there is nothing to anchor a cut on).
pub fn trim_silence(clip: &Clip, threshold_db: f64, pad_ms: f64) -> Clip {
    let ch = clip.channels as usize;
    let peak = clip.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let thr = (db_to_lin(threshold_db) as f32).max(peak * db_to_lin(TRIM_RELATIVE_DB) as f32);
    let loud = |f: usize| {
        clip.samples[f * ch..(f + 1) * ch]
            .iter()
            .any(|s| s.abs() > thr)
    };
    let frames = clip.frames();
    let Some(first) = (0..frames).find(|&f| loud(f)) else {
        return clip.clone();
    };
    let last = (0..frames).rev().find(|&f| loud(f)).unwrap_or(first);
    let pad = ms_to_frames(pad_ms, clip.sample_rate);
    let start = first.saturating_sub(pad);
    let end = (last + 1 + pad).min(frames);
    Clip {
        samples: clip.samples[start * ch..end * ch].to_vec(),
        sample_rate: clip.sample_rate,
        channels: clip.channels,
    }
}

/// Cut a generated clip down to its event, at most `target_s` long: from 5 ms before the level first comes within
/// `EVENT_ONSET_DB` of the peak, to 30 ms after the last frame above `EVENT_TAIL_DB` (or `target_s`, whichever is
/// sooner), with a 2 ms fade-in and a 10 ms fade-out. Used when a short sound was generated longer than asked
/// (models smear very short clips). A silent clip is just cut to `target_s`.
pub fn extract_event(clip: &Clip, target_s: f64) -> Clip {
    let ch = clip.channels as usize;
    let frames = clip.frames();
    let target =
        ((target_s * f64::from(clip.sample_rate)).round() as usize).clamp(1, frames.max(1));
    let peak = clip.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let level = |f: usize| {
        clip.samples[f * ch..(f + 1) * ch]
            .iter()
            .fold(0.0f32, |m, s| m.max(s.abs()))
    };
    let (start, end) = if peak <= 0.0 {
        (0, target.min(frames))
    } else {
        let on = peak * db_to_lin(EVENT_ONSET_DB) as f32;
        let tail = peak * db_to_lin(EVENT_TAIL_DB) as f32;
        let first = (0..frames).find(|&f| level(f) >= on).unwrap_or(0);
        let start = first.saturating_sub(ms_to_frames(5.0, clip.sample_rate));
        let limit = (start + target).min(frames);
        let last = (start..limit)
            .rev()
            .find(|&f| level(f) >= tail)
            .unwrap_or(first);
        (
            start,
            (last + 1 + ms_to_frames(30.0, clip.sample_rate)).min(limit),
        )
    };
    let mut out = Clip {
        samples: clip.samples[start * ch..end.max(start + 1) * ch].to_vec(),
        sample_rate: clip.sample_rate,
        channels: clip.channels,
    };
    fade_in(&mut out, 2.0);
    fade_out(&mut out, 10.0);
    out
}

/// Raised-cosine gain at position `t` in 0..=1 (0 -> 0.0, 1 -> 1.0).
fn smooth(t: f64) -> f32 {
    (0.5 * (1.0 - (std::f64::consts::PI * t).cos())) as f32
}

/// Fade the start in over `ms` (raised cosine; capped at the clip length).
pub fn fade_in(clip: &mut Clip, ms: f64) {
    let ch = clip.channels as usize;
    let n = ms_to_frames(ms, clip.sample_rate).min(clip.frames());
    for f in 0..n {
        let g = smooth(f as f64 / n as f64);
        for s in &mut clip.samples[f * ch..(f + 1) * ch] {
            *s *= g;
        }
    }
}

/// Fade the end out over `ms` (raised cosine; the last frame reaches 0).
pub fn fade_out(clip: &mut Clip, ms: f64) {
    let ch = clip.channels as usize;
    let frames = clip.frames();
    let n = ms_to_frames(ms, clip.sample_rate).min(frames);
    for i in 0..n {
        let f = frames - n + i;
        let g = smooth(1.0 - (i + 1) as f64 / n as f64);
        for s in &mut clip.samples[f * ch..(f + 1) * ch] {
            *s *= g;
        }
    }
}

/// Average all channels into one.
pub fn to_mono(clip: &Clip) -> Clip {
    let ch = clip.channels as usize;
    if ch == 1 {
        return clip.clone();
    }
    let samples = clip
        .samples
        .chunks_exact(ch)
        .map(|f| (f.iter().map(|&s| s as f64).sum::<f64>() / ch as f64) as f32)
        .collect();
    Clip {
        samples,
        sample_rate: clip.sample_rate,
        channels: 1,
    }
}

pub fn apply_gain_db(clip: &mut Clip, db: f64) {
    let g = db_to_lin(db) as f32;
    for s in &mut clip.samples {
        *s *= g;
    }
}

/// `x = min(0.5 s, d/4)`: the crossfade length for a loop of `d` seconds.
pub fn loop_x_s(d: f64) -> f64 {
    (d / 4.0).min(0.5)
}

/// How much longer than `d` the backend must generate for a local loop: `2x`.
pub fn loop_extra_s(d: f64) -> f64 {
    2.0 * loop_x_s(d)
}

/// Make a loop of exactly `round(target_s * sr)` frames from a clip generated at `d + 2x`.
///
/// The last `x` is dropped (the model fades out there). The `x` seconds that follow the loop end
/// are the natural continuation of the last frame, so they are blended into the head with an
/// equal-power crossfade: when the loop wraps, the last frame is followed by (almost) the sample
/// that originally came next. Needs at least `d + x` of input.
pub fn loop_crossfade(clip: &Clip, target_s: f64) -> Result<Clip, AudioError> {
    if !(target_s.is_finite() && target_s > 0.0) {
        return Err(AudioError::Invalid(format!(
            "loop length {target_s} s is not valid"
        )));
    }
    let sr = clip.sample_rate as f64;
    let d = (target_s * sr).round() as usize;
    let x = (loop_x_s(target_s) * sr).round() as usize;
    if d == 0 || clip.frames() < d + x {
        return Err(AudioError::Invalid(format!(
            "a {target_s} s loop needs at least {:.3} s of audio, got {:.3} s",
            (d + x) as f64 / sr,
            clip.duration_s()
        )));
    }
    let ch = clip.channels as usize;
    let mut out = clip.samples[..d * ch].to_vec();
    for i in 0..x {
        let t = (i as f64 + 0.5) / x as f64;
        let (g_in, g_out) = ((t * FRAC_PI_2).sin() as f32, (t * FRAC_PI_2).cos() as f32);
        for c in 0..ch {
            let head = out[i * ch + c];
            let tail = clip.samples[(d + i) * ch + c];
            out[i * ch + c] = head * g_in + tail * g_out;
        }
    }
    Ok(Clip {
        samples: out,
        sample_rate: clip.sample_rate,
        channels: clip.channels,
    })
}

/// How well a clip loops (design §6 step 6).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Seam {
    /// `|last - first|` divided by the median `|step|`, per channel; the worst channel.
    /// About 1 means the wrap is as smooth as the signal itself.
    pub wrap_ratio: f64,
    /// RMS of the first 20 ms minus RMS of the last 20 ms, in dB (all channels).
    pub head_tail_rms_db: f64,
}

const SEAM_WINDOW_MS: f64 = 20.0;
/// Below this a step counts as zero, so a flat or silent channel can't divide by nothing.
const STEP_FLOOR: f64 = 1e-9;

pub fn seam(clip: &Clip) -> Seam {
    let ch = clip.channels as usize;
    let frames = clip.frames();
    if frames < 2 {
        return Seam {
            wrap_ratio: 0.0,
            head_tail_rms_db: 0.0,
        };
    }
    let mut wrap_ratio = 0f64;
    for c in 0..ch {
        let mut steps: Vec<f64> = (1..frames)
            .map(|f| (clip.at(f, c) as f64 - clip.at(f - 1, c) as f64).abs())
            .collect();
        let mid = steps.len() / 2;
        let median = *steps.select_nth_unstable_by(mid, f64::total_cmp).1;
        let jump = (clip.at(0, c) as f64 - clip.at(frames - 1, c) as f64).abs();
        wrap_ratio = wrap_ratio.max(jump / median.max(STEP_FLOOR));
    }
    let win = ms_to_frames(SEAM_WINDOW_MS, clip.sample_rate).clamp(1, frames / 2);
    let rms = |range: std::ops::Range<usize>| {
        let slice = &clip.samples[range.start * ch..range.end * ch];
        (slice.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / slice.len() as f64).sqrt()
    };
    let head = rms(0..win);
    let tail = rms(frames - win..frames);
    Seam {
        wrap_ratio,
        head_tail_rms_db: lin_to_db(head.max(1e-6)) - lin_to_db(tail.max(1e-6)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::{noise, sine};
    use super::*;

    #[test]
    fn trim_removes_padded_silence() {
        let sr = 48000;
        let body = sine(1000.0, -12.0, 0.2, sr, 2, 0.0);
        let lead = Clip::silence(sr as usize / 2, sr, 2);
        let mut samples = lead.samples.clone();
        samples.extend(&body.samples);
        samples.extend(&lead.samples);
        let padded = Clip::new(samples, sr, 2).unwrap();
        let t = trim_silence(&padded, TRIM_THRESHOLD_DB, TRIM_PAD_MS);
        let pad = ms_to_frames(TRIM_PAD_MS, sr);
        // The sine starts at 0, so its first/last loud frames are within a frame or two of the ends.
        let expected = body.frames() + 2 * pad;
        assert!(
            t.frames().abs_diff(expected) <= 3,
            "{} vs {expected}",
            t.frames()
        );
        assert!(t.frames() < padded.frames() / 2);
    }

    #[test]
    fn trim_keeps_all_silent_clip() {
        let c = Clip::silence(100, 48000, 1);
        assert_eq!(trim_silence(&c, -60.0, 5.0).frames(), 100);
    }

    #[test]
    fn mono_halves_channels() {
        let c = Clip::new(vec![1.0, 0.0, 0.5, 0.5], 48000, 2).unwrap();
        let m = to_mono(&c);
        assert_eq!(m.channels, 1);
        assert_eq!(m.samples, vec![0.5, 0.5]);
        assert_eq!(m.frames(), c.frames());
        let n = sine(440.0, -6.0, 0.1, 48000, 2, 0.0);
        assert_eq!(to_mono(&n).samples.len(), n.samples.len() / 2);
    }

    #[test]
    fn fades_reach_silence_and_leave_the_middle() {
        let mut c = Clip::new(vec![1.0; 4800], 48000, 1).unwrap();
        fade_in(&mut c, 10.0);
        fade_out(&mut c, 10.0);
        assert_eq!(c.samples[0], 0.0);
        assert_eq!(*c.samples.last().unwrap(), 0.0);
        assert_eq!(c.samples[2400], 1.0);
        assert!((c.samples[240] - 0.5).abs() < 0.01); // halfway through a 480-frame fade
        assert!(c.samples[470] > 0.99 && c.samples[100] < 0.2);
    }

    #[test]
    fn gain_scales() {
        let mut c = Clip::new(vec![0.5, -0.5], 48000, 1).unwrap();
        apply_gain_db(&mut c, -6.0206);
        assert!((c.samples[0] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn loop_extra_matches_design() {
        assert_eq!(loop_extra_s(2.0), 1.0); // x = 0.5
        assert_eq!(loop_extra_s(1.0), 0.5); // x = d/4 = 0.25
        assert_eq!(loop_extra_s(8.0), 1.0);
    }

    fn noise_loop(d: f64, seed: u64) -> (Clip, Clip) {
        let sr = 44100;
        let raw = noise(
            ((d + loop_extra_s(d)) * sr as f64) as usize,
            sr,
            2,
            0.5,
            seed,
        );
        let out = loop_crossfade(&raw, d).unwrap();
        (raw, out)
    }

    #[test]
    fn loop_crossfade_noise_has_small_wrap_and_exact_length() {
        let (_, out) = noise_loop(1.5, 1);
        assert_eq!(out.frames(), (1.5f64 * 44100.0).round() as usize);
        assert!(seam(&out).wrap_ratio < 3.0, "{}", seam(&out).wrap_ratio);
    }

    #[test]
    fn loop_crossfade_wrap_is_as_smooth_as_the_signal_itself() {
        // For white noise the wrap step is one more random step, so the ratio is a random
        // variable around 1 (a few percent of draws exceed 3). Check the distribution.
        let ratios: Vec<f64> = (1..=40u64)
            .map(|s| seam(&noise_loop(1.0, s).1).wrap_ratio)
            .collect();
        let mut sorted = ratios.clone();
        sorted.sort_by(f64::total_cmp);
        assert!(sorted[20] < 2.0, "median {}", sorted[20]);
        assert!(
            ratios.iter().filter(|&&r| r < 3.0).count() >= 34,
            "{ratios:?}"
        );
    }

    #[test]
    fn loop_crossfade_fixes_a_trending_signal() {
        // A slow ramp plus a little noise: a hard wrap jumps by the whole ramp height.
        let sr = 44100u32;
        let d = 1.0;
        let n = ((d + loop_extra_s(d)) * sr as f64) as usize;
        let jitter = noise(n, sr, 1, 0.01, 4);
        let samples: Vec<f32> = (0..n)
            .map(|i| i as f32 / n as f32 * 0.8 + jitter.samples[i])
            .collect();
        let raw = Clip::new(samples, sr, 1).unwrap();
        let naive = Clip::new(raw.samples[..sr as usize].to_vec(), sr, 1).unwrap();
        let fixed = loop_crossfade(&raw, d).unwrap();
        assert!(seam(&naive).wrap_ratio > 50.0);
        assert!(seam(&fixed).wrap_ratio < 5.0, "{}", seam(&fixed).wrap_ratio);
    }

    #[test]
    fn loop_crossfade_rejects_short_input() {
        let raw = noise(1000, 44100, 1, 0.5, 1);
        assert!(loop_crossfade(&raw, 1.0).is_err());
    }

    #[test]
    fn seam_of_a_cut_sine_is_large_and_of_a_whole_cycle_small() {
        let whole = sine(100.0, -6.0, 0.1, 48000, 1, 0.0); // 10 whole cycles, starts at 0
        let s = seam(&whole);
        assert!(s.wrap_ratio < 3.0, "{}", s.wrap_ratio);
        assert!(s.head_tail_rms_db.abs() < 3.0);
        let cut = sine(100.0, -6.0, 0.105, 48000, 1, 1.0);
        assert!(seam(&cut).wrap_ratio > 3.0);
    }

    #[test]
    fn seam_reports_level_difference() {
        let mut c = Clip::new(vec![0.5; 4800], 48000, 1).unwrap();
        for s in &mut c.samples[3000..] {
            *s = 0.05;
        }
        let s = seam(&c);
        assert!((s.head_tail_rms_db - 20.0).abs() < 0.1);
    }

    #[test]
    fn extract_event_finds_a_short_burst_in_a_long_noisy_clip() {
        // 2 s at 48 kHz: noise at -50 dBFS, a 0.2 s burst at 0.8 s
        let sr = 48_000u32;
        let mut rng = 0x1234_5678u32;
        let mut samples = Vec::with_capacity(2 * sr as usize);
        for f in 0..2 * sr as usize {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let noise = ((rng as f32 / u32::MAX as f32) - 0.5) * 2.0 * 0.003;
            let t = f as f32 / sr as f32;
            let burst = if (0.8..1.0).contains(&t) {
                (t * 900.0 * std::f32::consts::TAU).sin() * 0.8
            } else {
                0.0
            };
            samples.push(noise + burst);
        }
        let clip = Clip {
            samples,
            sample_rate: sr,
            channels: 1,
        };
        let ev = extract_event(&clip, 0.5);
        let dur = ev.frames() as f64 / f64::from(sr);
        assert!((0.2..=0.5).contains(&dur), "event length {dur}");
        // it starts at the burst, not at the beginning of the clip
        let lead = ev.samples.iter().position(|s| s.abs() > 0.1).unwrap() as f64 / f64::from(sr);
        assert!(lead < 0.01, "lead-in {lead}");
    }

    #[test]
    fn trim_silence_cuts_room_tone_louder_than_the_absolute_threshold() {
        let sr = 48_000u32;
        let mut samples = vec![0.002f32; sr as usize]; // -54 dBFS room tone: above -60 dBFS, 53 dB under the peak
        for (i, s) in samples.iter_mut().enumerate().skip(20_000).take(4_800) {
            *s = ((i as f32) * 0.1).sin() * 0.9;
        }
        let clip = Clip {
            samples,
            sample_rate: sr,
            channels: 1,
        };
        let t = trim_silence(&clip, TRIM_THRESHOLD_DB, TRIM_PAD_MS);
        assert!(t.frames() < 6_000, "kept {} frames", t.frames());
    }
}
