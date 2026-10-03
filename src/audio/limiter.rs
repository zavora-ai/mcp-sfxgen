//! True-peak limiter (design §6 step 4).
//!
//! * **Detection:** 4x polyphase oversampling (12 taps per phase, 48 taps in all, Kaiser-windowed
//!   sinc) finds the inter-sample peak of every frame interval on every channel.
//! * **Gain:** the gain each peak needs goes through a sliding minimum over the 1.5 ms look-ahead
//!   window (monotonic deque), then a moving average of the same length, which turns each drop
//!   into a ramp that reaches its target before the peak. Recovery is a one-pole release of
//!   50 ms. One gain is applied to all channels.
//! * **Check:** the output is measured with `ebur128`'s true peak; if it is still above
//!   `ceiling * 1.001`, one safety rescale brings it down.
//!
//! The classic streaming form delays the audio by the look-ahead and flushes the tail. Working
//! on the whole clip at once gives the same output without the delay, so the output is the same
//! length as the input and nothing needs compensating.

use std::collections::VecDeque;

use serde::Serialize;

use super::{AudioError, Clip, db_to_lin, lin_to_db, loudness};

const PHASES: usize = 4;
const TAPS: usize = 12;
/// Taps before the interpolation point's left neighbour: a window covers `x[n-5] ..= x[n+6]`.
const TAPS_BEFORE: usize = 5;
const KAISER_BETA: f64 = 5.0;
const LOOKAHEAD_MS: f64 = 1.5;
const RELEASE_MS: f64 = 50.0;
/// The gain is computed for a target this much under the ceiling, so the interpolator's small
/// error against `ebur128` doesn't leave the result a hair over.
const TARGET_MARGIN: f64 = 0.998;
/// The post-check tolerance (spec: `ceiling * 1.001`).
const VERIFY_TOLERANCE: f64 = 1.001;
/// The safety rescale aims a hair under the ceiling so float rounding can't leave it over.
const SAFETY_TARGET: f64 = 0.9995;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct LimiterReport {
    /// Largest gain reduction applied, in dB (0 when the clip was already under the ceiling).
    pub max_gain_reduction_db: f64,
    /// Set when the final check still found the peak over the ceiling: the dB of the extra
    /// rescale (negative).
    pub safety_rescale_db: Option<f64>,
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..60 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-14 * sum {
            break;
        }
    }
    sum
}

fn sinc(t: f64) -> f64 {
    if t.abs() < 1e-12 {
        1.0
    } else {
        (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
    }
}

/// Polyphase coefficients: phase `p` interpolates at `n + p/4` from `x[n-5] ..= x[n+6]`.
/// Each phase is normalised to unity gain at DC.
fn phase_taps() -> [[f32; TAPS]; PHASES] {
    let half = (TAPS / 2) as f64;
    let norm = bessel_i0(KAISER_BETA);
    let mut out = [[0f32; TAPS]; PHASES];
    for (p, taps) in out.iter_mut().enumerate() {
        let frac = p as f64 / PHASES as f64;
        let mut c = [0f64; TAPS];
        for (j, v) in c.iter_mut().enumerate() {
            let t = j as f64 - TAPS_BEFORE as f64 - frac;
            let r = (t / half).abs().min(1.0);
            *v = sinc(t) * bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / norm;
        }
        let sum: f64 = c.iter().sum();
        for (o, v) in taps.iter_mut().zip(c) {
            *o = (v / sum) as f32;
        }
    }
    out
}

/// Highest oversampled absolute value in each frame interval `[n, n+1)`, over all channels.
fn interval_peaks(clip: &Clip) -> Vec<f32> {
    let taps = phase_taps();
    let ch = clip.channels as usize;
    let frames = clip.frames();
    let mut peaks = vec![0f32; frames];
    let mut padded = vec![0f32; frames + TAPS];
    for c in 0..ch {
        for f in 0..frames {
            padded[f + TAPS_BEFORE] = clip.samples[f * ch + c];
        }
        for (n, peak) in peaks.iter_mut().enumerate() {
            let win = &padded[n..n + TAPS];
            // Phase 0 is the sample itself.
            let mut best = padded[n + TAPS_BEFORE].abs();
            for ph in &taps[1..] {
                let v: f32 = win.iter().zip(ph).map(|(x, t)| x * t).sum();
                best = best.max(v.abs());
            }
            *peak = peak.max(best);
        }
    }
    peaks
}

/// `out[i] = min(g[i ..= i + l])`, with a monotonic deque.
fn sliding_min_forward(g: &[f64], l: usize) -> Vec<f64> {
    let n = g.len();
    let mut out = vec![1.0; n];
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut next = 0;
    for (i, o) in out.iter_mut().enumerate() {
        let hi = (i + l).min(n - 1);
        while next <= hi {
            while dq.back().is_some_and(|&b| g[b] >= g[next]) {
                dq.pop_back();
            }
            dq.push_back(next);
            next += 1;
        }
        while dq.front().is_some_and(|&f| f < i) {
            dq.pop_front();
        }
        *o = g[*dq.front().expect("window is never empty")];
    }
    out
}

/// The gain curve (linear, one value per frame) that keeps `peaks` under `target`.
fn gain_curve(peaks: &[f32], target: f64, sample_rate: u32) -> Vec<f64> {
    let n = peaks.len();
    let l = ((LOOKAHEAD_MS * sample_rate as f64 / 1000.0).round() as usize).max(1);
    // Gain each interval needs; a frame must satisfy the interval on either side of it.
    let need: Vec<f64> = peaks
        .iter()
        .map(|&p| {
            if p as f64 > target {
                target / p as f64
            } else {
                1.0
            }
        })
        .collect();
    let per_frame: Vec<f64> = (0..n)
        .map(|f| {
            if f == 0 {
                need[0]
            } else {
                need[f].min(need[f - 1])
            }
        })
        .collect();
    let m = sliding_min_forward(&per_frame, l);

    // Attack: the mean of m over the last l + 1 frames (m before the start counts as m[0]),
    // so every drop becomes a linear ramp that lands on its value exactly at the peak.
    let at = |j: isize| m[j.max(0) as usize];
    let w = (l + 1) as f64;
    let mut sum = w * m[0];
    let mut attack = Vec::with_capacity(n);
    attack.push(sum / w);
    for (i, &mi) in m.iter().enumerate().skip(1) {
        sum += mi - at(i as isize - l as isize - 1);
        attack.push(sum / w);
    }

    // Release: one pole, 50 ms, never above the attack curve.
    let a = (-1.0 / (RELEASE_MS / 1000.0 * sample_rate as f64)).exp();
    let mut gain = Vec::with_capacity(n);
    let mut prev = attack[0].min(1.0);
    gain.push(prev);
    for &att in &attack[1..] {
        let rel = 1.0 - a * (1.0 - prev);
        prev = att.min(rel).min(1.0);
        gain.push(prev);
    }
    gain
}

/// Limit `clip` so its true peak stays at or under `ceiling_dbtp` (for example -1.0).
/// The output has the same length as the input; a clip already under the ceiling is returned
/// sample-identical.
pub fn limit_true_peak(
    clip: &Clip,
    ceiling_dbtp: f64,
) -> Result<(Clip, LimiterReport), AudioError> {
    let ceiling = db_to_lin(ceiling_dbtp);
    let mut out = clip.clone();
    let mut report = LimiterReport::default();
    if clip.is_empty() {
        return Ok((out, report));
    }

    let peaks = interval_peaks(clip);
    let top = peaks.iter().copied().fold(0f32, f32::max) as f64;
    if top > ceiling {
        let gain = gain_curve(&peaks, ceiling * TARGET_MARGIN, clip.sample_rate);
        let ch = clip.channels as usize;
        let mut min_gain = 1.0f64;
        for (frame, &g) in out.samples.chunks_exact_mut(ch).zip(&gain) {
            min_gain = min_gain.min(g);
            let g = g as f32;
            for s in frame {
                *s *= g;
            }
        }
        report.max_gain_reduction_db = -lin_to_db(min_gain);
    }

    let tp = loudness::true_peak_linear(&out)?;
    if tp > ceiling * VERIFY_TOLERANCE {
        let scale = ceiling * SAFETY_TARGET / tp;
        for s in &mut out.samples {
            *s *= scale as f32;
        }
        report.safety_rescale_db = Some(lin_to_db(scale));
    }
    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::{noise, sine};
    use super::super::loudness::measure;
    use super::*;

    const CEIL: f64 = -1.0;

    fn tp(c: &Clip) -> f64 {
        measure(c).unwrap().true_peak_dbtp
    }

    #[test]
    fn plus3_sine_comes_out_under_ceiling() {
        for sr in [44100u32, 48000] {
            let c = sine(1000.0, 3.0, 1.0, sr, 2, 0.0);
            assert!(tp(&c) > 2.9);
            let (out, rep) = limit_true_peak(&c, CEIL).unwrap();
            assert_eq!(out.frames(), c.frames());
            let t = tp(&out);
            assert!(t <= CEIL, "sr {sr}: {t}");
            assert!(t > CEIL - 0.3, "over-limited: {t}");
            assert!(rep.max_gain_reduction_db > 3.9, "{rep:?}");
        }
    }

    #[test]
    fn inter_sample_peaks_are_caught() {
        // fs/4 at 45 degrees: every sample is +-0.707 of the true peak, so the sample peak is
        // 1.0 while the true peak is +3 dBTP.
        let c = sine(11025.0, 3.0103, 1.0, 44100, 2, std::f64::consts::FRAC_PI_4);
        let m = measure(&c).unwrap();
        assert!(m.sample_peak_dbfs < 0.1 && m.true_peak_dbtp > 2.9, "{m:?}");
        let (out, rep) = limit_true_peak(&c, CEIL).unwrap();
        assert!(tp(&out) <= CEIL, "{}", tp(&out));
        assert!(rep.max_gain_reduction_db > 3.9);
    }

    #[test]
    fn loud_noise_and_a_burst_come_out_under_ceiling() {
        let mut c = noise(44100, 44100, 2, 1.0, 11);
        for s in &mut c.samples {
            *s *= 1.4; // peaks near +3 dBFS
        }
        // A silent lead-in and a single hot click: the look-ahead has to anticipate it.
        let mut lead = Clip::silence(2000, 44100, 2);
        lead.samples.extend(&c.samples[..400]);
        let (out, _) = limit_true_peak(&lead, CEIL).unwrap();
        assert!(tp(&out) <= CEIL, "{}", tp(&out));
        let (out, _) = limit_true_peak(&c, CEIL).unwrap();
        assert!(tp(&out) <= CEIL, "{}", tp(&out));
    }

    #[test]
    fn peak_at_the_very_start_is_limited() {
        let mut c = Clip::silence(4000, 44100, 1);
        c.samples[0] = 1.5;
        c.samples[1] = -1.5;
        let (out, _) = limit_true_peak(&c, CEIL).unwrap();
        assert!(tp(&out) <= CEIL, "{}", tp(&out));
    }

    #[test]
    fn clean_signal_is_sample_identical() {
        let c = sine(1000.0, -6.0, 1.0, 44100, 2, 0.4);
        let (out, rep) = limit_true_peak(&c, CEIL).unwrap();
        assert_eq!(out, c);
        assert_eq!(rep, LimiterReport::default());
    }

    #[test]
    fn gain_recovers_after_a_peak() {
        // A hot click, then a quiet tone: after ~50 ms the tone should be back near its level.
        let sr = 44100;
        let mut c = sine(500.0, -20.0, 1.0, sr, 1, 0.0);
        c.samples[100] = 1.8;
        let (out, _) = limit_true_peak(&c, CEIL).unwrap();
        let tail = |c: &Clip| {
            let s = &c.samples[sr as usize * 3 / 4..];
            s.iter().fold(0f32, |a, &b| a.max(b.abs()))
        };
        assert!((tail(&out) - tail(&c)).abs() < 1e-3 * tail(&c) + 1e-4);
    }

    #[test]
    fn sliding_min_matches_brute_force() {
        let g: Vec<f64> = (0..200).map(|i| ((i * 37 % 91) as f64) / 91.0).collect();
        for l in [1, 3, 10] {
            let fast = sliding_min_forward(&g, l);
            for i in 0..g.len() {
                let hi = (i + l).min(g.len() - 1);
                let want = g[i..=hi].iter().copied().fold(f64::INFINITY, f64::min);
                assert_eq!(fast[i], want);
            }
        }
    }

    #[test]
    fn detector_sees_the_inter_sample_peak() {
        let c = sine(11025.0, 3.0103, 0.1, 44100, 1, std::f64::consts::FRAC_PI_4);
        // Away from the clip edges, where the zero padding rings.
        let peaks = interval_peaks(&c);
        let top = peaks[50..peaks.len() - 50]
            .iter()
            .copied()
            .fold(0f32, f32::max);
        let db = lin_to_db(top as f64);
        assert!((db - 3.0103).abs() < 0.1, "{db}");
    }
}
