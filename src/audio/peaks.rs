//! Waveform peaks for a review UI: min/max int8 pairs per bucket.

use base64::{Engine, engine::general_purpose::STANDARD};

use super::Clip;

/// Number of buckets stored in sidecars.
pub const DEFAULT_BUCKETS: usize = 512;

/// `(min, max)` per bucket over all channels, scaled so full scale is +-127.
/// A bucket with no frames (a clip shorter than `buckets`) is `(0, 0)`.
pub fn peaks(clip: &Clip, buckets: usize) -> Vec<(i8, i8)> {
    let ch = clip.channels as usize;
    let frames = clip.frames();
    let scale = |v: f32| (v.clamp(-1.0, 1.0) * 127.0).round() as i8;
    (0..buckets)
        .map(|b| {
            let (lo, hi) = (b * frames / buckets, (b + 1) * frames / buckets);
            if lo >= hi {
                return (0, 0);
            }
            let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
            for &s in &clip.samples[lo * ch..hi * ch] {
                min = min.min(s);
                max = max.max(s);
            }
            (scale(min), scale(max))
        })
        .collect()
}

/// `buckets` pairs as bytes `min0, max0, min1, max1, ...` (two's complement), base64 (standard).
pub fn peaks_base64(clip: &Clip, buckets: usize) -> String {
    let bytes: Vec<u8> = peaks(clip, buckets)
        .into_iter()
        .flat_map(|(lo, hi)| [lo as u8, hi as u8])
        .collect();
    STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::sine;
    use super::*;

    #[test]
    fn length_and_base64_size() {
        let c = sine(440.0, -6.0, 1.0, 44100, 2, 0.0);
        assert_eq!(peaks(&c, 512).len(), 512);
        let b64 = peaks_base64(&c, 512);
        assert_eq!(b64.len(), 1368); // 1024 bytes -> ceil(1024 / 3) * 4
        let raw = STANDARD.decode(&b64).unwrap();
        assert_eq!(raw.len(), 1024);
    }

    #[test]
    fn scaling_and_sign() {
        let c = Clip::new(vec![1.0, -1.0, 0.5, -0.25], 8000, 2).unwrap();
        assert_eq!(peaks(&c, 1), vec![(-127, 127)]);
        let c = Clip::new(vec![0.5, -0.25], 8000, 1).unwrap();
        let p = peaks(&c, 2);
        assert_eq!(p, vec![(64, 64), (-32, -32)]);
    }

    #[test]
    fn short_clip_has_empty_buckets() {
        let c = Clip::new(vec![0.5; 3], 8000, 1).unwrap();
        let p = peaks(&c, 8);
        assert_eq!(p.len(), 8);
        assert!(p.iter().any(|&(_, hi)| hi == 64));
    }
}
