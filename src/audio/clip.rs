//! The in-memory audio clip.

use super::AudioError;

/// Interleaved 32-bit float audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// Interleaved samples: `frame0_ch0, frame0_ch1, frame1_ch0, ...`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Clip {
    /// Build a clip, checking that the layout is consistent.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Result<Clip, AudioError> {
        if channels == 0 {
            return Err(AudioError::Invalid(
                "a clip needs at least one channel".into(),
            ));
        }
        if sample_rate == 0 {
            return Err(AudioError::Invalid("sample rate must be above zero".into()));
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(AudioError::Invalid(format!(
                "{} samples is not a whole number of {}-channel frames",
                samples.len(),
                channels
            )));
        }
        Ok(Clip {
            samples,
            sample_rate,
            channels,
        })
    }

    /// A clip of `frames` frames of silence.
    pub fn silence(frames: usize, sample_rate: u32, channels: u16) -> Clip {
        Clip {
            samples: vec![0.0; frames * channels as usize],
            sample_rate,
            channels,
        }
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    pub fn duration_s(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Headerless 16-bit little-endian PCM (ElevenLabs `pcm_*` formats), interleaved.
    /// Rejects odd lengths and lengths that are not whole frames.
    pub fn from_s16le_bytes(
        bytes: &[u8],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Clip, AudioError> {
        let bytes_per_frame = 2 * channels as usize;
        if channels == 0 || !bytes.len().is_multiple_of(bytes_per_frame) {
            return Err(AudioError::BadPcmLength {
                len: bytes.len(),
                bytes_per_frame,
            });
        }
        let samples = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();
        Clip::new(samples, sample_rate, channels)
    }

    /// Headerless 32-bit little-endian float PCM, interleaved (the Python worker's raw files).
    pub fn from_f32le_bytes(
        bytes: &[u8],
        sample_rate: u32,
        channels: u16,
    ) -> Result<Clip, AudioError> {
        let bytes_per_frame = 4 * channels as usize;
        if channels == 0 || !bytes.len().is_multiple_of(bytes_per_frame) {
            return Err(AudioError::BadPcmLength {
                len: bytes.len(),
                bytes_per_frame,
            });
        }
        let samples = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Clip::new(samples, sample_rate, channels)
    }

    /// Sample `frame` of channel `ch`.
    #[inline]
    pub fn at(&self, frame: usize, ch: usize) -> f32 {
        self.samples[frame * self.channels as usize + ch]
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::Clip;

    /// A sine of `freq` Hz at `dbfs` peak, `phase` radians, identical on all channels.
    pub fn sine(freq: f64, dbfs: f64, secs: f64, sr: u32, channels: u16, phase: f64) -> Clip {
        let frames = (secs * sr as f64).round() as usize;
        let amp = 10f64.powf(dbfs / 20.0);
        let mut samples = Vec::with_capacity(frames * channels as usize);
        for n in 0..frames {
            let v = (amp * (2.0 * std::f64::consts::PI * freq * n as f64 / sr as f64 + phase).sin())
                as f32;
            for _ in 0..channels {
                samples.push(v);
            }
        }
        Clip {
            samples,
            sample_rate: sr,
            channels,
        }
    }

    /// Deterministic white-ish noise in [-amp, amp] (xorshift), independent per channel.
    pub fn noise(frames: usize, sr: u32, channels: u16, amp: f32, seed: u64) -> Clip {
        let mut s = seed.max(1);
        let mut samples = Vec::with_capacity(frames * channels as usize);
        for _ in 0..frames * channels as usize {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            samples.push(((u * 2.0 - 1.0) as f32) * amp);
        }
        Clip {
            samples,
            sample_rate: sr,
            channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s16le_decodes_stereo() {
        let bytes = [0x00, 0x40, 0x00, 0xC0, 0xFF, 0x7F, 0x00, 0x80];
        let c = Clip::from_s16le_bytes(&bytes, 44100, 2).unwrap();
        assert_eq!(c.frames(), 2);
        assert_eq!(c.samples, vec![0.5, -0.5, 32767.0 / 32768.0, -1.0]);
    }

    #[test]
    fn odd_lengths_are_rejected() {
        assert!(matches!(
            Clip::from_s16le_bytes(&[0, 0, 0], 44100, 2),
            Err(AudioError::BadPcmLength { .. })
        ));
        // Even but not a whole stereo frame.
        assert!(Clip::from_s16le_bytes(&[0, 0, 0, 0, 0, 0], 44100, 2).is_err());
        assert!(Clip::from_f32le_bytes(&[0; 6], 44100, 1).is_err());
    }

    #[test]
    fn f32le_round_trip() {
        let src = [0.25f32, -0.75, 1.5, 0.0];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let c = Clip::from_f32le_bytes(&bytes, 48000, 2).unwrap();
        assert_eq!(c.samples, src);
        assert_eq!(c.frames(), 2);
        assert!((c.duration_s() - 2.0 / 48000.0).abs() < 1e-12);
    }

    #[test]
    fn new_validates() {
        assert!(Clip::new(vec![0.0; 3], 44100, 2).is_err());
        assert!(Clip::new(vec![], 44100, 0).is_err());
        assert!(Clip::new(vec![], 0, 1).is_err());
    }
}
