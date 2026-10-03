//! WAV reading and writing (`hound`). Quantisation to 16/24 bit happens here.

use std::path::Path;

use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use super::{AudioError, Clip};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BitDepth {
    #[default]
    Pcm16,
    Pcm24,
    Float32,
}

impl BitDepth {
    pub fn bits(self) -> u16 {
        match self {
            BitDepth::Pcm16 => 16,
            BitDepth::Pcm24 => 24,
            BitDepth::Float32 => 32,
        }
    }
}

/// Write a WAV. `Pcm16` gets TPDF dither drawn from a generator seeded with `dither_seed`, so
/// the same clip and seed always give the same bytes. `Pcm24` rounds without dither;
/// `Float32` is written as is.
pub fn write_wav(
    path: &Path,
    clip: &Clip,
    depth: BitDepth,
    dither_seed: u64,
) -> Result<(), AudioError> {
    let mut rng = StdRng::seed_from_u64(dither_seed);
    write_wav_with_rng(path, clip, depth, &mut rng)
}

/// As [`write_wav`], with the caller's random generator for the dither.
pub fn write_wav_with_rng(
    path: &Path,
    clip: &Clip,
    depth: BitDepth,
    rng: &mut impl Rng,
) -> Result<(), AudioError> {
    write_wav_impl(path, clip, depth, Some(rng))
}

/// Write without dither: `Pcm16` and `Pcm24` round to the nearest step. For raw copies of what a backend delivered,
/// which must come back bit for bit (a clip read from 16-bit PCM is written to the same 16-bit values).
pub fn write_wav_exact(path: &Path, clip: &Clip, depth: BitDepth) -> Result<(), AudioError> {
    write_wav_impl::<rand::rngs::StdRng>(path, clip, depth, None)
}

fn write_wav_impl<R: Rng>(
    path: &Path,
    clip: &Clip,
    depth: BitDepth,
    mut rng: Option<&mut R>,
) -> Result<(), AudioError> {
    let spec = hound::WavSpec {
        channels: clip.channels,
        sample_rate: clip.sample_rate,
        bits_per_sample: depth.bits(),
        sample_format: match depth {
            BitDepth::Float32 => hound::SampleFormat::Float,
            _ => hound::SampleFormat::Int,
        },
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    match depth {
        BitDepth::Float32 => {
            for &s in &clip.samples {
                writer.write_sample(s)?;
            }
        }
        BitDepth::Pcm24 => {
            for &s in &clip.samples {
                let v = (s as f64 * 8_388_608.0)
                    .round()
                    .clamp(-8_388_608.0, 8_388_607.0);
                writer.write_sample(v as i32)?;
            }
        }
        BitDepth::Pcm16 => {
            for &s in &clip.samples {
                // TPDF: the sum of two uniform variables, +-1 LSB peak.
                let dither = match rng.as_mut() {
                    Some(rng) => rng.random::<f64>() + rng.random::<f64>() - 1.0,
                    None => 0.0,
                };
                let v = (s as f64 * 32_768.0 + dither)
                    .round()
                    .clamp(-32_768.0, 32_767.0);
                writer.write_sample(v as i16)?;
            }
        }
    }
    writer.finalize()?;
    Ok(())
}

/// Read a WAV (16/24/32-bit integer or 32-bit float) into an `f32` clip.
pub fn read_wav(path: &Path) -> Result<Clip, AudioError> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            let scale = 1.0 / (1u64 << (bits - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f64 * scale) as f32))
                .collect::<Result<_, _>>()?
        }
        (fmt, bits) => {
            return Err(AudioError::UnsupportedFormat(format!(
                "{fmt:?} WAV with {bits} bits per sample"
            )));
        }
    };
    Clip::new(samples, spec.sample_rate, spec.channels)
}

#[cfg(test)]
mod tests {
    use super::super::clip::testutil::sine;
    use super::*;

    fn max_err(a: &Clip, b: &Clip) -> f32 {
        assert_eq!(a.samples.len(), b.samples.len());
        a.samples
            .iter()
            .zip(&b.samples)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn round_trip_each_depth() {
        let dir = tempfile::tempdir().unwrap();
        let clip = sine(1000.0, -6.0, 0.1, 44100, 2, 0.3);
        for (depth, tol) in [
            (BitDepth::Float32, 0.0),
            (BitDepth::Pcm24, 1.0 / 8_388_608.0),
            (BitDepth::Pcm16, 2.0 / 32_768.0),
        ] {
            let p = dir.path().join(format!("{depth:?}.wav"));
            write_wav(&p, &clip, depth, 7).unwrap();
            let back = read_wav(&p).unwrap();
            assert_eq!(back.sample_rate, 44100);
            assert_eq!(back.channels, 2);
            assert_eq!(back.frames(), clip.frames());
            assert!(max_err(&clip, &back) <= tol, "{depth:?}");
        }
    }

    #[test]
    fn exact_writes_return_the_same_16_bit_values() {
        let dir = tempfile::tempdir().unwrap();
        let pcm: Vec<u8> = (0..4000u32)
            .flat_map(|i| (((i * 37) % 65536) as u16).to_le_bytes())
            .collect();
        let clip = Clip::from_s16le_bytes(&pcm, 44100, 2).unwrap();
        let p = dir.path().join("exact.wav");
        write_wav_exact(&p, &clip, BitDepth::Pcm16).unwrap();
        let back = read_wav(&p).unwrap();
        assert_eq!(back.samples, clip.samples, "bit for bit");
        // The dithered writer would have moved some of them.
        write_wav(&p.with_file_name("dithered.wav"), &clip, BitDepth::Pcm16, 3).unwrap();
        let dithered = read_wav(&p.with_file_name("dithered.wav")).unwrap();
        assert_ne!(dithered.samples, clip.samples);
    }

    #[test]
    fn dither_is_deterministic_per_seed() {
        let dir = tempfile::tempdir().unwrap();
        let clip = sine(440.0, -30.0, 0.05, 44100, 1, 0.0);
        let write = |name: &str, seed| {
            let p = dir.path().join(name);
            write_wav(&p, &clip, BitDepth::Pcm16, seed).unwrap();
            std::fs::read(p).unwrap()
        };
        assert_eq!(write("a.wav", 1), write("b.wav", 1));
        assert_ne!(write("c.wav", 1), write("d.wav", 2));
    }

    #[test]
    fn clipping_is_clamped_not_wrapped() {
        let dir = tempfile::tempdir().unwrap();
        let clip = Clip::new(vec![2.0, -2.0, 1.0, -1.0], 8000, 1).unwrap();
        for depth in [BitDepth::Pcm16, BitDepth::Pcm24] {
            let p = dir.path().join("c.wav");
            write_wav(&p, &clip, depth, 0).unwrap();
            let back = read_wav(&p).unwrap();
            assert!(back.samples[0] > 0.999 && back.samples[1] <= -0.999);
        }
    }

    #[test]
    fn reads_32bit_int() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("i32.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&p, spec).unwrap();
        w.write_sample(i32::MAX / 2).unwrap();
        w.write_sample(i32::MIN / 2).unwrap();
        w.finalize().unwrap();
        let c = read_wav(&p).unwrap();
        assert!((c.samples[0] - 0.5).abs() < 1e-6 && (c.samples[1] + 0.5).abs() < 1e-6);
    }
}
