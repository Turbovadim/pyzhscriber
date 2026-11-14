use anyhow::{Context, Result, ensure};
use std::fs::File;
use std::io::{Read, Seek};

/// Returns mono 16 kHz audio for Ogg Opus, or rewinds other formats for Rodio.
pub fn try_decode(file: &mut File) -> Result<Option<Vec<f32>>> {
    let mut signature = [0; 4];
    let count = file.read(&mut signature)?;
    file.rewind()?;
    if count != 4 || &signature != b"OggS" {
        return Ok(None);
    }
    let mut packets = ogg::PacketReader::new(&mut *file);
    let header = packets.read_packet_expected().context("read Ogg header")?;
    if !header.data.starts_with(b"OpusHead") {
        drop(packets);
        file.rewind()?;
        return Ok(None);
    }
    let data = &header.data;
    ensure!(
        data.len() >= 19 && data[8] > 0 && data[8] < 16,
        "invalid Opus header"
    );
    ensure!(
        header.first_in_stream() && header.last_in_page(),
        "invalid Opus header page"
    );
    ensure!(
        data[18] == 0,
        "Opus channel mapping is unsupported; use mono or stereo mapping family 0"
    );
    let channels = match data[9] {
        1 => opus::Channels::Mono,
        2 => opus::Channels::Stereo,
        _ => anyhow::bail!("Opus input must be mono or stereo"),
    };
    let channel_count = usize::from(data[9]);
    let pre_skip = usize::from(u16::from_le_bytes([data[10], data[11]]));
    let gain = i16::from_le_bytes([data[16], data[17]]);
    let serial = header.stream_serial();
    let tags = packets.read_packet_expected().context("read Opus tags")?;
    ensure!(
        tags.stream_serial() == serial
            && tags.data.starts_with(b"OpusTags")
            && tags.data.len() >= 16,
        "invalid Opus tags"
    );
    let mut decoder = opus::Decoder::new(48_000, channels)?;
    decoder.set_gain(i32::from(gain))?;
    // RFC 7845 counts pre-skip and page granules in 48 kHz frames, not samples
    // interleaved across channels. A packet can contain up to 120 ms of audio.
    let mut buffer = vec![0.0; 5760 * channel_count];
    let mut mono = Vec::new();
    let mut granule_offset = None;
    let mut ended = false;
    while let Some(packet) = packets.read_packet().context("read Ogg Opus packet")? {
        ensure!(
            !ended && packet.stream_serial() == serial,
            "chained or multiplexed Ogg Opus streams are not supported"
        );
        ensure!(!packet.data.is_empty(), "empty Opus packet");
        let frames = decoder
            .decode_float(&packet.data, &mut buffer, false)
            .context("decode Opus packet")?;
        mono.extend(
            buffer[..frames * channel_count]
                .chunks_exact(channel_count)
                .map(|frame| frame.iter().sum::<f32>() / channel_count as f32),
        );
        if packet.last_in_page() {
            let granule = packet.absgp_page();
            ensure!(granule != u64::MAX, "missing Opus page position");
            let decoded = mono.len() as u64;
            let offset = *granule_offset.get_or_insert_with(|| granule.saturating_sub(decoded));
            if packet.last_in_stream() {
                let end = granule
                    .checked_sub(offset)
                    .context("invalid Opus end position")?;
                ensure!(
                    end <= decoded && end >= decoded - frames as u64 && end >= pre_skip as u64,
                    "invalid Opus end trimming"
                );
                mono.truncate(end as usize);
                ended = true;
            } else {
                ensure!(
                    granule.checked_sub(offset) == Some(decoded),
                    "discontinuous Opus stream"
                );
            }
        }
    }
    // Some voice-message recorders finish on a complete page without setting
    // EOS. Keep those decoded frames; without EOS there is no end trim to apply.
    // Packet/page parsing errors above still reject damaged input.
    if !ended {
        log::warn!("Ogg Opus has no end marker; using all complete decoded packets");
    }
    ensure!(mono.len() > pre_skip, "Opus file contains no audio");
    Ok(Some(crate::audio::resample(
        &mono[pre_skip..],
        48_000,
        16_000,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogg::writing::{PacketWriteEndInfo, PacketWriter};

    fn fixture(channels: u8, gain: i16, end: u64) -> Vec<u8> {
        fixture_with_end_marker(channels, gain, end, true)
    }

    fn fixture_with_end_marker(channels: u8, gain: i16, end: u64, eos: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut writer = PacketWriter::new(&mut bytes);
        let mut header = b"OpusHead".to_vec();
        header.extend([1, channels]);
        header.extend(312_u16.to_le_bytes());
        header.extend(48_000_u32.to_le_bytes());
        header.extend(gain.to_le_bytes());
        header.push(0);
        writer
            .write_packet(header, 1, PacketWriteEndInfo::EndPage, 0)
            .unwrap();
        writer
            .write_packet(
                b"OpusTags\0\0\0\0\0\0\0\0".to_vec(),
                1,
                PacketWriteEndInfo::EndPage,
                0,
            )
            .unwrap();
        let layout = if channels == 1 {
            opus::Channels::Mono
        } else {
            opus::Channels::Stereo
        };
        let mut encoder = opus::Encoder::new(48_000, layout, opus::Application::Audio).unwrap();
        for packet in 0..2 {
            let samples = (0..960)
                .flat_map(|frame| {
                    let sample = (((packet * 960 + frame) as f32 * 440.0 / 48_000.0)
                        * std::f32::consts::TAU)
                        .sin()
                        * 0.1;
                    std::iter::repeat_n(sample, channels as usize)
                })
                .collect::<Vec<_>>();
            let mut encoded = vec![0; 4000];
            let length = encoder.encode_float(&samples, &mut encoded).unwrap();
            encoded.truncate(length);
            writer
                .write_packet(
                    encoded,
                    1,
                    if packet == 0 || !eos {
                        PacketWriteEndInfo::EndPage
                    } else {
                        PacketWriteEndInfo::EndStream
                    },
                    if packet == 0 { 960 } else { end },
                )
                .unwrap();
        }
        bytes
    }

    #[test]
    fn complete_audio_without_eos_matches_a_finalized_stream() {
        let path =
            std::env::temp_dir().join(format!("pyzh-opus-no-eos-{}.ogg", std::process::id()));
        for channels in [1, 2] {
            std::fs::write(&path, fixture(channels, 0, 1920)).unwrap();
            let expected = try_decode(&mut File::open(&path).unwrap())
                .unwrap()
                .unwrap();
            std::fs::write(&path, fixture_with_end_marker(channels, 0, 1920, false)).unwrap();
            let actual = try_decode(&mut File::open(&path).unwrap())
                .unwrap()
                .unwrap();
            assert_eq!(actual.len(), 536);
            assert_eq!(actual, expected);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn opus_trims_frames_applies_gain_and_rejects_bad_endings() {
        let path = std::env::temp_dir().join(format!("pyzh-opus-{}.bin", std::process::id()));
        for channels in [1, 2] {
            let mut energy = Vec::new();
            for gain in [0, 1536] {
                std::fs::write(&path, fixture(channels, gain, 1311)).unwrap();
                let samples = try_decode(&mut File::open(&path).unwrap())
                    .unwrap()
                    .unwrap();
                assert_eq!(samples.len(), 333); // (1311 - 312) / 3, independent of channels.
                energy.push(samples.iter().map(|sample| sample * sample).sum::<f32>());
            }
            assert!((energy[1] / energy[0] - 10_f32.powf(6.0 / 10.0)).abs() < 0.01);
            std::fs::write(&path, fixture(channels, 0, 1)).unwrap();
            assert!(try_decode(&mut File::open(&path).unwrap()).is_err());
            let mut truncated = fixture(channels, 0, 1311);
            truncated.truncate(truncated.len() - 10);
            std::fs::write(&path, truncated).unwrap();
            assert!(try_decode(&mut File::open(&path).unwrap()).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }
}
