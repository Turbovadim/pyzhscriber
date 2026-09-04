/// Below this frame energy the microphone delivered nothing usable. The gate is
/// otherwise relative to the loudest frame, because a quiet microphone still
/// carries speech that an absolute threshold would discard entirely.
const SILENCE_FLOOR: f32 = 0.0005;

/// Removes leading and trailing silence with a relative energy gate.
/// It keeps 180 ms of padding so consonants at either edge survive.
/// Only genuinely silent recordings come back empty.
pub fn trim_silence(audio: &[f32]) -> Vec<f32> {
    const FRAME: usize = 320;
    const PADDING: usize = 2_880;
    if audio.len() < FRAME * 2 {
        return audio.to_vec();
    }

    let energies = audio
        .chunks(FRAME)
        .map(|frame| {
            let sum = frame.iter().map(|sample| sample * sample).sum::<f32>();
            (sum / frame.len() as f32).sqrt()
        })
        .collect::<Vec<_>>();
    let peak = energies.iter().copied().fold(0.0_f32, f32::max);
    if peak < SILENCE_FLOOR {
        return Vec::new();
    }
    // The loudest frame always clears a purely relative gate, so audio that has
    // any signal in it can never be trimmed away completely.
    let threshold = peak * 0.08;
    let Some(first_frame) = energies.iter().position(|energy| *energy >= threshold) else {
        return audio.to_vec();
    };
    let last_frame = energies
        .iter()
        .rposition(|energy| *energy >= threshold)
        .unwrap_or(first_frame);
    let start = (first_frame * FRAME).saturating_sub(PADDING);
    let end = ((last_frame + 1) * FRAME + PADDING).min(audio.len());
    audio[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_silence_but_keeps_padding() {
        let mut audio = vec![0.0; 8_000];
        audio.extend(vec![0.2; 16_000]);
        audio.extend(vec![0.0; 8_000]);
        let trimmed = trim_silence(&audio);
        assert!(trimmed.len() > 16_000);
        assert!(trimmed.len() < audio.len());
    }

    #[test]
    fn rejects_silent_recording() {
        assert!(trim_silence(&vec![0.0; 16_000]).is_empty());
    }

    #[test]
    fn keeps_speech_from_a_quiet_microphone() {
        let mut audio = vec![0.0; 8_000];
        audio.extend((0..16_000).map(|index| if index % 2 == 0 { 0.006 } else { -0.006 }));
        audio.extend(vec![0.0; 8_000]);
        let trimmed = trim_silence(&audio);
        assert!(!trimmed.is_empty());
        assert!(trimmed.len() >= 16_000);
    }
}
