use rodio::OutputStreamBuilder;
use std::io::{BufReader, Cursor};

#[derive(Clone, Copy)]
pub enum FeedbackSound {
    Start,
    Stop,
}

pub fn play(sound: FeedbackSound) {
    std::thread::spawn(move || {
        if let Err(error) = play_blocking(sound) {
            log::warn!("feedback sound failed: {error}");
        }
    });
}

fn play_blocking(sound: FeedbackSound) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: &'static [u8] = match sound {
        FeedbackSound::Start => include_bytes!("../resources/marimba_start.wav"),
        FeedbackSound::Stop => include_bytes!("../resources/marimba_stop.wav"),
    };
    let stream = OutputStreamBuilder::from_default_device()?.open_stream()?;
    let sink = rodio::play(stream.mixer(), BufReader::new(Cursor::new(bytes)))?;
    sink.set_volume(0.7);
    sink.sleep_until_end();
    Ok(())
}
