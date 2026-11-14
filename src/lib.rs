mod app;
mod assets;
mod audio;
mod cli;
mod cli_install;
mod cloud;
mod controller;
mod dialogs;
mod history;
mod hotkey;
mod ipc;
mod logging;
mod models;
mod opus_audio;
mod output;
mod permissions;
mod post_process;
mod settings;
mod sound;
mod startup;
mod theme;
mod transcription;
mod tray;
mod ui;
mod vad;

pub fn main_entry() {
    logging::init();
    match cli::run_if_requested() {
        Ok(true) => {}
        Ok(false) => app::run(),
        Err(error) => {
            eprintln!("Pyzhscriber: {error:#}");
            std::process::exit(1);
        }
    }
}

/// Renders deterministic off-screen screenshots without focusing a macOS window.
#[cfg(feature = "visual-qa")]
pub fn render_visual_qa(output: &std::path::Path) -> anyhow::Result<()> {
    use gpui::{AppContext as _, HeadlessAppContext};
    use std::sync::Arc;

    std::fs::create_dir_all(output)?;
    let controller = controller::AppController::new()?;
    let platform = gpui_platform::current_platform(true);
    let mut cx = HeadlessAppContext::with_platform(
        platform.text_system(),
        Arc::new(assets::Assets),
        gpui_platform::current_headless_renderer,
    );
    let window = cx.open_window(gpui::size(gpui::px(960.0), gpui::px(700.0)), {
        let controller = controller.clone();
        move |_, cx| cx.new(|cx| ui::RootView::new(controller, cx))
    })?;
    window.update(&mut cx, |view, _, cx| view.prepare_qa_general(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("general.png"))?;

    window.update(&mut cx, |view, _, cx| {
        view.open_qa_dropdown("route-model-1", cx)
    })?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("general-dropdown.png"))?;

    window.update(&mut cx, |view, _, cx| view.prepare_qa_single_languages(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("general-languages.png"))?;

    window.update(&mut cx, |view, _, cx| view.prepare_qa_shortcut_capture(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("general-shortcut-capture.png"))?;

    for page in ["models", "audio", "output", "history", "advanced"] {
        window.update(&mut cx, |view, _, cx| view.prepare_qa_page(page, cx))?;
        cx.run_until_parked();
        cx.capture_screenshot(window.into())?
            .save(output.join(format!("{page}.png")))?;
    }

    window.update(&mut cx, |view, _, cx| view.prepare_qa_model_filters(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("models-filtered.png"))?;

    // Cloud sits below the fold, so it gets a window tall enough to show it.
    let tall = cx.open_window(gpui::size(gpui::px(960.0), gpui::px(1500.0)), {
        let controller = controller.clone();
        move |_, cx| cx.new(|cx| ui::RootView::new(controller, cx))
    })?;
    tall.update(&mut cx, |view, _, cx| view.prepare_qa_cloud(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(tall.into())?
        .save(output.join("general-cloud.png"))?;

    window.update(&mut cx, |view, _, cx| view.prepare_qa_error(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("error.png"))?;

    window.update(&mut cx, |view, _, cx| view.prepare_qa_permissions(cx))?;
    cx.run_until_parked();
    cx.capture_screenshot(window.into())?
        .save(output.join("permissions.png"))?;

    let recording_overlay = cx.open_window(ui::overlay_window_size(), {
        let controller = controller.clone();
        move |_, cx| {
            cx.new(|_| {
                ui::RecordingOverlay::qa(controller, controller::RuntimeStatus::Recording, false)
            })
        }
    })?;
    cx.run_until_parked();
    cx.capture_screenshot(recording_overlay.into())?
        .save(output.join("overlay-recording.png"))?;

    let transcribing_overlay = cx.open_window(ui::overlay_window_size(), {
        let controller = controller.clone();
        move |_, cx| {
            cx.new(|_| {
                ui::RecordingOverlay::qa(controller, controller::RuntimeStatus::Transcribing, false)
            })
        }
    })?;
    cx.run_until_parked();
    cx.capture_screenshot(transcribing_overlay.into())?
        .save(output.join("overlay-transcribing.png"))?;

    let bubbly_overlay = cx.open_window(ui::overlay_window_size(), {
        let controller = controller.clone();
        move |_, cx| {
            cx.new(|_| {
                ui::RecordingOverlay::qa(controller, controller::RuntimeStatus::Recording, true)
            })
        }
    })?;
    cx.run_until_parked();
    cx.capture_screenshot(bubbly_overlay.into())?
        .save(output.join("overlay-bubbly.png"))?;

    for (name, opened_ago, leaving_ago) in [
        ("overlay-bubbly-opening", 0.15, None),
        ("overlay-bubbly-closing", 5.0, Some(0.2)),
    ] {
        let frame = cx.open_window(ui::overlay_window_size(), {
            let controller = controller.clone();
            move |_, cx| {
                cx.new(|_| {
                    ui::RecordingOverlay::qa(controller, controller::RuntimeStatus::Recording, true)
                        .qa_frame(opened_ago, leaving_ago)
                })
            }
        })?;
        cx.run_until_parked();
        cx.capture_screenshot(frame.into())?
            .save(output.join(format!("{name}.png")))?;
    }
    Ok(())
}
