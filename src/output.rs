use crate::settings::{AppSettings, PasteMethod};
use anyhow::Result;
use arboard::Clipboard;
use enigo::{Enigo, Keyboard, Settings};
use std::thread;
use std::time::Duration;

pub fn deliver(text: &str, settings: &AppSettings) -> Result<()> {
    log::info!(
        "delivering {} chars via {:?}; frontmost app: {}",
        text.chars().count(),
        settings.paste_method,
        frontmost_bundle_id()
    );
    match settings.paste_method {
        PasteMethod::DirectTyping => {
            let mut enigo = Enigo::new(&Settings::default())?;
            enigo.text(text)?;
        }
        PasteMethod::ClipboardOnly => {
            Clipboard::new()?.set_text(text)?;
        }
        PasteMethod::CommandV => paste_with_clipboard(text, settings)?,
    }
    Ok(())
}

fn paste_with_clipboard(text: &str, settings: &AppSettings) -> Result<()> {
    let mut clipboard = Clipboard::new()?;
    let previous = settings
        .preserve_clipboard
        .then(|| clipboard.get_text().ok())
        .flatten();
    clipboard.set_text(text)?;
    thread::sleep(Duration::from_millis(settings.paste_delay_ms));

    press_paste_shortcut()?;
    log::info!("posted paste keystroke");

    if settings.auto_submit {
        thread::sleep(Duration::from_millis(60));
        press_return()?;
    }
    if let Some(previous) = previous {
        // Give the target app time to read the clipboard before restoring it.
        thread::sleep(Duration::from_millis(400));
        clipboard.set_text(previous)?;
    }
    Ok(())
}

/// Synthesizes Command-V with CoreGraphics. Unlike enigo's default private event
/// source, the combined session source makes the Command modifier flag reliable
/// across apps that consult the live keyboard state.
#[cfg(target_os = "macos")]
fn press_paste_shortcut() -> Result<()> {
    const KEY_V: u16 = 9;
    macos_key::post_with_command(KEY_V)
}

#[cfg(target_os = "macos")]
fn press_return() -> Result<()> {
    const KEY_RETURN: u16 = 36;
    macos_key::post_plain(KEY_RETURN)
}

#[cfg(target_os = "macos")]
mod macos_key {
    use anyhow::{Result, anyhow};
    use std::ffi::c_void;

    const HID_EVENT_TAP: u32 = 0;
    const COMBINED_SESSION_STATE: i32 = 0;
    const FLAG_COMMAND: u64 = 1 << 20;

    unsafe extern "C" {
        fn CGEventSourceCreate(state_id: i32) -> *mut c_void;
        fn CGEventCreateKeyboardEvent(
            source: *mut c_void,
            keycode: u16,
            key_down: bool,
        ) -> *mut c_void;
        fn CGEventSetFlags(event: *mut c_void, flags: u64);
        fn CGEventPost(tap: u32, event: *mut c_void);
        fn CFRelease(cf: *const c_void);
    }

    fn post(keycode: u16, flags: u64) -> Result<()> {
        unsafe {
            let source = CGEventSourceCreate(COMBINED_SESSION_STATE);
            let down = CGEventCreateKeyboardEvent(source, keycode, true);
            let up = CGEventCreateKeyboardEvent(source, keycode, false);
            if down.is_null() || up.is_null() {
                if !source.is_null() {
                    CFRelease(source);
                }
                return Err(anyhow!("could not create paste keyboard event"));
            }
            CGEventSetFlags(down, flags);
            CGEventSetFlags(up, flags);
            CGEventPost(HID_EVENT_TAP, down);
            CGEventPost(HID_EVENT_TAP, up);
            CFRelease(down);
            CFRelease(up);
            if !source.is_null() {
                CFRelease(source);
            }
        }
        Ok(())
    }

    pub fn post_with_command(keycode: u16) -> Result<()> {
        post(keycode, FLAG_COMMAND)
    }

    pub fn post_plain(keycode: u16) -> Result<()> {
        post(keycode, 0)
    }
}

#[cfg(not(target_os = "macos"))]
fn press_paste_shortcut() -> Result<()> {
    Err(anyhow::anyhow!("paste shortcut is only supported on macOS"))
}

#[cfg(not(target_os = "macos"))]
fn press_return() -> Result<()> {
    Err(anyhow::anyhow!("return key is only supported on macOS"))
}

#[cfg(target_os = "macos")]
fn frontmost_bundle_id() -> String {
    use objc2_app_kit::NSWorkspace;
    match NSWorkspace::sharedWorkspace().frontmostApplication() {
        Some(app) => app
            .bundleIdentifier()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "<unknown>".into()),
        None => "<none>".into(),
    }
}

#[cfg(not(target_os = "macos"))]
fn frontmost_bundle_id() -> String {
    String::new()
}
