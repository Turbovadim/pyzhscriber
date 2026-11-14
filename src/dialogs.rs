#[cfg(target_os = "macos")]
pub fn prompt(title: &str, detail: &str, value: &str, secret: bool) -> Option<String> {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSAlert, NSAlertFirstButtonReturn, NSTextField};
    use objc2_foundation::{NSSize, NSString};

    let mtm = MainThreadMarker::new()?;
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str(title));
    alert.setInformativeText(&NSString::from_str(detail));
    alert.addButtonWithTitle(&NSString::from_str("Save"));
    alert.addButtonWithTitle(&NSString::from_str("Cancel"));

    let field = NSTextField::textFieldWithString(&NSString::from_str(value), mtm);
    field.setFrameSize(NSSize::new(420.0, 24.0));
    if secret {
        field.setStringValue(&NSString::from_str(""));
        field.setPlaceholderString(Some(&NSString::from_str(if value.is_empty() {
            "API key"
        } else {
            "API key is already saved. Enter a replacement."
        })));
    }
    alert.setAccessoryView(Some(&field));
    let response = alert.runModal();
    (response == NSAlertFirstButtonReturn).then(|| field.stringValue().to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn prompt(_title: &str, _detail: &str, _value: &str, _secret: bool) -> Option<String> {
    None
}

/// Asks before a destructive action; `action` labels the confirming button.
#[cfg(target_os = "macos")]
pub fn confirm(title: &str, detail: &str, action: &str) -> bool {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSAlert, NSAlertFirstButtonReturn};
    use objc2_foundation::NSString;

    let Some(mtm) = MainThreadMarker::new() else {
        return false;
    };
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str(title));
    alert.setInformativeText(&NSString::from_str(detail));
    alert.addButtonWithTitle(&NSString::from_str(action));
    alert.addButtonWithTitle(&NSString::from_str("Cancel"));
    alert.runModal() == NSAlertFirstButtonReturn
}

#[cfg(not(target_os = "macos"))]
pub fn confirm(_title: &str, _detail: &str, _action: &str) -> bool {
    false
}

pub fn open_privacy_settings(section: &str) {
    #[cfg(target_os = "macos")]
    {
        let anchor = match section {
            "microphone" => "Privacy_Microphone",
            "input-monitoring" => "Privacy_ListenEvent",
            _ => "Privacy_Accessibility",
        };
        let _ = std::process::Command::new("open")
            .arg(format!(
                "x-apple.systempreferences:com.apple.preference.security?{anchor}"
            ))
            .spawn();
    }
}
