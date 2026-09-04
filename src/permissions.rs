#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermissionStatus {
    NotDetermined,
    Denied,
    Granted,
}

impl PermissionStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::NotDetermined => "Not requested",
            Self::Denied => "Not granted",
            Self::Granted => "Granted",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PermissionStatuses {
    pub microphone: PermissionStatus,
    pub accessibility: PermissionStatus,
    pub input_monitoring: PermissionStatus,
}

impl PermissionStatuses {
    pub fn current() -> Self {
        Self {
            microphone: microphone_status(),
            accessibility: accessibility_status(),
            input_monitoring: input_monitoring_status(),
        }
    }

    /// The microphone records, Accessibility pastes the transcript, and Input
    /// Monitoring lets the keyboard tap see events from other apps. Without the
    /// last one macOS still creates the tap but delivers only our own events, so
    /// the global shortcut silently does nothing outside this app.
    pub fn all_granted(self) -> bool {
        self.microphone == PermissionStatus::Granted
            && self.accessibility == PermissionStatus::Granted
            && self.input_monitoring == PermissionStatus::Granted
    }
}

/// Ground truth for the global shortcut: creates the same listen-only keyboard
/// tap rdev uses and releases it right away.
#[cfg(target_os = "macos")]
pub fn event_tap_available() -> bool {
    use std::ffi::c_void;

    type TapCallback =
        unsafe extern "C" fn(*mut c_void, u32, *mut c_void, *mut c_void) -> *mut c_void;

    unsafe extern "C" {
        fn CGEventTapCreate(
            tap: u32,
            place: u32,
            options: u32,
            events_of_interest: u64,
            callback: TapCallback,
            user_info: *mut c_void,
        ) -> *mut c_void;
        fn CFRelease(cf: *const c_void);
    }

    unsafe extern "C" fn passthrough(
        _proxy: *mut c_void,
        _kind: u32,
        event: *mut c_void,
        _user_info: *mut c_void,
    ) -> *mut c_void {
        event
    }

    const HID_EVENT_TAP: u32 = 0;
    const HEAD_INSERT: u32 = 0;
    const LISTEN_ONLY: u32 = 1;
    // kCGEventKeyDown, kCGEventKeyUp, kCGEventFlagsChanged
    const KEYBOARD_EVENTS: u64 = (1 << 10) | (1 << 11) | (1 << 12);

    let tap = unsafe {
        CGEventTapCreate(
            HID_EVENT_TAP,
            HEAD_INSERT,
            LISTEN_ONLY,
            KEYBOARD_EVENTS,
            passthrough,
            std::ptr::null_mut(),
        )
    };
    if tap.is_null() {
        return false;
    }
    unsafe { CFRelease(tap) };
    true
}

#[cfg(not(target_os = "macos"))]
pub fn event_tap_available() -> bool {
    true
}

#[cfg(target_os = "macos")]
pub fn input_monitoring_status() -> PermissionStatus {
    unsafe extern "C" {
        fn CGPreflightListenEventAccess() -> bool;
    }

    if unsafe { CGPreflightListenEventAccess() } {
        PermissionStatus::Granted
    } else {
        PermissionStatus::Denied
    }
}

#[cfg(not(target_os = "macos"))]
pub fn input_monitoring_status() -> PermissionStatus {
    PermissionStatus::Granted
}

/// macOS only prompts the first time. Once the entry is an explicit deny the
/// request returns immediately, so fall back to opening the settings pane.
#[cfg(target_os = "macos")]
pub fn request_input_monitoring() {
    unsafe extern "C" {
        fn CGRequestListenEventAccess() -> bool;
    }

    if !unsafe { CGRequestListenEventAccess() } {
        crate::dialogs::open_privacy_settings("input-monitoring");
    }
}

#[cfg(not(target_os = "macos"))]
pub fn request_input_monitoring() {}

#[cfg(target_os = "macos")]
pub fn accessibility_status() -> PermissionStatus {
    unsafe extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }

    if unsafe { AXIsProcessTrusted() } {
        PermissionStatus::Granted
    } else {
        PermissionStatus::Denied
    }
}

#[cfg(not(target_os = "macos"))]
pub fn accessibility_status() -> PermissionStatus {
    PermissionStatus::Granted
}

#[cfg(target_os = "macos")]
pub fn request_accessibility() {
    use objc2::{class, msg_send, runtime::AnyObject};
    use std::ffi::c_void;

    unsafe extern "C" {
        static kAXTrustedCheckOptionPrompt: *const c_void;
        static kCFBooleanTrue: *const c_void;
        fn AXIsProcessTrustedWithOptions(options: *const c_void) -> bool;
    }

    unsafe {
        // NSDictionary and CFDictionary are toll-free bridged. Passing the documented
        // prompt option makes macOS show the Accessibility permission dialog.
        let options: *mut AnyObject = msg_send![
            class!(NSDictionary),
            dictionaryWithObject: kCFBooleanTrue,
            forKey: kAXTrustedCheckOptionPrompt
        ];
        let _ = AXIsProcessTrustedWithOptions(options.cast());
    }
}

#[cfg(not(target_os = "macos"))]
pub fn request_accessibility() {}

#[cfg(target_os = "macos")]
pub fn microphone_status() -> PermissionStatus {
    use objc2::{class, msg_send};
    use objc2_foundation::NSString;

    let media_type = NSString::from_str("soun");
    let status: isize = unsafe {
        msg_send![
            class!(AVCaptureDevice),
            authorizationStatusForMediaType: &*media_type
        ]
    };
    match status {
        0 => PermissionStatus::NotDetermined,
        3 => PermissionStatus::Granted,
        _ => PermissionStatus::Denied,
    }
}

#[cfg(not(target_os = "macos"))]
pub fn microphone_status() -> PermissionStatus {
    PermissionStatus::Granted
}

#[cfg(target_os = "macos")]
pub fn request_microphone() {
    use block2::RcBlock;
    use objc2::{class, msg_send, runtime::Bool};
    use objc2_foundation::NSString;

    if microphone_status() == PermissionStatus::Denied {
        crate::dialogs::open_privacy_settings("microphone");
        return;
    }

    let media_type = NSString::from_str("soun");
    let completion = RcBlock::new(|_granted: Bool| {});
    unsafe {
        let _: () = msg_send![
            class!(AVCaptureDevice),
            requestAccessForMediaType: &*media_type,
            completionHandler: &*completion
        ];
    }
}

#[cfg(not(target_os = "macos"))]
pub fn request_microphone() {}
