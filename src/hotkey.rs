//! Global keyboard tap behind the transcription shortcut.
//!
//! Runs a listen-only HID event tap on a dedicated thread with its own run loop,
//! so it neither depends on the UI main loop nor blocks it. The callback only
//! forwards events over a channel; macOS disables taps whose callback stalls.

use anyhow::{Result, anyhow};
use std::sync::mpsc;

pub const KEY_ESCAPE: u16 = 53;
pub const KEY_RIGHT_COMMAND: u16 = 54;
pub const KEY_LEFT_COMMAND: u16 = 55;
pub const KEY_LEFT_SHIFT: u16 = 56;
pub const KEY_LEFT_OPTION: u16 = 58;
pub const KEY_RIGHT_SHIFT: u16 = 60;
pub const KEY_RIGHT_OPTION: u16 = 61;
pub const KEY_LEFT_CONTROL: u16 = 59;
pub const KEY_RIGHT_CONTROL: u16 = 62;
pub const KEY_FUNCTION: u16 = 63;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeyEvent {
    pub key_code: u16,
    pub pressed: bool,
    pub command: bool,
    pub option: bool,
    pub shift: bool,
    pub control: bool,
}

/// Starts the tap and returns once it is running. Fails if macOS refuses the
/// tap, which happens without Accessibility (or Input Monitoring) access.
#[cfg(target_os = "macos")]
pub fn listen(sender: mpsc::Sender<KeyEvent>) -> Result<()> {
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("pyzhscriber-hotkey".into())
        .spawn(move || macos::run(sender, ready_tx))?;
    ready_rx
        .recv()
        .map_err(|_| anyhow!("hotkey thread exited before the event tap started"))?
}

#[cfg(not(target_os = "macos"))]
pub fn listen(_sender: mpsc::Sender<KeyEvent>) -> Result<()> {
    Err(anyhow!("global shortcuts are only supported on macOS"))
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::time::Duration;

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
        fn CGEventTapEnable(tap: *mut c_void, enable: bool);
        fn CGEventGetFlags(event: *mut c_void) -> u64;
        fn CGEventGetIntegerValueField(event: *mut c_void, field: u32) -> i64;
        fn CFMachPortCreateRunLoopSource(
            allocator: *const c_void,
            port: *mut c_void,
            order: isize,
        ) -> *mut c_void;
        fn CFRunLoopGetCurrent() -> *mut c_void;
        fn CFRunLoopAddSource(run_loop: *mut c_void, source: *mut c_void, mode: *const c_void);
        fn CFRunLoopRun();
        static kCFRunLoopCommonModes: *const c_void;
    }

    const HID_EVENT_TAP: u32 = 0;
    const HEAD_INSERT: u32 = 0;
    const LISTEN_ONLY: u32 = 1;
    const KEY_DOWN: u32 = 10;
    const KEY_UP: u32 = 11;
    const FLAGS_CHANGED: u32 = 12;
    const TAP_DISABLED_BY_TIMEOUT: u32 = 0xFFFF_FFFE;
    const TAP_DISABLED_BY_USER_INPUT: u32 = 0xFFFF_FFFF;
    const KEYBOARD_EVENTS: u64 = (1 << KEY_DOWN) | (1 << KEY_UP) | (1 << FLAGS_CHANGED);
    const KEYBOARD_EVENT_KEYCODE: u32 = 9;

    const FLAG_SHIFT: u64 = 1 << 17;
    const FLAG_CONTROL: u64 = 1 << 18;
    const FLAG_OPTION: u64 = 1 << 19;
    const FLAG_COMMAND: u64 = 1 << 20;
    const FLAG_FUNCTION: u64 = 1 << 23;

    struct TapState {
        tap: Cell<*mut c_void>,
        held_modifiers: Cell<HashSet<u16>>,
        sender: mpsc::Sender<KeyEvent>,
    }

    pub fn run(sender: mpsc::Sender<KeyEvent>, ready: mpsc::Sender<Result<()>>) {
        let state = Box::leak(Box::new(TapState {
            tap: Cell::new(std::ptr::null_mut()),
            held_modifiers: Cell::new(HashSet::new()),
            sender,
        }));
        unsafe {
            let tap = CGEventTapCreate(
                HID_EVENT_TAP,
                HEAD_INSERT,
                LISTEN_ONLY,
                KEYBOARD_EVENTS,
                callback,
                (state as *mut TapState).cast(),
            );
            if tap.is_null() {
                let _ = ready.send(Err(anyhow!(
                    "macOS refused the keyboard event tap; grant Accessibility access"
                )));
                return;
            }
            state.tap.set(tap);
            let source = CFMachPortCreateRunLoopSource(std::ptr::null(), tap, 0);
            if source.is_null() {
                let _ = ready.send(Err(anyhow!(
                    "could not create the event tap run loop source"
                )));
                return;
            }
            CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);
            CGEventTapEnable(tap, true);
            let _ = ready.send(Ok(()));
            start_watchdog(tap);
            CFRunLoopRun();
        }
    }

    /// macOS disables an event tap whose run loop is starved for too long, which
    /// happens while Metal transcription saturates the machine. Left alone, the
    /// tap only recovers on the next event that flushes the disable notification,
    /// so a shortcut press right after transcription is silently dropped. This
    /// keeps the tap enabled independently of the callback.
    fn start_watchdog(tap: *mut c_void) {
        let tap_addr = tap as usize;
        std::thread::Builder::new()
            .name("pyzhscriber-hotkey-watchdog".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_millis(500));
                    unsafe { CGEventTapEnable(tap_addr as *mut c_void, true) };
                }
            })
            .ok();
    }

    /// Modifier keys arrive as FlagsChanged without an up/down direction. The
    /// device-specific flag bit tells whether the key is now held; keyboards
    /// that omit those bits fall back to toggling against the last known state.
    fn modifier_masks(key_code: u16) -> Option<(u64, u64)> {
        let (device, generic) = match key_code {
            KEY_RIGHT_COMMAND => (0x10, FLAG_COMMAND),
            KEY_LEFT_COMMAND => (0x08, FLAG_COMMAND),
            KEY_LEFT_SHIFT => (0x02, FLAG_SHIFT),
            KEY_RIGHT_SHIFT => (0x04, FLAG_SHIFT),
            KEY_LEFT_OPTION => (0x20, FLAG_OPTION),
            KEY_RIGHT_OPTION => (0x40, FLAG_OPTION),
            KEY_LEFT_CONTROL => (0x01, FLAG_CONTROL),
            KEY_RIGHT_CONTROL => (0x2000, FLAG_CONTROL),
            KEY_FUNCTION => (FLAG_FUNCTION, FLAG_FUNCTION),
            _ => return None,
        };
        Some((device, generic))
    }

    unsafe extern "C" fn callback(
        _proxy: *mut c_void,
        kind: u32,
        event: *mut c_void,
        user_info: *mut c_void,
    ) -> *mut c_void {
        let state = unsafe { &*(user_info as *const TapState) };
        match kind {
            TAP_DISABLED_BY_TIMEOUT | TAP_DISABLED_BY_USER_INPUT => {
                log::warn!("keyboard event tap was disabled by macOS; re-enabling");
                unsafe { CGEventTapEnable(state.tap.get(), true) };
                return event;
            }
            KEY_DOWN | KEY_UP | FLAGS_CHANGED => {}
            _ => return event,
        }
        let key_code = unsafe { CGEventGetIntegerValueField(event, KEYBOARD_EVENT_KEYCODE) } as u16;
        let flags = unsafe { CGEventGetFlags(event) };
        let pressed = match kind {
            KEY_DOWN => true,
            KEY_UP => false,
            _ => {
                let Some((device_mask, generic_mask)) = modifier_masks(key_code) else {
                    return event;
                };
                let mut held = state.held_modifiers.take();
                let pressed = if flags & generic_mask == 0 {
                    false
                } else if flags & device_mask != 0 {
                    true
                } else {
                    !held.contains(&key_code)
                };
                if pressed {
                    held.insert(key_code);
                } else {
                    held.remove(&key_code);
                }
                state.held_modifiers.set(held);
                pressed
            }
        };
        let _ = state.sender.send(KeyEvent {
            key_code,
            pressed,
            command: flags & FLAG_COMMAND != 0,
            option: flags & FLAG_OPTION != 0,
            shift: flags & FLAG_SHIFT != 0,
            control: flags & FLAG_CONTROL != 0,
        });
        event
    }
}

/// Modifier keys that can be part of a chord. The fn key is deliberately not
/// here: macOS also sets its flag for arrows, Home and friends, so it only
/// works reliably as a lone shortcut.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Modifiers {
    pub control: bool,
    pub option: bool,
    pub shift: bool,
    pub command: bool,
}

impl Modifiers {
    fn from_event(event: KeyEvent) -> Self {
        Self {
            control: event.control,
            option: event.option,
            shift: event.shift,
            command: event.command,
        }
    }

    fn is_empty(self) -> bool {
        self == Self::default()
    }

    /// `⌃⌥⇧⌘` in the order macOS prints them.
    pub fn symbols(self) -> String {
        let mut out = String::new();
        for (held, symbol) in [
            (self.control, '⌃'),
            (self.option, '⌥'),
            (self.shift, '⇧'),
            (self.command, '⌘'),
        ] {
            if held {
                out.push(symbol);
            }
        }
        out
    }
}

/// The transcription shortcut: either a single modifier key such as Right
/// Command, or a chord like `option+command+space`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Shortcut {
    /// One modifier key on its own, identified by its side-specific key code.
    Modifier(u16),
    Chord {
        key_code: u16,
        modifiers: Modifiers,
    },
}

/// (key code, id used in settings, label shown in the UI)
const KEYS: &[(u16, &str, &str)] = &[
    (0, "a", "A"),
    (1, "s", "S"),
    (2, "d", "D"),
    (3, "f", "F"),
    (4, "h", "H"),
    (5, "g", "G"),
    (6, "z", "Z"),
    (7, "x", "X"),
    (8, "c", "C"),
    (9, "v", "V"),
    (10, "section", "§"),
    (11, "b", "B"),
    (12, "q", "Q"),
    (13, "w", "W"),
    (14, "e", "E"),
    (15, "r", "R"),
    (16, "y", "Y"),
    (17, "t", "T"),
    (18, "1", "1"),
    (19, "2", "2"),
    (20, "3", "3"),
    (21, "4", "4"),
    (22, "6", "6"),
    (23, "5", "5"),
    (24, "=", "="),
    (25, "9", "9"),
    (26, "7", "7"),
    (27, "-", "-"),
    (28, "8", "8"),
    (29, "0", "0"),
    (30, "]", "]"),
    (31, "o", "O"),
    (32, "u", "U"),
    (33, "[", "["),
    (34, "i", "I"),
    (35, "p", "P"),
    (36, "return", "Return"),
    (37, "l", "L"),
    (38, "j", "J"),
    (39, "'", "'"),
    (40, "k", "K"),
    (41, ";", ";"),
    (42, "\\", "\\"),
    (43, ",", ","),
    (44, "/", "/"),
    (45, "n", "N"),
    (46, "m", "M"),
    (47, ".", "."),
    (48, "tab", "Tab"),
    (49, "space", "Space"),
    (50, "`", "`"),
    (51, "delete", "Delete"),
    (53, "escape", "Esc"),
    (64, "f17", "F17"),
    (65, "keypad_.", "Keypad ."),
    (67, "keypad_*", "Keypad *"),
    (69, "keypad_+", "Keypad +"),
    (71, "clear", "Clear"),
    (75, "keypad_/", "Keypad /"),
    (76, "keypad_enter", "Keypad Enter"),
    (78, "keypad_-", "Keypad -"),
    (79, "f18", "F18"),
    (80, "f19", "F19"),
    (81, "keypad_=", "Keypad ="),
    (82, "keypad_0", "Keypad 0"),
    (83, "keypad_1", "Keypad 1"),
    (84, "keypad_2", "Keypad 2"),
    (85, "keypad_3", "Keypad 3"),
    (86, "keypad_4", "Keypad 4"),
    (87, "keypad_5", "Keypad 5"),
    (88, "keypad_6", "Keypad 6"),
    (89, "keypad_7", "Keypad 7"),
    (91, "keypad_8", "Keypad 8"),
    (92, "keypad_9", "Keypad 9"),
    (96, "f5", "F5"),
    (97, "f6", "F6"),
    (98, "f7", "F7"),
    (99, "f3", "F3"),
    (100, "f8", "F8"),
    (101, "f9", "F9"),
    (103, "f11", "F11"),
    (105, "f13", "F13"),
    (107, "f14", "F14"),
    (109, "f10", "F10"),
    (111, "f12", "F12"),
    (113, "f15", "F15"),
    (114, "help", "Help"),
    (115, "home", "Home"),
    (116, "page_up", "Page Up"),
    (117, "forward_delete", "Fwd Delete"),
    (118, "f4", "F4"),
    (119, "end", "End"),
    (120, "f2", "F2"),
    (121, "page_down", "Page Down"),
    (122, "f1", "F1"),
    (123, "left", "←"),
    (124, "right", "→"),
    (125, "down", "↓"),
    (126, "up", "↑"),
];

/// (key code, id used in settings, label shown in the UI)
const MODIFIER_KEYS: &[(u16, &str, &str)] = &[
    (KEY_RIGHT_COMMAND, "command_right", "Right ⌘"),
    (KEY_LEFT_COMMAND, "command_left", "Left ⌘"),
    (KEY_LEFT_SHIFT, "shift_left", "Left ⇧"),
    (KEY_RIGHT_SHIFT, "shift_right", "Right ⇧"),
    (KEY_LEFT_OPTION, "option_left", "Left ⌥"),
    (KEY_RIGHT_OPTION, "option_right", "Right ⌥"),
    (KEY_LEFT_CONTROL, "control_left", "Left ⌃"),
    (KEY_RIGHT_CONTROL, "control_right", "Right ⌃"),
    (KEY_FUNCTION, "fn", "fn"),
];

fn is_modifier_key(key_code: u16) -> bool {
    MODIFIER_KEYS.iter().any(|(code, _, _)| *code == key_code)
}

/// Function keys are the only non-modifier keys allowed without a modifier,
/// since a bare letter or Space would fire while typing.
fn is_function_key(key_code: u16) -> bool {
    KEYS.iter().any(|(code, id, _)| {
        *code == key_code && id.starts_with('f') && id.len() > 1 && id[1..].parse::<u8>().is_ok()
    })
}

fn key_id(key_code: u16) -> String {
    KEYS.iter()
        .find(|(code, _, _)| *code == key_code)
        .map_or_else(|| format!("key{key_code}"), |(_, id, _)| (*id).to_string())
}

fn key_label(key_code: u16) -> String {
    KEYS.iter()
        .find(|(code, _, _)| *code == key_code)
        .map_or_else(
            || format!("Key {key_code}"),
            |(_, _, label)| (*label).to_string(),
        )
}

fn key_from_id(id: &str) -> Option<u16> {
    if let Some(code) = id.strip_prefix("key") {
        if let Ok(code) = code.parse() {
            return Some(code);
        }
    }
    KEYS.iter()
        .find(|(_, key, _)| *key == id)
        .map(|(code, _, _)| *code)
}

impl Shortcut {
    /// Parses a settings binding such as `command_right`, `fn` or
    /// `control+option+space`.
    pub fn parse(binding: &str) -> Option<Self> {
        if let Some((code, _, _)) = MODIFIER_KEYS.iter().find(|(_, id, _)| *id == binding) {
            return Some(Self::Modifier(*code));
        }
        let mut modifiers = Modifiers::default();
        let mut key_code = None;
        for part in binding.split('+') {
            match part {
                "control" | "ctrl" => modifiers.control = true,
                "option" | "alt" => modifiers.option = true,
                "shift" => modifiers.shift = true,
                "command" | "cmd" => modifiers.command = true,
                _ => key_code = Some(key_from_id(part)?),
            }
        }
        Some(Self::Chord {
            key_code: key_code?,
            modifiers,
        })
    }

    /// Human label such as `Right ⌘` or `⌥ ⌘ Space`.
    pub fn label(self) -> String {
        match self {
            Self::Modifier(code) => MODIFIER_KEYS
                .iter()
                .find(|(key, _, _)| *key == code)
                .map_or_else(
                    || format!("Key {code}"),
                    |(_, _, label)| (*label).to_string(),
                ),
            Self::Chord {
                key_code,
                modifiers,
            } => {
                let symbols = modifiers.symbols();
                let mut parts: Vec<String> = symbols.chars().map(String::from).collect();
                parts.push(key_label(key_code));
                parts.join(" ")
            }
        }
    }

    /// The main key went down. Chords need exactly their modifiers so that a
    /// binding like ⌥ Space does not also fire on ⌥ ⌘ Space.
    pub fn is_pressed(self, event: KeyEvent) -> bool {
        if !event.pressed {
            return false;
        }
        match self {
            Self::Modifier(code) => event.key_code == code,
            Self::Chord {
                key_code,
                modifiers,
            } => event.key_code == key_code && Modifiers::from_event(event) == modifiers,
        }
    }

    /// Any key that is part of the shortcut went up, in whatever order.
    pub fn is_released(self, event: KeyEvent) -> bool {
        if event.pressed {
            return false;
        }
        match self {
            Self::Modifier(code) => event.key_code == code,
            Self::Chord {
                key_code,
                modifiers,
            } => {
                event.key_code == key_code
                    || (modifiers.command
                        && matches!(event.key_code, KEY_LEFT_COMMAND | KEY_RIGHT_COMMAND))
                    || (modifiers.option
                        && matches!(event.key_code, KEY_LEFT_OPTION | KEY_RIGHT_OPTION))
                    || (modifiers.shift
                        && matches!(event.key_code, KEY_LEFT_SHIFT | KEY_RIGHT_SHIFT))
                    || (modifiers.control
                        && matches!(event.key_code, KEY_LEFT_CONTROL | KEY_RIGHT_CONTROL))
            }
        }
    }
}

impl std::fmt::Display for Shortcut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Modifier(code) => {
                let id = MODIFIER_KEYS
                    .iter()
                    .find(|(key, _, _)| *key == code)
                    .map_or("fn", |(_, id, _)| *id);
                f.write_str(id)
            }
            Self::Chord {
                key_code,
                modifiers,
            } => {
                for (held, id) in [
                    (modifiers.control, "control"),
                    (modifiers.option, "option"),
                    (modifiers.shift, "shift"),
                    (modifiers.command, "command"),
                ] {
                    if held {
                        f.write_str(id)?;
                        f.write_str("+")?;
                    }
                }
                f.write_str(&key_id(key_code))
            }
        }
    }
}

/// Outcome of feeding one key event to a [`ShortcutRecorder`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecorderStep {
    /// Still waiting; carries the modifiers currently held for a live preview.
    Listening(String),
    /// A bare key that would fire while typing was pressed.
    NeedsModifier,
    Captured(Shortcut),
    Cancelled,
}

/// Turns raw tap events into a shortcut: a chord is captured on the first
/// non-modifier key down, a lone modifier on its release if nothing else was
/// pressed while it was held. Escape on its own cancels.
#[derive(Debug, Default)]
pub struct ShortcutRecorder {
    held: Vec<u16>,
    chorded: bool,
}

impl ShortcutRecorder {
    pub fn feed(&mut self, event: KeyEvent) -> RecorderStep {
        let key = event.key_code;
        if is_modifier_key(key) {
            if event.pressed {
                if !self.held.contains(&key) {
                    self.held.push(key);
                }
                self.chorded |= self.held.len() > 1;
                return RecorderStep::Listening(self.held_label());
            }
            let lone = !self.chorded && self.held == [key];
            self.held.retain(|held| *held != key);
            if self.held.is_empty() {
                self.chorded = false;
            }
            return if lone {
                RecorderStep::Captured(Shortcut::Modifier(key))
            } else {
                RecorderStep::Listening(self.held_label())
            };
        }
        if !event.pressed {
            return RecorderStep::Listening(self.held_label());
        }
        let modifiers = Modifiers::from_event(event);
        if key == KEY_ESCAPE && modifiers.is_empty() {
            return RecorderStep::Cancelled;
        }
        if modifiers.is_empty() && !is_function_key(key) {
            self.chorded = true;
            return RecorderStep::NeedsModifier;
        }
        self.chorded = true;
        RecorderStep::Captured(Shortcut::Chord {
            key_code: key,
            modifiers,
        })
    }

    fn held_label(&self) -> String {
        self.held
            .iter()
            .filter_map(|code| MODIFIER_KEYS.iter().find(|(key, _, _)| key == code))
            .map(|(_, _, label)| label.rsplit(' ').next().unwrap_or(label))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_SPACE: u16 = 49;

    fn event(key_code: u16, pressed: bool, command: bool, option: bool) -> KeyEvent {
        KeyEvent {
            key_code,
            pressed,
            command,
            option,
            shift: false,
            control: false,
        }
    }

    #[test]
    fn right_command_matches_only_itself() {
        let shortcut = Shortcut::parse("command_right").unwrap();
        assert!(shortcut.is_pressed(event(KEY_RIGHT_COMMAND, true, true, false)));
        assert!(!shortcut.is_pressed(event(KEY_LEFT_COMMAND, true, true, false)));
        assert!(shortcut.is_released(event(KEY_RIGHT_COMMAND, false, false, false)));
        assert!(!shortcut.is_released(event(KEY_LEFT_COMMAND, false, false, false)));
    }

    #[test]
    fn chords_need_exact_modifiers_and_release_on_any_part() {
        let shortcut = Shortcut::parse("option+command+space").unwrap();
        assert!(shortcut.is_pressed(event(KEY_SPACE, true, true, true)));
        assert!(!shortcut.is_pressed(event(KEY_SPACE, true, false, true)));
        assert!(shortcut.is_released(event(KEY_LEFT_OPTION, false, true, false)));
        assert!(shortcut.is_released(event(KEY_SPACE, false, true, true)));
        assert!(!shortcut.is_released(event(KEY_LEFT_SHIFT, false, true, true)));
        let narrower = Shortcut::parse("option+space").unwrap();
        assert!(!narrower.is_pressed(event(KEY_SPACE, true, true, true)));
        assert_eq!(Shortcut::parse("hyper+space"), None);
        assert_eq!(Shortcut::parse("option+command"), None);
    }

    #[test]
    fn bindings_round_trip_and_keep_legacy_ids() {
        for binding in [
            "command_right",
            "fn",
            "option+space",
            "control+option+shift+command+k",
            "f5",
            "command+key200",
        ] {
            assert_eq!(Shortcut::parse(binding).unwrap().to_string(), binding);
        }
        assert_eq!(Shortcut::parse("command_right").unwrap().label(), "Right ⌘");
        assert_eq!(
            Shortcut::parse("option+command+space").unwrap().label(),
            "⌥ ⌘ Space"
        );
        assert_eq!(
            Shortcut::parse("command+key200").unwrap().label(),
            "⌘ Key 200"
        );
    }

    #[test]
    fn recorder_captures_chords_lone_modifiers_and_rejects_bare_keys() {
        let mut recorder = ShortcutRecorder::default();
        assert_eq!(
            recorder.feed(event(KEY_LEFT_OPTION, true, false, true)),
            RecorderStep::Listening("⌥".into())
        );
        assert_eq!(
            recorder.feed(event(KEY_SPACE, true, false, true)),
            RecorderStep::Captured(Shortcut::parse("option+space").unwrap())
        );

        let mut recorder = ShortcutRecorder::default();
        recorder.feed(event(KEY_RIGHT_COMMAND, true, true, false));
        assert_eq!(
            recorder.feed(event(KEY_RIGHT_COMMAND, false, false, false)),
            RecorderStep::Captured(Shortcut::Modifier(KEY_RIGHT_COMMAND))
        );

        // Two modifiers held together never resolve to a lone modifier.
        let mut recorder = ShortcutRecorder::default();
        recorder.feed(event(KEY_LEFT_COMMAND, true, true, false));
        recorder.feed(event(KEY_LEFT_OPTION, true, true, true));
        recorder.feed(event(KEY_LEFT_OPTION, false, true, false));
        assert_eq!(
            recorder.feed(event(KEY_LEFT_COMMAND, false, false, false)),
            RecorderStep::Listening(String::new())
        );

        let mut recorder = ShortcutRecorder::default();
        assert_eq!(
            recorder.feed(event(0, true, false, false)),
            RecorderStep::NeedsModifier
        );
        assert_eq!(
            recorder.feed(event(96, true, false, false)),
            RecorderStep::Captured(Shortcut::parse("f5").unwrap())
        );
        assert_eq!(
            recorder.feed(event(KEY_ESCAPE, true, false, false)),
            RecorderStep::Cancelled
        );
    }
}
