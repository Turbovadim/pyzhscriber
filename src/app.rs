use crate::assets::Assets;
use crate::controller::AppController;
use crate::tray::AppTray;
use crate::ui::RootView;
use gpui::{
    App, AppContext, Bounds, KeyBinding, Menu, MenuItem, OsAction, SystemMenuType, Window,
    WindowBounds, WindowHandle, WindowOptions, actions, point, px, size,
};
use gpui_platform::application;

#[derive(Clone, Copy)]
struct MainWindow(WindowHandle<RootView>);

impl gpui::Global for MainWindow {}

pub fn run() {
    let instance_path = std::env::temp_dir().join("com.yolki.pyzhscriber.instance.lock");
    let instance = single_instance::SingleInstance::new(&instance_path.to_string_lossy())
        .expect("create single-instance lock");
    if !instance.is_single() {
        return;
    }
    log::info!(
        "Pyzhscriber {} starting from {}",
        env!("CARGO_PKG_VERSION"),
        std::env::current_exe()
            .map(|path| path.display().to_string())
            .unwrap_or_default()
    );
    let controller = AppController::new().expect("initialize Pyzhscriber");
    let _cli_server = crate::ipc::Server::start(&controller)
        .map_err(|error| log::error!("CLI server failed to start: {error:#}"))
        .ok();
    // Always start the tap: it reports its own failure, and Input Monitoring can
    // be granted while the app is running.
    controller.start();

    let application = application().with_assets(Assets);
    application.on_reopen(|cx| {
        if let Some(MainWindow(window)) = cx.try_global::<MainWindow>().copied() {
            show_main_window(window, cx);
        }
    });
    application.run(move |cx: &mut App| {
        gpui_tokio::init(cx);
        let controller_for_quit = controller.clone();
        cx.on_app_quit(move |_| {
            // Free native models synchronously: GPUI limits how long it waits
            // for async quit hooks, but Metal teardown requires every buffer gone.
            controller_for_quit.shutdown();
            std::future::ready(())
        })
        .detach();
        let settings = controller.settings();
        // Set the policy before creating windows or the menu bar item. Recording
        // uses a non-activating panel, keeping the dictation target frontmost.
        set_dock_visible(!settings.hide_from_dock);
        let visible = !settings.start_hidden;
        let bounds = Bounds::centered(None, size(px(960.0), px(700.0)), cx);
        let controller_for_window = controller.clone();
        let main_window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(gpui::TitlebarOptions {
                        title: Some(env!("PYZHSCRIBER_APP_NAME").into()),
                        appears_transparent: true,
                        traffic_light_position: Some(point(px(14.0), px(14.0))),
                    }),
                    is_movable: true,
                    is_minimizable: true,
                    show: visible,
                    focus: visible,
                    window_min_size: Some(size(px(820.0), px(580.0))),
                    app_id: Some(env!("PYZHSCRIBER_BUNDLE_ID").into()),
                    tabbing_identifier: Some("pyzhscriber".into()),
                    ..Default::default()
                },
                move |window, cx| {
                    follow_active_space(window);
                    window.on_window_should_close(cx, |window, cx| {
                        dismiss_main_window(window, cx);
                        false
                    });
                    cx.new(|cx| RootView::new(controller_for_window.clone(), cx))
                },
            )
            .expect("open Pyzhscriber window");
        cx.set_global(MainWindow(main_window));
        register_menus(cx, controller.clone(), main_window);
        AppTray::listen(controller.clone(), main_window, cx);
        // An accessory app orders new windows in behind whatever is frontmost, so
        // without activating here the window is on screen but buried and the
        // launch looks like nothing happened. Recording still never activates,
        // which is what keeps the synthesized paste in the user's text field.
        if visible {
            cx.activate(true);
        }
    });
}

pub(crate) fn show_main_window(main_window: WindowHandle<RootView>, cx: &mut App) {
    cx.activate(true);
    let _ = main_window.update(cx, |_, window, _| window.activate_window());
}

/// GPUI owns the returned NSWindow for the lifetime of `window`.
#[cfg(target_os = "macos")]
fn native_window(window: &Window) -> Option<*mut objc2::runtime::AnyObject> {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let handle = HasWindowHandle::window_handle(window).ok()?;
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return None;
    };
    let view = handle.ns_view.as_ptr().cast::<AnyObject>();
    let native: *mut AnyObject = unsafe { msg_send![view, window] };
    (!native.is_null()).then_some(native)
}

/// A window reopened from another Space stays assigned to the Space it was
/// closed on, which leaves it invisible until something drags it on screen.
/// Moving it to the active Space makes it appear where the user asked for it.
#[cfg(target_os = "macos")]
fn follow_active_space(window: &Window) {
    use objc2::msg_send;

    const MOVE_TO_ACTIVE_SPACE: usize = 1 << 1;

    if let Some(native) = native_window(window) {
        unsafe {
            let behavior: usize = msg_send![native, collectionBehavior];
            let _: () = msg_send![native, setCollectionBehavior: behavior | MOVE_TO_ACTIVE_SPACE];
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn follow_active_space(_: &Window) {}

/// Keep RootView alive to handle recording events, but remove its window from
/// the screen so showing the recording panel cannot restore it with the app.
#[cfg(target_os = "macos")]
fn dismiss_main_window(window: &Window, _: &mut App) {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;

    if let Some(native) = native_window(window) {
        unsafe {
            let _: () = msg_send![native, orderOut: std::ptr::null::<AnyObject>()];
        }
    }
}

/// Open means on screen or minimized to the Dock.
#[cfg(target_os = "macos")]
pub(crate) fn is_main_window_open(window: &Window) -> bool {
    use objc2::msg_send;

    native_window(window).is_some_and(|native| unsafe {
        let visible: bool = msg_send![native, isVisible];
        let minimized: bool = msg_send![native, isMiniaturized];
        visible || minimized
    })
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn is_main_window_open(_: &Window) -> bool {
    true
}

#[cfg(not(target_os = "macos"))]
fn dismiss_main_window(_: &Window, cx: &mut App) {
    cx.hide();
}

#[cfg(target_os = "macos")]
pub(crate) fn set_dock_visible(visible: bool) {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    if let Some(mtm) = MainThreadMarker::new() {
        let app = NSApplication::sharedApplication(mtm);
        let policy = if visible {
            NSApplicationActivationPolicy::Regular
        } else {
            NSApplicationActivationPolicy::Accessory
        };
        if app.activationPolicy() != policy {
            app.setActivationPolicy(policy);
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn set_dock_visible(_: bool) {}

actions!(
    pyzhscriber,
    [
        Quit,
        ToggleRecording,
        CancelRecording,
        Minimize,
        Zoom,
        Hide,
        HideOthers,
        ShowAll,
        CloseWindow,
        Cut,
        Copy,
        Paste,
        SelectAll
    ]
);

/// Key context no GPUI element sets. Edit bindings live here so they only
/// give the Edit menu its shortcuts: the menu sends the native `paste:` etc.
/// to native text fields (the dialogs), while key handling in GPUI views,
/// such as paste into the model search, stays untouched.
const NATIVE_TEXT_CONTEXT: &str = "NativeTextField";

fn register_menus(
    cx: &mut App,
    controller: std::sync::Arc<AppController>,
    main_window: gpui::WindowHandle<RootView>,
) {
    cx.bind_keys([
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("cmd-m", Minimize, None),
        KeyBinding::new("cmd-h", Hide, None),
        KeyBinding::new("alt-cmd-h", HideOthers, None),
        KeyBinding::new("cmd-w", CloseWindow, None),
        KeyBinding::new("cmd-x", Cut, Some(NATIVE_TEXT_CONTEXT)),
        KeyBinding::new("cmd-c", Copy, Some(NATIVE_TEXT_CONTEXT)),
        KeyBinding::new("cmd-v", Paste, Some(NATIVE_TEXT_CONTEXT)),
        KeyBinding::new("cmd-a", SelectAll, Some(NATIVE_TEXT_CONTEXT)),
    ]);
    cx.on_action(|_: &Quit, cx| cx.quit());
    cx.on_action(|_: &Hide, cx| cx.hide());
    cx.on_action(|_: &HideOthers, cx| cx.hide_other_apps());
    cx.on_action(|_: &ShowAll, cx| cx.unhide_other_apps());
    cx.on_action(move |_: &CloseWindow, cx| {
        cx.defer(move |cx| {
            let _ = main_window.update(cx, |_, window, cx| dismiss_main_window(window, cx));
        });
    });
    let window_to_minimize = main_window;
    cx.on_action(move |_: &Minimize, cx| {
        cx.defer(move |cx| {
            let _ = window_to_minimize.update(cx, |_, window, _| window.minimize_window());
        });
    });
    cx.on_action(move |_: &Zoom, cx| {
        cx.defer(move |cx| {
            let _ = main_window.update(cx, |_, window, _| window.zoom_window());
        });
    });
    let toggle_controller = controller.clone();
    cx.on_action(move |_: &ToggleRecording, _| toggle_controller.toggle_recording());
    cx.on_action(move |_: &CancelRecording, _| controller.cancel());
    cx.set_menus([
        Menu::new("Pyzhscriber").items([
            MenuItem::action("Start or Stop Recording", ToggleRecording),
            MenuItem::action("Cancel Recording", CancelRecording),
            MenuItem::separator(),
            MenuItem::os_submenu("Services", SystemMenuType::Services),
            MenuItem::separator(),
            MenuItem::action("Hide Pyzhscriber", Hide),
            MenuItem::action("Hide Others", HideOthers),
            MenuItem::action("Show All", ShowAll),
            MenuItem::separator(),
            MenuItem::action("Quit Pyzhscriber", Quit),
        ]),
        Menu::new("Edit").items([
            MenuItem::os_action("Cut", Cut, OsAction::Cut),
            MenuItem::os_action("Copy", Copy, OsAction::Copy),
            MenuItem::os_action("Paste", Paste, OsAction::Paste),
            MenuItem::os_action("Select All", SelectAll, OsAction::SelectAll),
        ]),
        Menu::new("Window").items([
            MenuItem::action("Close Window", CloseWindow),
            MenuItem::action("Minimize", Minimize),
            MenuItem::action("Zoom", Zoom),
        ]),
    ]);
}
