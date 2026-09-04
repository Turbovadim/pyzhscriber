use crate::assets::Assets;
use crate::controller::AppController;
use crate::tray::AppTray;
use crate::ui::RootView;
use gpui::{
    App, AppContext, Bounds, KeyBinding, Menu, MenuItem, SystemMenuType, WindowBounds,
    WindowOptions, actions, point, px, size,
};
use gpui_platform::application;

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
    // Always start the tap: it reports its own failure, and Input Monitoring can
    // be granted while the app is running.
    controller.start();

    application().with_assets(Assets).run(move |cx: &mut App| {
        gpui_tokio::init(cx);
        // A regular app so Pyzhscriber has a Dock icon and a menu bar. What keeps
        // the synthesized paste in the user's text field is that nothing on the
        // recording path activates: the overlay is a non-activating panel, so the
        // app they are dictating into stays frontmost throughout.
        set_regular_activation_policy();
        let settings = controller.settings();
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
                    window.on_window_should_close(cx, |_, cx| {
                        cx.hide();
                        false
                    });
                    cx.new(|cx| RootView::new(controller_for_window.clone(), cx))
                },
            )
            .expect("open Pyzhscriber window");
        register_menus(cx, controller.clone(), main_window);
        if settings.show_tray_icon {
            match AppTray::new() {
                Ok(tray) => tray.listen(controller.clone(), main_window, cx),
                Err(error) => log::warn!("menu bar icon unavailable: {error:#}"),
            }
        }
        // An accessory app orders new windows in behind whatever is frontmost, so
        // without activating here the window is on screen but buried and the
        // launch looks like nothing happened. Recording still never activates,
        // which is what keeps the synthesized paste in the user's text field.
        if visible {
            cx.activate(true);
        }
    });
}

#[cfg(target_os = "macos")]
fn set_regular_activation_policy() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};

    if let Some(mtm) = MainThreadMarker::new() {
        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_regular_activation_policy() {}

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
        CloseWindow
    ]
);

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
    ]);
    cx.on_action(|_: &Quit, cx| cx.quit());
    cx.on_action(|_: &Hide, cx| cx.hide());
    cx.on_action(|_: &HideOthers, cx| cx.hide_other_apps());
    cx.on_action(|_: &ShowAll, cx| cx.unhide_other_apps());
    cx.on_action(|_: &CloseWindow, cx| cx.hide());
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
        Menu::new("Window").items([
            MenuItem::action("Minimize", Minimize),
            MenuItem::action("Zoom", Zoom),
        ]),
    ]);
}
