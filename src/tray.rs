use crate::controller::AppController;
use crate::ui::RootView;
use anyhow::{Context, Result};
use gpui::WindowHandle;
use std::sync::Arc;
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

pub struct AppTray {
    _tray: TrayIcon,
    open_id: tray_icon::menu::MenuId,
    toggle_id: tray_icon::menu::MenuId,
    quit_id: tray_icon::menu::MenuId,
}

impl AppTray {
    pub fn new() -> Result<Self> {
        let menu = Menu::new();
        let open = MenuItem::new("Open Pyzhscriber", true, None);
        let toggle = MenuItem::new("Start or Stop Recording", true, None);
        let quit = MenuItem::new("Quit Pyzhscriber", true, None);
        menu.append(&open)?;
        menu.append(&toggle)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit)?;

        let image = image::load_from_memory(include_bytes!("../assets/app-icon.png"))?.into_rgba8();
        let (width, height) = image.dimensions();
        let icon = Icon::from_rgba(image.into_raw(), width, height)
            .context("create Pyzhscriber tray icon")?;
        let tray = TrayIconBuilder::new()
            .with_tooltip("Pyzhscriber")
            .with_menu(Box::new(menu))
            .with_icon(icon)
            .with_icon_as_template(false)
            .build()?;
        #[cfg(target_os = "macos")]
        if let Some(item) = tray.ns_status_item() {
            // Preserve the item's position across launches, including when the
            // menu bar is crowded enough to put new items in the overflow.
            item.setAutosaveName(Some(&objc2_foundation::NSString::from_str("Pyzhscriber")));
        }
        Ok(Self {
            _tray: tray,
            open_id: open.id().clone(),
            toggle_id: toggle.id().clone(),
            quit_id: quit.id().clone(),
        })
    }

    pub fn listen(
        controller: Arc<AppController>,
        main_window: WindowHandle<RootView>,
        cx: &mut gpui::App,
    ) {
        cx.spawn(async move |cx| {
            let mut tray = None;
            let mut previous_preferences = None;
            loop {
                let settings = controller.settings();
                let preferences = (settings.show_tray_icon, settings.hide_from_dock);
                if previous_preferences != Some(preferences) {
                    if settings.show_tray_icon || settings.hide_from_dock {
                        if tray.is_none() {
                            match Self::new() {
                                Ok(icon) => tray = Some(icon),
                                Err(error) => log::warn!("menu bar icon unavailable: {error:#}"),
                            }
                        }
                    } else {
                        tray = None;
                    }
                    previous_preferences = Some(preferences);
                }
                // A hidden Dock icon comes back while the window is open, and stays
                // if creating the menu bar icon failed.
                let window_open = cx.update(|cx| {
                    main_window
                        .update(cx, |_, window, _| crate::app::is_main_window_open(window))
                        .unwrap_or(false)
                });
                crate::app::set_dock_visible(
                    !settings.hide_from_dock || tray.is_none() || window_open,
                );
                while let Ok(event) = MenuEvent::receiver().try_recv() {
                    let Some(tray) = tray.as_ref() else { continue };
                    if event.id == tray.open_id {
                        cx.update(|cx| {
                            crate::app::show_main_window(main_window, cx);
                        });
                    } else if event.id == tray.toggle_id {
                        controller.toggle_recording();
                    } else if event.id == tray.quit_id {
                        cx.update(|cx| cx.quit());
                        return;
                    }
                }
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(150))
                    .await;
            }
        })
        .detach();
    }
}
