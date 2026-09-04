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
        Ok(Self {
            _tray: tray,
            open_id: open.id().clone(),
            toggle_id: toggle.id().clone(),
            quit_id: quit.id().clone(),
        })
    }

    pub fn listen(
        self,
        controller: Arc<AppController>,
        main_window: WindowHandle<RootView>,
        cx: &mut gpui::App,
    ) {
        cx.spawn(async move |cx| {
            loop {
                while let Ok(event) = MenuEvent::receiver().try_recv() {
                    if event.id == self.open_id {
                        cx.update(|cx| {
                            // Accessory apps stay in the background, so bring the
                            // whole app forward before focusing its window.
                            cx.activate(true);
                            let _ = main_window.update(cx, |_, window, _| window.activate_window());
                        });
                    } else if event.id == self.toggle_id {
                        controller.toggle_recording();
                    } else if event.id == self.quit_id {
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
