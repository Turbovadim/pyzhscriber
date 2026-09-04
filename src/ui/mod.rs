use crate::audio::AudioRecorder;
use crate::controller::{AppController, RuntimeEvent, RuntimeStatus, ShortcutCapture};
use crate::dialogs;
use crate::history::HistoryEntry;
use crate::hotkey::Shortcut;
use crate::models::{
    ModelChoice, display_size, language_label, supports_english, supports_language,
    supports_russian,
};
use crate::permissions::{self, PermissionStatus, PermissionStatuses};
use crate::settings::{AppSettings, PasteMethod, RouteSetting, TranscriptionMode};
use crate::theme;
use gpui::{
    AnimationExt, AnyElement, Bounds, Context, CursorStyle, FocusHandle, FontWeight, Hsla,
    KeyDownEvent, MouseButton, Path, Pixels, Render, SharedString, Size, SpringAnimation,
    SpringConfig, Window, WindowBackgroundAppearance, WindowBounds, WindowHandle, WindowKind,
    WindowOptions, canvas, deferred, div, point, prelude::*, px, size, svg,
};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Page {
    General,
    Models,
    Audio,
    Output,
    History,
    Advanced,
}

struct SelectOption {
    value: String,
    label: String,
}

impl SelectOption {
    fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
        }
    }
}

/// The shared frame around a group of settings rows. Separators are drawn by the
/// frame rather than by each row, so the last row never rules a line across the
/// rounded bottom edge, and clipping keeps row hover states inside the corners.
/// Dropdown panels are `deferred`, which paints them outside this mask.
fn framed(children: Vec<AnyElement>) -> impl IntoElement {
    let last = children.len().saturating_sub(1);
    div()
        .w_full()
        .rounded_lg()
        .border_1()
        .border_color(theme::border())
        .bg(theme::raised())
        .overflow_hidden()
        .children(children.into_iter().enumerate().map(|(index, child)| {
            div()
                .w_full()
                .when(index < last, |row| {
                    row.border_b_1().border_color(theme::border())
                })
                .child(child)
        }))
}

pub struct RootView {
    controller: Arc<AppController>,
    settings: AppSettings,
    status: RuntimeStatus,
    page: Page,
    models: Vec<ModelChoice>,
    microphones: Vec<String>,
    history: Vec<HistoryEntry>,
    downloads: HashMap<String, f32>,
    overlay: Option<WindowHandle<RecordingOverlay>>,
    open_dropdown: Option<SharedString>,
    permission_statuses: PermissionStatuses,
    permission_setup_visible: bool,
    permission_polling: bool,
    shortcut_capture: ShortcutCapture,
    /// History entry whose text was just copied, cleared after a short delay.
    copied_entry: Option<i64>,
    /// Models tab filters. The catalog runs to dozens of entries, most of which
    /// are irrelevant to any one user, so the list is narrowed before display.
    model_search: String,
    model_search_focus: FocusHandle,
    model_language: ModelLanguage,
    model_streaming_only: bool,
    model_translate_only: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelLanguage {
    Any,
    English,
    Russian,
    Multilingual,
}

impl ModelLanguage {
    fn value(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::English => "en",
            Self::Russian => "ru",
            Self::Multilingual => "multi",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Any => "All languages",
            Self::English => "English",
            Self::Russian => "Russian",
            Self::Multilingual => "Multilingual",
        }
    }

    fn from_value(value: &str) -> Self {
        match value {
            "en" => Self::English,
            "ru" => Self::Russian,
            "multi" => Self::Multilingual,
            _ => Self::Any,
        }
    }

    fn matches(self, model: &ModelChoice) -> bool {
        match self {
            Self::Any => true,
            Self::English => supports_english(model),
            Self::Russian => supports_russian(model),
            Self::Multilingual => model.languages.len() > 1,
        }
    }
}

impl RootView {
    pub fn new(controller: Arc<AppController>, cx: &mut Context<Self>) -> Self {
        let receiver = controller.events();
        cx.spawn(async move |this, cx| {
            while let Ok(event) = receiver.recv().await {
                let _ = this.update(cx, |this, cx| {
                    match event {
                        RuntimeEvent::Status(status) => {
                            this.status = status;
                            this.sync_overlay(cx);
                        }
                        RuntimeEvent::Transcript(entry) => {
                            this.history.insert(0, entry);
                            this.history.truncate(this.settings.history_limit);
                        }
                        RuntimeEvent::Settings(settings) => {
                            this.settings = settings;
                            this.sync_overlay(cx);
                        }
                        RuntimeEvent::DownloadProgress { model_id, progress } => {
                            if progress >= 1.0 {
                                this.downloads.remove(&model_id);
                            } else {
                                this.downloads.insert(model_id, progress);
                            }
                        }
                        RuntimeEvent::ModelsChanged(models) => this.models = models,
                        RuntimeEvent::ShortcutCapture(capture) => this.shortcut_capture = capture,
                    }
                    cx.notify();
                });
            }
        })
        .detach();

        let settings = controller.settings();
        let status = controller.status();
        let permission_statuses = PermissionStatuses::current();
        let models = controller.models.default_choices();
        let history = controller
            .history
            .recent(settings.history_limit)
            .unwrap_or_default();
        Self {
            controller,
            settings,
            status,
            page: Page::General,
            models,
            microphones: AudioRecorder::input_devices(),
            history,
            downloads: HashMap::new(),
            overlay: None,
            open_dropdown: None,
            permission_statuses,
            permission_setup_visible: !permission_statuses.all_granted(),
            permission_polling: false,
            shortcut_capture: ShortcutCapture::Idle,
            copied_entry: None,
            model_search: String::new(),
            model_search_focus: cx.focus_handle().tab_stop(true),
            model_language: ModelLanguage::Any,
            model_streaming_only: false,
            model_translate_only: false,
        }
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_general(&mut self, cx: &mut Context<Self>) {
        self.page = Page::General;
        self.settings.transcription_mode = TranscriptionMode::Routed;
        self.open_dropdown = None;
        self.permission_setup_visible = false;
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn open_qa_dropdown(&mut self, id: &'static str, cx: &mut Context<Self>) {
        self.open_dropdown = Some(SharedString::from(id));
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_shortcut_capture(&mut self, cx: &mut Context<Self>) {
        self.prepare_qa_general(cx);
        self.shortcut_capture = ShortcutCapture::Listening("⌥ ⌘".into());
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_single_languages(&mut self, cx: &mut Context<Self>) {
        self.page = Page::General;
        self.settings.transcription_mode = TranscriptionMode::Single;
        if let Some(model) = self
            .models
            .iter()
            .find(|model| model.installed && model.languages.len() > 1)
        {
            self.settings.selected_model = model.id.clone();
        }
        self.settings.allowed_languages = vec!["en".into(), "ru".into()];
        self.open_dropdown = Some("single-languages".into());
        self.permission_setup_visible = false;
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_page(&mut self, page: &'static str, cx: &mut Context<Self>) {
        self.page = match page {
            "models" => Page::Models,
            "audio" => Page::Audio,
            "output" => Page::Output,
            "history" => Page::History,
            "advanced" => Page::Advanced,
            _ => Page::General,
        };
        self.open_dropdown = None;
        self.permission_setup_visible = false;
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_permissions(&mut self, cx: &mut Context<Self>) {
        self.permission_statuses = PermissionStatuses {
            microphone: PermissionStatus::NotDetermined,
            accessibility: PermissionStatus::Denied,
            input_monitoring: PermissionStatus::Denied,
        };
        self.permission_setup_visible = true;
        self.open_dropdown = None;
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_model_filters(&mut self, cx: &mut Context<Self>) {
        self.page = Page::Models;
        self.model_language = ModelLanguage::Russian;
        self.model_streaming_only = true;
        self.open_dropdown = None;
        cx.notify();
    }

    fn sync_overlay(&mut self, cx: &mut Context<Self>) {
        let visible = self.settings.show_overlay
            && matches!(
                self.status,
                RuntimeStatus::Recording | RuntimeStatus::Transcribing
            );
        if !visible {
            // The pill plays its exit first; the window is only torn down once
            // that has finished and recording has not started again.
            if let Some(handle) = self.overlay {
                let Ok(leave) = handle.update(cx, |overlay, _, cx| {
                    overlay.begin_exit();
                    cx.notify();
                    overlay.leave_seconds()
                }) else {
                    self.overlay = None;
                    return;
                };
                cx.spawn(async move |this, cx| {
                    smol::Timer::after(Duration::from_secs_f32(leave + 0.05)).await;
                    this.update(cx, |this, cx| {
                        let Some(handle) = this.overlay else {
                            return;
                        };
                        let still_leaving = handle
                            .update(cx, |overlay, _, _| overlay.leaving.is_some())
                            .unwrap_or(true);
                        if still_leaving {
                            let _ = handle.update(cx, |_, window, _| window.remove_window());
                            this.overlay = None;
                        }
                    })
                    .ok();
                })
                .detach();
            }
            return;
        }
        let bubbly = self.settings.bubbly_overlay;
        if let Some(handle) = self.overlay {
            let status = self.status.clone();
            let _ = handle.update(cx, |overlay, _, cx| {
                overlay.set_status(status);
                overlay.bubbly = bubbly;
                cx.notify();
            });
            return;
        }

        let overlay_size = overlay_window_size();
        let bounds = cx
            .primary_display()
            .map(|display| {
                let screen = display.visible_bounds();
                Bounds::new(
                    point(
                        screen.origin.x + (screen.size.width - overlay_size.width) / 2.0,
                        screen.origin.y + screen.size.height
                            - overlay_size.height
                            - px(OVERLAY_SCREEN_GAP),
                    ),
                    overlay_size,
                )
            })
            .unwrap_or_else(|| Bounds::centered(None, overlay_size, cx));
        let status = self.status.clone();
        self.overlay = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: None,
                    // PopUp is the only kind that stays visible while another app is
                    // active: it is a non-activating panel that joins every space.
                    focus: false,
                    kind: WindowKind::PopUp,
                    is_movable: false,
                    is_resizable: false,
                    is_minimizable: false,
                    window_background: WindowBackgroundAppearance::Transparent,
                    app_id: Some(env!("PYZHSCRIBER_BUNDLE_ID").into()),
                    ..Default::default()
                },
                {
                    let controller = self.controller.clone();
                    move |window, cx| {
                        strip_window_chrome(window);
                        cx.new(|cx| RecordingOverlay::new(controller.clone(), status, bubbly, cx))
                    }
                },
            )
            .ok();
    }

    fn request_permission(&mut self, permission: PermissionKind, cx: &mut Context<Self>) {
        match permission {
            PermissionKind::Microphone => permissions::request_microphone(),
            PermissionKind::Accessibility => permissions::request_accessibility(),
            PermissionKind::InputMonitoring => permissions::request_input_monitoring(),
        }
        self.refresh_permissions();
        self.start_permission_polling(cx);
        cx.notify();
    }

    fn refresh_permissions(&mut self) {
        self.permission_statuses = PermissionStatuses::current();
        self.controller.start();
        if self.permission_statuses.all_granted() {
            self.permission_setup_visible = false;
        }
    }

    fn start_permission_polling(&mut self, cx: &mut Context<Self>) {
        if self.permission_polling {
            return;
        }
        self.permission_polling = true;
        cx.spawn(async move |this, cx| {
            for _ in 0..30 {
                smol::Timer::after(Duration::from_secs(1)).await;
                let mut done = false;
                if this
                    .update(cx, |this, cx| {
                        this.refresh_permissions();
                        done = this.permission_statuses.all_granted();
                        cx.notify();
                    })
                    .is_err()
                {
                    return;
                }
                if done {
                    break;
                }
            }
            let _ = this.update(cx, |this, cx| {
                this.permission_polling = false;
                cx.notify();
            });
        })
        .detach();
    }

    fn save(&mut self, mutate: impl FnOnce(&mut AppSettings), cx: &mut Context<Self>) {
        let mut settings = self.settings.clone();
        mutate(&mut settings);
        if let Err(error) = self.controller.save_settings(settings.clone()) {
            self.status = RuntimeStatus::Error(error.to_string());
        } else {
            self.settings = settings;
        }
        cx.notify();
    }

    fn close_dropdown(&mut self, cx: &mut Context<Self>) {
        if self.open_dropdown.take().is_some() {
            cx.notify();
        }
    }

    fn select<F>(
        &self,
        id: impl Into<SharedString>,
        options: Vec<SelectOption>,
        selected_value: &str,
        width: f32,
        cx: &Context<Self>,
        on_select: F,
    ) -> AnyElement
    where
        F: Fn(&mut Self, String, &mut Context<Self>) + 'static,
    {
        let dropdown_id = id.into();
        let id = dropdown_id.as_ref().to_string();
        let is_open = self.open_dropdown.as_ref() == Some(&dropdown_id);
        let current_label = options
            .iter()
            .find(|option| option.value == selected_value)
            .map(|option| option.label.clone())
            .unwrap_or_else(|| selected_value.to_string());
        let toggle_id = dropdown_id.clone();
        let mut container = div().relative().w(px(width)).min_w(px(180.0)).child(
            div()
                .id(SharedString::from(format!("{id}-trigger")))
                .h(px(34.0))
                .w_full()
                .px_3()
                .rounded_md()
                .border_1()
                .border_color(if is_open {
                    theme::accent()
                } else {
                    theme::border()
                })
                .bg(if is_open {
                    theme::accent_dim()
                } else {
                    theme::input()
                })
                .flex()
                .items_center()
                .justify_between()
                .gap_3()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(theme::text())
                .hover(|style| style.border_color(theme::accent()).bg(theme::accent_dim()))
                .cursor_pointer()
                .child(div().flex_1().min_w_0().truncate().child(current_label))
                .child(
                    svg()
                        .path(if is_open {
                            "icons/chevron-up.svg"
                        } else {
                            "icons/chevron-down.svg"
                        })
                        .size(px(15.0))
                        .text_color(theme::text()),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    if this.open_dropdown.as_ref() == Some(&toggle_id) {
                        this.open_dropdown = None;
                    } else {
                        this.open_dropdown = Some(toggle_id.clone());
                    }
                    cx.notify();
                })),
        );

        if is_open {
            let mut panel = div()
                .id(SharedString::from(format!("{id}-panel")))
                .absolute()
                .top(px(38.0))
                .left_0()
                .w_full()
                .max_h(px(250.0))
                .overflow_y_scroll()
                .py_1()
                .rounded_md()
                .border_1()
                .border_color(theme::border())
                .bg(theme::raised())
                .occlude()
                .on_mouse_move(|_, _, cx| cx.stop_propagation())
                .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .on_mouse_down_out(cx.listener(|this, _, _, cx| this.close_dropdown(cx)));

            let on_select = std::rc::Rc::new(on_select);
            for (index, option) in options.into_iter().enumerate() {
                let selected = option.value == selected_value;
                let value = option.value.clone();
                let on_select = on_select.clone();
                panel = panel.child(
                    div()
                        .id(SharedString::from(format!("{id}-option-{index}")))
                        .min_h(px(34.0))
                        .w_full()
                        .px_3()
                        .py_2()
                        .text_sm()
                        .text_color(if selected {
                            theme::text()
                        } else {
                            theme::muted()
                        })
                        .bg(if selected {
                            theme::accent_dim()
                        } else {
                            theme::raised()
                        })
                        .hover(move |style| {
                            style
                                .bg(if selected {
                                    theme::accent_dim()
                                } else {
                                    theme::hover()
                                })
                                .text_color(theme::text())
                        })
                        .cursor_pointer()
                        .child(
                            div()
                                .w_full()
                                .flex()
                                .items_center()
                                .justify_between()
                                .gap_2()
                                .child(option.label)
                                .when(selected, |row| {
                                    row.child(
                                        svg()
                                            .path("icons/check.svg")
                                            .size(px(14.0))
                                            .text_color(theme::accent()),
                                    )
                                }),
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            on_select(this, value.clone(), cx);
                            this.close_dropdown(cx);
                        })),
                );
            }
            container = container.child(deferred(panel).with_priority(20));
        }
        container.into_any_element()
    }

    fn control_row(&self, label: &'static str, control: AnyElement) -> AnyElement {
        self.detailed_control_row(label, None, control)
    }

    fn detailed_control_row(
        &self,
        label: &'static str,
        detail: Option<SharedString>,
        control: AnyElement,
    ) -> AnyElement {
        div()
            .w_full()
            .min_h(px(52.0))
            .px_4()
            .py_2()
            .flex()
            .items_center()
            .justify_between()
            .gap_6()
            .child(
                div()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme::text())
                            .child(label),
                    )
                    .when_some(detail, |column, detail| {
                        column.child(div().text_xs().text_color(theme::muted()).child(detail))
                    }),
            )
            .child(control)
            .into_any_element()
    }

    /// Keycap-style button showing the transcription shortcut. Clicking it
    /// records the next key press through the global tap, so the capture works
    /// even though the settings window never takes keyboard focus.
    fn shortcut_control(&self, cx: &Context<Self>) -> AnyElement {
        let capturing = self.shortcut_capture != ShortcutCapture::Idle;
        let (label, muted) = match &self.shortcut_capture {
            ShortcutCapture::Idle => (
                Shortcut::parse(&self.settings.shortcut)
                    .map(Shortcut::label)
                    .unwrap_or_else(|| self.settings.shortcut.clone()),
                false,
            ),
            ShortcutCapture::Listening(held) if held.is_empty() => {
                ("Press keys…".to_string(), true)
            }
            ShortcutCapture::Listening(held) => (held.clone(), false),
            ShortcutCapture::NeedsModifier => ("Add ⌃ ⌥ ⇧ or ⌘".to_string(), true),
        };
        div()
            .id("shortcut-control")
            .h(px(34.0))
            .min_w(px(200.0))
            .px_3()
            .rounded_md()
            .border_1()
            .border_color(if capturing {
                theme::accent()
            } else {
                theme::border()
            })
            .bg(if capturing {
                theme::accent_dim()
            } else {
                theme::input()
            })
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .text_sm()
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(if muted { theme::muted() } else { theme::text() })
            .hover(|style| style.border_color(theme::accent()).bg(theme::accent_dim()))
            .cursor_pointer()
            .child(label)
            .child(
                div()
                    .text_xs()
                    .font_weight(FontWeight::NORMAL)
                    .text_color(theme::muted())
                    .child(if capturing { "Esc cancels" } else { "Change" }),
            )
            .on_click(cx.listener(|this, _, _, cx| {
                cx.stop_propagation();
                if this.shortcut_capture != ShortcutCapture::Idle {
                    this.controller.cancel_shortcut_capture();
                } else if let Err(error) = this.controller.begin_shortcut_capture() {
                    this.status = RuntimeStatus::Error(error.to_string());
                    cx.notify();
                }
            }))
            .when(capturing, |control| {
                control.on_mouse_down_out(cx.listener(|this, _, _, _| {
                    this.controller.cancel_shortcut_capture();
                }))
            })
            .into_any_element()
    }

    fn switch<F>(
        &self,
        id: &'static str,
        enabled: bool,
        cx: &Context<Self>,
        on_toggle: F,
    ) -> AnyElement
    where
        F: Fn(&mut AppSettings, bool) + 'static,
    {
        div()
            .id(id)
            .relative()
            .w(px(42.0))
            .h(px(24.0))
            .p(px(3.0))
            .rounded_full()
            .bg(if enabled {
                theme::accent()
            } else {
                theme::hover()
            })
            .border_1()
            .border_color(if enabled {
                theme::accent()
            } else {
                theme::border()
            })
            .cursor_pointer()
            .child(
                div()
                    .relative()
                    .size(px(16.0))
                    .rounded_full()
                    .bg(theme::text())
                    .with_spring(
                        SharedString::from(format!("{id}-thumb-spring")),
                        SpringAnimation::new(SpringConfig::new(520.0, 34.0, 1.0))
                            .to(px(if enabled { 18.0 } else { 0.0 })),
                        |thumb, left| thumb.left(left),
                    ),
            )
            .on_click(cx.listener(move |this, _, _, cx| {
                this.save(|settings| on_toggle(settings, !enabled), cx);
            }))
            .into_any_element()
    }

    fn sidebar_item(
        &self,
        page: Page,
        label: &'static str,
        icon: &'static str,
        cx: &Context<Self>,
    ) -> AnyElement {
        let selected = self.page == page;
        div()
            .id(SharedString::from(format!("nav-{page:?}")))
            .h(px(36.0))
            .px_3()
            .mx_2()
            .rounded_md()
            .flex()
            .items_center()
            .gap_2p5()
            .text_sm()
            .font_weight(if selected {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            })
            .text_color(if selected {
                theme::text()
            } else {
                theme::muted()
            })
            .bg(if selected {
                theme::accent_dim()
            } else {
                theme::raised()
            })
            .hover(move |style| {
                style
                    .bg(if selected {
                        theme::accent_dim()
                    } else {
                        theme::hover()
                    })
                    .text_color(theme::text())
            })
            .cursor_pointer()
            .child(svg().path(icon).size(px(16.0)).text_color(if selected {
                theme::accent()
            } else {
                theme::muted()
            }))
            .child(label)
            .on_click(cx.listener(move |this, _, _, cx| {
                this.page = page;
                cx.notify();
            }))
            .into_any_element()
    }

    fn section(&self, title: &'static str, children: Vec<AnyElement>) -> AnyElement {
        div()
            .w_full()
            .pt_4()
            .pb_2()
            .flex()
            .flex_col()
            .gap_2p5()
            .child(
                div()
                    .px_2()
                    .text_xs()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme::muted())
                    .child(title),
            )
            .child(framed(children))
            .into_any_element()
    }

    fn page_header(&self, title: &'static str, detail: impl Into<SharedString>) -> AnyElement {
        div()
            .w_full()
            .pb_3()
            .flex()
            .items_end()
            .justify_between()
            .gap_6()
            .child(
                div()
                    .text_2xl()
                    .font_weight(FontWeight::BOLD)
                    .text_color(theme::text())
                    .child(title),
            )
            .child(
                div()
                    .pb_1()
                    .text_xs()
                    .text_color(theme::muted())
                    .child(detail.into()),
            )
            .into_any_element()
    }

    fn row(&self, label: impl Into<SharedString>, value: impl Into<SharedString>) -> AnyElement {
        div()
            .w_full()
            .min_h(px(52.0))
            .px_4()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .child(
                div()
                    .text_sm()
                    .text_color(theme::text())
                    .child(label.into()),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme::muted())
                    .child(value.into()),
            )
            .into_any_element()
    }

    fn action_row<F>(
        &self,
        id: impl Into<SharedString>,
        label: impl Into<SharedString>,
        value: impl Into<SharedString>,
        cx: &Context<Self>,
        on_click: F,
    ) -> AnyElement
    where
        F: Fn(&mut Self, &mut Context<Self>) + 'static,
    {
        let id = id.into();
        div()
            .id(id)
            .w_full()
            .min_h(px(52.0))
            .px_4()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .hover(|style| style.bg(theme::hover()))
            .cursor_pointer()
            .child(
                div()
                    .text_sm()
                    .text_color(theme::text())
                    .child(label.into()),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme::accent())
                    .child(value.into()),
            )
            .on_click(cx.listener(move |this, _, _, cx| on_click(this, cx)))
            .into_any_element()
    }

    fn toggle_row<F>(
        &self,
        id: &'static str,
        label: &'static str,
        enabled: bool,
        cx: &Context<Self>,
        on_toggle: F,
    ) -> AnyElement
    where
        F: Fn(&mut AppSettings, bool) + 'static,
    {
        self.control_row(label, self.switch(id, enabled, cx, on_toggle))
    }

    fn model_name(&self, id: &str) -> String {
        self.controller.models.model_name(id)
    }

    fn model_options(&self, language: Option<&str>) -> Vec<SelectOption> {
        self.models
            .iter()
            .filter(|model| model.installed)
            .filter(|model| language.is_none_or(|language| supports_language(model, language)))
            .map(|model| SelectOption::new(model.id.clone(), model.name.clone()))
            .collect()
    }

    fn model_languages(&self, model_id: &str) -> Vec<String> {
        self.models
            .iter()
            .find(|model| model.id == model_id)
            .map(|model| model.languages.clone())
            .unwrap_or_default()
    }

    fn route_language_options(&self, route_index: usize) -> Vec<SelectOption> {
        let used = self
            .settings
            .routes
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != route_index)
            .map(|(_, route)| route.language.as_str())
            .collect::<BTreeSet<_>>();
        self.models
            .iter()
            .filter(|model| model.installed)
            .flat_map(|model| model.languages.iter())
            .filter(|language| !used.contains(language.as_str()))
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|language| SelectOption::new(language.clone(), language_label(&language)))
            .collect()
    }

    fn add_route(&mut self, cx: &mut Context<Self>) {
        let used = self
            .settings
            .routes
            .iter()
            .map(|route| route.language.as_str())
            .collect::<BTreeSet<_>>();
        let Some((language, model)) = self.models.iter().find_map(|model| {
            if !model.installed {
                return None;
            }
            model
                .languages
                .iter()
                .find(|language| !used.contains(language.as_str()))
                .map(|language| (language.clone(), model.id.clone()))
        }) else {
            return;
        };
        self.save(
            |settings| settings.routes.push(RouteSetting::new(language, model)),
            cx,
        );
    }

    fn route_row(&self, index: usize, route: &RouteSetting, cx: &Context<Self>) -> AnyElement {
        let language = route.language.clone();
        let can_remove = self.settings.routes.len() > 1;
        div()
            .w_full()
            .min_h(px(58.0))
            .px_4()
            .py_2()
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .child(self.select(
                SharedString::from(format!("route-language-{index}")),
                self.route_language_options(index),
                &route.language,
                150.0,
                cx,
                move |this, value, cx| {
                    let current_model = this
                        .settings
                        .routes
                        .get(index)
                        .map(|route| route.model.clone())
                        .unwrap_or_default();
                    let model = this
                        .models
                        .iter()
                        .find(|model| model.id == current_model && supports_language(model, &value))
                        .or_else(|| {
                            this.models
                                .iter()
                                .find(|model| model.installed && supports_language(model, &value))
                        })
                        .map(|model| model.id.clone());
                    this.save(
                        |settings| {
                            if let Some(route) = settings.routes.get_mut(index) {
                                route.language = value;
                                if let Some(model) = model {
                                    route.model = model;
                                }
                            }
                        },
                        cx,
                    );
                },
            ))
            .child(self.select(
                SharedString::from(format!("route-model-{index}")),
                self.model_options(Some(&language)),
                &route.model,
                300.0,
                cx,
                move |this, value, cx| {
                    this.save(
                        |settings| {
                            if let Some(route) = settings.routes.get_mut(index) {
                                route.model = value;
                            }
                        },
                        cx,
                    );
                },
            ))
            .child(
                div()
                    .id(SharedString::from(format!("remove-route-{index}")))
                    .size(px(30.0))
                    .rounded_md()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(if can_remove {
                        theme::muted()
                    } else {
                        theme::border()
                    })
                    .child(
                        svg()
                            .path("icons/x.svg")
                            .size(px(14.0))
                            .text_color(if can_remove {
                                theme::muted()
                            } else {
                                theme::border()
                            }),
                    )
                    .when(can_remove, |button| {
                        button
                            .hover(|style| style.bg(theme::hover()).text_color(theme::danger()))
                            .cursor_pointer()
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.save(
                                    |settings| {
                                        settings.routes.remove(index);
                                    },
                                    cx,
                                );
                            }))
                    }),
            )
            .into_any_element()
    }

    fn language_limit_control(&self, cx: &Context<Self>) -> AnyElement {
        let id = SharedString::from("single-languages");
        let is_open = self.open_dropdown.as_ref() == Some(&id);
        let supported = self.model_languages(&self.settings.selected_model);
        let selected = self
            .settings
            .allowed_languages
            .iter()
            .filter(|language| supported.contains(language))
            .cloned()
            .collect::<Vec<_>>();
        let label = match selected.as_slice() {
            [] => "All supported".to_string(),
            [language] => language_label(language),
            languages => format!("{} selected", languages.len()),
        };
        let toggle_id = id.clone();
        let mut control = div().relative().w(px(300.0)).min_w(px(180.0)).child(
            div()
                .id("single-languages-trigger")
                .h(px(34.0))
                .w_full()
                .px_3()
                .rounded_md()
                .border_1()
                .border_color(if is_open {
                    theme::accent()
                } else {
                    theme::border()
                })
                .bg(if is_open {
                    theme::accent_dim()
                } else {
                    theme::input()
                })
                .flex()
                .items_center()
                .justify_between()
                .gap_3()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(theme::text())
                .hover(|style| style.border_color(theme::accent()).bg(theme::accent_dim()))
                .cursor_pointer()
                .child(div().flex_1().min_w_0().truncate().child(label))
                .child(
                    svg()
                        .path(if is_open {
                            "icons/chevron-up.svg"
                        } else {
                            "icons/chevron-down.svg"
                        })
                        .size(px(15.0)),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    if this.open_dropdown.as_ref() == Some(&toggle_id) {
                        this.open_dropdown = None;
                    } else {
                        this.open_dropdown = Some(toggle_id.clone());
                    }
                    cx.notify();
                })),
        );

        if is_open {
            let mut panel = div()
                .id("single-languages-panel")
                .absolute()
                .top(px(38.0))
                .left_0()
                .w_full()
                .max_h(px(250.0))
                .overflow_y_scroll()
                .py_1()
                .rounded_md()
                .border_1()
                .border_color(theme::border())
                .bg(theme::raised())
                .occlude()
                .on_mouse_move(|_, _, cx| cx.stop_propagation())
                .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .on_mouse_down_out(cx.listener(|this, _, _, cx| this.close_dropdown(cx)));

            let all_selected = selected.is_empty();
            panel = panel.child(self.language_limit_option(
                "all",
                "All supported",
                all_selected,
                cx,
                |this, cx| {
                    this.save(|settings| settings.allowed_languages.clear(), cx);
                },
            ));
            for language in supported {
                let checked = selected.contains(&language);
                let option_language = language.clone();
                panel = panel.child(self.language_limit_option(
                    &language,
                    &language_label(&language),
                    checked,
                    cx,
                    move |this, cx| {
                        let language = option_language.clone();
                        this.save(
                            |settings| {
                                if let Some(index) = settings
                                    .allowed_languages
                                    .iter()
                                    .position(|selected| selected == &language)
                                {
                                    settings.allowed_languages.remove(index);
                                } else {
                                    settings.allowed_languages.push(language);
                                }
                            },
                            cx,
                        );
                    },
                ));
            }
            control = control.child(deferred(panel).with_priority(20));
        }
        control.into_any_element()
    }

    fn language_limit_option<F>(
        &self,
        id: &str,
        label: &str,
        checked: bool,
        cx: &Context<Self>,
        on_click: F,
    ) -> AnyElement
    where
        F: Fn(&mut Self, &mut Context<Self>) + 'static,
    {
        div()
            .id(SharedString::from(format!("single-language-{id}")))
            .min_h(px(34.0))
            .w_full()
            .px_3()
            .py_2()
            .flex()
            .items_center()
            .justify_between()
            .text_sm()
            .text_color(if checked {
                theme::text()
            } else {
                theme::muted()
            })
            .bg(if checked {
                theme::accent_dim()
            } else {
                theme::raised()
            })
            .hover(move |style| {
                style
                    .bg(if checked {
                        theme::accent_dim()
                    } else {
                        theme::hover()
                    })
                    .text_color(theme::text())
            })
            .cursor_pointer()
            .child(label.to_string())
            .when(checked, |row| {
                row.child(
                    svg()
                        .path("icons/check.svg")
                        .size(px(14.0))
                        .text_color(theme::accent()),
                )
            })
            .on_click(cx.listener(move |this, _, _, cx| {
                cx.stop_propagation();
                on_click(this, cx);
            }))
            .into_any_element()
    }

    fn mode_control(&self, cx: &Context<Self>) -> AnyElement {
        let routed = self.settings.transcription_mode == TranscriptionMode::Routed;
        div()
            .w(px(310.0))
            .h(px(34.0))
            .rounded_md()
            .border_1()
            .border_color(theme::border())
            .bg(theme::input())
            .overflow_hidden()
            .flex()
            .child(
                div()
                    .id("mode-routed")
                    .h_full()
                    .flex_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_sm()
                    .text_color(if routed {
                        theme::text()
                    } else {
                        theme::muted()
                    })
                    .bg(if routed {
                        theme::accent_dim()
                    } else {
                        theme::input()
                    })
                    .cursor_pointer()
                    .child("Routed")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.save(
                            |settings| settings.transcription_mode = TranscriptionMode::Routed,
                            cx,
                        );
                    })),
            )
            .child(
                div()
                    .id("mode-single")
                    .h_full()
                    .flex_1()
                    .border_l_1()
                    .border_color(theme::border())
                    .flex()
                    .items_center()
                    .justify_center()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_sm()
                    .text_color(if routed {
                        theme::muted()
                    } else {
                        theme::text()
                    })
                    .bg(if routed {
                        theme::input()
                    } else {
                        theme::accent_dim()
                    })
                    .cursor_pointer()
                    .child("Single model")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.save(
                            |settings| settings.transcription_mode = TranscriptionMode::Single,
                            cx,
                        );
                    })),
            )
            .into_any_element()
    }

    fn permission_row(
        &self,
        kind: PermissionKind,
        title: &'static str,
        detail: &'static str,
        status: PermissionStatus,
        cx: &Context<Self>,
    ) -> AnyElement {
        let granted = status == PermissionStatus::Granted;
        let action = match status {
            PermissionStatus::NotDetermined => "Grant access",
            PermissionStatus::Denied => "Open Settings",
            PermissionStatus::Granted => "Granted",
        };
        div()
            .id(SharedString::from(format!("permission-{kind:?}")))
            .w_full()
            .min_h(px(66.0))
            .px_4()
            .py_3()
            .flex()
            .items_center()
            .justify_between()
            .gap_6()
            .child(
                div()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme::text())
                            .child(title),
                    )
                    .child(div().text_xs().text_color(theme::muted()).child(detail)),
            )
            .child(
                div()
                    .id(SharedString::from(format!("permission-{kind:?}-action")))
                    .min_w(px(116.0))
                    .h(px(32.0))
                    .px_3()
                    .rounded_md()
                    .border_1()
                    .border_color(if granted {
                        theme::border()
                    } else {
                        theme::accent()
                    })
                    .bg(if granted {
                        theme::input()
                    } else {
                        theme::accent_dim()
                    })
                    .flex()
                    .items_center()
                    .justify_center()
                    .gap_2()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(if granted {
                        theme::success()
                    } else {
                        theme::text()
                    })
                    .when(granted, |button| {
                        button.child(
                            svg()
                                .path("icons/circle-check.svg")
                                .size(px(14.0))
                                .text_color(theme::success()),
                        )
                    })
                    .child(action)
                    .when(!granted, |button| {
                        button
                            .cursor_pointer()
                            .hover(|style| style.bg(theme::hover()))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.request_permission(kind, cx);
                            }))
                    }),
            )
            .into_any_element()
    }

    fn render_permission_setup(&self, cx: &Context<Self>) -> AnyElement {
        let all_granted = self.permission_statuses.all_granted();
        div()
            .size_full()
            .bg(theme::background())
            .text_color(theme::text())
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .w(px(520.0))
                    .px_5()
                    .flex()
                    .flex_col()
                    .gap_5()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .text_sm()
                            .font_weight(FontWeight::BOLD)
                            .text_color(theme::accent())
                            .child("Pyzhscriber"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_2xl()
                                    .font_weight(FontWeight::BOLD)
                                    .child(if all_granted {
                                        "You’re ready"
                                    } else {
                                        "One quick setup"
                                    }),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(theme::muted())
                                    .child("Pyzhscriber needs these macOS permissions to record speech and insert the transcript."),
                            ),
                    )
                    .child(framed(vec![
                        self.permission_row(
                            PermissionKind::Microphone,
                            "Microphone",
                            "Records only while your transcription shortcut is active",
                            self.permission_statuses.microphone,
                            cx,
                        ),
                        self.permission_row(
                            PermissionKind::Accessibility,
                            "Accessibility",
                            "Inserts the finished transcript into the active app",
                            self.permission_statuses.accessibility,
                            cx,
                        ),
                        self.permission_row(
                            PermissionKind::InputMonitoring,
                            "Input Monitoring",
                            "Detects your shortcut while other apps are active",
                            self.permission_statuses.input_monitoring,
                            cx,
                        ),
                    ]))
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::muted())
                            .child(if self.permission_polling {
                                "Waiting for macOS…"
                            } else {
                                "After granting access, return here. This screen updates automatically."
                            }),
                    ),
            )
            .into_any_element()
    }

    fn render_general(&self, cx: &Context<Self>) -> AnyElement {
        let recording = self.status == RuntimeStatus::Recording;
        let status_detail = match &self.status {
            RuntimeStatus::Error(error) => error.clone(),
            _ => self.status.label().to_string(),
        };
        div()
            .flex()
            .flex_col()
            .child(self.page_header("General", status_detail.clone()))
            .child(self.section("Transcription", {
                let mut rows = vec![
                    self.action_row(
                        "toggle-recording",
                        if recording {
                            "Stop recording"
                        } else {
                            "Start recording"
                        },
                        status_detail,
                        cx,
                        |this, _| this.controller.toggle_recording(),
                    ),
                    self.control_row("Transcription mode", self.mode_control(cx)),
                ];
                if self.settings.transcription_mode == TranscriptionMode::Routed {
                    rows.extend(
                        self.settings
                            .routes
                            .iter()
                            .enumerate()
                            .map(|(index, route)| self.route_row(index, route, cx)),
                    );
                    rows.push(
                        div()
                            .id("add-route")
                            .w_full()
                            .h(px(44.0))
                            .px_4()
                            .flex()
                            .items_center()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme::accent())
                            .hover(|style| style.bg(theme::hover()))
                            .cursor_pointer()
                            .child("+ Add language")
                            .on_click(cx.listener(|this, _, _, cx| this.add_route(cx)))
                            .into_any_element(),
                    );
                } else {
                    rows.push(self.control_row(
                        "Model",
                        self.select(
                            "single-model",
                            self.model_options(None),
                            &self.settings.selected_model,
                            340.0,
                            cx,
                            |this, value, cx| {
                                let supported = this.model_languages(&value);
                                this.save(
                                    |settings| {
                                        settings.selected_model = value;
                                        settings
                                            .allowed_languages
                                            .retain(|language| supported.contains(language));
                                    },
                                    cx,
                                );
                            },
                        ),
                    ));
                    if self.model_languages(&self.settings.selected_model).len() > 1 {
                        rows.push(self.detailed_control_row(
                            "Languages",
                            Some("Limit automatic language detection".into()),
                            self.language_limit_control(cx),
                        ));
                    }
                }
                rows.push(
                    self.detailed_control_row(
                        "Transcribe shortcut",
                        Some(
                            if self.shortcut_capture == ShortcutCapture::Idle {
                                "Modifier chord, function key, or a single modifier"
                            } else {
                                "Press the new shortcut. Release a lone modifier to pick it."
                            }
                            .into(),
                        ),
                        self.shortcut_control(cx),
                    ),
                );
                rows.push(
                    self.detailed_control_row(
                        "Push to talk",
                        Some(
                            if self.settings.push_to_talk {
                                "Hold the shortcut to record, release to transcribe"
                            } else {
                                "Press once to start, again to stop"
                            }
                            .into(),
                        ),
                        self.switch(
                            "push-to-talk",
                            self.settings.push_to_talk,
                            cx,
                            |settings, value| settings.push_to_talk = value,
                        ),
                    ),
                );
                rows.push(self.toggle_row(
                    "recording-overlay",
                    "Recording overlay",
                    self.settings.show_overlay,
                    cx,
                    |settings, value| settings.show_overlay = value,
                ));
                if self.settings.show_overlay {
                    rows.push(self.detailed_control_row(
                        "Bubbly overlay",
                        Some("The pill jiggles softly with your voice".into()),
                        self.switch(
                            "bubbly-overlay",
                            self.settings.bubbly_overlay,
                            cx,
                            |settings, value| settings.bubbly_overlay = value,
                        ),
                    ));
                }
                rows
            }))
            .child(self.section(
                "Permissions",
                vec![
                    self.permission_row(
                        PermissionKind::Microphone,
                        "Microphone",
                        "Records only while your transcription shortcut is active",
                        self.permission_statuses.microphone,
                        cx,
                    ),
                    self.permission_row(
                        PermissionKind::Accessibility,
                        "Accessibility",
                        "Inserts the finished transcript into the active app",
                        self.permission_statuses.accessibility,
                        cx,
                    ),
                    self.permission_row(
                        PermissionKind::InputMonitoring,
                        "Input Monitoring",
                        "Detects your shortcut while other apps are active",
                        self.permission_statuses.input_monitoring,
                        cx,
                    ),
                ],
            ))
            .into_any_element()
    }

    /// One icon-only filter toggle, tinted while it is narrowing the list.
    fn filter_toggle<F>(
        &self,
        id: &'static str,
        icon: &'static str,
        label: &'static str,
        active: bool,
        cx: &Context<Self>,
        on_click: F,
    ) -> AnyElement
    where
        F: Fn(&mut Self, &mut Context<Self>) + 'static,
    {
        div()
            .id(id)
            .h(px(32.0))
            .px_3()
            .rounded_md()
            .border_1()
            .border_color(if active {
                theme::accent()
            } else {
                theme::border()
            })
            .bg(if active {
                theme::accent_dim()
            } else {
                theme::input()
            })
            .flex()
            .items_center()
            .gap_2()
            .text_xs()
            .font_weight(FontWeight::MEDIUM)
            .text_color(if active {
                theme::text()
            } else {
                theme::muted()
            })
            .child(svg().path(icon).size(px(13.0)).text_color(if active {
                theme::accent()
            } else {
                theme::muted()
            }))
            .child(label)
            .hover(move |style| {
                style.bg(if active {
                    theme::accent_dim()
                } else {
                    theme::hover()
                })
            })
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, _, cx| on_click(this, cx)))
            .into_any_element()
    }

    fn on_model_search_key_down(
        &mut self,
        event: &KeyDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let keystroke = &event.keystroke;
        let handled = match keystroke.key.as_str() {
            "backspace" | "delete" => {
                self.model_search.pop();
                true
            }
            "escape" => {
                self.model_search.clear();
                true
            }
            "v" if keystroke.modifiers.secondary() => {
                if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
                    self.model_search
                        .push_str(&text.replace(['\n', '\r', '\t'], " "));
                }
                true
            }
            _ if !keystroke.modifiers.control
                && !keystroke.modifiers.platform
                && !keystroke.modifiers.function =>
            {
                if let Some(text) = &keystroke.key_char {
                    self.model_search
                        .push_str(&text.replace(['\n', '\r', '\t'], " "));
                    true
                } else {
                    false
                }
            }
            _ => false,
        };

        if handled {
            cx.stop_propagation();
            cx.notify();
        }
    }

    fn model_search(&self, cx: &Context<Self>) -> AnyElement {
        let empty = self.model_search.is_empty();
        let text = if empty {
            SharedString::from("Search name or description")
        } else {
            SharedString::from(self.model_search.clone())
        };

        div()
            .id("model-search")
            .key_context("ModelSearch")
            .track_focus(&self.model_search_focus)
            .h(px(34.0))
            .w(px(280.0))
            .min_w(px(220.0))
            .px_3()
            .rounded_md()
            .border_1()
            .border_color(theme::border())
            .bg(theme::input())
            .flex()
            .items_center()
            .gap_2()
            .text_sm()
            .text_color(if empty { theme::muted() } else { theme::text() })
            .cursor(CursorStyle::IBeam)
            .focus(|style| style.border_color(theme::accent()).bg(theme::accent_dim()))
            .child(div().flex_1().min_w_0().truncate().child(text))
            .when(!empty, |input| {
                input.child(
                    div()
                        .id("clear-model-search")
                        .size(px(20.0))
                        .flex()
                        .items_center()
                        .justify_center()
                        .rounded_md()
                        .cursor_pointer()
                        .hover(|style| style.bg(theme::hover()))
                        .child(
                            svg()
                                .path("icons/x.svg")
                                .size(px(12.0))
                                .text_color(theme::muted()),
                        )
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.model_search.clear();
                            cx.stop_propagation();
                            cx.notify();
                        })),
                )
            })
            .on_click(cx.listener(|this, _, window, cx| {
                this.model_search_focus.focus(window, cx);
            }))
            .on_key_down(cx.listener(Self::on_model_search_key_down))
            .into_any_element()
    }

    fn render_models(&self, cx: &Context<Self>) -> AnyElement {
        let matching = self
            .models
            .iter()
            .filter(|model| model_matches_search(model, &self.model_search))
            .filter(|model| self.model_language.matches(model))
            .filter(|model| !self.model_streaming_only || model.streaming)
            .filter(|model| !self.model_translate_only || model.translate)
            .collect::<Vec<_>>();
        let (installed_models, available_models): (Vec<&ModelChoice>, Vec<&ModelChoice>) = matching
            .iter()
            .copied()
            .partition(|model| model.installed || self.downloads.contains_key(&model.id));
        let total = self.models.len();
        let shown = matching.len();

        let card = |model: &ModelChoice| {
            let languages = if model.languages.len() == 1 {
                model.languages[0].to_uppercase()
            } else {
                format!("{} languages", model.languages.len())
            };
            let model_id = model.id.clone();
            let installed = model.installed;
            let download_progress = self.downloads.get(&model.id).copied();
            let status = download_progress.map_or_else(
                || {
                    if installed {
                        "Installed".to_string()
                    } else {
                        "Download".to_string()
                    }
                },
                |progress| format!("{:.0}%", progress * 100.0),
            );
            div()
                .id(SharedString::from(format!("model-{}", model.id)))
                .w_full()
                .p_4()
                .rounded_lg()
                .border_1()
                .border_color(theme::border())
                .bg(theme::raised())
                .flex()
                .items_start()
                .justify_between()
                .gap_6()
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_lg()
                                .font_weight(FontWeight::BOLD)
                                .text_color(theme::text())
                                .child(model.name.clone()),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme::muted())
                                .child(model.description.clone()),
                        )
                        .child(div().text_xs().text_color(theme::muted()).child(format!(
                            "{}  ·  {}  ·  {}  ·  {}",
                            model.architecture,
                            model.quant,
                            languages,
                            display_size(model.size_bytes)
                        )))
                        .child(
                            div()
                                .pt_1()
                                .flex()
                                .gap_5()
                                .child(model_metric("Accuracy", model.accuracy_score))
                                .child(model_metric("Speed", model.speed_score)),
                        )
                        .when_some(download_progress, |column, progress| {
                            column.child(
                                div()
                                    .mt_1()
                                    .w_full()
                                    .h(px(3.0))
                                    .rounded_full()
                                    .bg(theme::hover())
                                    .child(
                                        div()
                                            .h_full()
                                            .w(px(360.0 * progress.clamp(0.0, 1.0)))
                                            .rounded_full()
                                            .bg(theme::accent()),
                                    ),
                            )
                        }),
                )
                .child(
                    div()
                        .mt_1()
                        .min_w(px(80.0))
                        .h(px(32.0))
                        .px_3()
                        .rounded_md()
                        .border_1()
                        .border_color(if installed {
                            theme::border()
                        } else {
                            theme::accent()
                        })
                        .bg(if installed {
                            theme::input()
                        } else {
                            theme::accent_dim()
                        })
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(if installed {
                            theme::success()
                        } else {
                            theme::text()
                        })
                        .child(status),
                )
                .when(!installed, |row| {
                    row.cursor_pointer()
                        .on_click(cx.listener(move |this, _, _, _| {
                            this.controller.download_model(model_id.clone());
                        }))
                })
        };

        div()
            .flex()
            .flex_col()
            .child(self.page_header(
                "Models",
                if shown == total {
                    format!("{total} models in the shared cache")
                } else {
                    format!("{shown} of {total} models")
                },
            ))
            .child(
                div()
                    .pb_1()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(self.model_search(cx))
                    .child(self.filter_toggle(
                        "filter-streaming",
                        "icons/audio-lines.svg",
                        "Streaming",
                        self.model_streaming_only,
                        cx,
                        |this, cx| {
                            this.model_streaming_only = !this.model_streaming_only;
                            cx.notify();
                        },
                    ))
                    .child(self.filter_toggle(
                        "filter-translate",
                        "icons/languages.svg",
                        "Translation",
                        self.model_translate_only,
                        cx,
                        |this, cx| {
                            this.model_translate_only = !this.model_translate_only;
                            cx.notify();
                        },
                    ))
                    .child(
                        self.select(
                            "filter-language",
                            [
                                ModelLanguage::Any,
                                ModelLanguage::English,
                                ModelLanguage::Russian,
                                ModelLanguage::Multilingual,
                            ]
                            .into_iter()
                            .map(|language| SelectOption::new(language.value(), language.label()))
                            .collect(),
                            self.model_language.value(),
                            160.0,
                            cx,
                            |this, value, cx| {
                                this.model_language = ModelLanguage::from_value(&value);
                                cx.notify();
                            },
                        ),
                    ),
            )
            .child(
                div()
                    .pt_2()
                    .pb_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .when(shown == 0, |column| {
                        column.child(
                            div()
                                .py_8()
                                .text_sm()
                                .text_color(theme::muted())
                                .child("No models match."),
                        )
                    })
                    .when(!installed_models.is_empty(), |column| {
                        column
                            .child(self.model_group_label("Your models"))
                            .children(installed_models.iter().copied().map(card))
                    })
                    .when(!available_models.is_empty(), |column| {
                        column
                            .child(self.model_group_label("Available"))
                            .children(available_models.iter().copied().map(card))
                    }),
            )
            .into_any_element()
    }

    fn model_group_label(&self, title: &'static str) -> AnyElement {
        div()
            .pt_2()
            .px_2()
            .text_xs()
            .font_weight(FontWeight::MEDIUM)
            .text_color(theme::muted())
            .child(title)
            .into_any_element()
    }

    fn render_audio(&self, cx: &Context<Self>) -> AnyElement {
        let mut microphone_options = vec![SelectOption::new("__default__", "System default")];
        microphone_options.extend(
            self.microphones
                .iter()
                .map(|name| SelectOption::new(name.clone(), name.clone())),
        );
        let selected_microphone = self
            .settings
            .selected_microphone
            .as_deref()
            .unwrap_or("__default__");
        div()
            .flex()
            .flex_col()
            .child(self.page_header("Audio", "16 kHz mono processing"))
            .child(self.section(
                "Microphone",
                vec![
                    self.control_row(
                        "Input device",
                        self.select(
                            "microphone-select",
                            microphone_options,
                            selected_microphone,
                            360.0,
                            cx,
                            |this, value, cx| {
                                this.save(
                                    |settings| {
                                        settings.selected_microphone =
                                            (value != "__default__").then_some(value)
                                    },
                                    cx,
                                );
                            },
                        ),
                    ),
                    self.toggle_row(
                        "vad",
                        "Voice activity detection",
                        self.settings.vad_enabled,
                        cx,
                        |settings, value| settings.vad_enabled = value,
                    ),
                    self.toggle_row(
                        "feedback",
                        "Audio feedback",
                        self.settings.audio_feedback,
                        cx,
                        |settings, value| settings.audio_feedback = value,
                    ),
                ],
            ))
            .into_any_element()
    }

    fn render_output(&self, cx: &Context<Self>) -> AnyElement {
        let mut text_rows = vec![self.control_row(
            "Insertion method",
            self.select(
                "paste-method-select",
                vec![
                    SelectOption::new("direct_typing", "Direct typing"),
                    SelectOption::new("command_v", "Clipboard + Command V"),
                    SelectOption::new("clipboard_only", "Clipboard only"),
                ],
                paste_method_value(self.settings.paste_method),
                260.0,
                cx,
                |this, value, cx| {
                    let method = match value.as_str() {
                        "clipboard_only" => PasteMethod::ClipboardOnly,
                        "direct_typing" => PasteMethod::DirectTyping,
                        _ => PasteMethod::CommandV,
                    };
                    this.save(|settings| settings.paste_method = method, cx);
                },
            ),
        )];
        if self.settings.paste_method == PasteMethod::CommandV {
            text_rows.push(self.toggle_row(
                "preserve-clipboard",
                "Restore clipboard after paste",
                self.settings.preserve_clipboard,
                cx,
                |settings, value| settings.preserve_clipboard = value,
            ));
        }
        text_rows.push(self.toggle_row(
            "trailing-space",
            "Append trailing space",
            self.settings.append_trailing_space,
            cx,
            |settings, value| settings.append_trailing_space = value,
        ));
        if self.settings.paste_method == PasteMethod::CommandV {
            text_rows.push(self.toggle_row(
                "auto-submit",
                "Submit after paste",
                self.settings.auto_submit,
                cx,
                |settings, value| settings.auto_submit = value,
            ));
        }
        text_rows.extend([
            self.action_row(
                "custom-terms",
                "Custom terms",
                format!("{} terms", self.settings.custom_words.len()),
                cx,
                |this, cx| {
                    let current = this.settings.custom_words.join(", ");
                    if let Some(value) = dialogs::prompt(
                        "Custom terms",
                        "Comma or newline separated. Whisper uses these as an initial prompt.",
                        &current,
                        false,
                    ) {
                        let words = value
                            .split([',', '\n'])
                            .map(str::trim)
                            .filter(|word| !word.is_empty())
                            .map(str::to_string)
                            .collect();
                        this.save(|settings| settings.custom_words = words, cx);
                    }
                },
            ),
            self.toggle_row(
                "fillers",
                "Remove filler words",
                self.settings.filler_word_removal_enabled,
                cx,
                |settings, value| settings.filler_word_removal_enabled = value,
            ),
        ]);
        let mut cleanup_rows = vec![self.toggle_row(
            "post-process",
            "Enable cleanup",
            self.settings.post_process_enabled,
            cx,
            |settings, value| settings.post_process_enabled = value,
        )];
        if self.settings.post_process_enabled {
            cleanup_rows.extend([
                self.action_row(
                    "post-process-url",
                    "AI endpoint",
                    self.settings.post_process_base_url.clone(),
                    cx,
                    |this, cx| {
                        if let Some(value) = dialogs::prompt(
                            "AI endpoint",
                            "OpenAI-compatible base URL",
                            &this.settings.post_process_base_url,
                            false,
                        ) {
                            this.save(|settings| settings.post_process_base_url = value, cx);
                        }
                    },
                ),
                self.action_row(
                    "post-process-model",
                    "AI model",
                    if self.settings.post_process_model.is_empty() {
                        "Not set".to_string()
                    } else {
                        self.settings.post_process_model.clone()
                    },
                    cx,
                    |this, cx| {
                        if let Some(value) = dialogs::prompt(
                            "AI model",
                            "Model name exposed by the endpoint",
                            &this.settings.post_process_model,
                            false,
                        ) {
                            this.save(|settings| settings.post_process_model = value, cx);
                        }
                    },
                ),
                self.action_row(
                    "post-process-key",
                    "API key",
                    if self.settings.post_process_api_key.is_empty() {
                        "Not set"
                    } else {
                        "Saved"
                    },
                    cx,
                    |this, cx| {
                        if let Some(value) = dialogs::prompt(
                            "API key",
                            "Saved locally in Pyzhscriber settings.",
                            &this.settings.post_process_api_key,
                            true,
                        ) && !value.is_empty()
                        {
                            this.save(|settings| settings.post_process_api_key = value, cx);
                        }
                    },
                ),
            ]);
        }

        div()
            .flex()
            .flex_col()
            .child(self.page_header("Output", "Text insertion and cleanup"))
            .child(self.section("Text", text_rows))
            .child(self.section("AI cleanup", cleanup_rows))
            .into_any_element()
    }

    fn render_history(&self, cx: &Context<Self>) -> AnyElement {
        if self.history.is_empty() {
            return div()
                .flex()
                .flex_col()
                .child(self.page_header("History", "Recent transcriptions"))
                .child(self.section(
                    "Transcriptions",
                    vec![self.row("No transcriptions yet", "")],
                ))
                .into_any_element();
        }
        let mut rows =
            vec![
                self.action_row(
                    "clear-history",
                    "Clear history",
                    "Clear",
                    cx,
                    |this, cx| match this.controller.history.clear() {
                        Ok(()) => {
                            this.history.clear();
                            cx.notify();
                        }
                        Err(error) => this.status = RuntimeStatus::Error(error.to_string()),
                    },
                ),
            ];
        rows.extend(self.history.iter().map(|entry| {
            let text = entry.text.clone();
            let id = entry.id;
            let copied = self.copied_entry == Some(id);
            div()
                .id(SharedString::from(format!("history-{}", entry.id)))
                .min_h(px(64.0))
                .px_4()
                .py_3()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .text_sm()
                        .text_color(theme::text())
                        .child(entry.text.clone()),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_4()
                        .text_xs()
                        .text_color(theme::muted())
                        .child(format!(
                            "{:.1}s  ·  {}",
                            entry.audio_seconds,
                            entry
                                .created_at
                                .with_timezone(&chrono::Local)
                                .format("%H:%M")
                        ))
                        .child(if copied {
                            div()
                                .flex()
                                .items_center()
                                .gap_1()
                                .text_color(theme::success())
                                .child(
                                    svg()
                                        .path("icons/check.svg")
                                        .size(px(13.0))
                                        .text_color(theme::success()),
                                )
                                .child("Copied")
                        } else {
                            div().child(
                                svg()
                                    .path("icons/copy.svg")
                                    .size(px(13.0))
                                    .text_color(theme::muted()),
                            )
                        }),
                )
                .hover(|style| style.bg(theme::hover()))
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, cx| {
                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                        let _ = clipboard.set_text(text.clone());
                        this.copied_entry = Some(id);
                        cx.spawn(async move |this, cx| {
                            smol::Timer::after(Duration::from_millis(1500)).await;
                            let _ = this.update(cx, |this, cx| {
                                if this.copied_entry == Some(id) {
                                    this.copied_entry = None;
                                    cx.notify();
                                }
                            });
                        })
                        .detach();
                    } else {
                        this.status = RuntimeStatus::Error("Clipboard is unavailable".into());
                    }
                    cx.notify();
                }))
                .into_any_element()
        }));
        div()
            .flex()
            .flex_col()
            .child(self.page_header("History", "Recent transcriptions"))
            .child(self.section("Transcriptions", rows))
            .into_any_element()
    }

    fn render_advanced(&self, cx: &Context<Self>) -> AnyElement {
        let unload_value = self
            .settings
            .model_unload_minutes
            .map_or_else(|| "never".to_string(), |minutes| minutes.to_string());
        let retention_value = self
            .settings
            .recording_retention_days
            .map_or_else(|| "forever".to_string(), |days| days.to_string());
        let history_value = self.settings.history_limit.to_string();
        div()
            .flex()
            .flex_col()
            .child(self.page_header("Advanced", "Runtime and startup"))
            .child(self.section(
                "Runtime",
                vec![
                    self.row(
                        "Loaded model",
                        self.controller
                            .loaded_model_id()
                            .map(|id| self.model_name(&id))
                            .unwrap_or_else(|| "None".into()),
                    ),
                    self.control_row(
                        "Unload models after",
                        self.select(
                            "unload-timeout-select",
                            vec![
                                SelectOption::new("never", "Never"),
                                SelectOption::new("1", "1 minute"),
                                SelectOption::new("5", "5 minutes"),
                                SelectOption::new("15", "15 minutes"),
                                SelectOption::new("30", "30 minutes"),
                            ],
                            &unload_value,
                            220.0,
                            cx,
                            |this, value, cx| {
                                this.save(
                                    |settings| {
                                        settings.model_unload_minutes = value.parse().ok()
                                    },
                                    cx,
                                );
                            },
                        ),
                    ),
                    self.action_row(
                        "unload-model",
                        "Unload current model now",
                        "Unload",
                        cx,
                        |this, cx| {
                            this.controller.unload_model();
                            cx.notify();
                        },
                    ),
                ],
            ))
            .child(self.section(
                "History",
                vec![
                    self.control_row(
                        "Entries kept",
                        self.select(
                            "history-limit-select",
                            vec![
                                SelectOption::new("20", "20 entries"),
                                SelectOption::new("50", "50 entries"),
                                SelectOption::new("100", "100 entries"),
                                SelectOption::new("250", "250 entries"),
                            ],
                            &history_value,
                            220.0,
                            cx,
                            |this, value, cx| {
                                if let Ok(limit) = value.parse() {
                                    this.save(|settings| settings.history_limit = limit, cx);
                                }
                            },
                        ),
                    ),
                    self.control_row(
                        "Keep recordings",
                        self.select(
                            "recording-retention-select",
                            vec![
                                SelectOption::new("forever", "Forever"),
                                SelectOption::new("1", "1 day"),
                                SelectOption::new("7", "7 days"),
                                SelectOption::new("30", "30 days"),
                            ],
                            &retention_value,
                            220.0,
                            cx,
                            |this, value, cx| {
                                this.save(
                                    |settings| {
                                        settings.recording_retention_days = value.parse().ok()
                                    },
                                    cx,
                                );
                            },
                        ),
                    ),
                ],
            ))
            .child(self.section(
                "Startup",
                vec![
                    self.toggle_row(
                        "autostart",
                        "Launch at login",
                        self.settings.autostart_enabled,
                        cx,
                        |settings, value| settings.autostart_enabled = value,
                    ),
                    self.toggle_row(
                        "tray",
                        "Show menu bar icon",
                        self.settings.show_tray_icon,
                        cx,
                        |settings, value| settings.show_tray_icon = value,
                    ),
                    self.toggle_row(
                        "start-hidden",
                        "Start hidden",
                        self.settings.start_hidden,
                        cx,
                        |settings, value| settings.start_hidden = value,
                    ),
                ],
            ))
            .into_any_element()
    }
}

// Seconds, all read straight off the wall clock during render.
const OVERLAY_ENTER: f32 = 0.26;
const OVERLAY_LEAVE: f32 = 0.16;
const OVERLAY_SETTLE: f32 = 0.14;
const OVERLAY_FRAME: Duration = Duration::from_millis(16);
const OVERLAY_PILL_WIDTH: f32 = 148.0;
const OVERLAY_PILL_HEIGHT: f32 = 42.0;
const OVERLAY_WINDOW_WIDTH: f32 = 196.0;
const OVERLAY_WINDOW_HEIGHT: f32 = 88.0;
/// The window is larger than the pill so the entrance slide, the overshoot, and
/// the shadow have room; this gap keeps the pill's centre where it sat before.
const OVERLAY_SCREEN_GAP: f32 = 5.0;
/// Bubbly mode: the jelly entrance needs longer than the plain pop to settle,
/// and both the bars and the pill are underdamped springs instead of a
/// smoothing rate. Both sit near half-critical damping: one soft overshoot
/// and done, the way glass wobbles rather than the way rubber bounces. The
/// pill's spring is the softer one so its wobble reads as a whole body
/// reacting rather than as a ninth bar.
const BUBBLY_ENTER: f32 = 0.44;
const BUBBLY_LEAVE: f32 = 0.32;
const BAR_STIFFNESS: f32 = 420.0;
const BAR_DAMPING: f32 = 24.0;
const PILL_STIFFNESS: f32 = 200.0;
const PILL_DAMPING: f32 = 13.0;
/// Cap on the spring step so a stalled frame cannot push them past stability.
const SPRING_STEP: f32 = 0.04;

pub(crate) fn overlay_window_size() -> Size<Pixels> {
    size(px(OVERLAY_WINDOW_WIDTH), px(OVERLAY_WINDOW_HEIGHT))
}

/// The bottom-centered recorder pill. Its motion is driven by wall clock time
/// rather than per-property animations: one render reads the clock and derives
/// the entrance, the exit, and the bar heights from it, so the states can
/// interrupt each other at any point. The pill keeps one shape throughout;
/// transcription is usually over in a few hundred milliseconds, so a second
/// layout for it would never finish playing.
pub(crate) struct RecordingOverlay {
    status: RuntimeStatus,
    controller: Arc<AppController>,
    qa_levels: Option<[f32; 9]>,
    /// A second smoothing stage on top of the audio thread's: the mic callback
    /// runs far faster than the display, so bars sampled raw look like noise.
    /// Fast attack keeps a shout instant, slow release keeps the decay calm.
    bars: [f32; 9],
    /// Bar velocities for the bubbly spring; idle in the plain mode.
    velocity: [f32; 9],
    /// The pill's own spring, chasing the loudest bar: a burst of voice swells
    /// the pill and the lag past it rings down as a wobble. Bubbly mode only.
    flex: f32,
    flex_velocity: f32,
    /// Flexes and jiggles: springy bars and a jelly pill.
    bubbly: bool,
    opened: Instant,
    /// Set when the pill starts shrinking away; its window closes once the exit
    /// has played out. Cleared if recording resumes mid-exit.
    leaving: Option<Instant>,
    /// Start of the current status, used to settle the bars once the mic stops.
    changed: Instant,
    painted: Instant,
}

impl RecordingOverlay {
    fn new(
        controller: Arc<AppController>,
        status: RuntimeStatus,
        bubbly: bool,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.spawn(async move |this, cx| {
            loop {
                smol::Timer::after(OVERLAY_FRAME).await;
                if this.update(cx, |_, cx| cx.notify()).is_err() {
                    break;
                }
            }
        })
        .detach();
        let now = Instant::now();
        Self {
            status,
            controller,
            qa_levels: None,
            bars: [0.0; 9],
            velocity: [0.0; 9],
            flex: 0.0,
            flex_velocity: 0.0,
            bubbly,
            opened: now,
            leaving: None,
            changed: now,
            painted: now,
        }
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn qa(controller: Arc<AppController>, status: RuntimeStatus, bubbly: bool) -> Self {
        // Screenshots must be reproducible, so the pill is pinned to its settled
        // state instead of whatever frame the clock happens to land on.
        let settled = Instant::now() - Duration::from_secs(5);
        Self {
            status,
            controller,
            qa_levels: Some([0.24, 0.52, 0.78, 0.42, 0.94, 0.66, 0.35, 0.7, 0.28]),
            bars: [0.24, 0.52, 0.78, 0.42, 0.94, 0.66, 0.35, 0.7, 0.28],
            velocity: [0.0; 9],
            // Pinned to the peak of the levels above so the pill sits at rest.
            flex: 0.94,
            flex_velocity: 0.0,
            bubbly,
            opened: settled,
            leaving: None,
            changed: settled,
            painted: settled,
        }
    }

    /// Pins the clock to a moment inside the entrance or the exit.
    #[cfg(feature = "visual-qa")]
    pub(crate) fn qa_frame(mut self, opened_ago: f32, leaving_ago: Option<f32>) -> Self {
        let now = Instant::now();
        self.opened = now - Duration::from_secs_f32(opened_ago);
        self.leaving = leaving_ago.map(|ago| now - Duration::from_secs_f32(ago));
        self
    }

    fn set_status(&mut self, status: RuntimeStatus) {
        // Reappearing mid-exit rewinds the entrance to the size the pill still
        // has on screen, so it grows back instead of blinking to full size.
        if self.leaving.take().is_some() {
            let visible = self.presence();
            self.opened = Instant::now() - Duration::from_secs_f32(visible * OVERLAY_ENTER);
        }
        if self.status != status {
            self.status = status;
            self.changed = Instant::now();
        }
    }

    fn begin_exit(&mut self) {
        if self.leaving.is_none() {
            self.leaving = Some(Instant::now());
        }
    }

    /// How long the exit plays before the window can be torn down.
    fn leave_seconds(&self) -> f32 {
        if self.bubbly {
            BUBBLY_LEAVE
        } else {
            OVERLAY_LEAVE
        }
    }

    /// How much of the pill is on screen, 0 when fully hidden and 1 when settled.
    fn presence(&self) -> f32 {
        let entered = ease_out(elapsed_fraction(self.opened, OVERLAY_ENTER));
        let left = self
            .leaving
            .map(|start| {
                let gone = elapsed_fraction(start, self.leave_seconds());
                if self.bubbly {
                    // Holds through the anticipation lift, fades on the drop.
                    ease_in(((gone - 0.25) / 0.75).clamp(0.0, 1.0))
                } else {
                    ease_in(gone)
                }
            })
            .unwrap_or(0.0);
        entered * (1.0 - left)
    }

    /// Time since the last frame, capped so a stall does not become a jump.
    fn tick(&mut self) -> f32 {
        let now = Instant::now();
        let delta = (now - self.painted).as_secs_f32().min(0.1);
        self.painted = now;
        delta
    }

    /// Advances the render-rate bar smoothing and returns the drawn heights.
    fn sample_bars(&mut self, settled: f32, delta: f32) -> [f32; 9] {
        let age = self.opened.elapsed().as_secs_f32();
        let levels = self
            .qa_levels
            .unwrap_or_else(|| self.controller.mic_levels());
        // Silence has to look silent, so the resting bars are a flat row of dots
        // with one shared, barely visible pulse; anything per-bar reads as a
        // faint signal.
        let idle = 0.02 + 0.015 * (age * 2.2).sin();
        let mut drawn = self.bars;
        for (index, level) in levels.into_iter().enumerate() {
            let target = level.max(idle);
            if self.bubbly {
                // Underdamped, so a bar overshoots the level and jiggles back.
                // A spring turns meter flutter into jitter, so only voice that
                // clearly clears the breathing floor drives it; the knee is
                // soft so a rising level fades in rather than snapping.
                let voice = ((level - idle) / 0.06).clamp(0.0, 1.0);
                let target = idle + (level - idle).max(0.0) * voice;
                let dt = delta.min(SPRING_STEP);
                let pull = (target - self.bars[index]) * BAR_STIFFNESS;
                let drag = self.velocity[index] * BAR_DAMPING;
                self.velocity[index] += (pull - drag) * dt;
                self.bars[index] += self.velocity[index] * dt;
            } else {
                let rate = if target > self.bars[index] { 26.0 } else { 9.0 };
                let step = (rate * delta).min(1.0);
                self.bars[index] += (target - self.bars[index]) * step;
            }
            // Transcribing has no live input, so the bars drop back to the idle
            // floor rather than taking on a shape of their own.
            drawn[index] = self.bars[index] + (idle - self.bars[index]) * settled;
        }
        drawn
    }

    /// Advances the pill spring toward the loudest bar and returns its position.
    fn flex_pill(&mut self, peak: f32, delta: f32) -> f32 {
        let dt = delta.min(SPRING_STEP);
        let pull = (peak - self.flex) * PILL_STIFFNESS;
        let drag = self.flex_velocity * PILL_DAMPING;
        self.flex_velocity += (pull - drag) * dt;
        self.flex += self.flex_velocity * dt;
        self.flex
    }
}

/// What bubbly mode adds to a frame; every field is zero in the plain mode.
#[derive(Clone, Copy, Default)]
struct Jelly {
    /// Extra scale from the pill spring, so voice faintly swells the body.
    swell: f32,
    /// Wider-and-flatter when positive, taller-and-thinner when negative.
    squash: f32,
}

/// The pill's size and position for this frame of its entrance or exit.
struct Shape {
    width: f32,
    height: f32,
    rise: f32,
}

impl Render for RecordingOverlay {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let presence = self.presence();
        let age = self.opened.elapsed().as_secs_f32();
        let leaving = self.leaving.map(|start| start.elapsed().as_secs_f32());
        let bubbly = self.bubbly;

        let shape = if bubbly {
            // A droplet: it lands at nearly full height first, then stretches
            // wide with one soft overshoot, arriving from below with a small
            // overshoot upward. The stretch starts a beat late so the droplet
            // shape is actually seen.
            let t = elapsed_fraction(self.opened, BUBBLY_ENTER);
            let stretch = ease_out_back(((t - 0.08) / 0.92).clamp(0.0, 1.0));
            let mut shape = Shape {
                width: 0.4 + 0.6 * stretch,
                height: 0.82 + 0.18 * ease_out(t),
                rise: (1.0 - ease_out_back(t)) * 22.0,
            };
            if let Some(start) = self.leaving {
                // Anticipation: a small lift and swell, then it squeezes back
                // into a droplet and drops away.
                let gone = elapsed_fraction(start, BUBBLY_LEAVE);
                let lift = ease_out((gone / 0.3).min(1.0));
                let drop = ease_in(((gone - 0.25) / 0.75).clamp(0.0, 1.0));
                shape.width *= 1.0 + 0.04 * lift - 0.55 * drop;
                shape.height *= 1.0 + 0.02 * lift - 0.3 * drop;
                shape.rise += -4.0 * lift + 28.0 * drop;
            }
            shape
        } else {
            // The entrance overshoots; the exit must not, so it tracks presence.
            let grow = if self.leaving.is_some() {
                presence
            } else {
                ease_out_back(elapsed_fraction(self.opened, OVERLAY_ENTER))
            };
            let scale = 0.74 + 0.26 * grow;
            Shape {
                width: scale,
                height: scale,
                rise: (1.0 - presence) * 16.0,
            }
        };

        // Releasing the shortcut just lets the bars fall back to rest; on a fast
        // transcription this runs straight into the exit as one gesture.
        let settled = if self.status == RuntimeStatus::Transcribing {
            ease_out(elapsed_fraction(self.changed, OVERLAY_SETTLE))
        } else {
            0.0
        };

        let delta = self.tick();
        let bars = self.sample_bars(settled, delta);
        let velocity = self.velocity;
        let peak = bars.iter().copied().fold(0.0_f32, f32::max);
        // How much voice is showing, 0 at the resting floor.
        let activity = ((peak - 0.06) / 0.2).clamp(0.0, 1.0) * (1.0 - settled);

        let jelly = if bubbly {
            let flex = self.flex_pill(peak, delta);
            // The spring's lag behind the level is the wobble.
            Jelly {
                swell: 0.02 * flex.clamp(0.0, 1.2),
                squash: ((flex - peak) * 0.12).clamp(-0.05, 0.08),
            }
        } else {
            Jelly::default()
        };
        let scale = shape.height * (1.0 + jelly.swell);
        let rise = shape.rise;
        let pill_width =
            OVERLAY_PILL_WIDTH * shape.width * (1.0 + jelly.swell) * (1.0 + jelly.squash);
        let pill_height = OVERLAY_PILL_HEIGHT * scale * (1.0 - jelly.squash);
        let accent = theme::accent();

        let bars =
            bars.into_iter()
                .zip(velocity)
                .enumerate()
                .map(move |(index, (level, velocity))| {
                    let span = if index % 2 == 0 { 17.0 } else { 21.0 };
                    // Squash and stretch: a bar fattens a touch while it is moving
                    // fast, and the row rides a slow, shallow travelling wave
                    // that only comes up with the voice, so silence stays still.
                    let (stretch, wave) = if bubbly {
                        (
                            (velocity.abs() * 0.012).min(0.12),
                            (age * 2.6 + index as f32 * 0.7).sin() * 0.5 * activity,
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    // Bubbly bars bloom from the centre outward once the droplet
                    // has started stretching, and fold back in, outer bars
                    // first, ahead of the exit's drop.
                    let reveal = if bubbly {
                        let from_center = (index as f32 - 4.0).abs();
                        let bloom = ease_out(
                            ((age / BUBBLY_ENTER - 0.1 - from_center * 0.08) / 0.4).clamp(0.0, 1.0),
                        );
                        let fold = leaving
                            .map(|gone| {
                                ease_in(
                                    ((gone - (4.0 - from_center) * 0.03) / 0.16).clamp(0.0, 1.0),
                                )
                            })
                            .unwrap_or(0.0);
                        bloom * (1.0 - fold)
                    } else {
                        1.0
                    };
                    div()
                        .relative()
                        .top(px(wave))
                        .w(px(3.0 * (1.0 + stretch)))
                        .h(px((5.0 + level.max(0.0) * span * reveal) * scale))
                        .rounded_full()
                        .bg(accent.opacity((0.5 + 0.5 * level.clamp(0.0, 1.0)) * reveal))
                });

        div()
            .size_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .child(
                div()
                    .mt(px(rise))
                    .w(px(pill_width))
                    .h(px(pill_height))
                    .px_4()
                    // The shadow is quad-only, so it keeps a capsule shape; the
                    // difference from the squircle vanishes inside the blur.
                    .rounded_full()
                    .shadow_lg()
                    .opacity(presence)
                    .flex()
                    .items_center()
                    .child(
                        // The body and its 1px border are two anti-aliased fills,
                        // drawn a pixel inside the element so both stay whole.
                        canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                let border = mix(
                                    theme::border(),
                                    accent,
                                    (0.1 + 0.32 * peak) * (1.0 - settled),
                                );
                                window.paint_path(squircle(bounds, 1.0), border);
                                window.paint_path(squircle(bounds, 2.0), theme::raised());
                            },
                        )
                        .absolute()
                        .inset_0(),
                    )
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .items_center()
                            .justify_center()
                            .gap(px(3.0))
                            .children(bars),
                    ),
            )
    }
}

/// A capsule with continuous caps, inset from `bounds` by `inset` on every
/// side. The tips are true semicircles, exactly as with `rounded_full`; what
/// changes is the junction with the flat edges, where a short easing curve
/// takes the curvature from zero up to the arc's instead of switching to it
/// abruptly. The outline is all quadratic curves because that is the only
/// geometry GPUI anti-aliases; the long edges bow outward by half a pixel for
/// the same reason.
fn squircle(bounds: Bounds<Pixels>, inset: f32) -> Path<Pixels> {
    /// Where the easing hands over to the circular arc, measured from the
    /// top of the cap. Larger means a longer, more visible blend.
    const BLEND: f32 = 40.0_f32.to_radians();
    const SEGMENTS: usize = 4;
    let left = f32::from(bounds.origin.x) + inset;
    let top = f32::from(bounds.origin.y) + inset;
    let width = f32::from(bounds.size.width) - inset * 2.0;
    let height = f32::from(bounds.size.height) - inset * 2.0;
    let r = height / 2.0;
    let (right, bottom) = (left + width, top + height);
    let middle = top + r;
    let (sin, cos) = BLEND.sin_cos();

    // The easing is a cubic whose control points sit on the flat edge, which
    // pins its curvature to zero there; their spacing is what makes its
    // curvature at the far end match the arc's, so the join is invisible.
    let arc_x = r * sin;
    let arc_y = r * cos;
    let k2 = r * (BLEND / 2.0).tan();
    let c2_x = arc_x - k2 * cos;
    let c1_x = c2_x - 1.5 * k2 * k2 / (r * sin);
    let reach = (c2_x - c1_x) - c1_x;
    let reach = reach.min(width / 2.0 - r).max(0.0);

    // A cap point at horizontal `side` (-1 left, 1 right) and vertical `half`,
    // with `t` in 0..=1 running from the flat edge to the tip.
    let cap = move |side: f32, half: f32, t: f32| {
        let cx = if side < 0.0 { left + r } else { right - r };
        let (x, y) = if t < 0.5 {
            // The cubic easing, from the edge into the arc.
            let t = t * 2.0;
            let (p0, c1, c2, p1) = ((-reach, 0.0), (c1_x, 0.0), (c2_x, 0.0), (arc_x, r - arc_y));
            let u = 1.0 - t;
            let w = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            (
                p0.0 * w.0 + c1.0 * w.1 + c2.0 * w.2 + p1.0 * w.3,
                p0.1 * w.0 + c1.1 * w.1 + c2.1 * w.2 + p1.1 * w.3,
            )
        } else {
            // The circular arc, from the blend angle to the tip.
            let angle = BLEND + (t - 0.5) * 2.0 * (std::f32::consts::FRAC_PI_2 - BLEND);
            (r * angle.sin(), r - r * angle.cos())
        };
        point(px(cx + side * x), px(middle + half * (y - r)))
    };
    // Sweeps a cap half as quadratics, each fitted through its midpoint.
    let sweep = |path: &mut Path<Pixels>, side: f32, half: f32, from: f32, to: f32| {
        let step = (to - from) / SEGMENTS as f32;
        for segment in 0..SEGMENTS {
            let t0 = from + step * segment as f32;
            let t1 = t0 + step;
            let a = cap(side, half, t0);
            let m = cap(side, half, (t0 + t1) / 2.0);
            let b = cap(side, half, t1);
            let ctrl = point(m.x * 2.0 - (a.x + b.x) / 2.0, m.y * 2.0 - (a.y + b.y) / 2.0);
            path.curve_to(b, ctrl);
        }
    };

    let mut path = Path::new(cap(-1.0, -1.0, 0.0));
    path.curve_to(
        cap(1.0, -1.0, 0.0),
        point(px((left + right) / 2.0), px(top - 1.0)),
    );
    sweep(&mut path, 1.0, -1.0, 0.0, 0.5);
    sweep(&mut path, 1.0, -1.0, 0.5, 1.0);
    sweep(&mut path, 1.0, 1.0, 1.0, 0.5);
    sweep(&mut path, 1.0, 1.0, 0.5, 0.0);
    path.curve_to(
        cap(-1.0, 1.0, 0.0),
        point(px((left + right) / 2.0), px(bottom + 1.0)),
    );
    sweep(&mut path, -1.0, 1.0, 0.0, 0.5);
    sweep(&mut path, -1.0, 1.0, 0.5, 1.0);
    sweep(&mut path, -1.0, -1.0, 1.0, 0.5);
    sweep(&mut path, -1.0, -1.0, 0.5, 0.0);
    path
}

fn elapsed_fraction(since: Instant, seconds: f32) -> f32 {
    (since.elapsed().as_secs_f32() / seconds).clamp(0.0, 1.0)
}

fn ease_out(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn ease_in(t: f32) -> f32 {
    t * t
}

/// Ease out with a small overshoot, which is what gives the pill its pop.
fn ease_out_back(t: f32) -> f32 {
    let c = 1.70158;
    1.0 + (c + 1.0) * (t - 1.0).powi(3) + c * (t - 1.0).powi(2)
}

/// Interpolates two theme colours along the shorter way around the hue circle.
fn mix(from: Hsla, to: Hsla, amount: f32) -> Hsla {
    let amount = amount.clamp(0.0, 1.0);
    let mut hue_delta = to.h - from.h;
    if hue_delta > 0.5 {
        hue_delta -= 1.0;
    } else if hue_delta < -0.5 {
        hue_delta += 1.0;
    }
    Hsla {
        h: (from.h + hue_delta * amount).rem_euclid(1.0),
        s: from.s + (to.s - from.s) * amount,
        l: from.l + (to.l - from.l) * amount,
        a: from.a + (to.a - from.a) * amount,
    }
}

/// GPUI builds every window with a titled style mask, and macOS 26 paints its
/// glass theme frame behind the content. The recording overlay draws its own
/// pill, so the frame is stripped down to a borderless panel that also stays
/// visible while another app is active.
#[cfg(target_os = "macos")]
fn strip_window_chrome(window: &Window) {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    // Borderless drops the glass theme frame; the non-activating panel bit must
    // survive it, otherwise clicking the overlay would activate the app.
    const BORDERLESS_NONACTIVATING_PANEL: usize = 1 << 7;

    // Disambiguated: gpui's own inherent window_handle() returns AnyWindowHandle.
    let Ok(handle) = HasWindowHandle::window_handle(window) else {
        return;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return;
    };
    let view = handle.ns_view.as_ptr().cast::<AnyObject>();
    unsafe {
        let native: *mut AnyObject = msg_send![view, window];
        if native.is_null() {
            return;
        }
        let _: () = msg_send![native, setStyleMask: BORDERLESS_NONACTIVATING_PANEL];
        let _: () = msg_send![native, setOpaque: false];
        // The pill renders its own shadow; a window shadow would trace the frame.
        let _: () = msg_send![native, setHasShadow: false];
        let _: () = msg_send![native, setHidesOnDeactivate: false];
    }
}

#[cfg(not(target_os = "macos"))]
fn strip_window_chrome(_window: &Window) {}

#[derive(Clone, Copy, Debug)]
enum PermissionKind {
    Microphone,
    Accessibility,
    InputMonitoring,
}

impl Render for RootView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.permission_setup_visible {
            return self.render_permission_setup(cx);
        }
        let content = match self.page {
            Page::General => self.render_general(cx),
            Page::Models => self.render_models(cx),
            Page::Audio => self.render_audio(cx),
            Page::Output => self.render_output(cx),
            Page::History => self.render_history(cx),
            Page::Advanced => self.render_advanced(cx),
        };
        let status_label = self.status.label().to_string();
        div()
            .size_full()
            .bg(theme::background())
            .text_color(theme::text())
            .flex()
            .child(
                div()
                    .w(px(196.0))
                    .flex_shrink_0()
                    .h_full()
                    .bg(theme::raised())
                    .pt(px(48.0))
                    .pb_4()
                    .border_r_1()
                    .border_color(theme::border())
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .px_4()
                            .pb_4()
                            .text_lg()
                            .font_weight(FontWeight::BOLD)
                            .child("Pyzhscriber"),
                    )
                    .child(self.sidebar_item(
                        Page::General,
                        "General",
                        "icons/sliders-horizontal.svg",
                        cx,
                    ))
                    .child(self.sidebar_item(Page::Models, "Models", "icons/package.svg", cx))
                    .child(self.sidebar_item(Page::Audio, "Audio", "icons/zap.svg", cx))
                    .child(self.sidebar_item(Page::Output, "Output", "icons/copy.svg", cx))
                    .child(self.sidebar_item(Page::History, "History", "icons/history.svg", cx))
                    .child(self.sidebar_item(Page::Advanced, "Advanced", "icons/settings.svg", cx))
                    .child(div().flex_1())
                    .child(
                        div()
                            .mx_3()
                            .pt_3()
                            .border_t_1()
                            .border_color(theme::border())
                            .flex()
                            .items_center()
                            .gap_2()
                            .text_xs()
                            .text_color(theme::muted())
                            .child(div().size(px(7.0)).rounded_full().bg(
                                if matches!(self.status, RuntimeStatus::Error(_)) {
                                    theme::danger()
                                } else {
                                    theme::success()
                                },
                            ))
                            .child(status_label),
                    ),
            )
            .child(
                div()
                    .id("window-content")
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .pt(px(48.0))
                    .px(px(40.0))
                    .pb_8()
                    .overflow_y_scroll()
                    .child(content),
            )
            .into_any_element()
    }
}

fn paste_method_value(method: PasteMethod) -> &'static str {
    match method {
        PasteMethod::CommandV => "command_v",
        PasteMethod::ClipboardOnly => "clipboard_only",
        PasteMethod::DirectTyping => "direct_typing",
    }
}

fn model_metric(label: &'static str, score: f32) -> AnyElement {
    let score = score.clamp(0.0, 1.0);
    div()
        .w(px(148.0))
        .flex()
        .items_center()
        .gap_2()
        .child(
            div()
                .w(px(64.0))
                .text_xs()
                .text_color(theme::muted())
                .child(label),
        )
        .child(
            div()
                .w(px(74.0))
                .h(px(6.0))
                .rounded_full()
                .bg(theme::hover())
                .child(
                    div()
                        .h_full()
                        .w(px(74.0 * score))
                        .rounded_full()
                        .bg(theme::accent()),
                ),
        )
        .into_any_element()
}

fn model_matches_search(model: &ModelChoice, query: &str) -> bool {
    let query = query.trim();
    if query.is_empty() {
        return true;
    }

    let query = query.to_lowercase();
    model.name.to_lowercase().contains(&query) || model.description.to_lowercase().contains(&query)
}

#[cfg(test)]
mod tests {
    use super::model_matches_search;
    use crate::models::ModelChoice;

    fn model(name: &str, description: &str) -> ModelChoice {
        ModelChoice {
            id: "test/model.gguf".into(),
            name: name.into(),
            description: description.into(),
            architecture: "test".into(),
            languages: vec!["en".into()],
            quant: "Q8_0".into(),
            size_bytes: 1,
            speed_score: 1.0,
            accuracy_score: 1.0,
            installed: false,
            recommended: false,
            streaming: false,
            translate: false,
        }
    }

    #[test]
    fn model_search_matches_name_and_description_case_insensitively() {
        let model = model(
            "Parakeet Unified EN 0.6B",
            "Fast, accurate live English transcription",
        );

        assert!(model_matches_search(&model, "parakeet"));
        assert!(model_matches_search(&model, "ACCURATE"));
        assert!(!model_matches_search(&model, "translation"));
    }

    #[test]
    fn empty_model_search_matches_everything() {
        let model = model("Canary", "Tiny and instant");

        assert!(model_matches_search(&model, ""));
        assert!(model_matches_search(&model, "   "));
    }
}
