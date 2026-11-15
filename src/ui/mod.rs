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
use crate::settings::{AppSettings, CloudProvider, PasteMethod, RouteSetting, TranscriptionMode};
use crate::theme;
use gpui::{
    AnimationExt, AnyElement, App, Bounds, Context, CursorStyle, FocusHandle, FontWeight, Hsla,
    KeyDownEvent, MouseButton, Path, Pixels, Render, SharedString, Size, SpringAnimation,
    SpringConfig, Window, WindowBackgroundAppearance, WindowBounds, WindowHandle, WindowKind,
    WindowOptions, canvas, deferred, div, point, prelude::*, px, relative, size, svg,
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

/// A group of settings rows ruled by hairlines, with no frame around it.
fn framed(children: Vec<AnyElement>) -> impl IntoElement {
    let last = children.len().saturating_sub(1);
    div()
        .w_full()
        .border_t_1()
        .border_color(theme::separator())
        .children(children.into_iter().enumerate().map(|(index, child)| {
            div()
                .w_full()
                .when(index < last, |row| {
                    row.border_b_1().border_color(theme::separator())
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
    sensitivity_drag: Option<Bounds<Pixels>>,
    permission_statuses: PermissionStatuses,
    permission_setup_visible: bool,
    permission_polling: bool,
    shortcut_capture: ShortcutCapture,
    /// History entry whose text was just copied, cleared after a short delay.
    copied_entry: Option<i64>,
    cli_install_result: Option<Result<String, String>>,
    /// Models tab filters. The catalog runs to dozens of entries, most of which
    /// are irrelevant to any one user, so the list is narrowed before display.
    model_search: String,
    model_search_focus: FocusHandle,
    model_language: ModelLanguage,
    model_streaming_only: bool,
    model_translate_only: bool,
    /// Query typed into whichever searchable dropdown is open.
    dropdown_search: String,
    dropdown_search_focus: FocusHandle,
    /// OpenRouter's transcription models, fetched the first time its model
    /// menu opens. Until then the built-in suggestions are shown.
    openrouter_models: Option<Vec<crate::cloud::CloudModel>>,
    openrouter_fetching: bool,
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
                            this.settings = *settings;
                            this.sync_overlay(cx);
                        }
                        RuntimeEvent::DownloadProgress { model_id, progress } => {
                            if progress >= 1.0 {
                                this.downloads.remove(&model_id);
                            } else {
                                this.downloads.insert(model_id, progress);
                            }
                        }
                        RuntimeEvent::DownloadStopped { model_id } => {
                            this.downloads.remove(&model_id);
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
        let models = controller.models.choices();
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
            sensitivity_drag: None,
            permission_statuses,
            permission_setup_visible: !permission_statuses.all_granted(),
            permission_polling: false,
            shortcut_capture: ShortcutCapture::Idle,
            copied_entry: None,
            cli_install_result: None,
            model_search: String::new(),
            model_search_focus: cx.focus_handle().tab_stop(true),
            model_language: ModelLanguage::Any,
            model_streaming_only: false,
            model_translate_only: false,
            dropdown_search: String::new(),
            dropdown_search_focus: cx.focus_handle(),
            openrouter_models: None,
            openrouter_fetching: false,
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
    pub(crate) fn prepare_qa_cloud(&mut self, cx: &mut Context<Self>) {
        self.prepare_qa_general(cx);
        self.shortcut_capture = ShortcutCapture::Idle;
        self.settings.cloud_enabled = true;
        self.settings.cloud_provider = CloudProvider::OpenRouter;
        self.settings.cloud_accounts.insert(
            CloudProvider::OpenRouter,
            crate::settings::CloudAccount {
                api_key: "qa".into(),
                ..Default::default()
            },
        );
        self.openrouter_models = Some(
            [
                ("openai/whisper-large-v3", "OpenAI: Whisper Large V3"),
                (
                    "openai/whisper-large-v3-turbo",
                    "OpenAI: Whisper Large V3 Turbo",
                ),
                ("elevenlabs/scribe-v2", "ElevenLabs: Scribe v2"),
            ]
            .map(|(id, name)| crate::cloud::CloudModel {
                id: id.into(),
                name: name.into(),
            })
            .into(),
        );
        self.dropdown_search = "whisper".into();
        self.open_dropdown = Some("cloud-model".into());
        cx.notify();
    }

    #[cfg(feature = "visual-qa")]
    pub(crate) fn prepare_qa_error(&mut self, cx: &mut Context<Self>) {
        self.prepare_qa_page("models", cx);
        self.status = RuntimeStatus::Error("Download failed: checksum mismatch".into());
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
                    // The pill never becomes the active window, and GPUI caps
                    // inactive windows at 30 fps unless told otherwise.
                    inactive_frame_interval: None,
                    app_id: Some(env!("PYZHSCRIBER_BUNDLE_ID").into()),
                    ..Default::default()
                },
                {
                    let controller = self.controller.clone();
                    move |window, cx| {
                        strip_window_chrome(window);
                        cx.new(|_| RecordingOverlay::new(controller.clone(), status, bubbly))
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

    /// Shows a native alert once the current event has been handled, then
    /// applies its answer. NSAlert spins a nested run loop that keeps running
    /// GPUI's queued tasks; called from an event handler, where app state is
    /// still borrowed, those tasks panic and abort the app.
    fn after_modal<T: 'static>(
        cx: &mut Context<Self>,
        ask: impl FnOnce() -> Option<T> + 'static,
        apply: impl FnOnce(&mut Self, T, &mut Context<Self>) + 'static,
    ) {
        cx.spawn(async move |this, cx| {
            if let Some(answer) = ask() {
                this.update(cx, |this, cx| apply(this, answer, cx)).ok();
            }
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
        self.select_with(id, options, selected_value, width, false, cx, on_select)
    }

    /// A select whose panel opens with a search field narrowing the options
    /// by label or value.
    fn searchable_select<F>(
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
        self.select_with(id, options, selected_value, width, true, cx, on_select)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_with<F>(
        &self,
        id: impl Into<SharedString>,
        options: Vec<SelectOption>,
        selected_value: &str,
        width: f32,
        searchable: bool,
        cx: &Context<Self>,
        on_select: F,
    ) -> AnyElement
    where
        F: Fn(&mut Self, String, &mut Context<Self>) + 'static,
    {
        let id = id.into();
        let label = options
            .iter()
            .find(|option| option.value == selected_value)
            .map_or_else(|| selected_value.to_string(), |option| option.label.clone());
        let on_select = std::rc::Rc::new(on_select);
        let query = if searchable {
            self.dropdown_search.as_str()
        } else {
            ""
        };
        self.dropdown(id.clone(), label, width, searchable, cx, || {
            options
                .into_iter()
                .filter(|option| matches_query(&[&option.label, &option.value], query))
                .enumerate()
                .map(|(index, option)| {
                    let on_select = on_select.clone();
                    let value = option.value.clone();
                    self.dropdown_option(
                        format!("{id}-option-{index}"),
                        option.label,
                        option.value == selected_value,
                        cx,
                        move |this, cx| {
                            on_select(this, value.clone(), cx);
                            this.close_dropdown(cx);
                        },
                    )
                })
                .collect()
        })
    }

    /// A trigger that toggles `open_dropdown`, with its option panel built only
    /// while open and deferred so it paints above later rows.
    fn dropdown(
        &self,
        id: SharedString,
        label: impl Into<SharedString>,
        width: f32,
        searchable: bool,
        cx: &Context<Self>,
        options: impl FnOnce() -> Vec<AnyElement>,
    ) -> AnyElement {
        let is_open = self.open_dropdown.as_ref() == Some(&id);
        let toggle_id = id.clone();
        div()
            .relative()
            .w(px(width))
            .min_w(px(180.0))
            .child(
                div()
                    .id(SharedString::from(format!("{id}-trigger")))
                    .h(px(30.0))
                    .w_full()
                    .px_2p5()
                    .rounded_md()
                    .border_1()
                    .border_color(if is_open {
                        theme::border_strong()
                    } else {
                        theme::border()
                    })
                    .bg(if is_open {
                        theme::hover()
                    } else {
                        theme::input()
                    })
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .text_sm()
                    .text_color(theme::text())
                    .hover(|style| {
                        style
                            .border_color(theme::border_strong())
                            .bg(theme::hover())
                    })
                    .cursor_pointer()
                    .child(div().flex_1().min_w_0().truncate().child(label.into()))
                    .child(
                        svg()
                            .path(if is_open {
                                "icons/chevron-up.svg"
                            } else {
                                "icons/chevron-down.svg"
                            })
                            .size(px(14.0))
                            .flex_shrink_0()
                            .text_color(theme::muted()),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        if this.open_dropdown.as_ref() == Some(&toggle_id) {
                            this.open_dropdown = None;
                        } else {
                            this.open_dropdown = Some(toggle_id.clone());
                            this.dropdown_search.clear();
                            if searchable {
                                this.dropdown_search_focus.focus(window, cx);
                            }
                            if toggle_id == "cloud-model" {
                                this.fetch_openrouter_models(cx);
                            }
                        }
                        cx.notify();
                    })),
            )
            .when(is_open, |container| {
                container.child(
                    deferred(
                        div()
                            .id(SharedString::from(format!("{id}-panel")))
                            .absolute()
                            .top(px(34.0))
                            .left_0()
                            .w_full()
                            .flex()
                            .flex_col()
                            .rounded_md()
                            .border_1()
                            .border_color(theme::border())
                            .bg(theme::raised())
                            .shadow_lg()
                            .occlude()
                            .on_mouse_move(|_, _, cx| cx.stop_propagation())
                            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                            .on_mouse_down_out(
                                cx.listener(|this, _, _, cx| this.close_dropdown(cx)),
                            )
                            .when(searchable, |panel| {
                                panel.child(div().p_1().child(self.search_field(
                                    SharedString::from(format!("{id}-search")),
                                    "Search or type a model id",
                                    &self.dropdown_search,
                                    &self.dropdown_search_focus,
                                    |this| &mut this.dropdown_search,
                                    cx,
                                )))
                            })
                            .child(
                                div()
                                    .id(SharedString::from(format!("{id}-options")))
                                    .max_h(px(250.0))
                                    .overflow_y_scroll()
                                    .py_1()
                                    .children(options()),
                            ),
                    )
                    .with_priority(20),
                )
            })
            .into_any_element()
    }

    fn dropdown_option<F>(
        &self,
        id: impl Into<SharedString>,
        label: impl Into<SharedString>,
        checked: bool,
        cx: &Context<Self>,
        on_click: F,
    ) -> AnyElement
    where
        F: Fn(&mut Self, &mut Context<Self>) + 'static,
    {
        div()
            .id(id.into())
            .min_h(px(30.0))
            .w_full()
            .px_2p5()
            .py_1p5()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .text_sm()
            .text_color(if checked {
                theme::text()
            } else {
                theme::muted()
            })
            .hover(|style| style.bg(theme::hover()).text_color(theme::text()))
            .cursor_pointer()
            .child(label.into())
            .when(checked, |row| {
                row.child(
                    svg()
                        .path("icons/check.svg")
                        .size(px(14.0))
                        .flex_shrink_0()
                        .text_color(theme::accent()),
                )
            })
            .on_click(cx.listener(move |this, _, _, cx| {
                cx.stop_propagation();
                on_click(this, cx);
            }))
            .into_any_element()
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
            .min_h(px(44.0))
            .px_3()
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
                    .child(div().text_sm().text_color(theme::text()).child(label))
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
            .h(px(30.0))
            .min_w(px(180.0))
            .px_2p5()
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
            .text_color(if muted { theme::muted() } else { theme::text() })
            .hover(|style| {
                style
                    .border_color(theme::border_strong())
                    .bg(theme::hover())
            })
            .cursor_pointer()
            .child(if muted {
                div().child(label)
            } else {
                div()
                    .px_1p5()
                    .rounded_sm()
                    .border_1()
                    .border_b_2()
                    .border_color(theme::border_strong())
                    .bg(theme::selection())
                    .text_xs()
                    .font_weight(FontWeight::MEDIUM)
                    .child(label)
            })
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

    fn sensitivity_slider(&self, cx: &Context<Self>) -> AnyElement {
        let sensitivity = self.settings.overlay_sensitivity.clamp(0.25, 6.0);
        let fraction = (sensitivity / 0.25).ln() / (6.0_f32 / 0.25).ln();
        let bounds = std::rc::Rc::new(std::cell::Cell::new(Bounds::<Pixels>::default()));
        let painted_bounds = bounds.clone();
        let click_bounds = bounds.clone();
        let view = cx.entity().downgrade();
        div()
            .flex()
            .items_center()
            .gap_3()
            .child(
                div()
                    .id("voice-sensitivity")
                    .relative()
                    .w(px(160.0))
                    .h(px(24.0))
                    .cursor_pointer()
                    .child(
                        canvas(
                            move |area, _, _| painted_bounds.set(area),
                            move |_, _, window, _| {
                                let moving_view = view.clone();
                                window.on_mouse_event(
                                    move |event: &gpui::MouseMoveEvent, phase, _, cx| {
                                        if !phase.capture() {
                                            return;
                                        }
                                        let _ = moving_view.update(cx, |this, cx| {
                                            if let Some(bounds) = this.sensitivity_drag {
                                                if event.pressed_button == Some(MouseButton::Left) {
                                                    this.set_sensitivity_at(
                                                        event.position.x,
                                                        bounds,
                                                        cx,
                                                    );
                                                } else {
                                                    this.sensitivity_drag = None;
                                                }
                                            }
                                        });
                                    },
                                );
                                let released_view = view.clone();
                                window.on_mouse_event(
                                    move |event: &gpui::MouseUpEvent, phase, _, cx| {
                                        if !phase.capture() || event.button != MouseButton::Left {
                                            return;
                                        }
                                        let _ = released_view.update(cx, |this, cx| {
                                            if let Some(bounds) = this.sensitivity_drag.take() {
                                                this.set_sensitivity_at(
                                                    event.position.x,
                                                    bounds,
                                                    cx,
                                                );
                                            }
                                        });
                                    },
                                );
                            },
                        )
                        .absolute()
                        .inset_0(),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(px(10.0))
                            .w_full()
                            .h(px(4.0))
                            .rounded_full()
                            .bg(theme::border()),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(px(10.0))
                            .w(px(160.0 * fraction))
                            .h(px(4.0))
                            .rounded_full()
                            .bg(theme::accent_gradient()),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(px(148.0 * fraction))
                            .top(px(6.0))
                            .size(px(12.0))
                            .rounded_full()
                            .bg(theme::text()),
                    )
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &gpui::MouseDownEvent, _, cx| {
                            this.sensitivity_drag = Some(click_bounds.get());
                            this.set_sensitivity_at(event.position.x, click_bounds.get(), cx);
                        }),
                    ),
            )
            .child(
                div()
                    .w(px(44.0))
                    .text_sm()
                    .text_color(theme::muted())
                    .child(format!("{sensitivity:.2}×")),
            )
            .into_any_element()
    }

    fn set_sensitivity_at(&mut self, x: Pixels, bounds: Bounds<Pixels>, cx: &mut Context<Self>) {
        if bounds.size.width <= px(0.0) {
            return;
        }
        let fraction = ((x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0);
        let sensitivity = (0.25 * (6.0_f32 / 0.25).powf(fraction) * 100.0).round() / 100.0;
        if self.settings.overlay_sensitivity != sensitivity {
            self.save(|settings| settings.overlay_sensitivity = sensitivity, cx);
        }
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
            .w(px(36.0))
            .h(px(20.0))
            .p(px(2.0))
            .rounded_full()
            .bg(if enabled {
                theme::accent_gradient()
            } else {
                theme::hover().into()
            })
            .border_1()
            .border_color(if enabled {
                gpui::transparent_black()
            } else {
                theme::border()
            })
            .cursor_pointer()
            .child(
                div()
                    .relative()
                    .size(px(14.0))
                    .rounded_full()
                    .bg(if enabled {
                        theme::background()
                    } else {
                        theme::text()
                    })
                    .with_spring(
                        SharedString::from(format!("{id}-thumb-spring")),
                        SpringAnimation::new(SpringConfig::new(520.0, 34.0, 1.0))
                            .to(px(if enabled { 16.0 } else { 0.0 })),
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
            .h(px(30.0))
            .px_3()
            .mx_2()
            .rounded_md()
            .flex()
            .items_center()
            .gap_2p5()
            .text_sm()
            .text_color(if selected {
                theme::text()
            } else {
                theme::muted()
            })
            .relative()
            .when(selected, |item| {
                item.bg(theme::selection()).child(
                    div()
                        .absolute()
                        .left_0()
                        .top(px(8.0))
                        .w(px(3.0))
                        .h(px(14.0))
                        .rounded_full()
                        .bg(theme::accent()),
                )
            })
            .hover(move |style| {
                style
                    .bg(if selected {
                        theme::selection()
                    } else {
                        theme::hover()
                    })
                    .text_color(theme::text())
            })
            .cursor_pointer()
            .child(svg().path(icon).size(px(15.0)).text_color(if selected {
                theme::accent()
            } else {
                theme::muted()
            }))
            .child(label)
            .on_click(cx.listener(move |this, _, _, cx| {
                this.page = page;
                // Picks up models added or removed outside the app.
                this.models = this.controller.models.choices();
                cx.notify();
            }))
            .into_any_element()
    }

    fn section(&self, title: &'static str, children: Vec<AnyElement>) -> AnyElement {
        self.section_with(title, None, children)
    }

    /// A titled group of rows, with an optional control at the right of the title.
    fn section_with(
        &self,
        title: &'static str,
        trailing: Option<AnyElement>,
        children: Vec<AnyElement>,
    ) -> AnyElement {
        div()
            .w_full()
            .pb_6()
            .flex()
            .flex_col()
            .child(self.section_title(title, trailing))
            .child(framed(children))
            .into_any_element()
    }

    fn section_title(&self, title: &'static str, trailing: Option<AnyElement>) -> AnyElement {
        div()
            .h(px(36.0))
            .px_3()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .child(
                div()
                    .text_base()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme::text())
                    .child(title),
            )
            .children(trailing)
            .into_any_element()
    }

    fn row(&self, label: impl Into<SharedString>, value: impl Into<SharedString>) -> AnyElement {
        div()
            .w_full()
            .min_h(px(44.0))
            .px_3()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .child(
                div()
                    .flex_shrink_0()
                    .text_sm()
                    .text_color(theme::text())
                    .child(label.into()),
            )
            .child(
                div()
                    .min_w_0()
                    .truncate()
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
        self.tinted_action_row(id, label, value, theme::accent(), cx, on_click)
    }

    /// An action row whose value is tinted, e.g. danger for destructive actions.
    fn tinted_action_row<F>(
        &self,
        id: impl Into<SharedString>,
        label: impl Into<SharedString>,
        value: impl Into<SharedString>,
        tint: Hsla,
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
            .min_h(px(44.0))
            .px_3()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .hover(|style| style.bg(theme::hover()))
            .cursor_pointer()
            .child(
                div()
                    .flex_shrink_0()
                    .text_sm()
                    .text_color(theme::text())
                    .child(label.into()),
            )
            .child(
                div()
                    .min_w_0()
                    .truncate()
                    .text_sm()
                    .text_color(tint)
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

    /// Installed models for a language, plus `current` when it is not
    /// installed, so the trigger shows its name instead of a raw catalog id.
    fn model_options(&self, language: Option<&str>, current: &str) -> Vec<SelectOption> {
        let mut options = self
            .models
            .iter()
            .filter(|model| model.installed)
            .filter(|model| language.is_none_or(|language| supports_language(model, language)))
            .map(|model| SelectOption::new(model.id.clone(), model.name.clone()))
            .collect::<Vec<_>>();
        if !options.iter().any(|option| option.value == current) {
            options.insert(0, SelectOption::new(current, self.model_name(current)));
        }
        options
    }

    /// Starts a download unless one is already running for this model.
    fn download_model(&mut self, model_id: String, cx: &mut Context<Self>) {
        if self.downloads.contains_key(&model_id) {
            return;
        }
        self.downloads.insert(model_id.clone(), 0.0);
        self.controller.download_model(model_id);
        cx.notify();
    }

    /// "Installed" with a delete action for models in our own folder; models
    /// found in a shared Hugging Face cache are left for their owner to manage.
    fn installed_status(&self, model: &ModelChoice, cx: &Context<Self>) -> AnyElement {
        let id = model.id.clone();
        let name = model.name.clone();
        div()
            .h(px(28.0))
            .flex()
            .items_center()
            .gap_2()
            .text_xs()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1p5()
                    .text_color(theme::muted())
                    .child(
                        svg()
                            .path("icons/check.svg")
                            .size(px(13.0))
                            .text_color(theme::accent()),
                    )
                    .child("Installed"),
            )
            .when(model.removable, |status| {
                status.child(
                    div()
                        .id(SharedString::from(format!("delete-{id}")))
                        .px_2()
                        .h(px(24.0))
                        .flex()
                        .items_center()
                        .rounded_md()
                        .text_color(theme::muted())
                        .cursor_pointer()
                        .hover(|style| style.bg(theme::hover()).text_color(theme::danger()))
                        .child("Delete")
                        .on_click(cx.listener(move |_, _, _, cx| {
                            let name = name.clone();
                            let id = id.clone();
                            Self::after_modal(
                                cx,
                                move || {
                                    dialogs::confirm(
                                        &format!("Delete {name}?"),
                                        "You can download it again later.",
                                        "Delete",
                                    )
                                    .then_some(())
                                },
                                move |this, (), cx| {
                                    if let Err(error) = this.controller.remove_model(&id) {
                                        this.status = RuntimeStatus::Error(format!("{error:#}"));
                                    }
                                    this.models = this.controller.models.choices();
                                    cx.notify();
                                },
                            );
                        })),
                )
            })
            .into_any_element()
    }

    /// Download button for a catalog model: resumes an interrupted download,
    /// and shows progress with a cancel while one runs. `None` once the model
    /// is installed or when the id is not in the catalog.
    fn download_button(
        &self,
        model_id: &str,
        with_size: bool,
        cx: &Context<Self>,
    ) -> Option<AnyElement> {
        let model = self.models.iter().find(|model| model.id == model_id)?;
        if model.installed {
            return None;
        }
        let progress = self.downloads.get(model_id).copied();
        let label = match progress {
            Some(progress) => format!("{:.0}%", progress * 100.0),
            None if model.partial_bytes > 0 => format!(
                "Resume {:.0}%",
                model.partial_bytes as f64 / model.size_bytes as f64 * 100.0
            ),
            None if with_size => format!("Download {}", display_size(model.size_bytes)),
            None => "Download".to_string(),
        };
        let id = model.id.clone();
        Some(
            div()
                .id(SharedString::from(format!("download-{model_id}")))
                .flex_shrink_0()
                .min_w(px(76.0))
                // Sizes differ in digit widths, so a fixed width keeps stacked buttons aligned.
                .when(with_size, |button| button.w(px(128.0)))
                .h(px(28.0))
                .px_2p5()
                .rounded_md()
                .border_1()
                .border_color(theme::border())
                .bg(theme::input())
                .flex()
                .items_center()
                .justify_center()
                .gap_2()
                .text_xs()
                .font_weight(FontWeight::MEDIUM)
                .text_color(theme::text())
                .cursor_pointer()
                .hover(|style| {
                    style
                        .border_color(theme::border_strong())
                        .bg(theme::hover())
                })
                .child(label)
                .when(progress.is_some(), |button| {
                    button.child(
                        svg()
                            .path("icons/x.svg")
                            .size(px(12.0))
                            .text_color(theme::muted()),
                    )
                })
                .on_click(cx.listener(move |this, _, _, cx| {
                    cx.stop_propagation();
                    if this.downloads.contains_key(&id) {
                        this.controller.cancel_download(&id);
                    } else {
                        this.download_model(id.clone(), cx);
                    }
                }))
                .into_any_element(),
        )
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

    /// The first unrouted language an installed model covers, with that model.
    fn next_route(&self) -> Option<(String, String)> {
        let used = self
            .settings
            .routes
            .iter()
            .map(|route| route.language.as_str())
            .collect::<BTreeSet<_>>();
        self.models
            .iter()
            .filter(|model| model.installed)
            .find_map(|model| {
                model
                    .languages
                    .iter()
                    .find(|language| !used.contains(language.as_str()))
                    .map(|language| (language.clone(), model.id.clone()))
            })
    }

    fn add_route(&mut self, cx: &mut Context<Self>) {
        let Some((language, model)) = self.next_route() else {
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
            .min_h(px(44.0))
            .px_3()
            .py_1p5()
            .flex()
            .items_center()
            .gap_3()
            .child(self.select(
                SharedString::from(format!("route-language-{index}")),
                self.route_language_options(index),
                &route.language,
                180.0,
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
                self.model_options(Some(&language), &route.model),
                &route.model,
                320.0,
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
            .child(div().flex_1())
            .children(self.download_button(&route.model, true, cx))
            .child(
                div()
                    .id(SharedString::from(format!("remove-route-{index}")))
                    .flex_shrink_0()
                    .size(px(28.0))
                    .rounded_md()
                    .flex()
                    .items_center()
                    .justify_center()
                    .when(can_remove, |button| {
                        button
                            .child(
                                svg()
                                    .path("icons/x.svg")
                                    .size(px(14.0))
                                    .text_color(theme::muted()),
                            )
                            .hover(|style| style.bg(theme::hover()))
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
        let mut supported = self.model_languages(&self.settings.selected_model);
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
            languages => languages
                .iter()
                .map(|language| language.to_uppercase())
                .collect::<Vec<_>>()
                .join(", "),
        };
        // Picked languages float to the top so a long list still shows them.
        supported.sort_by_key(|language| !selected.contains(language));
        self.dropdown("single-languages".into(), label, 300.0, false, cx, || {
            let mut options = vec![self.dropdown_option(
                "single-language-all",
                "All supported",
                selected.is_empty(),
                cx,
                |this, cx| this.save(|settings| settings.allowed_languages.clear(), cx),
            )];
            options.extend(supported.into_iter().map(|language| {
                let checked = selected.contains(&language);
                let label = language_label(&language);
                self.dropdown_option(
                    format!("single-language-{language}"),
                    label,
                    checked,
                    cx,
                    move |this, cx| {
                        let language = language.clone();
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
                )
            }));
            options
        })
    }

    fn mode_control(&self, cx: &Context<Self>) -> AnyElement {
        let routed = self.settings.transcription_mode == TranscriptionMode::Routed;
        div()
            .w(px(240.0))
            .h(px(30.0))
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
                    .text_sm()
                    .text_color(if routed {
                        theme::text()
                    } else {
                        theme::muted()
                    })
                    .bg(if routed {
                        theme::selection()
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
                    .text_sm()
                    .text_color(if routed {
                        theme::muted()
                    } else {
                        theme::text()
                    })
                    .bg(if routed {
                        theme::input()
                    } else {
                        theme::selection()
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
            .min_h(px(56.0))
            .px_3()
            .py_2p5()
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
                    .child(div().text_sm().text_color(theme::text()).child(title))
                    .child(div().text_xs().text_color(theme::muted()).child(detail)),
            )
            .child(
                div()
                    .id(SharedString::from(format!("permission-{kind:?}-action")))
                    .flex_shrink_0()
                    .h(px(28.0))
                    .px_3()
                    .rounded_md()
                    .flex()
                    .items_center()
                    .justify_center()
                    .gap_1p5()
                    .text_xs()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(if granted {
                        theme::muted()
                    } else {
                        theme::background()
                    })
                    .when(!granted, |button| button.bg(theme::text()))
                    .when(granted, |button| {
                        button.child(
                            svg()
                                .path("icons/check.svg")
                                .size(px(13.0))
                                .text_color(theme::accent()),
                        )
                    })
                    .child(action)
                    .when(!granted, |button| {
                        button
                            .cursor_pointer()
                            .hover(|style| style.opacity(0.85))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.request_permission(kind, cx);
                            }))
                    }),
            )
            .into_any_element()
    }

    fn permission_rows(&self, cx: &Context<Self>) -> Vec<AnyElement> {
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
        ]
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
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(FontWeight::SEMIBOLD)
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
                    .child(framed(self.permission_rows(cx)))
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
        div()
            .flex()
            .flex_col()
            .child(self.section("Transcription", {
                let mut rows = vec![
                    self.action_row(
                        "toggle-recording",
                        "Manual recording",
                        if recording { "Stop" } else { "Start" },
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
                    rows.extend(self.next_route().is_some().then(|| {
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
                            .into_any_element()
                    }));
                } else {
                    let selected_model = &self.settings.selected_model;
                    rows.push(
                        self.control_row(
                            "Model",
                            div()
                                .flex()
                                .items_center()
                                .gap_3()
                                .children(self.download_button(selected_model, true, cx))
                                .child(self.select(
                                    "single-model",
                                    self.model_options(None, selected_model),
                                    selected_model,
                                    340.0,
                                    cx,
                                    |this, value, cx| {
                                        let supported = this.model_languages(&value);
                                        this.save(
                                            |settings| {
                                                settings.selected_model = value;
                                                settings.allowed_languages.retain(|language| {
                                                    supported.contains(language)
                                                });
                                            },
                                            cx,
                                        );
                                    },
                                ))
                                .into_any_element(),
                        ),
                    );
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
                        "Voice sensitivity",
                        Some("How strongly the pill reacts to your voice".into()),
                        self.sensitivity_slider(cx),
                    ));
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
            .child(self.section("Cloud", self.cloud_rows(cx)))
            .child(self.section(
                "Permissions",
                if self.permission_statuses.all_granted() {
                    vec![self.row("All granted", "Microphone, Accessibility, Input Monitoring")]
                } else {
                    self.permission_rows(cx)
                },
            ))
            .into_any_element()
    }

    fn cloud_rows(&self, cx: &Context<Self>) -> Vec<AnyElement> {
        let enabled = self.settings.cloud_enabled;
        let provider = self.settings.cloud_provider;
        let mut rows = vec![
            self.detailed_control_row(
                "Cloud dictation",
                Some(
                    if enabled {
                        let delivery = if crate::cloud::streams(&self.settings) {
                            "streams to"
                        } else {
                            "is sent to"
                        };
                        format!(
                            "Audio {delivery} {}. Local models step in if it fails.",
                            provider.label()
                        )
                    } else {
                        "Use a paid provider instead of local models".to_string()
                    }
                    .into(),
                ),
                self.switch("cloud-enabled", enabled, cx, |settings, value| {
                    settings.cloud_enabled = value;
                }),
            ),
        ];
        if !enabled {
            return rows;
        }
        let account = self
            .settings
            .cloud_accounts
            .get(&provider)
            .cloned()
            .unwrap_or_default();

        rows.push(
            self.control_row(
                "Provider",
                self.select(
                    "cloud-provider",
                    CloudProvider::ALL
                        .iter()
                        .map(|provider| {
                            SelectOption::new(format!("{provider:?}"), provider.label())
                        })
                        .collect(),
                    &format!("{provider:?}"),
                    260.0,
                    cx,
                    |this, value, cx| {
                        if let Some(provider) = CloudProvider::ALL
                            .into_iter()
                            .find(|provider| format!("{provider:?}") == value)
                        {
                            this.save(|settings| settings.cloud_provider = provider, cx);
                        }
                    },
                ),
            ),
        );

        let model = crate::cloud::model(&self.settings);
        let mut model_options = match &self.openrouter_models {
            Some(models) if provider == CloudProvider::OpenRouter => models
                .iter()
                .map(|model| SelectOption::new(&model.id, &model.name))
                .collect::<Vec<_>>(),
            _ => provider
                .models()
                .iter()
                .map(|model| {
                    let label = if crate::cloud::is_streaming(provider, model) {
                        format!("{model} · streaming")
                    } else {
                        model.to_string()
                    };
                    SelectOption::new(*model, label)
                })
                .collect(),
        };
        if !model_options.iter().any(|option| option.value == model) {
            let label = if model.is_empty() {
                "Not set"
            } else {
                model.as_str()
            };
            model_options.insert(0, SelectOption::new(model.clone(), label));
        }
        let typed = self.dropdown_search.trim();
        if !typed.is_empty() && !model_options.iter().any(|option| option.value == typed) {
            model_options.push(SelectOption::new(
                typed,
                format!("Use \u{201c}{typed}\u{201d}"),
            ));
        }
        rows.push(self.control_row(
            "Model",
            self.searchable_select(
                "cloud-model",
                model_options,
                &model,
                260.0,
                cx,
                move |this, value, cx| {
                    this.save(
                        |settings| {
                            settings.cloud_accounts.entry(provider).or_default().model = value;
                        },
                        cx,
                    );
                },
            ),
        ));

        rows.push(self.action_row(
            "cloud-key",
            "API key",
            if account.api_key.is_empty() {
                "Not set"
            } else {
                "Saved"
            },
            cx,
            move |this, cx| {
                let current = this
                    .settings
                    .cloud_accounts
                    .get(&provider)
                    .map(|account| account.api_key.clone())
                    .unwrap_or_default();
                let detail = match provider.key_page() {
                    Some(page) => {
                        format!("Create one at {page}. Saved locally in Pyzhscriber settings.")
                    }
                    None => "Saved locally in Pyzhscriber settings.".to_string(),
                };
                Self::after_modal(
                    cx,
                    move || {
                        dialogs::prompt("API key", &detail, &current, true)
                            .map(|value| value.trim().to_string())
                            .filter(|value| !value.is_empty())
                    },
                    move |this, value, cx| {
                        this.save(
                            |settings| {
                                settings.cloud_accounts.entry(provider).or_default().api_key =
                                    value;
                            },
                            cx,
                        );
                    },
                );
            },
        ));

        if provider == CloudProvider::Custom {
            rows.push(self.action_row(
                "cloud-url",
                "Server URL",
                if account.base_url.is_empty() {
                    "Not set".to_string()
                } else {
                    account.base_url.clone()
                },
                cx,
                move |this, cx| {
                    let current = this
                        .settings
                        .cloud_accounts
                        .get(&provider)
                        .map(|account| account.base_url.clone())
                        .unwrap_or_default();
                    Self::after_modal(
                        cx,
                        move || {
                            dialogs::prompt(
                                "Server URL",
                                "OpenAI-compatible base URL, e.g. http://localhost:8000/v1",
                                &current,
                                false,
                            )
                            .map(|value| value.trim().to_string())
                        },
                        move |this, value, cx| {
                            this.save(
                                |settings| {
                                    settings
                                        .cloud_accounts
                                        .entry(provider)
                                        .or_default()
                                        .base_url = value;
                                },
                                cx,
                            );
                        },
                    );
                },
            ));
        }
        rows
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
            .h(px(30.0))
            .px_2p5()
            .rounded_md()
            .border_1()
            .border_color(if active {
                theme::border_strong()
            } else {
                theme::border()
            })
            .bg(if active {
                theme::selection()
            } else {
                theme::input()
            })
            .flex()
            .items_center()
            .gap_2()
            .text_xs()
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
                    theme::selection()
                } else {
                    theme::hover()
                })
            })
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, _, cx| on_click(this, cx)))
            .into_any_element()
    }

    /// A one-line text field editing the string `field` points at. Keys are
    /// handled by hand since GPUI ships no text input.
    fn search_field(
        &self,
        id: SharedString,
        placeholder: &'static str,
        value: &str,
        focus: &FocusHandle,
        field: fn(&mut Self) -> &mut String,
        cx: &Context<Self>,
    ) -> AnyElement {
        let empty = value.is_empty();
        let text = if empty {
            SharedString::from(placeholder)
        } else {
            SharedString::from(value.to_string())
        };
        let focus_on_click = focus.clone();

        div()
            .id(id.clone())
            .track_focus(focus)
            .h(px(30.0))
            .w_full()
            .px_2p5()
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
            .focus(|style| style.border_color(theme::border_strong()))
            .child(div().flex_1().min_w_0().truncate().child(text))
            .when(!empty, |input| {
                input.child(
                    div()
                        .id(SharedString::from(format!("{id}-clear")))
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
                        .on_click(cx.listener(move |this, _, _, cx| {
                            field(this).clear();
                            cx.stop_propagation();
                            cx.notify();
                        })),
                )
            })
            .on_click(cx.listener(move |_, _, window, cx| {
                focus_on_click.focus(window, cx);
            }))
            .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                if edit_query(field(this), event, cx) {
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .into_any_element()
    }

    /// Loads OpenRouter's model list once per session, while its menu is the
    /// one being opened.
    fn fetch_openrouter_models(&mut self, cx: &mut Context<Self>) {
        if self.settings.cloud_provider != CloudProvider::OpenRouter
            || self.openrouter_models.is_some()
            || self.openrouter_fetching
        {
            return;
        }
        self.openrouter_fetching = true;
        cx.spawn(async move |this, cx| {
            let models = smol::unblock(crate::cloud::openrouter_models).await;
            this.update(cx, |this, cx| {
                this.openrouter_fetching = false;
                match models {
                    Ok(models) => this.openrouter_models = Some(models),
                    Err(error) => log::warn!("OpenRouter model list: {error:#}"),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
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
        let installed = self.models.iter().filter(|model| model.installed).count();

        let card = |model: &ModelChoice| {
            let languages = if model.languages.len() == 1 {
                model.languages[0].to_uppercase()
            } else {
                format!("{} languages", model.languages.len())
            };
            let download_progress = self.downloads.get(&model.id).copied();
            div()
                .id(SharedString::from(format!("model-{}", model.id)))
                .w_full()
                .px_3()
                .py_2p5()
                .border_t_1()
                .border_color(theme::separator())
                .flex()
                .items_center()
                .gap_6()
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap_2p5()
                                .child(
                                    div()
                                        .flex_shrink_0()
                                        .text_sm()
                                        .font_weight(FontWeight::MEDIUM)
                                        .text_color(theme::text())
                                        .child(model.name.clone()),
                                )
                                .child(
                                    div()
                                        .min_w_0()
                                        .truncate()
                                        .text_xs()
                                        .text_color(theme::muted())
                                        .child(format!(
                                            "{} · {} · {} · {}",
                                            model.architecture,
                                            model.quant,
                                            languages,
                                            display_size(model.size_bytes)
                                        )),
                                ),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme::muted())
                                .child(model.description.clone()),
                        )
                        .when_some(download_progress, |column, progress| {
                            column.child(
                                div()
                                    .mt_1()
                                    .w_full()
                                    .h(px(2.0))
                                    .rounded_full()
                                    .bg(theme::selection())
                                    .child(
                                        div()
                                            .h_full()
                                            .w(relative(progress.clamp(0.0, 1.0)))
                                            .rounded_full()
                                            .bg(theme::accent_gradient()),
                                    ),
                            )
                        }),
                )
                .child(
                    div()
                        .flex_shrink_0()
                        .flex()
                        .flex_col()
                        .gap_1p5()
                        .child(model_metric("Accuracy", model.accuracy_score))
                        .child(model_metric("Speed", model.speed_score)),
                )
                .child(
                    div()
                        .flex_shrink_0()
                        .w(px(132.0))
                        .flex()
                        .justify_end()
                        .child(
                            self.download_button(&model.id, false, cx)
                                .unwrap_or_else(|| self.installed_status(model, cx)),
                        ),
                )
        };

        div()
            .flex()
            .flex_col()
            .child(
                div()
                    .h(px(36.0))
                    .px_3()
                    .mb_4()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().flex_1().min_w(px(120.0)).child(self.search_field(
                        "model-search".into(),
                        "Search models",
                        &self.model_search,
                        &self.model_search_focus,
                        |this| &mut this.model_search,
                        cx,
                    )))
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
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_xs()
                            .text_color(theme::muted())
                            .child(if shown == total {
                                format!("{installed} installed, {total} in catalog")
                            } else {
                                format!("{shown} of {total} shown")
                            }),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .when(shown == 0, |column| {
                        column.child(
                            div()
                                .px_3()
                                .py_8()
                                .text_sm()
                                .text_color(theme::muted())
                                .child("No models match."),
                        )
                    })
                    .when(!installed_models.is_empty(), |column| {
                        column.child(
                            div()
                                .pb_6()
                                .child(self.section_title("Your models", None))
                                .children(installed_models.iter().copied().map(card)),
                        )
                    })
                    .when(!available_models.is_empty(), |column| {
                        column.child(
                            div()
                                .pb_6()
                                .child(self.section_title("Available", None))
                                .children(available_models.iter().copied().map(card)),
                        )
                    }),
            )
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
                match self.settings.custom_words.len() {
                    0 => "None".to_string(),
                    1 => "1 term".to_string(),
                    count => format!("{count} terms"),
                },
                cx,
                |this, cx| {
                    let current = this.settings.custom_words.join(", ");
                    Self::after_modal(
                        cx,
                        move || {
                            dialogs::prompt(
                                "Custom terms",
                                "Comma or newline separated. Whisper uses these as an initial prompt.",
                                &current,
                                false,
                            )
                        },
                        |this, value, cx| {
                            let words = value
                                .split([',', '\n'])
                                .map(str::trim)
                                .filter(|word| !word.is_empty())
                                .map(str::to_string)
                                .collect();
                            this.save(|settings| settings.custom_words = words, cx);
                        },
                    );
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
                        let current = this.settings.post_process_base_url.clone();
                        Self::after_modal(
                            cx,
                            move || {
                                dialogs::prompt(
                                    "AI endpoint",
                                    "OpenAI-compatible base URL",
                                    &current,
                                    false,
                                )
                            },
                            |this, value, cx| {
                                this.save(|settings| settings.post_process_base_url = value, cx);
                            },
                        );
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
                        let current = this.settings.post_process_model.clone();
                        Self::after_modal(
                            cx,
                            move || {
                                dialogs::prompt(
                                    "AI model",
                                    "Model name exposed by the endpoint",
                                    &current,
                                    false,
                                )
                            },
                            |this, value, cx| {
                                this.save(|settings| settings.post_process_model = value, cx);
                            },
                        );
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
                        let current = this.settings.post_process_api_key.clone();
                        Self::after_modal(
                            cx,
                            move || {
                                dialogs::prompt(
                                    "API key",
                                    "Saved locally in Pyzhscriber settings.",
                                    &current,
                                    true,
                                )
                                .filter(|value| !value.is_empty())
                            },
                            |this, value, cx| {
                                this.save(|settings| settings.post_process_api_key = value, cx);
                            },
                        );
                    },
                ),
            ]);
        }

        div()
            .flex()
            .flex_col()
            .child(self.section("Text", text_rows))
            .child(self.section("AI cleanup", cleanup_rows))
            .into_any_element()
    }

    fn render_history(&self, cx: &Context<Self>) -> AnyElement {
        if self.history.is_empty() {
            return div()
                .flex()
                .flex_col()
                .child(
                    div()
                        .px_3()
                        .py_2()
                        .text_sm()
                        .text_color(theme::muted())
                        .child("No transcriptions yet."),
                )
                .into_any_element();
        }
        let clear = div()
            .id("clear-history")
            .h(px(24.0))
            .px_2()
            .flex()
            .items_center()
            .rounded_md()
            .cursor_pointer()
            .hover(|style| style.bg(theme::hover()).text_color(theme::danger()))
            .child("Clear")
            .on_click(cx.listener(|_, _, _, cx| {
                Self::after_modal(
                    cx,
                    || {
                        dialogs::confirm(
                            "Clear history?",
                            "All transcriptions and their recordings will be deleted.",
                            "Clear",
                        )
                        .then_some(())
                    },
                    |this, (), cx| {
                        match this.controller.history.clear() {
                            Ok(()) => this.history.clear(),
                            Err(error) => this.status = RuntimeStatus::Error(error.to_string()),
                        }
                        cx.notify();
                    },
                );
            }));
        let rows = self.history.iter().map(|entry| {
            let text = entry.text.clone();
            let id = entry.id;
            let copied = self.copied_entry == Some(id);
            div()
                .id(SharedString::from(format!("history-{}", entry.id)))
                .px_3()
                .py_2p5()
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
                            history_time(entry.created_at)
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
        });
        div()
            .flex()
            .flex_col()
            .child(
                self.section_with(
                    "Transcriptions",
                    Some(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .text_xs()
                            .text_color(theme::muted())
                            .child(format!("{} entries", self.history.len()))
                            .child(clear)
                            .into_any_element(),
                    ),
                    rows.collect(),
                ),
            )
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
            .child(self.section("Command line", {
                let mut rows = vec![self.action_row(
                    "install-cli",
                    "Use Pyzhscriber in your terminal",
                    "Install CLI",
                    cx,
                    |this, cx| {
                        this.cli_install_result = Some(
                            crate::cli_install::install().map_err(|error| format!("{error:#}")),
                        );
                        cx.notify();
                    },
                )];
                if let Some(result) = &self.cli_install_result {
                    let (message, color) = match result {
                        Ok(message) => (message.clone(), theme::muted()),
                        Err(error) => (error.clone(), theme::danger()),
                    };
                    rows.push(
                        div()
                            .px_4()
                            .py_3()
                            .text_xs()
                            .text_color(color)
                            .child(message)
                            .into_any_element(),
                    );
                }
                rows
            }))
            .child(self.section("Runtime", {
                let loaded = self.controller.loaded_model_id();
                let mut rows = vec![
                    self.row(
                        "Loaded model",
                        loaded
                            .as_deref()
                            .map_or_else(|| "None".into(), |id| self.model_name(id)),
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
                                    |settings| settings.model_unload_minutes = value.parse().ok(),
                                    cx,
                                );
                            },
                        ),
                    ),
                ];
                if loaded.is_some() {
                    rows.push(self.action_row(
                        "unload-model",
                        "Free model memory now",
                        "Unload",
                        cx,
                        |this, cx| {
                            this.controller.unload_model();
                            cx.notify();
                        },
                    ));
                }
                rows
            }))
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
                        |settings, value| {
                            settings.show_tray_icon = value;
                            if !value {
                                settings.hide_from_dock = false;
                            }
                        },
                    ),
                    self.toggle_row(
                        "hide-from-dock",
                        "Hide from Dock",
                        self.settings.hide_from_dock,
                        cx,
                        |settings, value| {
                            settings.hide_from_dock = value;
                            if value {
                                settings.show_tray_icon = true;
                            }
                        },
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
    fn new(controller: Arc<AppController>, status: RuntimeStatus, bubbly: bool) -> Self {
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
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        // Redraws on every display refresh, so the motion runs at the
        // screen's own rate instead of a fixed timer's.
        window.request_animation_frame();
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
/// abruptly. The caps are quadratic curves, which GPUI anti-aliases; the long
/// edges are straight lines.
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
    let right = left + width;
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

    // The outline must stay convex: any triangle that overlaps another shows
    // as a darker patch as soon as the pill is translucent.
    let mut path = Path::new(cap(-1.0, -1.0, 0.0));
    path.line_to(cap(1.0, -1.0, 0.0));
    sweep(&mut path, 1.0, -1.0, 0.0, 0.5);
    sweep(&mut path, 1.0, -1.0, 0.5, 1.0);
    sweep(&mut path, 1.0, 1.0, 1.0, 0.5);
    sweep(&mut path, 1.0, 1.0, 0.5, 0.0);
    path.line_to(cap(-1.0, 1.0, 0.0));
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
                    .w(px(184.0))
                    .flex_shrink_0()
                    .h_full()
                    .bg(theme::sidebar_gradient())
                    .pt(px(48.0))
                    .pb_4()
                    .border_r_1()
                    .border_color(theme::separator())
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .px_5()
                            .pb_3()
                            .flex()
                            .items_center()
                            .gap_2p5()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(brand_mark())
                            .child("Pyzhscriber"),
                    )
                    .child(self.sidebar_item(
                        Page::General,
                        "General",
                        "icons/sliders-horizontal.svg",
                        cx,
                    ))
                    .child(self.sidebar_item(Page::Models, "Models", "icons/package.svg", cx))
                    .child(self.sidebar_item(Page::Audio, "Audio", "icons/mic.svg", cx))
                    .child(self.sidebar_item(Page::Output, "Output", "icons/type.svg", cx))
                    .child(self.sidebar_item(Page::History, "History", "icons/history.svg", cx))
                    .child(self.sidebar_item(Page::Advanced, "Advanced", "icons/settings.svg", cx))
                    .child(div().flex_1())
                    .child(
                        div()
                            .px_5()
                            .flex()
                            .items_center()
                            .gap_2()
                            .text_xs()
                            .text_color(theme::muted())
                            .child({
                                let color = match self.status {
                                    RuntimeStatus::Idle => theme::success(),
                                    RuntimeStatus::Recording | RuntimeStatus::Error(_) => {
                                        theme::danger()
                                    }
                                    RuntimeStatus::Transcribing => theme::accent(),
                                };
                                div()
                                    .size(px(13.0))
                                    .rounded_full()
                                    .bg(color.opacity(0.2))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child(div().size(px(7.0)).rounded_full().bg(color))
                            })
                            .child(status_label),
                    ),
            )
            .child(
                div()
                    .id("window-content")
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .pt(px(40.0))
                    .px(px(24.0))
                    .pb_4()
                    .overflow_y_scroll()
                    .children(self.error_banner(cx))
                    .child(content),
            )
            .into_any_element()
    }
}

impl RootView {
    /// The latest runtime error, shown above every page until dismissed, so a
    /// failure (e.g. a model download) is visible wherever it happens.
    fn error_banner(&self, cx: &Context<Self>) -> Option<AnyElement> {
        let RuntimeStatus::Error(error) = &self.status else {
            return None;
        };
        Some(
            div()
                .mb_4()
                .px_3()
                .py_2()
                .rounded_md()
                .border_1()
                .border_color(theme::danger())
                .flex()
                .items_start()
                .gap_3()
                .text_sm()
                .child(
                    svg()
                        .path("icons/circle-alert.svg")
                        .mt(px(2.0))
                        .size(px(16.0))
                        .flex_shrink_0()
                        .text_color(theme::danger()),
                )
                .child(div().flex_1().min_w_0().child(error.clone()))
                .child(
                    div()
                        .id("dismiss-error")
                        .flex_shrink_0()
                        .size(px(20.0))
                        .rounded_md()
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .hover(|style| style.bg(theme::hover()))
                        .child(
                            svg()
                                .path("icons/x.svg")
                                .size(px(12.0))
                                .text_color(theme::muted()),
                        )
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.status = RuntimeStatus::Idle;
                            cx.notify();
                        })),
                )
                .into_any_element(),
        )
    }
}

/// Time of day for today's entries, with the date for anything older.
fn history_time(created_at: chrono::DateTime<chrono::Utc>) -> String {
    use chrono::Datelike;
    let local = created_at.with_timezone(&chrono::Local);
    let now = chrono::Local::now();
    let format = if local.date_naive() == now.date_naive() {
        "%H:%M"
    } else if local.year() == now.year() {
        "%b %-d, %H:%M"
    } else {
        "%b %-d %Y, %H:%M"
    };
    local.format(format).to_string()
}

fn paste_method_value(method: PasteMethod) -> &'static str {
    match method {
        PasteMethod::CommandV => "command_v",
        PasteMethod::ClipboardOnly => "clipboard_only",
        PasteMethod::DirectTyping => "direct_typing",
    }
}

/// The app icon in miniature: sound bars that end in a text cursor.
fn brand_mark() -> impl IntoElement {
    let heights = [5.0, 11.0, 16.0, 9.0, 13.0];
    div()
        .h(px(16.0))
        .flex()
        .items_center()
        .gap(px(2.0))
        .children(heights.into_iter().enumerate().map(|(index, height)| {
            div().w(px(2.0)).h(px(height)).rounded_full().bg(mix(
                theme::accent(),
                theme::success(),
                index as f32 / 4.0,
            ))
        }))
        .child(div().ml(px(1.0)).w(px(1.5)).h(px(16.0)).bg(theme::text()))
}

fn model_metric(label: &'static str, score: f32) -> AnyElement {
    let score = score.clamp(0.0, 1.0);
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(
            div()
                .w(px(56.0))
                .text_xs()
                .text_color(theme::muted())
                .child(label),
        )
        .child(
            div()
                .w(px(64.0))
                .h(px(3.0))
                .rounded_full()
                .bg(theme::selection())
                .child(
                    div()
                        .h_full()
                        .w(px(64.0 * score))
                        .rounded_full()
                        .bg(theme::accent_gradient()),
                ),
        )
        .into_any_element()
}

fn model_matches_search(model: &ModelChoice, query: &str) -> bool {
    matches_query(&[&model.name, &model.description], query)
}

/// Whether any of `fields` contains the query, ignoring case. An empty query
/// matches everything.
fn matches_query(fields: &[&str], query: &str) -> bool {
    let query = query.trim().to_lowercase();
    query.is_empty()
        || fields
            .iter()
            .any(|field| field.to_lowercase().contains(&query))
}

/// Applies a key press to a search query. Returns whether it was consumed.
fn edit_query(query: &mut String, event: &KeyDownEvent, cx: &App) -> bool {
    let keystroke = &event.keystroke;
    match keystroke.key.as_str() {
        "backspace" | "delete" => {
            query.pop();
            true
        }
        "escape" => {
            query.clear();
            true
        }
        "v" if keystroke.modifiers.secondary(cx) => {
            if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
                query.push_str(&text.replace(['\n', '\r', '\t'], " "));
            }
            true
        }
        _ if !keystroke.modifiers.control
            && !keystroke.modifiers.platform
            && !keystroke.modifiers.function =>
        {
            match &keystroke.key_char {
                Some(text) => {
                    query.push_str(&text.replace(['\n', '\r', '\t'], " "));
                    true
                }
                None => false,
            }
        }
        _ => false,
    }
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
            removable: false,
            partial_bytes: 0,
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
