use gpui::{Background, Hsla, linear_color_stop, linear_gradient, rgb};

pub fn background() -> Hsla {
    rgb(0x191a22).into()
}

/// Popovers, which float above the page.
pub fn raised() -> Hsla {
    rgb(0x21232e).into()
}

pub fn hover() -> Hsla {
    rgb(0x323546).into()
}

/// The current choice in a nav, segmented control or filter.
pub fn selection() -> Hsla {
    rgb(0x3a3d50).into()
}

pub fn border() -> Hsla {
    rgb(0x444860).into()
}

/// Rules between rows. Barely above the background, so lists read as
/// grouped without the lines competing with the controls.
pub fn separator() -> Hsla {
    rgb(0x262834).into()
}

/// Control outlines on hover, focus and while open.
pub fn border_strong() -> Hsla {
    rgb(0x666b8a).into()
}

pub fn text() -> Hsla {
    rgb(0xf4f2f5).into()
}

pub fn muted() -> Hsla {
    rgb(0xaaa5b3).into()
}

pub fn accent() -> Hsla {
    rgb(0x7cf5c4).into()
}

pub fn accent_dim() -> Hsla {
    rgb(0x244a44).into()
}

pub fn input() -> Hsla {
    rgb(0x292b38).into()
}

pub fn success() -> Hsla {
    rgb(0x4cc9f0).into()
}

pub fn danger() -> Hsla {
    rgb(0xff6969).into()
}

/// The app icon's mint-to-cyan sweep, for fills that show a level or an "on" state.
pub fn accent_gradient() -> Background {
    linear_gradient(
        90.0,
        linear_color_stop(accent(), 0.0),
        linear_color_stop(success(), 1.0),
    )
}

/// The icon tile's top-to-bottom shading, used behind the sidebar.
pub fn sidebar_gradient() -> Background {
    linear_gradient(
        180.0,
        linear_color_stop(rgb(0x262835), 0.0),
        linear_color_stop(background(), 1.0),
    )
}
