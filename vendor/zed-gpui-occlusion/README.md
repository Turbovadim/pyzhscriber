# GPUI macOS occlusion patch

This is a self-contained Cargo workspace containing patched `gpui_apple` and
`gpui_macos` crates from Zed commit
`b5cd310b3b54dfc659fcc06453e6358843dfe216`. It requires Rust 1.98.1.

After a macOS window remains fully occluded for two seconds, the patch replaces
its hidden `CAMetalLayer` with a plain `CALayer`. It then releases the drawable
pool, path targets, atlas textures, and cached video textures. Before the window
becomes visible, it installs a fresh Metal layer at the latest drawable size and
forces one complete render. Shared renderer buffers stay warm, so hiding one
window does not disturb other visible windows.

The implementation is app-independent but macOS-specific. It cannot be a normal
add-on crate because GPUI does not expose its native window state or Metal
renderer. The package names intentionally match the upstream platform crates so
applications can replace them through Cargo's root `[patch]` table.

## Use from a Git repository

Publish this directory as the root of a Git repository, then keep every Zed
dependency on the upstream revision above and add:

```toml
[patch."https://github.com/zed-industries/zed"]
gpui_apple = { git = "https://github.com/your-org/gpui-occlusion", rev = "PATCH_COMMIT" }
gpui_macos = { git = "https://github.com/your-org/gpui-occlusion", rev = "PATCH_COMMIT" }
```

The application must place this table in its workspace root. Cargo ignores
patch tables declared by dependency crates. Pin `gpui`, `gpui_platform`,
`gpui_tokio`, and any other direct Zed dependencies to the exact upstream
revision recorded above.

For a vendored copy, replace the two Git entries with paths to
`crates/gpui_apple` and `crates/gpui_macos`, as this application does.

## Shader header

Upstream `gpui_apple` runs cbindgen over sibling `crates/gpui` sources to
produce `scene.h`. This workspace has no `gpui` sources, so the header is
committed as `crates/gpui_apple/src/scene.h` and `build.rs` uses it directly.
When moving to a new Zed revision, regenerate it from a Zed checkout at that
revision with upstream's `build.rs`, and copy `OUT_DIR/scene.h` here.

## Verify the standalone workspace

```sh
cargo test --manifest-path Cargo.toml --locked
```

The upstream source is licensed under Apache-2.0. See `LICENSE-APACHE`.
