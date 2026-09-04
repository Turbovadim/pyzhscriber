use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

const APP_NAME: &str = "Pyzhscriber";
const BUNDLE_ID: &str = "com.yolki.pyzhscriber";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rustc-env=PYZHSCRIBER_APP_NAME={APP_NAME}");
    println!("cargo:rustc-env=PYZHSCRIBER_BUNDLE_ID={BUNDLE_ID}");
    println!("cargo:rerun-if-changed=assets/app-icon.png");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=ApplicationServices");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        embed_macos_info_plist()?;
    }
    Ok(())
}

fn embed_macos_info_plist() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let plist_path = out_dir.join("Info.plist");
    let executable = env::var("CARGO_PKG_NAME")?;
    let version = env::var("CARGO_PKG_VERSION")?;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleDisplayName</key><string>{APP_NAME}</string>
    <key>CFBundleExecutable</key><string>{executable}</string>
    <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
    <key>CFBundleName</key><string>{APP_NAME}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>Pyzhscriber records speech only while the transcription shortcut is held.</string>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
"#
    );
    fs::write(&plist_path, plist)?;
    println!(
        "cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{}",
        plist_path.display()
    );
    Ok(())
}
