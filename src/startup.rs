use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;

pub fn set_enabled(enabled: bool) -> Result<()> {
    let path = launch_agent_path()?;
    if !enabled {
        if path.exists() {
            fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        }
        return Ok(());
    }

    let executable = std::env::current_exe().context("resolve Pyzhscriber executable")?;
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{bundle_id}</string>
  <key>ProgramArguments</key>
  <array><string>{executable}</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"#,
        bundle_id = env!("PYZHSCRIBER_BUNDLE_ID"),
        executable = xml_escape(&executable.to_string_lossy()),
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, plist).with_context(|| format!("write {}", path.display()))
}

fn launch_agent_path() -> Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("home directory is unavailable")?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", env!("PYZHSCRIBER_BUNDLE_ID"))))
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
