use anyhow::{Context, Result, bail};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::symlink;
use std::path::Path;

const PATH_SETUP: &str = "\n# Pyzhscriber CLI\ncase \":$PATH:\" in\n  *\":$HOME/.local/bin:\"*) ;;\n  *) export PATH=\"$HOME/.local/bin:$PATH\" ;;\nesac\n";

/// Installs for the current user without requiring administrator access.
pub fn install() -> Result<String> {
    let home = dirs::home_dir().context("home directory is unavailable")?;
    let executable = std::env::current_exe()?.canonicalize()?;
    anyhow::ensure!(
        !executable
            .components()
            .any(|part| part.as_os_str() == "AppTranslocation"),
        "Move Pyzhscriber to Applications and reopen it before installing the CLI"
    );
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    install_at(&home, &executable, &shell)
}

fn install_at(home: &Path, executable: &Path, shell: &str) -> Result<String> {
    let directory = home.join(".local/bin");
    fs::create_dir_all(&directory).context("create CLI directory")?;
    let link = directory.join("pyzhscriber");
    match fs::symlink_metadata(&link) {
        Ok(metadata) => {
            anyhow::ensure!(
                metadata.file_type().is_symlink() && fs::read_link(&link)? == executable,
                "{} already exists and points elsewhere; move it aside before installing the CLI",
                link.display()
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            symlink(executable, &link).context("install CLI link")?;
        }
        Err(error) => return Err(error).context("inspect existing CLI command"),
    }

    let profiles = match Path::new(shell).file_name().and_then(|name| name.to_str()) {
        Some("zsh") => vec![".zprofile", ".zshrc"],
        Some("bash") => {
            // Creating .bash_profile would hide an existing .bash_login or .profile.
            let login = [".bash_profile", ".bash_login", ".profile"]
                .into_iter()
                .find(|name| home.join(name).exists())
                .unwrap_or(".bash_profile");
            vec![login, ".bashrc"]
        }
        _ => return Ok(
            "CLI installed. Add ~/.local/bin to your shell's PATH, then run pyzhscriber --help."
                .into(),
        ),
    };
    for profile in profiles {
        let path = home.join(profile);
        let existing = match fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => bail!(
                "CLI link installed, but could not read {}: {error}",
                path.display()
            ),
        };
        if !existing.contains(PATH_SETUP) {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut file| file.write_all(PATH_SETUP.as_bytes()))
                .with_context(|| {
                    format!(
                        "CLI link installed, but could not update {}",
                        path.display()
                    )
                })?;
        }
    }
    Ok("CLI installed. Open a new terminal and run pyzhscriber --help.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installation_preserves_profiles_and_is_repeatable() {
        let root = std::env::temp_dir().join(format!("pyzh-install-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let executable = root.join("Pyzhscriber");
        fs::write(&executable, "fixture").unwrap();
        for shell in ["zsh", "bash", "fish"] {
            let home = root.join(shell);
            fs::create_dir_all(&home).unwrap();
            let profile = match shell {
                "bash" => ".bashrc",
                _ => ".zshrc",
            };
            fs::write(home.join(profile), "# existing settings\n").unwrap();
            if shell == "bash" {
                fs::write(home.join(".profile"), "# login settings\n").unwrap();
            }
            install_at(&home, &executable, shell).unwrap();
            if shell == "bash" {
                assert!(!home.join(".bash_profile").exists());
                assert!(
                    fs::read_to_string(home.join(".profile"))
                        .unwrap()
                        .contains(PATH_SETUP)
                );
            }
            let first = fs::read_to_string(home.join(profile)).unwrap();
            assert!(first.starts_with("# existing settings\n"));
            assert_eq!(first.contains(PATH_SETUP), shell != "fish");
            install_at(&home, &executable, shell).unwrap();
            assert_eq!(fs::read_to_string(home.join(profile)).unwrap(), first);
            let link = home.join(".local/bin/pyzhscriber");
            assert_eq!(fs::read_link(&link).unwrap(), executable);
            fs::remove_file(&link).unwrap();
            fs::write(&link, "another command").unwrap();
            assert!(install_at(&home, &executable, shell).is_err());
            assert_eq!(fs::read_to_string(&link).unwrap(), "another command");
        }
        fs::remove_dir_all(root).unwrap();
    }
}
