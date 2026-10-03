//! Secrets handling (requirements R9, design §7).
//!
//! The ElevenLabs key is read from an environment variable, then from the OS credential store. It lives in an
//! [`ApiKey`], whose `Debug` and `Display` never show it. [`scrub`] masks anything that looks like a key in text
//! that may reach an error message, a log line or a tool result.

use std::fmt;

/// An API key. No `Serialize`; `Debug` and `Display` print `ApiKey(<redacted>)`.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps a key. Surrounding whitespace is the caller's job (`resolve` trims).
    pub fn new(key: impl Into<String>) -> Self {
        ApiKey(key.into())
    }

    /// The secret itself. Use only to build the request header.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

impl fmt::Display for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// Where a key was found (never the key itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeySource {
    Env,
    Keychain,
    None,
}

/// Failure to consult the credential store. A key that simply isn't there is `Ok(None)`, not an error.
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("could not run the credential lookup: {0}")]
    Spawn(String),
    #[error(
        "the credential lookup did not finish within {0} s (is the Keychain locked or waiting on a prompt?)"
    )]
    Timeout(u64),
    #[error("the credential store refused the lookup: {0}")]
    Store(String),
}

/// Resolves the key: 1) `env_var` if set and non-empty; 2) the credential store (`security` CLI on macOS, the
/// `keyring` crate elsewhere). `Ok(None)` when neither has it.
pub fn resolve(env_var: &str, keychain_service: &str) -> Result<Option<ApiKey>, SecretError> {
    Ok(resolve_with_source(env_var, keychain_service)?.map(|(key, _)| key))
}

/// Reports where a key is, without handing it out. Lookup failures count as "no key" (they are logged).
pub fn presence(env_var: &str, keychain_service: &str) -> KeySource {
    match resolve_with_source(env_var, keychain_service) {
        Ok(Some((_key, source))) => source,
        Ok(None) => KeySource::None,
        Err(e) => {
            tracing::warn!(error = %scrub(&e.to_string()), "credential lookup failed; reporting no key");
            KeySource::None
        }
    }
}

fn resolve_with_source(
    env_var: &str,
    keychain_service: &str,
) -> Result<Option<(ApiKey, KeySource)>, SecretError> {
    if !env_var.is_empty()
        && let Ok(value) = std::env::var(env_var)
    {
        let value = value.trim();
        if !value.is_empty() {
            return Ok(Some((ApiKey::new(value), KeySource::Env)));
        }
    }
    Ok(store_lookup(keychain_service)?.map(|key| (key, KeySource::Keychain)))
}

fn current_user() -> Option<String> {
    ["USER", "LOGNAME", "USERNAME"]
        .iter()
        .filter_map(|v| std::env::var(v).ok())
        .map(|u| u.trim().to_string())
        .find(|u| !u.is_empty())
}

#[cfg(target_os = "macos")]
fn store_lookup(service: &str) -> Result<Option<ApiKey>, SecretError> {
    // An item made by the `security` CLI only lists `security` in its ACL, so a native read from this binary would
    // raise a prompt a headless server can't answer. Ask `security` itself.
    security_cli(
        std::ffi::OsStr::new("/usr/bin/security"),
        service,
        current_user().as_deref(),
        std::time::Duration::from_secs(5),
    )
}

#[cfg(not(target_os = "macos"))]
fn store_lookup(service: &str) -> Result<Option<ApiKey>, SecretError> {
    let user = current_user().unwrap_or_else(|| "mcp-sfxgen".to_string());
    let entry = match keyring::Entry::new(service, &user) {
        Ok(entry) => entry,
        Err(keyring::Error::NoDefaultStore) => {
            tracing::debug!("no credential store on this platform; treating as no key");
            return Ok(None);
        }
        Err(e) => return Err(SecretError::Store(scrub(&e.to_string()))),
    };
    match entry.get_password() {
        Ok(password) => {
            let password = password.trim();
            Ok((!password.is_empty()).then(|| ApiKey::new(password)))
        }
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(SecretError::Store(scrub(&e.to_string()))),
    }
}

/// `security find-generic-password -s <service> [-a <user>] -w`, no shell, killed after `timeout`.
/// Exit status 44 (`errSecItemNotFound`) means the item doesn't exist. Compiled for tests on every Unix.
#[cfg(any(target_os = "macos", test))]
fn security_cli(
    program: &std::ffi::OsStr,
    service: &str,
    account: Option<&str>,
    timeout: std::time::Duration,
) -> Result<Option<ApiKey>, SecretError> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let mut cmd = Command::new(program);
    cmd.arg("find-generic-password").arg("-s").arg(service);
    if let Some(account) = account {
        cmd.arg("-a").arg(account);
    }
    cmd.arg("-w")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| SecretError::Spawn(scrub(&e.to_string())))?;

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(SecretError::Timeout(timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(e) => return Err(SecretError::Spawn(scrub(&e.to_string()))),
        }
    };

    match status.code() {
        Some(0) => {
            let mut out = String::new();
            if let Some(stdout) = child.stdout.take() {
                let _ = stdout.take(16 * 1024).read_to_string(&mut out);
            }
            let out = out.trim();
            Ok((!out.is_empty()).then(|| ApiKey::new(out)))
        }
        Some(44) => Ok(None),
        Some(code) => Err(SecretError::Store(format!(
            "`security` exited with status {code}"
        ))),
        None => Err(SecretError::Store(
            "`security` was killed by a signal".into(),
        )),
    }
}

const MASK: &str = "sk_<redacted>";

/// Masks anything that looks like an ElevenLabs key (`sk_` plus 16 or more letters and digits).
pub fn scrub(s: &str) -> String {
    scrub_with_key(s, None)
}

/// [`scrub`], and also masks the exact key if given (whatever shape it has).
pub fn scrub_with_key(s: &str, key: Option<&ApiKey>) -> String {
    let replaced;
    let mut s = s;
    if let Some(key) = key.map(ApiKey::expose).filter(|k| !k.is_empty()) {
        replaced = s.replace(key, MASK);
        s = &replaced;
    }

    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let (mut copied, mut i) = (0, 0);
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"sk_" {
            let run = bytes[i + 3..]
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric())
                .count();
            if run >= 16 {
                // `i` and the end of the run both sit next to ASCII bytes, so they are char boundaries.
                out.push_str(&s[copied..i]);
                out.push_str(MASK);
                i += 3 + run;
                copied = i;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&s[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const KEY: &str = "sk_0123456789abcdef0123456789abcdef";

    #[test]
    fn api_key_never_prints() {
        let key = ApiKey::new(KEY);
        assert_eq!(format!("{key:?}"), "ApiKey(<redacted>)");
        assert_eq!(format!("{key}"), "ApiKey(<redacted>)");
        assert_eq!(format!("{key:#?}"), "ApiKey(<redacted>)");
        assert_eq!(key.expose(), KEY);
        // Nested in another struct's Debug output.
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Holder(ApiKey);
        assert!(!format!("{:?}", Holder(key)).contains("0123456789"));
    }

    #[test]
    fn scrub_masks_key_shapes() {
        assert_eq!(
            scrub("bad key sk_0123456789abcdef!"),
            "bad key sk_<redacted>!"
        );
        assert_eq!(
            scrub(&format!("a {KEY} b {KEY}")),
            "a sk_<redacted> b sk_<redacted>"
        );
        // Too short to be a key.
        assert_eq!(scrub("sk_short and task_id"), "sk_short and task_id");
        // Exactly 15 characters: untouched; 16: masked.
        assert_eq!(scrub("sk_123456789012345"), "sk_123456789012345");
        assert_eq!(scrub("sk_1234567890123456"), "sk_<redacted>");
        // Idempotent and unicode-safe.
        let once = scrub("héllo sk_ABCDEFGHIJKLMNOPQRST wörld");
        assert_eq!(once, "héllo sk_<redacted> wörld");
        assert_eq!(scrub(&once), once);
        assert_eq!(scrub(""), "");
    }

    #[test]
    fn scrub_masks_the_exact_key() {
        let key = ApiKey::new("plain-secret-value");
        assert_eq!(
            scrub_with_key("the key plain-secret-value is bad", Some(&key)),
            "the key sk_<redacted> is bad"
        );
        // An empty key must not blow up or mask everything.
        let empty = ApiKey::new("");
        assert_eq!(scrub_with_key("abc", Some(&empty)), "abc");
    }

    #[cfg(unix)]
    fn script(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn security_cli_outcomes() {
        let dir = tempfile::tempdir().unwrap();
        let t = Duration::from_secs(5);

        let found = script(
            dir.path(),
            "found",
            "printf '%s\\n' \"  sk_from_keychain_123456  \"",
        );
        let key = security_cli(found.as_os_str(), "svc", Some("me"), t)
            .unwrap()
            .unwrap();
        assert_eq!(key.expose(), "sk_from_keychain_123456");

        let missing = script(dir.path(), "missing", "exit 44");
        assert!(
            security_cli(missing.as_os_str(), "svc", None, t)
                .unwrap()
                .is_none()
        );

        let locked = script(dir.path(), "locked", "exit 36");
        let err = security_cli(locked.as_os_str(), "svc", None, t).unwrap_err();
        assert!(matches!(err, SecretError::Store(ref m) if m.contains("36")));

        let empty = script(dir.path(), "empty", "echo");
        assert!(
            security_cli(empty.as_os_str(), "svc", None, t)
                .unwrap()
                .is_none()
        );

        // The arguments are exactly what the spec says, with no shell in between.
        let args = script(dir.path(), "args", "printf '%s|' \"$@\"");
        let got = security_cli(args.as_os_str(), "my service; rm -rf /", Some("jk"), t)
            .unwrap()
            .unwrap();
        assert_eq!(
            got.expose(),
            "find-generic-password|-s|my service; rm -rf /|-a|jk|-w|"
        );

        assert!(matches!(
            security_cli(
                std::ffi::OsStr::new("/nonexistent/security"),
                "svc",
                None,
                t
            ),
            Err(SecretError::Spawn(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn security_cli_times_out_and_kills() {
        let dir = tempfile::tempdir().unwrap();
        let slow = script(dir.path(), "slow", "sleep 30");
        let started = std::time::Instant::now();
        let err =
            security_cli(slow.as_os_str(), "svc", None, Duration::from_millis(200)).unwrap_err();
        assert!(matches!(err, SecretError::Timeout(_)));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn env_var_wins_and_presence_hides_the_key() {
        // A unique variable name per test keeps parallel tests from interfering.
        let var = "SFX_CLOUD_TEST_KEY_ENV";
        // SAFETY: only this test touches this variable.
        unsafe { std::env::set_var(var, "  sk_env_value_0123456789  ") };
        let key = resolve(var, "no-such-service-sfx-cloud-test")
            .unwrap()
            .unwrap();
        assert_eq!(key.expose(), "sk_env_value_0123456789");
        assert_eq!(
            presence(var, "no-such-service-sfx-cloud-test"),
            KeySource::Env
        );

        unsafe { std::env::set_var(var, "   ") };
        // Whitespace-only counts as unset, so the store is consulted (and the service doesn't exist).
        let source = presence(var, "no-such-service-sfx-cloud-test");
        assert_ne!(source, KeySource::Env);
        unsafe { std::env::remove_var(var) };
    }
}
