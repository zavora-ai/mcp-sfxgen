//! mcp-sfxgen: sound effects from text over MCP, with local (Stable Audio 3 on MLX) and cloud (ElevenLabs) backends.
//!
//! See the README for what it does, how to set it up and how it protects a shared machine.

pub mod audio;
pub mod backend;
pub mod config;
pub mod error;
pub mod finish;
pub mod gpu_lock;
pub mod heavy;
pub mod jobs;
pub mod ledger;
pub mod memory;
pub mod output;
pub mod registry;
pub mod request;
pub mod result;
pub mod secrets;
pub mod server;

/// The registry manifest, embedded so the binary works from any working directory.
pub const MANIFEST_TOML: &str = include_str!("../mcp-server.toml");

#[cfg(test)]
mod tests {
    use super::MANIFEST_TOML;

    /// The README's download commands are the ones `list_models` prints for an empty cache.
    #[test]
    fn readme_download_commands_match_the_registry() {
        let readme = include_str!("../README.md");
        let registry =
            crate::registry::ModelRegistry::from_toml(crate::config::MODELS_EXAMPLE).unwrap();
        let empty = tempfile::tempdir().unwrap();
        let mut commands = 0;
        for model in registry.models() {
            for missing in model.missing_files(empty.path(), empty.path()) {
                commands += 1;
                assert!(
                    readme.contains(&missing.command),
                    "README lacks: {}",
                    missing.command
                );
            }
        }
        assert!(
            commands > 0,
            "the example registry has no downloads to check"
        );
    }

    #[test]
    fn manifest_is_valid() {
        let manifest =
            adk_mcp_sdk::ServerManifest::from_toml(MANIFEST_TOML).expect("manifest parses");
        assert_eq!(
            manifest.version,
            env!("CARGO_PKG_VERSION"),
            "manifest version matches the crate"
        );
        assert!(
            manifest.validate().is_empty(),
            "manifest errors: {:?}",
            manifest.validate()
        );
        assert_eq!(manifest.server_id, "mcp_sfxgen");
        assert_eq!(manifest.credentials, ["vault://zavora/elevenlabs-api-key"]);
        let tools: Vec<&str> = manifest.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            tools,
            [
                "generate_sfx",
                "get_job",
                "cancel_job",
                "list_models",
                "account_status",
                "unload_models"
            ]
        );
    }
}
