use std::fs;
use std::path::Path;

use color_eyre::eyre::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::app::metrics::MetricSample;

/// Increment this when the stored JSON schema changes.
pub const RUN_FILE_VERSION: u32 = 2;

fn default_run_version() -> u32 {
    0
}

/// RLlib-specific provenance for locating checkpoints and resume points.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RllibRunInfo {
    /// Absolute or project-relative trial directory containing checkpoints.
    pub trial_dir: Option<String>,
    /// Absolute or project-relative checkpoint used to start this run (if any).
    pub resume_from: Option<String>,
    /// Checkpoint frequency used during the run.
    pub checkpoint_frequency: Option<u64>,
    /// Index offset applied to checkpoints when resuming.
    pub checkpoint_index_offset: Option<u64>,
}

/// Representation of a stored training run with embedded metrics and logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedRun {
    #[serde(default = "default_run_version")]
    pub version: u32,
    pub id: String,
    pub name: String,
    pub experiment_name: String,
    pub training_mode: String,
    pub timestamp: u64,
    pub duration_seconds: f64,
    pub metrics: Vec<MetricSample>,
    #[serde(default)]
    pub training_output: Vec<String>,
    #[serde(default)]
    pub rllib_info: Option<RllibRunInfo>,
}

impl SavedRun {
    pub fn new(
        id: String,
        name: String,
        experiment_name: String,
        training_mode: String,
        timestamp: u64,
        duration_seconds: f64,
        metrics: Vec<MetricSample>,
        training_output: Vec<String>,
        rllib_info: Option<RllibRunInfo>,
    ) -> Self {
        Self {
            version: RUN_FILE_VERSION,
            id,
            name,
            experiment_name,
            training_mode,
            timestamp,
            duration_seconds,
            metrics,
            training_output,
            rllib_info,
        }
    }
}

pub fn load_saved_run(path: &Path) -> Result<SavedRun> {
    let data = fs::read_to_string(path)
        .wrap_err_with(|| format!("failed to read run file {}", path.display()))?;
    deserialize_saved_run(&data)
        .wrap_err_with(|| format!("failed to parse run file {}", path.display()))
}

pub fn deserialize_saved_run(data: &str) -> Result<SavedRun> {
    let run: SavedRun =
        serde_json::from_str(data).wrap_err("failed to deserialize saved run contents")?;
    Ok(run)
}

pub fn save_saved_run(path: &Path, run: &SavedRun) -> Result<()> {
    let json = serialize_saved_run(run)?;
    fs::write(path, json)
        .wrap_err_with(|| format!("failed to write saved run to {}", path.to_string_lossy()))?;
    Ok(())
}

pub fn serialize_saved_run(run: &SavedRun) -> Result<String> {
    let json = serde_json::to_string_pretty(run).wrap_err("failed to serialize saved run")?;
    Ok(json)
}
