use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use color_eyre::eyre::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::app::metrics::MetricSample;

/// Increment this when the stored JSON schema changes.
pub const RUN_FILE_VERSION: u32 = 3;
const METRICS_INDEX_STRIDE: u64 = 256;
const METRICS_PAGE_SIZE: u64 = 512;
const METRICS_CACHE_PAGES: usize = 16;

fn default_run_version() -> u32 {
    0
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunMetricsSummary {
    pub total_samples: u64,
    pub min_training_iteration: Option<u64>,
    pub max_training_iteration: Option<u64>,
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
    pub metrics_path: Option<String>,
    #[serde(default)]
    pub metrics_index_path: Option<String>,
    #[serde(default)]
    pub metrics_summary: Option<RunMetricsSummary>,
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
            metrics_path: None,
            metrics_index_path: None,
            metrics_summary: None,
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
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create directory {}", parent.display()))?;
    }

    let tmp_path = path.with_file_name(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("run.json")
    ));

    let file = fs::File::create(&tmp_path)
        .wrap_err_with(|| format!("failed to create temp run file {}", tmp_path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, run)
        .wrap_err("failed to serialize saved run")?;
    writer
        .flush()
        .wrap_err_with(|| format!("failed to flush temp run file {}", tmp_path.display()))?;

    if path.exists() {
        let _ = fs::remove_file(path);
    }
    fs::rename(&tmp_path, path)
        .wrap_err_with(|| format!("failed to move run file into place {}", path.display()))?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricsIndexEntry {
    line: u64,
    offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetricsIndexFile {
    stride: u64,
    total_lines: u64,
    entries: Vec<MetricsIndexEntry>,
    summary: RunMetricsSummary,
}

fn resolve_metrics_path(run_path: &Path, rel: &str) -> std::path::PathBuf {
    let candidate = std::path::PathBuf::from(rel);
    if candidate.is_absolute() {
        candidate
    } else {
        run_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(candidate)
    }
}

pub fn build_metrics_index(jsonl_path: &Path, idx_path: &Path) -> Result<RunMetricsSummary> {
    let file = fs::File::open(jsonl_path)
        .wrap_err_with(|| format!("failed to open metrics file {}", jsonl_path.display()))?;
    let mut reader = BufReader::new(file);

    let mut offset: u64 = 0;
    let mut line_no: u64 = 0;
    let mut entries: Vec<MetricsIndexEntry> = Vec::new();
    let mut buf = String::new();
    let mut first_iter: Option<u64> = None;
    let mut last_iter: Option<u64> = None;

    loop {
        buf.clear();
        let this_offset = offset;
        let bytes = reader
            .read_line(&mut buf)
            .wrap_err_with(|| format!("failed reading {}", jsonl_path.display()))?;
        if bytes == 0 {
            break;
        }
        offset = offset.saturating_add(bytes as u64);

        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }

        if line_no % METRICS_INDEX_STRIDE == 0 {
            entries.push(MetricsIndexEntry {
                line: line_no,
                offset: this_offset,
            });
        }

        if let Ok(sample) = serde_json::from_str::<MetricSample>(trimmed) {
            if first_iter.is_none() {
                first_iter = sample.training_iteration();
            }
            last_iter = sample.training_iteration();
        }

        line_no = line_no.saturating_add(1);
    }

    let summary = RunMetricsSummary {
        total_samples: line_no,
        min_training_iteration: first_iter,
        max_training_iteration: last_iter,
    };

    let idx = MetricsIndexFile {
        stride: METRICS_INDEX_STRIDE,
        total_lines: line_no,
        entries,
        summary: summary.clone(),
    };
    let json = serde_json::to_string_pretty(&idx).wrap_err("failed to serialize metrics index")?;
    fs::write(idx_path, json).wrap_err_with(|| {
        format!(
            "failed to write metrics index {}",
            idx_path.to_string_lossy()
        )
    })?;
    Ok(summary)
}

#[derive(Debug, Default)]
struct MetricsPageCache {
    lru: std::collections::VecDeque<u64>,
    pages: std::collections::HashMap<u64, Vec<MetricSample>>,
    capacity_pages: usize,
}

impl MetricsPageCache {
    fn new(capacity_pages: usize) -> Self {
        Self {
            lru: std::collections::VecDeque::new(),
            pages: std::collections::HashMap::new(),
            capacity_pages,
        }
    }

    fn touch(&mut self, key: u64) {
        if let Some(pos) = self.lru.iter().position(|v| *v == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key);
    }

    fn get(&mut self, key: u64) -> Option<&Vec<MetricSample>> {
        if self.pages.contains_key(&key) {
            self.touch(key);
        }
        self.pages.get(&key)
    }

    fn insert(&mut self, key: u64, value: Vec<MetricSample>) {
        if self.pages.contains_key(&key) {
            self.pages.insert(key, value);
            self.touch(key);
            return;
        }
        self.pages.insert(key, value);
        self.touch(key);
        while self.pages.len() > self.capacity_pages {
            if let Some(evict) = self.lru.pop_front() {
                self.pages.remove(&evict);
            } else {
                break;
            }
        }
    }
}

#[derive(Debug)]
pub struct RunMetricsStream {
    jsonl_path: std::path::PathBuf,
    idx: MetricsIndexFile,
    cache: std::cell::RefCell<MetricsPageCache>,
}

impl RunMetricsStream {
    pub fn open(run_path: &Path, run: &SavedRun) -> Result<Option<Self>> {
        let Some(metrics_rel) = run.metrics_path.as_ref() else {
            return Ok(None);
        };

        let jsonl_path = resolve_metrics_path(run_path, metrics_rel);

        let idx = if let Some(idx_rel) = run.metrics_index_path.as_ref() {
            let idx_path = resolve_metrics_path(run_path, idx_rel);
            if idx_path.is_file() {
                let data = fs::read_to_string(&idx_path).wrap_err_with(|| {
                    format!("failed to read metrics index {}", idx_path.display())
                })?;
                serde_json::from_str::<MetricsIndexFile>(&data)
                    .wrap_err("failed to parse metrics index")?
            } else {
                // Create index if missing.
                let summary = build_metrics_index(&jsonl_path, &idx_path).unwrap_or_default();
                if let Ok(data) = fs::read_to_string(&idx_path) {
                    serde_json::from_str::<MetricsIndexFile>(&data).unwrap_or(MetricsIndexFile {
                        stride: METRICS_INDEX_STRIDE,
                        total_lines: summary.total_samples,
                        entries: Vec::new(),
                        summary,
                    })
                } else {
                    MetricsIndexFile {
                        stride: METRICS_INDEX_STRIDE,
                        total_lines: summary.total_samples,
                        entries: Vec::new(),
                        summary,
                    }
                }
            }
        } else {
            MetricsIndexFile {
                stride: METRICS_INDEX_STRIDE,
                total_lines: run
                    .metrics_summary
                    .as_ref()
                    .map(|s| s.total_samples)
                    .unwrap_or(0),
                entries: Vec::new(),
                summary: run.metrics_summary.clone().unwrap_or_default(),
            }
        };

        Ok(Some(Self {
            jsonl_path,
            idx,
            cache: std::cell::RefCell::new(MetricsPageCache::new(METRICS_CACHE_PAGES)),
        }))
    }

    pub fn len(&self) -> usize {
        self.idx.total_lines as usize
    }

    pub fn get(&self, index_from_oldest: usize) -> Result<Option<MetricSample>> {
        if index_from_oldest >= self.len() {
            return Ok(None);
        }
        let mut out = self.range(index_from_oldest, index_from_oldest + 1)?;
        Ok(out.pop())
    }

    pub fn range(&self, start_inclusive: usize, end_exclusive: usize) -> Result<Vec<MetricSample>> {
        let total = self.len();
        if total == 0 {
            return Ok(Vec::new());
        }
        let start = start_inclusive.min(total);
        let end = end_exclusive.min(total);
        if start >= end {
            return Ok(Vec::new());
        }

        let page_size = METRICS_PAGE_SIZE as usize;
        let start_page = (start as u64) / METRICS_PAGE_SIZE;
        let end_page = ((end - 1) as u64) / METRICS_PAGE_SIZE;

        let mut assembled: Vec<MetricSample> = Vec::with_capacity(end - start);
        for page in start_page..=end_page {
            let page_start = (page * METRICS_PAGE_SIZE) as usize;
            let page_end = ((page + 1) * METRICS_PAGE_SIZE) as usize;
            let want_start = start.max(page_start);
            let want_end = end.min(page_end);
            let local_start = want_start - page_start;
            let local_end = want_end - page_start;

            let samples = {
                let mut cache = self.cache.borrow_mut();
                if let Some(hit) = cache.get(page) {
                    hit.clone()
                } else {
                    let loaded = self.load_page(page, page_size)?;
                    cache.insert(page, loaded.clone());
                    loaded
                }
            };

            for sample in samples.into_iter().skip(local_start).take(local_end - local_start) {
                assembled.push(sample);
            }
        }

        Ok(assembled)
    }

    fn nearest_index_offset(&self, target_line: u64) -> u64 {
        if self.idx.entries.is_empty() {
            return 0;
        }
        let mut best = 0u64;
        for entry in &self.idx.entries {
            if entry.line <= target_line {
                best = entry.offset;
            } else {
                break;
            }
        }
        best
    }

    fn load_page(&self, page: u64, max_lines: usize) -> Result<Vec<MetricSample>> {
        let start_line = page.saturating_mul(METRICS_PAGE_SIZE);
        if start_line >= self.idx.total_lines {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&self.jsonl_path).wrap_err_with(|| {
            format!(
                "failed to open metrics file {}",
                self.jsonl_path.display()
            )
        })?;
        let mut reader = BufReader::new(file);
        let offset = self.nearest_index_offset(start_line);
        reader
            .seek(SeekFrom::Start(offset))
            .wrap_err("failed to seek metrics file")?;

        let mut current_line = start_line.saturating_sub(start_line % METRICS_INDEX_STRIDE);
        let mut buf = String::new();
        while current_line < start_line {
            buf.clear();
            let bytes = reader.read_line(&mut buf)?;
            if bytes == 0 {
                return Ok(Vec::new());
            }
            current_line += 1;
        }

        let mut samples = Vec::new();
        for _ in 0..max_lines {
            if current_line >= self.idx.total_lines {
                break;
            }
            buf.clear();
            let bytes = reader.read_line(&mut buf)?;
            if bytes == 0 {
                break;
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(sample) = serde_json::from_str::<MetricSample>(trimmed) {
                samples.push(sample);
            }
            current_line += 1;
            if samples.len() >= max_lines {
                break;
            }
        }
        Ok(samples)
    }
}

pub fn run_metrics_summary(run: &SavedRun) -> RunMetricsSummary {
    if let Some(summary) = run.metrics_summary.clone() {
        return summary;
    }
    if run.metrics.is_empty() {
        return RunMetricsSummary::default();
    }
    let mut min_iter: Option<u64> = None;
    let mut max_iter: Option<u64> = None;
    for (idx, sample) in run.metrics.iter().enumerate() {
        let iter = sample.training_iteration().unwrap_or(idx as u64);
        min_iter = Some(min_iter.map_or(iter, |m| m.min(iter)));
        max_iter = Some(max_iter.map_or(iter, |m| m.max(iter)));
    }
    RunMetricsSummary {
        total_samples: run.metrics.len() as u64,
        min_training_iteration: min_iter,
        max_training_iteration: max_iter,
    }
}

pub fn load_run_metrics(run_path: &Path, run: &SavedRun) -> Result<Vec<MetricSample>> {
    if !run.metrics.is_empty() {
        return Ok(run.metrics.clone());
    }
    let Some(metrics_rel) = run.metrics_path.as_ref() else {
        return Ok(Vec::new());
    };
    let jsonl_path = resolve_metrics_path(run_path, metrics_rel);
    let file = fs::File::open(&jsonl_path)
        .wrap_err_with(|| format!("failed to open metrics file {}", jsonl_path.display()))?;
    let reader = BufReader::new(file);
    let mut metrics = Vec::new();
    for line in reader.lines().flatten() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(sample) = serde_json::from_str::<MetricSample>(trimmed) {
            metrics.push(sample);
        }
    }
    Ok(metrics)
}
