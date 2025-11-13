use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

const INDEX_FILENAME: &str = "index.json";
pub const PROJECT_CONFIG_DIR: &str = ".rlcontroller";
const PROJECT_METADATA_FILENAME: &str = "project.json";
const DEFAULT_PROJECT_NAME: &str = "project";
const RECENT_LIMIT: usize = 20;

#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub name: String,
    pub root_path: PathBuf,
    pub logs_path: PathBuf,
    pub last_used: SystemTime,
}

impl ProjectInfo {
    pub fn metadata_path(&self) -> PathBuf {
        self.root_path
            .join(PROJECT_CONFIG_DIR)
            .join(PROJECT_METADATA_FILENAME)
    }

    pub fn runs_dir(&self) -> PathBuf {
        self.root_path.join(PROJECT_CONFIG_DIR).join("runs")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectRecord {
    name: String,
    path: PathBuf,
    #[serde(default)]
    logs_path: Option<PathBuf>,
    last_used: u64,
}

impl From<&ProjectInfo> for ProjectRecord {
    fn from(info: &ProjectInfo) -> Self {
        Self {
            name: info.name.clone(),
            path: info.root_path.clone(),
            logs_path: Some(info.logs_path.clone()),
            last_used: system_time_to_unix(info.last_used),
        }
    }
}

impl From<ProjectRecord> for ProjectInfo {
    fn from(record: ProjectRecord) -> Self {
        let root_path = record.path;
        let logs_path = record.logs_path.unwrap_or_else(|| root_path.join("logs"));
        Self {
            name: record.name,
            root_path,
            logs_path,
            last_used: unix_to_system_time(record.last_used),
        }
    }
}

#[derive(Debug)]
pub struct ProjectManager {
    root: PathBuf,
    index_path: PathBuf,
}

impl ProjectManager {
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root).wrap_err_with(|| {
            format!("failed to create projects directory at {}", root.display())
        })?;
        let index_path = root.join(INDEX_FILENAME);
        Ok(Self { root, index_path })
    }

    pub fn default_project_dir_for(&self, name: &str) -> PathBuf {
        let slug = slugify(name);
        let unique = self.make_unique_slug(&self.root, &slug);
        self.root.join(&unique)
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectInfo>> {
        let mut records = self.load_index()?.unwrap_or_default();
        let mut infos = Vec::with_capacity(records.len());
        let mut index_changed = false;

        records.retain(|record| {
            let info: ProjectInfo = record.clone().into();
            if info.root_path.exists() {
                if !info.logs_path.exists() {
                    let _ = fs::create_dir_all(&info.logs_path);
                }
                infos.push(info);
                true
            } else {
                index_changed = true;
                false
            }
        });

        infos.sort_by(|a: &ProjectInfo, b: &ProjectInfo| {
            system_time_to_unix(b.last_used).cmp(&system_time_to_unix(a.last_used))
        });
        if infos.len() > RECENT_LIMIT {
            infos.truncate(RECENT_LIMIT);
        }

        if index_changed {
            self.save_index_from_projects(&infos)?;
        }

        Ok(infos)
    }

    pub fn register_project(&self, name: &str, root_path: PathBuf) -> Result<ProjectInfo> {
        let cleaned = name.trim();
        if cleaned.is_empty() {
            bail!("Project name cannot be empty");
        }

        let mut root = root_path;
        if !root.is_absolute() {
            root = std::env::current_dir()
                .wrap_err("failed to determine current directory")?
                .join(root);
        }

        fs::create_dir_all(&root)
            .wrap_err_with(|| format!("failed to create project directory {}", root.display()))?;
        let logs_path = root.join("logs");
        fs::create_dir_all(&logs_path)
            .wrap_err_with(|| format!("failed to create logs directory {}", logs_path.display()))?;
        let gdignore_path = logs_path.join(".gdignore");
        if !gdignore_path.exists() {
            fs::write(&gdignore_path, b"")
                .wrap_err_with(|| format!("failed to create {}", gdignore_path.display()))?;
        }

        let info = ProjectInfo {
            name: cleaned.to_string(),
            root_path: root.clone(),
            logs_path: logs_path.clone(),
            last_used: SystemTime::now(),
        };

        self.write_metadata(&info)?;

        let mut projects = self.list_projects()?;
        projects.push(info.clone());
        projects.sort_by(|a: &ProjectInfo, b: &ProjectInfo| {
            system_time_to_unix(b.last_used).cmp(&system_time_to_unix(a.last_used))
        });
        if projects.len() > RECENT_LIMIT {
            projects.truncate(RECENT_LIMIT);
        }
        self.save_index_from_projects(&projects)?;

        Ok(info)
    }

    pub fn mark_as_used(&self, project: &ProjectInfo) -> Result<()> {
        let mut projects = self.list_projects()?;
        let mut updated = false;

        for info in &mut projects {
            if info.logs_path == project.logs_path {
                info.last_used = SystemTime::now();
                updated = true;
                break;
            }
        }

        if !updated {
            projects.push(ProjectInfo {
                name: project.name.clone(),
                root_path: project.root_path.clone(),
                logs_path: project.logs_path.clone(),
                last_used: SystemTime::now(),
            });
        }

        projects.sort_by(|a: &ProjectInfo, b: &ProjectInfo| {
            system_time_to_unix(b.last_used).cmp(&system_time_to_unix(a.last_used))
        });
        if projects.len() > RECENT_LIMIT {
            projects.truncate(RECENT_LIMIT);
        }
        self.save_index_from_projects(&projects)
    }

    fn load_index(&self) -> Result<Option<Vec<ProjectRecord>>> {
        if !self.index_path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&self.index_path)
            .wrap_err_with(|| format!("failed to read {}", self.index_path.display()))?;
        if data.trim().is_empty() {
            return Ok(Some(Vec::new()));
        }
        let records: Vec<ProjectRecord> = serde_json::from_str(&data)
            .wrap_err_with(|| format!("failed to parse {}", self.index_path.display()))?;
        Ok(Some(records))
    }

    fn save_index_from_projects(&self, projects: &[ProjectInfo]) -> Result<()> {
        let records: Vec<ProjectRecord> = projects.iter().map(ProjectRecord::from).collect();
        self.save_index(&records)
    }

    fn save_index(&self, records: &[ProjectRecord]) -> Result<()> {
        let json = serde_json::to_string_pretty(records)
            .wrap_err_with(|| format!("failed to serialize {}", self.index_path.display()))?;
        let mut file = fs::File::create(&self.index_path).wrap_err_with(|| {
            format!("failed to open {} for writing", self.index_path.display())
        })?;
        file.write_all(json.as_bytes()).wrap_err_with(|| {
            format!(
                "failed to write project index to {}",
                self.index_path.display()
            )
        })?;
        Ok(())
    }

    fn write_metadata(&self, info: &ProjectInfo) -> Result<()> {
        let metadata = json!({
            "name": info.name,
            "logs_path": info.logs_path,
            "created": system_time_to_unix(info.last_used),
        });
        let path = info.metadata_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).wrap_err_with(|| {
                format!("failed to create metadata directory {}", parent.display())
            })?;
        }
        let json = serde_json::to_string_pretty(&metadata)
            .wrap_err_with(|| format!("failed to serialize metadata for {}", info.name))?;
        fs::write(&path, json)
            .wrap_err_with(|| format!("failed to write metadata file {}", path.display()))?;
        Ok(())
    }

    fn make_unique_slug(&self, parent: &Path, base: &str) -> String {
        let mut candidate = base.to_string();
        let mut counter = 1;

        while parent.join(&candidate).exists() {
            counter += 1;
            candidate = format!("{}-{}", base, counter);
        }

        candidate
    }
}

fn slugify(name: &str) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;

    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            previous_dash = false;
        } else if ch.is_whitespace() || matches!(ch, '-' | '_' | '.') {
            if !previous_dash && !slug.is_empty() {
                slug.push('-');
                previous_dash = true;
            }
        }
    }

    if slug.is_empty() {
        DEFAULT_PROJECT_NAME.to_string()
    } else {
        slug.trim_matches('-').to_string()
    }
}

fn system_time_to_unix(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_to_system_time(timestamp: u64) -> SystemTime {
    UNIX_EPOCH + std::time::Duration::from_secs(timestamp)
}
