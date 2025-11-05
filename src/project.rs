use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

const INDEX_FILENAME: &str = "index.json";
const PROJECT_METADATA_FILENAME: &str = "project.json";
const DEFAULT_PROJECT_NAME: &str = "project";
const RECENT_LIMIT: usize = 20;

#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub name: String,
    pub path: PathBuf,
    pub last_used: SystemTime,
}

impl ProjectInfo {
    pub fn metadata_path(&self) -> PathBuf {
        self.path.join(PROJECT_METADATA_FILENAME)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectRecord {
    name: String,
    path: PathBuf,
    last_used: u64,
}

impl From<&ProjectInfo> for ProjectRecord {
    fn from(info: &ProjectInfo) -> Self {
        Self {
            name: info.name.clone(),
            path: info.path.clone(),
            last_used: system_time_to_unix(info.last_used),
        }
    }
}

impl From<ProjectRecord> for ProjectInfo {
    fn from(record: ProjectRecord) -> Self {
        Self {
            name: record.name,
            path: record.path,
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

    // pub fn root(&self) -> &Path {
    //     &self.root
    // }

    pub fn list_projects(&self) -> Result<Vec<ProjectInfo>> {
        let mut records = self.load_index()?.unwrap_or_default();
        let mut infos = Vec::with_capacity(records.len());
        let mut index_changed = false;

        records.retain(|record| {
            if record.path.exists() {
                infos.push(record.clone().into());
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

    pub fn create_project(&self, name: &str) -> Result<ProjectInfo> {
        let cleaned = name.trim();
        if cleaned.is_empty() {
            bail!("Project name cannot be empty");
        }

        let slug = slugify(cleaned);
        let unique_slug = self.make_unique_slug(&slug);
        let project_path = self.root.join(&unique_slug);

        fs::create_dir_all(&project_path).wrap_err_with(|| {
            format!(
                "failed to create project directory {}",
                project_path.display()
            )
        })?;

        let info = ProjectInfo {
            name: cleaned.to_string(),
            path: project_path.clone(),
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
            if info.path == project.path {
                info.last_used = SystemTime::now();
                updated = true;
                break;
            }
        }

        if !updated {
            projects.push(ProjectInfo {
                name: project.name.clone(),
                path: project.path.clone(),
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
            "created": system_time_to_unix(info.last_used),
        });
        let path = info.metadata_path();
        let json = serde_json::to_string_pretty(&metadata)
            .wrap_err_with(|| format!("failed to serialize metadata for {}", info.name))?;
        fs::write(&path, json)
            .wrap_err_with(|| format!("failed to write metadata file {}", path.display()))?;
        Ok(())
    }

    fn make_unique_slug(&self, base: &str) -> String {
        let mut candidate = base.to_string();
        let mut counter = 1;

        while self.root.join(&candidate).exists() {
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
