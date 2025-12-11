use std::path::{Path, PathBuf};

use super::config::{ConfigField, ExportField};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBrowserTarget {
    Config(ConfigField),
    Export(ExportField),
    ProjectLocation,
    ProjectImportArchive,
    ProjectExportPath,
    SavedRun,
    SimulatorEnvPath,
    InterfaceAgentPath,
    ChartExport,
}

#[derive(Debug, Clone)]
pub enum FileBrowserKind {
    Directory {
        allow_create: bool,
        require_checkpoints: bool,
    },
    ExistingFile {
        extensions: Vec<String>,
    },
    OutputFile {
        extension: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBrowserState {
    Browsing,
    NamingFolder,
    NamingFile,
}

#[derive(Debug, Clone)]
pub enum FileBrowserEntry {
    Parent(PathBuf),
    Directory(PathBuf),
    File(PathBuf),
}

impl FileBrowserEntry {
    pub fn path(&self) -> &Path {
        match self {
            FileBrowserEntry::Parent(path)
            | FileBrowserEntry::Directory(path)
            | FileBrowserEntry::File(path) => path,
        }
    }

    pub fn is_parent(&self) -> bool {
        matches!(self, FileBrowserEntry::Parent(_))
    }

    pub fn display_name(&self) -> String {
        if self.is_parent() {
            return String::from("[..]");
        }

        self.path()
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.path().display().to_string())
    }
}
