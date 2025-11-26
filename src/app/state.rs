use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufRead, BufReader};
use std::ops::Index;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use color_eyre::{
    eyre::{bail, WrapErr},
    Result,
};

use super::config::{
    default_rllib_config_file, ConfigField, ExportConfig, ExportField, ExportMode, ExportState,
    PolicyType, RllibAlgorithm, RllibStopMode, TrainingConfig, TrainingMode,
    EXPORT_CONFIG_FILENAME, POLICY_TYPE_LIST, RLLIB_ALGORITHM_LIST, TRAINING_CONFIG_FILENAME,
};
use super::file_browser::{FileBrowserEntry, FileBrowserKind, FileBrowserState, FileBrowserTarget};
use super::metrics::{ChartData, ChartMetricKind, ChartMetricOption, MetricSample};
use super::runs::{self, SavedRun};
use crate::domain::projects::PROJECT_CONFIG_DIR;
use crate::domain::{ProjectInfo, ProjectManager};
use ratatui::style::Color;
use serde_json::{self, Value};

const TRAINING_BUFFER_LIMIT: usize = 512;
const EXPORT_BUFFER_LIMIT: usize = 512;
const METRIC_PREFIX: &str = "@METRIC ";
const SIM_EVENT_PREFIX: &str = "@SIM_EVENT ";
const SIM_ACTION_PREFIX: &str = "@SIM_ACTION ";
const INTERFACE_EVENT_PREFIX: &str = "@INTERFACE_EVENT ";
const INTERFACE_ACTION_PREFIX: &str = "@INTERFACE_ACTION ";
const TRAINING_METRIC_HISTORY_LIMIT: usize = 2048;
const SIM_EVENT_BUFFER_LIMIT: usize = 512;
const SIM_ACTION_AUTO_COMPACT_THRESHOLD: usize = 20;
const SIM_ACTION_VALUE_MAX_LEN: usize = 48;
const SIM_INFO_VALUE_MAX_LEN: usize = 64;
const SIM_ACTION_HISTORY_LIMIT: usize = 512;
const EMBEDDED_PYTHON_BIN: Option<&str> = option_env!("CONTROLLER_PYTHON_BIN");
const EMBEDDED_SCRIPT_ROOT: Option<&str> = option_env!("CONTROLLER_SCRIPTS_ROOT");
const PROJECT_LOCATION_MAX_LEN: usize = 4096;
const MAX_RUN_OVERLAYS: usize = 4;
const OVERLAY_COLORS: [Color; 6] = [
    Color::LightMagenta,
    Color::LightGreen,
    Color::LightYellow,
    Color::LightBlue,
    Color::LightRed,
    Color::White,
];

const RLLIB_STOP_MODE_CHOICES: [(&str, &str, &str); 2] = [
    (
        "time_seconds",
        "Time (seconds)",
        "Ends training once the configured wall-clock seconds have elapsed.",
    ),
    (
        "timesteps",
        "Environment Timesteps",
        "Stops when the requested total number of environment timesteps are collected.",
    ),
];

const RLLIB_BATCH_MODE_CHOICES: [(&str, &str, &str); 2] = [
    (
        "truncate_episodes",
        "Truncate Episodes",
        "Splits rollouts into fragments regardless of episode boundaries (default).",
    ),
    (
        "complete_episodes",
        "Complete Episodes",
        "Collects full episodes per batch; useful for on-policy algorithms needing trajectories.",
    ),
];

const RLLIB_FRAMEWORK_CHOICES: [(&str, &str, &str); 2] = [
    (
        "torch",
        "PyTorch",
        "Recommended backend with first-class ONNX export support in this toolkit.",
    ),
    (
        "tf",
        "TensorFlow",
        "Alternative backend if your project already depends on TensorFlow.",
    ),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabId {
    Home,
    Train,
    Metrics,
    Simulator,
    Interface,
    Export,
    Settings,
}

#[derive(Debug, Clone)]
pub struct Tab {
    pub title: &'static str,
    pub id: TabId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct StatusMessage {
    pub text: String,
    pub kind: StatusKind,
}

#[derive(Debug, Clone)]
struct RunOverlay {
    id: String,
    label: String,
    color: Color,
    path: PathBuf,
    run: SavedRun,
}

impl RunOverlay {
    fn metrics(&self) -> &[MetricSample] {
        &self.run.metrics
    }

    fn sample_matching(&self, iteration: Option<u64>) -> Option<&MetricSample> {
        if let Some(iter) = iteration {
            if let Some(sample) = self
                .run
                .metrics
                .iter()
                .find(|sample| sample.training_iteration() == Some(iter))
            {
                return Some(sample);
            }
        }
        self.run.metrics.last()
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[derive(Debug, Clone)]
struct ArchivedRunView {
    run: SavedRun,
    path: PathBuf,
    label: String,
}

impl ArchivedRunView {
    fn new(run: SavedRun, path: PathBuf) -> Self {
        let label = format!("{} [{}]", run.experiment_name, run.training_mode);
        Self { run, path, label }
    }

    fn metrics(&self) -> &[MetricSample] {
        &self.run.metrics
    }

    fn logs(&self) -> &[String] {
        &self.run.training_output
    }

    fn id(&self) -> &str {
        &self.run.id
    }
}

#[derive(Debug, Clone)]
pub struct PolicyComparisonData {
    pub baseline_label: String,
    pub reward_mean: Option<(f64, f64)>,
    pub reward_min: Option<(f64, f64)>,
    pub reward_max: Option<(f64, f64)>,
    pub episode_len_mean: Option<(f64, f64)>,
    pub completed_episodes: Option<(u64, u64)>,
}

impl PolicyComparisonData {}

#[derive(Debug, Clone)]
pub struct ConfigChoice {
    pub label: String,
    pub value: String,
    pub description: String,
}

impl ConfigChoice {
    fn new(
        label: impl Into<String>,
        value: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            description: description.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct ChoiceMenuState {
    field: ConfigField,
    options: Vec<ConfigChoice>,
    selected: usize,
}

impl ChoiceMenuState {
    fn selected_choice(&self) -> Option<&ConfigChoice> {
        self.options.get(self.selected)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChoiceMenuView<'a> {
    pub field: ConfigField,
    pub options: &'a [ConfigChoice],
    pub selected: usize,
}
#[derive(Debug, Clone)]
pub struct ChartOverlaySeries {
    pub label: String,
    pub color: Color,
    pub points: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    CreatingProject,
    EditingConfig,
    EditingAdvancedConfig,
    SelectingConfigOption,
    AdvancedConfig,
    BrowsingFiles,
    Help,
    ConfirmQuit,
    EditingExport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectCreationStage {
    Name,
    Location,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsFocus {
    History,
    Summary,
    Policies,
    Chart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulatorFocus {
    Events,
    Actions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulatorMode {
    Single,
    Multi,
}

impl SimulatorMode {
    pub fn label(self) -> &'static str {
        match self {
            SimulatorMode::Single => "Single-Agent",
            SimulatorMode::Multi => "Multi-Agent",
        }
    }

    fn arg(self) -> &'static str {
        match self {
            SimulatorMode::Single => "single",
            SimulatorMode::Multi => "multi",
        }
    }

    fn toggle(&mut self) {
        *self = match self {
            SimulatorMode::Single => SimulatorMode::Multi,
            SimulatorMode::Multi => SimulatorMode::Single,
        };
    }
}

#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    pub mode: SimulatorMode,
    pub env_path: String,
    pub show_window: bool,
    pub step_delay: f64,
    pub restart_delay: f64,
    pub max_episodes: Option<u32>,
    pub max_steps: Option<u32>,
    pub auto_restart: bool,
    pub log_tracebacks: bool,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            mode: SimulatorMode::Single,
            env_path: String::new(),
            show_window: false,
            step_delay: 0.0,
            restart_delay: 2.0,
            max_episodes: None,
            max_steps: None,
            auto_restart: true,
            log_tracebacks: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulatorEventSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct SimulatorEventEntry {
    pub timestamp: Option<String>,
    pub kind: String,
    pub message: String,
    pub severity: SimulatorEventSeverity,
}

#[derive(Debug, Clone)]
pub struct SimulatorAgentRow {
    pub episode: Option<u64>,
    pub step: Option<u64>,
    pub agent_id: String,
    pub policy: Option<String>,
    pub action: String,
    pub reward: Option<f64>,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct SimulatorActionMeta {
    episode: Option<u64>,
    step: Option<u64>,
    mode: SimulatorMode,
    total_agents: usize,
}

impl SimulatorActionMeta {
    pub fn episode(self) -> Option<u64> {
        self.episode
    }

    pub fn step(self) -> Option<u64> {
        self.step
    }

    pub fn mode(self) -> SimulatorMode {
        self.mode
    }

    pub fn total_agents(self) -> usize {
        self.total_agents
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentType {
    StableBaselines3,
    Rllib,
}

impl AgentType {
    pub fn label(self) -> &'static str {
        match self {
            AgentType::StableBaselines3 => "SB3",
            AgentType::Rllib => "RLlib",
        }
    }

    fn arg(self) -> &'static str {
        match self {
            AgentType::StableBaselines3 => "sb3",
            AgentType::Rllib => "rllib",
        }
    }

    fn toggle(&mut self) {
        *self = match self {
            AgentType::StableBaselines3 => AgentType::Rllib,
            AgentType::Rllib => AgentType::StableBaselines3,
        };
    }
}

#[derive(Debug, Clone)]
pub struct InterfaceConfig {
    pub agent_type: AgentType,
    pub agent_path: String,
    pub mode: SimulatorMode,
    pub show_window: bool,
    pub step_delay: f64,
    pub restart_delay: f64,
    pub auto_restart: bool,
    pub log_tracebacks: bool,
    // RLlib-specific
    pub rllib_checkpoint_number: Option<u32>,
    pub rllib_policy_id: String,
    // SB3-specific
    pub sb3_algo: String,
}

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            agent_type: AgentType::StableBaselines3,
            agent_path: String::new(),
            mode: SimulatorMode::Single,
            show_window: false,
            step_delay: 0.0,
            restart_delay: 2.0,
            auto_restart: true,
            log_tracebacks: false,
            rllib_checkpoint_number: None,
            rllib_policy_id: String::new(),
            sb3_algo: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InterfaceAgentRow {
    pub episode: Option<u64>,
    pub step: Option<u64>,
    pub agent_id: String,
    pub policy: Option<String>,
    pub observation: Option<String>,
    pub action: String,
    pub reward: Option<f64>,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct InterfaceActionMeta {
    episode: Option<u64>,
    step: Option<u64>,
    mode: SimulatorMode,
    total_agents: usize,
}

impl InterfaceActionMeta {
    pub fn episode(self) -> Option<u64> {
        self.episode
    }

    pub fn step(self) -> Option<u64> {
        self.step
    }

    pub fn mode(self) -> SimulatorMode {
        self.mode
    }

    pub fn total_agents(self) -> usize {
        self.total_agents
    }
}

#[derive(Debug, Clone)]
pub struct InterfaceEventEntry {
    pub timestamp: Option<String>,
    pub kind: String,
    pub message: String,
    pub severity: SimulatorEventSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceFocus {
    Events,
    Actions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsField {
    AnimationsEnabled,
    AnimationSpeed,
    AutoScrollTrainingLog,
}

impl SettingsField {
    pub fn label(self) -> &'static str {
        match self {
            SettingsField::AnimationsEnabled => "Animations",
            SettingsField::AnimationSpeed => "Animation Speed",
            SettingsField::AutoScrollTrainingLog => "Training Log Auto-scroll",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            SettingsField::AnimationsEnabled => {
                "Toggle spinners and pulsing highlights throughout the UI."
            }
            SettingsField::AnimationSpeed => "Adjust how fast animated indicators update.",
            SettingsField::AutoScrollTrainingLog => {
                "Follow the latest training output automatically when new lines arrive."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationSpeed {
    Slow,
    Normal,
    Fast,
}

impl AnimationSpeed {
    fn label(self) -> &'static str {
        match self {
            AnimationSpeed::Slow => "Slow",
            AnimationSpeed::Normal => "Normal",
            AnimationSpeed::Fast => "Fast",
        }
    }

    fn interval_ms(self) -> u64 {
        match self {
            AnimationSpeed::Slow => 400,
            AnimationSpeed::Normal => 200,
            AnimationSpeed::Fast => 120,
        }
    }

    fn step(self, direction: i32) -> Self {
        use AnimationSpeed::*;
        let speeds = [Slow, Normal, Fast];
        let idx = speeds.iter().position(|s| s == &self).unwrap_or(1) as i32;
        let len = speeds.len() as i32;
        let mut next = idx + direction;
        if next < 0 {
            next = 0;
        } else if next >= len {
            next = len - 1;
        }
        speeds[next as usize]
    }
}

#[derive(Debug, Clone)]
pub struct ControllerSettings {
    animations_enabled: bool,
    animation_speed: AnimationSpeed,
    auto_scroll_training_log: bool,
}

impl Default for ControllerSettings {
    fn default() -> Self {
        Self {
            animations_enabled: true,
            animation_speed: AnimationSpeed::Normal,
            auto_scroll_training_log: true,
        }
    }
}

impl ControllerSettings {
    fn toggle_animations(&mut self) {
        self.animations_enabled = !self.animations_enabled;
    }

    fn toggle_auto_scroll(&mut self) {
        self.auto_scroll_training_log = !self.auto_scroll_training_log;
    }

    fn change_speed(&mut self, direction: i32) {
        self.animation_speed = self.animation_speed.step(direction);
    }

    pub fn animations_enabled(&self) -> bool {
        self.animations_enabled
    }

    pub fn animation_speed(&self) -> AnimationSpeed {
        self.animation_speed
    }

    pub fn auto_scroll_training_log(&self) -> bool {
        self.auto_scroll_training_log
    }
}

const SETTINGS_FIELDS: [SettingsField; 3] = [
    SettingsField::AnimationsEnabled,
    SettingsField::AnimationSpeed,
    SettingsField::AutoScrollTrainingLog,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFocus {
    Fields,
    Output,
}

#[derive(Debug)]
enum TrainingEvent {
    Line(String),
    Error(String),
    Finished(Option<i32>),
}

#[derive(Debug)]
enum ExportEvent {
    Line(String),
    Error(String),
    Finished(Option<i32>),
}

#[derive(Debug)]
enum SimulatorEvent {
    Line(String),
    Error(String),
    Finished(Option<i32>),
}

#[derive(Debug)]
enum InterfaceEvent {
    Line(String),
    Error(String),
    Finished(Option<i32>),
}

pub struct App {
    tabs: Vec<Tab>,
    active_tab_index: usize,
    should_quit: bool,

    project_manager: ProjectManager,
    projects: Vec<ProjectInfo>,
    active_project: Option<ProjectInfo>,
    selected_project: usize,

    input_mode: InputMode,
    project_name_buffer: String,
    project_location_buffer: String,
    project_creation_stage: ProjectCreationStage,
    config_edit_buffer: String,
    active_config_field: Option<ConfigField>,
    config_return_mode: Option<InputMode>,
    choice_menu: Option<ChoiceMenuState>,
    advanced_fields: Vec<ConfigField>,
    advanced_selection: usize,
    file_browser_path: PathBuf,
    file_browser_entries: Vec<FileBrowserEntry>,
    file_browser_selected: usize,
    file_browser_target: Option<FileBrowserTarget>,
    file_browser_kind: FileBrowserKind,
    file_browser_state: FileBrowserState,
    file_browser_default_name: Option<String>,
    file_browser_input: String,

    status: Option<StatusMessage>,

    training_config: TrainingConfig,
    training_config_valid: bool,
    advanced_validation_errors: HashMap<ConfigField, String>,
    training_output: Vec<String>,
    training_output_scroll: usize,
    training_receiver: Option<Receiver<TrainingEvent>>,
    training_cancel: Option<Sender<()>>,
    training_running: bool,
    training_metrics: Vec<MetricSample>,
    metrics_timeline: Vec<MetricSample>,
    metrics_resume_iteration: Option<u64>,
    metrics_resume_label: Option<String>,
    saved_run_overlays: Vec<RunOverlay>,
    selected_overlay_index: Option<usize>,
    archived_run_view: Option<ArchivedRunView>,
    overlay_color_cursor: usize,
    current_run_start: Option<SystemTime>,
    metrics_history_index: usize,
    metrics_chart_index: usize,
    metrics_focus: MetricsFocus,
    metrics_history_scroll: usize,
    metrics_summary_scroll: usize,
    metrics_policies_scroll: usize,
    metrics_policies_expanded: bool,
    metrics_policies_horizontal_scroll: usize,
    metric_timer_start: Option<Instant>,
    metric_last_sample_time: Option<Instant>,

    simulator_config: SimulatorConfig,
    simulator_running: bool,
    simulator_receiver: Option<Receiver<SimulatorEvent>>,
    simulator_cancel: Option<Sender<()>>,
    simulator_event_log: Vec<SimulatorEventEntry>,
    simulator_event_scroll: usize,
    simulator_focus: SimulatorFocus,
    simulator_actions: Vec<SimulatorAgentRow>,
    simulator_actions_scroll: usize,
    simulator_compact_view: bool,
    simulator_compact_user_override: bool,
    simulator_action_meta: Option<SimulatorActionMeta>,
    simulator_status_line: Option<String>,

    interface_config: InterfaceConfig,
    interface_running: bool,
    interface_receiver: Option<Receiver<InterfaceEvent>>,
    interface_cancel: Option<Sender<()>>,
    interface_event_log: Vec<InterfaceEventEntry>,
    interface_event_scroll: usize,
    interface_focus: InterfaceFocus,
    interface_actions: Vec<InterfaceAgentRow>,
    interface_actions_scroll: usize,
    interface_compact_view: bool,
    interface_compact_user_override: bool,
    interface_action_meta: Option<InterfaceActionMeta>,
    interface_status_line: Option<String>,

    export_mode: ExportMode,
    export_config: ExportConfig,
    export_focus: ExportFocus,
    export_fields: Vec<ExportField>,
    export_selection: usize,
    export_edit_buffer: String,
    active_export_field: Option<ExportField>,
    export_return_mode: Option<InputMode>,
    export_output: Vec<String>,
    export_output_scroll: usize,
    export_receiver: Option<Receiver<ExportEvent>>,
    export_cancel: Option<Sender<()>>,
    export_running: bool,

    // Python environment check results
    python_sb3_available: Option<bool>,
    python_ray_available: Option<bool>,
    python_check_user_triggered: bool,

    controller_settings: ControllerSettings,
    settings_selection: usize,
    ui_animation_anchor: Instant,
}

impl App {
    pub fn new() -> Result<Self> {
        let project_root = default_projects_root()?;
        let project_manager = ProjectManager::new(project_root)?;
        let projects = project_manager.list_projects()?;

        let tabs = vec![
            Tab {
                title: "Home",
                id: TabId::Home,
            },
            Tab {
                title: "Train",
                id: TabId::Train,
            },
            Tab {
                title: "Metrics",
                id: TabId::Metrics,
            },
            Tab {
                title: "Simulator",
                id: TabId::Simulator,
            },
            Tab {
                title: "Interface",
                id: TabId::Interface,
            },
            Tab {
                title: "Export",
                id: TabId::Export,
            },
            Tab {
                title: "Settings",
                id: TabId::Settings,
            },
        ];

        let mut app = Self {
            tabs,
            active_tab_index: 0,
            should_quit: false,
            project_manager,
            projects,
            active_project: None,
            selected_project: 0,
            input_mode: InputMode::Normal,
            project_name_buffer: String::new(),
            project_location_buffer: String::new(),
            project_creation_stage: ProjectCreationStage::Name,
            config_edit_buffer: String::new(),
            active_config_field: None,
            config_return_mode: None,
            choice_menu: None,
            advanced_fields: Vec::new(),
            advanced_selection: 0,
            file_browser_path: std::env::current_dir().unwrap_or_default(),
            file_browser_entries: Vec::new(),
            file_browser_selected: 0,
            file_browser_target: None,
            file_browser_kind: FileBrowserKind::Directory {
                allow_create: true,
                require_checkpoints: false,
            },
            file_browser_state: FileBrowserState::Browsing,
            file_browser_default_name: None,
            file_browser_input: String::new(),
            status: None,
            training_config: TrainingConfig::default(),
            training_config_valid: false,
            advanced_validation_errors: HashMap::new(),
            training_output: Vec::new(),
            training_output_scroll: 0,
            training_receiver: None,
            training_running: false,
            training_cancel: None,
            training_metrics: Vec::new(),
            metrics_timeline: Vec::new(),
            metrics_resume_iteration: None,
            metrics_resume_label: None,
            saved_run_overlays: Vec::new(),
            selected_overlay_index: None,
            archived_run_view: None,
            overlay_color_cursor: 0,
            current_run_start: None,
            metrics_history_index: 0,
            metrics_chart_index: 0,
            metrics_focus: MetricsFocus::History,
            metrics_history_scroll: 0,
            metrics_summary_scroll: 0,
            metrics_policies_scroll: 0,
            metrics_policies_expanded: false,
            metrics_policies_horizontal_scroll: 0,
            metric_timer_start: None,
            metric_last_sample_time: None,
            simulator_config: SimulatorConfig::default(),
            simulator_running: false,
            simulator_receiver: None,
            simulator_cancel: None,
            simulator_event_log: Vec::new(),
            simulator_event_scroll: 0,
            simulator_focus: SimulatorFocus::Events,
            simulator_actions: Vec::new(),
            simulator_actions_scroll: 0,
            simulator_compact_view: false,
            simulator_compact_user_override: false,
            simulator_action_meta: None,
            simulator_status_line: None,

            interface_config: InterfaceConfig::default(),
            interface_running: false,
            interface_receiver: None,
            interface_cancel: None,
            interface_event_log: Vec::new(),
            interface_event_scroll: 0,
            interface_focus: InterfaceFocus::Events,
            interface_actions: Vec::new(),
            interface_actions_scroll: 0,
            interface_compact_view: false,
            interface_compact_user_override: false,
            interface_action_meta: None,
            interface_status_line: None,

            export_mode: ExportMode::StableBaselines3,
            export_config: ExportConfig::default(),
            export_focus: ExportFocus::Fields,
            export_fields: Vec::new(),
            export_selection: 0,
            export_edit_buffer: String::new(),
            active_export_field: None,
            export_return_mode: None,
            export_output: Vec::new(),
            export_output_scroll: 0,
            export_receiver: None,
            export_cancel: None,
            export_running: false,

            python_sb3_available: None,
            python_ray_available: None,
            python_check_user_triggered: false,

            controller_settings: ControllerSettings::default(),
            settings_selection: 0,
            ui_animation_anchor: Instant::now(),
        };

        app.ensure_selection_valid();
        app.rebuild_export_fields();
        app.check_python_environment();

        Ok(app)
    }

    pub fn tabs(&self) -> &[Tab] {
        &self.tabs
    }

    pub fn active_index(&self) -> usize {
        self.active_tab_index
    }

    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active_tab_index]
    }

    pub fn activate(&mut self, tab: TabId) {
        if let Some(index) = self.tabs.iter().position(|t| t.id == tab) {
            self.active_tab_index = index;
        }
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    pub fn projects(&self) -> &[ProjectInfo] {
        &self.projects
    }

    pub fn selected_project_index(&self) -> Option<usize> {
        if self.projects.is_empty() {
            None
        } else {
            Some(self.selected_project)
        }
    }

    pub fn selected_project(&self) -> Option<&ProjectInfo> {
        self.selected_project_index()
            .and_then(|index| self.projects.get(index))
    }

    pub fn active_project(&self) -> Option<&ProjectInfo> {
        self.active_project.as_ref()
    }

    pub fn input_mode(&self) -> InputMode {
        self.input_mode
    }

    pub fn project_name_buffer(&self) -> &str {
        &self.project_name_buffer
    }

    pub fn status(&self) -> Option<&StatusMessage> {
        self.status.as_ref()
    }

    pub fn python_sb3_available(&self) -> Option<bool> {
        self.python_sb3_available
    }

    pub fn python_ray_available(&self) -> Option<bool> {
        self.python_ray_available
    }

    pub fn python_check_hint_visible(&self) -> bool {
        !self.python_check_user_triggered
    }

    pub fn refresh_python_environment(&mut self) {
        self.python_check_user_triggered = true;
        self.set_status("Checking Python environment...", StatusKind::Info);
        self.check_python_environment();
    }

    fn check_python_environment(&mut self) {
        let python_cmd = determine_python_command();
        let base_dir = controller_scripts_root()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));

        let check_script = match find_script(&base_dir, "check_py_env.py") {
            Ok(script) => script,
            Err(err) => {
                self.python_sb3_available = None;
                self.python_ray_available = None;
                self.set_status(
                    format!("Python check script not found: {err}"),
                    StatusKind::Warning,
                );
                return;
            }
        };

        let output = Command::new(&python_cmd).arg(&check_script).output();

        match output {
            Ok(result) => {
                match result.status.code() {
                    Some(0) => {
                        // Both available
                        self.python_sb3_available = Some(true);
                        self.python_ray_available = Some(true);
                    }
                    Some(2) => {
                        // Only SB3 available
                        self.python_sb3_available = Some(true);
                        self.python_ray_available = Some(false);
                    }
                    Some(3) => {
                        // Only Ray available
                        self.python_sb3_available = Some(false);
                        self.python_ray_available = Some(true);
                    }
                    Some(4) => {
                        // Neither available
                        self.python_sb3_available = Some(false);
                        self.python_ray_available = Some(false);
                    }
                    _ => {
                        // Unknown error
                        self.python_sb3_available = None;
                        self.python_ray_available = None;
                    }
                }
            }
            Err(_) => {
                // Failed to run check
                self.python_sb3_available = None;
                self.python_ray_available = None;
            }
        }
    }

    pub fn training_output(&self) -> &[String] {
        &self.training_output
    }

    pub fn simulator_config(&self) -> &SimulatorConfig {
        &self.simulator_config
    }

    pub fn simulator_focus(&self) -> SimulatorFocus {
        self.simulator_focus
    }

    pub fn simulator_events(&self) -> &[SimulatorEventEntry] {
        &self.simulator_event_log
    }

    pub fn simulator_event_scroll(&self) -> usize {
        self.simulator_event_scroll
    }

    pub fn simulator_actions(&self) -> &[SimulatorAgentRow] {
        &self.simulator_actions
    }

    pub fn simulator_actions_scroll(&self) -> usize {
        self.simulator_actions_scroll
    }

    pub fn simulator_action_meta(&self) -> Option<SimulatorActionMeta> {
        self.simulator_action_meta
    }

    pub fn simulator_status_line(&self) -> Option<&str> {
        self.simulator_status_line.as_deref()
    }

    pub fn simulator_compact_view(&self) -> bool {
        self.simulator_compact_view
    }

    pub fn is_simulator_running(&self) -> bool {
        self.simulator_running
    }

    pub fn interface_config(&self) -> &InterfaceConfig {
        &self.interface_config
    }

    pub fn interface_focus(&self) -> InterfaceFocus {
        self.interface_focus
    }

    pub fn interface_events(&self) -> &[InterfaceEventEntry] {
        &self.interface_event_log
    }

    pub fn interface_event_scroll(&self) -> usize {
        self.interface_event_scroll
    }

    pub fn interface_actions(&self) -> &[InterfaceAgentRow] {
        &self.interface_actions
    }

    pub fn interface_actions_scroll(&self) -> usize {
        self.interface_actions_scroll
    }

    pub fn interface_action_meta(&self) -> Option<InterfaceActionMeta> {
        self.interface_action_meta
    }

    pub fn interface_status_line(&self) -> Option<&str> {
        self.interface_status_line.as_deref()
    }

    pub fn interface_compact_view(&self) -> bool {
        self.interface_compact_view
    }

    pub fn is_interface_running(&self) -> bool {
        self.interface_running
    }

    pub fn metrics_log_lines(&self) -> Option<&[String]> {
        self.archived_run_view.as_ref().map(|view| view.logs())
    }

    pub fn is_viewing_saved_run(&self) -> bool {
        self.archived_run_view.is_some()
    }

    pub fn viewed_run_label(&self) -> Option<&str> {
        self.archived_run_view
            .as_ref()
            .map(|view| view.label.as_str())
    }

    pub fn metrics_source_hint(&self) -> Option<String> {
        self.archived_run_view.as_ref().map(|view| {
            let file = view
                .path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("unknown");
            format!(
                "Viewing saved run: {} (file: {}) — press 'v' or 'o' to return, 'O' to switch runs",
                view.label, file
            )
        })
    }

    pub fn has_saved_run_overlays(&self) -> bool {
        !self.saved_run_overlays.is_empty()
    }

    pub fn selected_overlay_label(&self) -> Option<&str> {
        self.selected_overlay().map(|overlay| overlay.label())
    }

    fn selected_overlay(&self) -> Option<&RunOverlay> {
        self.selected_overlay_index
            .and_then(|idx| self.saved_run_overlays.get(idx))
            .or_else(|| self.saved_run_overlays.first())
    }

    fn selected_overlay_sample(&self) -> Option<&MetricSample> {
        let overlay = self.selected_overlay()?;
        let target_iteration = self
            .selected_metric_sample()
            .and_then(|sample| sample.training_iteration());
        overlay.sample_matching(target_iteration)
    }

    fn live_sample_for_iteration(&self, iteration: Option<u64>) -> Option<&MetricSample> {
        if self.training_metrics.is_empty() {
            return None;
        }
        if let Some(iter) = iteration {
            if let Some(sample) = self
                .training_metrics
                .iter()
                .rev()
                .find(|sample| sample.training_iteration() == Some(iter))
            {
                return Some(sample);
            }
        }
        self.training_metrics.last()
    }

    fn normalize_selected_overlay_index(&mut self) {
        if let Some(idx) = self.selected_overlay_index {
            if self.saved_run_overlays.is_empty() {
                self.selected_overlay_index = None;
            } else if idx >= self.saved_run_overlays.len() {
                self.selected_overlay_index = Some(self.saved_run_overlays.len() - 1);
            }
        } else if !self.saved_run_overlays.is_empty() {
            self.selected_overlay_index = Some(self.saved_run_overlays.len() - 1);
        }
    }

    fn handle_overlay_evicted(&mut self, removed: &RunOverlay) {
        if let Some(idx) = self.selected_overlay_index {
            if idx == 0 {
                self.selected_overlay_index = None;
            } else {
                self.selected_overlay_index = Some(idx - 1);
            }
        }
        if self.archived_run_view.as_ref().map(|view| view.id()) == Some(removed.id.as_str()) {
            self.drop_archived_run_view();
        }
        self.normalize_selected_overlay_index();
    }

    pub fn toggle_selected_overlay_view(&mut self) {
        if self.archived_run_view.is_some() {
            self.clear_archived_run_view();
            return;
        }
        if self.saved_run_overlays.is_empty() {
            self.set_status("Load a saved run first", StatusKind::Warning);
            return;
        }
        self.normalize_selected_overlay_index();
        if let Some(idx) = self.selected_overlay_index {
            if let Some(overlay) = self.saved_run_overlays.get(idx) {
                self.set_archived_run_view(overlay.run.clone(), overlay.path.clone());
            }
        }
    }

    pub fn cycle_saved_run_overlay(&mut self, direction: i32) {
        if self.saved_run_overlays.is_empty() {
            self.set_status("No saved run overlays loaded yet", StatusKind::Info);
            return;
        }
        let len = self.saved_run_overlays.len() as i32;
        let current = self
            .selected_overlay_index
            .unwrap_or(0)
            .min(self.saved_run_overlays.len().saturating_sub(1)) as i32;
        let next_idx = (current + direction).rem_euclid(len) as usize;
        self.selected_overlay_index = Some(next_idx);
        if let Some(overlay) = self.saved_run_overlays.get(next_idx) {
            let label = overlay.label().to_string();
            let run = overlay.run.clone();
            let path = overlay.path.clone();
            self.set_status(format!("Selected run overlay: {label}"), StatusKind::Info);
            if self.archived_run_view.is_some() {
                self.set_archived_run_view(run, path);
            }
        }
    }

    pub fn training_output_scroll(&self) -> usize {
        self.training_output_scroll
    }

    pub fn scroll_training_output_up(&mut self, lines: usize) {
        if self.training_output.is_empty() {
            self.training_output_scroll = 0;
            return;
        }
        let max_offset = self.training_output.len().saturating_sub(1);
        self.training_output_scroll = self
            .training_output_scroll
            .saturating_add(lines)
            .min(max_offset);
    }

    pub fn scroll_training_output_down(&mut self, lines: usize) {
        if lines >= self.training_output_scroll {
            self.training_output_scroll = 0;
        } else {
            self.training_output_scroll -= lines;
        }
    }

    pub fn reset_training_output_scroll(&mut self) {
        self.training_output_scroll = 0;
    }

    pub fn clear_training_output(&mut self) {
        if self.training_output.is_empty() {
            self.set_status("Training output is already clear", StatusKind::Info);
            return;
        }
        self.training_output.clear();
        self.training_output_scroll = 0;
        self.set_status("Training output cleared", StatusKind::Info);
    }

    fn drop_archived_run_view(&mut self) -> bool {
        if self.archived_run_view.take().is_some() {
            self.metrics_history_index = 0;
            self.metrics_summary_scroll = 0;
            self.metrics_policies_scroll = 0;
            true
        } else {
            false
        }
    }

    pub fn clear_archived_run_view(&mut self) {
        if self.drop_archived_run_view() {
            self.set_status("Returned to live metrics", StatusKind::Info);
        }
    }

    fn should_activate_archived_run_view(&self) -> bool {
        !self.training_running && self.training_metrics.is_empty()
    }

    fn set_archived_run_view(&mut self, run: SavedRun, path: PathBuf) {
        self.drop_archived_run_view();
        let view = ArchivedRunView::new(run, path);
        if let Some(idx) = self
            .saved_run_overlays
            .iter()
            .position(|overlay| overlay.id == view.id())
        {
            self.selected_overlay_index = Some(idx);
        }
        let label = view.label.clone();
        self.archived_run_view = Some(view);
        self.set_status(
            format!("Viewing saved run metrics: {label}"),
            StatusKind::Info,
        );
    }

    pub fn training_metrics_history(&self) -> &[MetricSample] {
        if let Some(view) = &self.archived_run_view {
            view.metrics()
        } else {
            &self.metrics_timeline
        }
    }

    pub fn metrics_history_selected_index(&self) -> usize {
        let history = self.training_metrics_history();
        if history.is_empty() {
            0
        } else {
            self.metrics_history_index
                .min(history.len().saturating_sub(1))
        }
    }

    pub fn metrics_sample_at(&self, offset_from_latest: usize) -> Option<&MetricSample> {
        let history = self.training_metrics_history();
        if history.is_empty() {
            return None;
        }
        let len = history.len();
        if offset_from_latest >= len {
            None
        } else {
            history.get(len - 1 - offset_from_latest)
        }
    }

    pub fn selected_metric_sample(&self) -> Option<&MetricSample> {
        let index = self.metrics_history_selected_index();
        self.metrics_sample_at(index)
    }

    pub fn metrics_history_move_newer(&mut self) {
        if self.metrics_history_index > 0 {
            self.metrics_history_index -= 1;
        }
    }

    pub fn metrics_history_move_older(&mut self) {
        let len = self.training_metrics_history().len();
        if self.metrics_history_index + 1 < len {
            self.metrics_history_index += 1;
        } else if len > 0 {
            self.metrics_history_index = len - 1;
        }
    }

    pub fn metrics_history_page_newer(&mut self, count: usize) {
        if count >= self.metrics_history_index {
            self.metrics_history_index = 0;
        } else {
            self.metrics_history_index -= count;
        }
    }

    pub fn metrics_history_page_older(&mut self, count: usize) {
        let history = self.training_metrics_history();
        if history.is_empty() {
            return;
        }
        let max_index = history.len() - 1;
        let new_index = self.metrics_history_index.saturating_add(count);
        self.metrics_history_index = new_index.min(max_index);
    }

    pub fn metrics_history_to_latest(&mut self) {
        self.metrics_history_index = 0;
    }

    pub fn metrics_history_to_oldest(&mut self) {
        let history = self.training_metrics_history();
        if !history.is_empty() {
            self.metrics_history_index = history.len() - 1;
        }
    }

    pub fn available_chart_metrics(&self) -> Vec<ChartMetricOption> {
        let mut options = Vec::new();
        options.push(ChartMetricOption::new(
            "Episode reward mean",
            ChartMetricKind::EpisodeRewardMean,
        ));
        options.push(ChartMetricOption::new(
            "Episode length mean",
            ChartMetricKind::EpisodeLenMean,
        ));

        if let Some(latest) = self.latest_training_metric() {
            if latest.env_throughput().is_some() || latest.env_steps_this_iter().is_some() {
                options.push(ChartMetricOption::new(
                    "Env throughput (steps/s)",
                    ChartMetricKind::EnvThroughput,
                ));
            }

            for (name, _) in latest.custom_metrics().iter() {
                options.push(ChartMetricOption::new(
                    format!("Custom metric: {name}"),
                    ChartMetricKind::CustomMetric(name.clone()),
                ));
            }

            // Add overlay options if there are multiple policies
            let num_policies = latest.policies().len();
            if num_policies > 1 {
                options.push(ChartMetricOption::new(
                    format!("All Policies - Reward ({num_policies} policies)"),
                    ChartMetricKind::AllPoliciesRewardMean,
                ));
                options.push(ChartMetricOption::new(
                    format!("All Policies - Episode Length ({num_policies} policies)"),
                    ChartMetricKind::AllPoliciesEpisodeLenMean,
                ));

                // Collect all unique learner stat keys
                let mut learner_stat_keys = std::collections::HashSet::new();
                for (_, metrics) in latest.policies() {
                    for key in metrics.learner_stats().keys() {
                        learner_stat_keys.insert(key.clone());
                    }
                }

                // Add overlay options for each learner stat
                for key in learner_stat_keys {
                    options.push(ChartMetricOption::new(
                        format!("All Policies - {} ({num_policies} policies)", key),
                        ChartMetricKind::AllPoliciesLearnerStat(key),
                    ));
                }
            }

            for (policy_id, metrics) in latest.policies() {
                if metrics.reward_mean().is_some() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} reward mean"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyRewardMean,
                    ));
                }
                if metrics.episode_len_mean().is_some() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} episode length mean"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyEpisodeLenMean,
                    ));
                }
                for (key, _) in metrics.learner_stats().iter() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} {key}"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyLearnerStat(key.clone()),
                    ));
                }
                for (key, _) in metrics.custom_metrics().iter() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} custom {key}"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyCustomMetric(key.clone()),
                    ));
                }
            }
        }

        options
    }

    fn ensure_chart_metric_index(&mut self) {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            self.metrics_chart_index = 0;
        } else if self.metrics_chart_index >= options.len() {
            self.metrics_chart_index = options.len() - 1;
        }
    }

    pub fn current_chart_metric(&self) -> Option<ChartMetricOption> {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            None
        } else {
            let index = self
                .metrics_chart_index
                .min(options.len().saturating_sub(1));
            Some(options[index].clone())
        }
    }

    pub fn cycle_chart_metric_next(&mut self) {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            self.metrics_chart_index = 0;
            return;
        }
        self.metrics_chart_index = (self.metrics_chart_index + 1) % options.len();
    }

    pub fn cycle_chart_metric_previous(&mut self) {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            self.metrics_chart_index = 0;
            return;
        }
        if self.metrics_chart_index == 0 {
            self.metrics_chart_index = options.len() - 1;
        } else {
            self.metrics_chart_index -= 1;
        }
    }

    pub fn chart_data(&self, max_points: usize) -> Option<ChartData> {
        let metric = self.current_chart_metric()?;
        let samples = self.training_metrics_history();
        if samples.is_empty() {
            return None;
        }

        let max_points = max_points.max(1);
        let len = samples.len();
        let selected_idx = self.metrics_history_selected_index();
        let selected_pos = len.saturating_sub(1).saturating_sub(selected_idx);
        let start = if selected_pos + max_points >= len {
            len.saturating_sub(max_points)
        } else {
            selected_pos
        };
        let end = (start + max_points).min(len);
        let mut points = Vec::new();

        for (idx, sample) in samples.iter().enumerate().take(end).skip(start) {
            if let Some(value) = App::chart_value_for_sample(sample, &metric) {
                let x = sample
                    .training_iteration()
                    .map(|iter| iter as f64)
                    .unwrap_or_else(|| idx as f64);
                points.push((x, value));
            }
        }

        if points.is_empty() {
            return None;
        }

        Some(ChartData {
            label: metric.label().to_string(),
            points,
        })
    }

    pub fn selected_chart_value(&self) -> Option<f64> {
        let sample = self.selected_metric_sample()?;
        let metric = self.current_chart_metric()?;
        App::chart_value_for_sample(sample, &metric)
    }

    pub fn resume_marker_iteration(&self) -> Option<u64> {
        self.metrics_resume_iteration
    }

    pub fn resume_marker_value(&self, option: &ChartMetricOption) -> Option<f64> {
        let iteration = self.metrics_resume_iteration?;
        let sample = self
            .training_metrics_history()
            .iter()
            .find(|s| s.training_iteration() == Some(iteration))?;
        App::chart_value_for_sample(sample, option)
    }

    pub fn latest_chart_value(&self, option: &ChartMetricOption) -> Option<f64> {
        let sample = self.training_metrics_history().last()?;
        App::chart_value_for_sample(sample, option)
    }

    fn chart_value_for_sample(sample: &MetricSample, option: &ChartMetricOption) -> Option<f64> {
        match option.kind() {
            ChartMetricKind::EpisodeRewardMean => sample.episode_reward_mean(),
            ChartMetricKind::EpisodeLenMean => sample.episode_len_mean(),
            ChartMetricKind::EnvThroughput => sample.env_throughput().or_else(|| {
                match (sample.env_steps_this_iter(), sample.time_this_iter_s()) {
                    (Some(steps), Some(time)) if time > 0.0 => Some(steps as f64 / time.max(1e-9)),
                    _ => None,
                }
            }),
            ChartMetricKind::CustomMetric(name) => sample.custom_metrics().get(name).copied(),
            ChartMetricKind::PolicyRewardMean => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.reward_mean()),
            ChartMetricKind::PolicyEpisodeLenMean => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.episode_len_mean()),
            ChartMetricKind::PolicyLearnerStat(key) => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.learner_stats().get(key).copied()),
            ChartMetricKind::PolicyCustomMetric(key) => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.custom_metrics().get(key).copied()),
            // Multi-policy overlays return None for single-value lookups
            ChartMetricKind::AllPoliciesRewardMean
            | ChartMetricKind::AllPoliciesEpisodeLenMean
            | ChartMetricKind::AllPoliciesLearnerStat(_) => None,
        }
    }

    /// Get multi-series chart data for overlay charts
    pub fn chart_multi_series_data(
        &self,
        option: &ChartMetricOption,
    ) -> Vec<(String, Vec<(f64, f64)>)> {
        let samples = self.training_metrics_history();

        match option.kind() {
            ChartMetricKind::AllPoliciesRewardMean => {
                // Collect all policy IDs
                let mut policy_ids = std::collections::HashSet::new();
                for sample in samples {
                    for id in sample.policies().keys() {
                        policy_ids.insert(id.clone());
                    }
                }

                // Sort policy IDs
                let mut sorted_ids: Vec<_> = policy_ids.into_iter().collect();
                sorted_ids.sort_by(|a, b| {
                    crate::ui::alphanumeric_sort_key(a).cmp(&crate::ui::alphanumeric_sort_key(b))
                });

                // Build series for each policy
                sorted_ids
                    .into_iter()
                    .map(|policy_id| {
                        let data: Vec<(f64, f64)> = samples
                            .iter()
                            .filter_map(|sample| {
                                let x = sample.training_iteration()? as f64;
                                let y = sample
                                    .policies()
                                    .get(&policy_id)
                                    .and_then(|m| m.reward_mean())?;
                                Some((x, y))
                            })
                            .collect();
                        (policy_id, data)
                    })
                    .collect()
            }
            ChartMetricKind::AllPoliciesEpisodeLenMean => {
                let mut policy_ids = std::collections::HashSet::new();
                for sample in samples {
                    for id in sample.policies().keys() {
                        policy_ids.insert(id.clone());
                    }
                }

                let mut sorted_ids: Vec<_> = policy_ids.into_iter().collect();
                sorted_ids.sort_by(|a, b| {
                    crate::ui::alphanumeric_sort_key(a).cmp(&crate::ui::alphanumeric_sort_key(b))
                });

                sorted_ids
                    .into_iter()
                    .map(|policy_id| {
                        let data: Vec<(f64, f64)> = samples
                            .iter()
                            .filter_map(|sample| {
                                let x = sample.training_iteration()? as f64;
                                let y = sample
                                    .policies()
                                    .get(&policy_id)
                                    .and_then(|m| m.episode_len_mean())?;
                                Some((x, y))
                            })
                            .collect();
                        (policy_id, data)
                    })
                    .collect()
            }
            ChartMetricKind::AllPoliciesLearnerStat(stat_key) => {
                let mut policy_ids = std::collections::HashSet::new();
                for sample in samples {
                    for id in sample.policies().keys() {
                        policy_ids.insert(id.clone());
                    }
                }

                let mut sorted_ids: Vec<_> = policy_ids.into_iter().collect();
                sorted_ids.sort_by(|a, b| {
                    crate::ui::alphanumeric_sort_key(a).cmp(&crate::ui::alphanumeric_sort_key(b))
                });

                sorted_ids
                    .into_iter()
                    .map(|policy_id| {
                        let data: Vec<(f64, f64)> = samples
                            .iter()
                            .filter_map(|sample| {
                                let x = sample.training_iteration()? as f64;
                                let y = sample
                                    .policies()
                                    .get(&policy_id)
                                    .and_then(|m| m.learner_stats().get(stat_key).copied())?;
                                Some((x, y))
                            })
                            .collect();
                        (policy_id, data)
                    })
                    .collect()
            }
            // Non-overlay types return empty
            _ => Vec::new(),
        }
    }

    pub fn policy_comparison(&self, policy_id: &str) -> Option<PolicyComparisonData> {
        let overlay_sample = self.selected_overlay_sample()?;
        let overlay_metrics = overlay_sample.policies().get(policy_id)?;
        let live_sample = self.live_sample_for_iteration(overlay_sample.training_iteration())?;
        let live_metrics = live_sample.policies().get(policy_id)?;
        let baseline_label = self.selected_overlay_label()?.to_string();

        Some(PolicyComparisonData {
            baseline_label,
            reward_mean: Self::pair_f64(live_metrics.reward_mean(), overlay_metrics.reward_mean()),
            reward_min: Self::pair_f64(live_metrics.reward_min(), overlay_metrics.reward_min()),
            reward_max: Self::pair_f64(live_metrics.reward_max(), overlay_metrics.reward_max()),
            episode_len_mean: Self::pair_f64(
                live_metrics.episode_len_mean(),
                overlay_metrics.episode_len_mean(),
            ),
            completed_episodes: Self::pair_u64(
                live_metrics.completed_episodes(),
                overlay_metrics.completed_episodes(),
            ),
        })
    }

    fn pair_f64(live: Option<f64>, baseline: Option<f64>) -> Option<(f64, f64)> {
        match (live, baseline) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    fn pair_u64(live: Option<u64>, baseline: Option<u64>) -> Option<(u64, u64)> {
        match (live, baseline) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    // Metrics panel focus and scrolling
    pub fn metrics_focus(&self) -> MetricsFocus {
        self.metrics_focus
    }

    pub fn animations_enabled(&self) -> bool {
        self.controller_settings.animations_enabled()
    }

    pub fn animation_phase(&self) -> usize {
        if !self.controller_settings.animations_enabled() {
            return 0;
        }
        let interval = self
            .controller_settings
            .animation_speed()
            .interval_ms()
            .max(1);
        let elapsed = self.ui_animation_anchor.elapsed().as_millis();
        ((elapsed / interval as u128) % usize::MAX as u128) as usize
    }

    pub fn spinner_char(&self) -> Option<char> {
        if !self.controller_settings.animations_enabled() {
            return None;
        }
        const FRAMES: [char; 4] = ['|', '/', '-', '\\'];
        Some(FRAMES[self.animation_phase() % FRAMES.len()])
    }

    pub fn settings_fields(&self) -> &'static [SettingsField] {
        &SETTINGS_FIELDS
    }

    pub fn settings_selection_index(&self) -> usize {
        self.settings_selection
    }

    pub fn current_settings_field(&self) -> SettingsField {
        SETTINGS_FIELDS[self.settings_selection]
    }

    pub fn select_next_setting(&mut self) {
        self.settings_selection = (self.settings_selection + 1) % self.settings_fields().len();
    }

    pub fn select_previous_setting(&mut self) {
        if self.settings_selection == 0 {
            self.settings_selection = self.settings_fields().len() - 1;
        } else {
            self.settings_selection -= 1;
        }
    }

    pub fn adjust_setting(&mut self, direction: i32) {
        match self.current_settings_field() {
            SettingsField::AnimationsEnabled => {
                self.controller_settings.toggle_animations();
                let state = if self.controller_settings.animations_enabled() {
                    "enabled"
                } else {
                    "disabled"
                };
                self.set_status(format!("Animations {state}"), StatusKind::Info);
            }
            SettingsField::AnimationSpeed => {
                self.controller_settings.change_speed(direction);
                self.set_status(
                    format!(
                        "Animation speed {}",
                        self.controller_settings.animation_speed().label()
                    ),
                    StatusKind::Info,
                );
            }
            SettingsField::AutoScrollTrainingLog => {
                self.controller_settings.toggle_auto_scroll();
                let state = if self.controller_settings.auto_scroll_training_log() {
                    "enabled"
                } else {
                    "paused"
                };
                self.set_status(format!("Log auto-scroll {state}"), StatusKind::Info);
                if self.controller_settings.auto_scroll_training_log() {
                    self.training_output_scroll = 0;
                }
            }
        }
    }

    fn project_relative_display(&self, path: &Path) -> String {
        if let Some(project) = &self.active_project {
            if let Ok(relative) = path.strip_prefix(&project.root_path) {
                return relative.to_string_lossy().to_string();
            }
        }
        path.to_string_lossy().to_string()
    }

    fn find_rllib_trial_dir(&self, sample: &MetricSample) -> Option<PathBuf> {
        let project = self.active_project.as_ref()?;
        let base = project.logs_path.join("rllib");
        let trial_hint = sample.trial_id().map(|s| s.to_string());
        let mut newest: Option<(SystemTime, PathBuf)> = None;

        let experiments = fs::read_dir(&base).ok()?;
        for experiment in experiments.flatten() {
            let path = experiment.path();
            if !path.is_dir() {
                continue;
            }
            let trials = match fs::read_dir(&path) {
                Ok(entries) => entries,
                Err(_) => continue,
            };
            for entry in trials.flatten() {
                let trial_path = entry.path();
                if !trial_path.is_dir() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(ref hint) = trial_hint {
                    if name.contains(hint) {
                        return Some(trial_path);
                    }
                }
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        if newest.as_ref().map(|(ts, _)| modified > *ts).unwrap_or(true) {
                            newest = Some((modified, trial_path.clone()));
                        }
                    }
                }
            }
        }

        newest.map(|(_, path)| path)
    }

    fn select_checkpoint_dir(
        &mut self,
        sample: &MetricSample,
        trial_dir: &Path,
    ) -> Result<Option<(PathBuf, u32)>> {
        let mut checkpoints: Vec<(u32, PathBuf)> = Vec::new();
        let entries = match fs::read_dir(trial_dir) {
            Ok(entries) => entries,
            Err(error) => {
                self.set_status(
                    format!(
                        "Failed to read trial directory {}: {error}",
                        trial_dir.display()
                    ),
                    StatusKind::Error,
                );
                return Ok(None);
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(number) = Self::checkpoint_number_from_path(&path) {
                checkpoints.push((number, path));
            }
        }

        if checkpoints.is_empty() {
            return Ok(None);
        }

        checkpoints.sort_by_key(|(num, _)| *num);
        let target = sample.checkpoints().map(|n| n as u32);
        let chosen = if let Some(t) = target {
            checkpoints
                .iter()
                .rev()
                .find(|(num, _)| *num <= t)
                .cloned()
                .unwrap_or_else(|| checkpoints.last().cloned().unwrap())
        } else {
            checkpoints.last().cloned().unwrap()
        };

        let (num, path) = chosen;
        Ok(Some((path, num)))
    }

    pub fn apply_selected_checkpoint_to_config(&mut self) -> Result<()> {
        if self.training_config.mode != TrainingMode::MultiAgent {
            self.set_status(
                "Resume shortcut is available for RLlib runs only.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        if self.active_project.is_none() {
            self.set_status("Select a project first", StatusKind::Warning);
            return Ok(());
        }

        let sample = match self.selected_metric_sample().cloned() {
            Some(sample) => sample,
            None => {
                self.set_status(
                    "Select a metric sample in the history to choose a checkpoint.",
                    StatusKind::Warning,
                );
                return Ok(());
            }
        };

        let trial_dir = match self.find_rllib_trial_dir(&sample) {
            Some(path) => path,
            None => {
                self.set_status(
                    "Could not locate a matching RLlib run directory for this sample.",
                    StatusKind::Warning,
                );
                return Ok(());
            }
        };

        let (checkpoint_dir, checkpoint_number) = match self.select_checkpoint_dir(&sample, &trial_dir)?
        {
            Some((dir, num)) => (dir, num),
            None => {
                self.set_status(
                    "No checkpoints found in the matched RLlib run directory.",
                    StatusKind::Warning,
                );
                return Ok(());
            }
        };

        let resume_dir = trial_dir.parent().unwrap_or(&trial_dir).to_path_buf();
        let resume_display = self.project_relative_display(&resume_dir);
        let checkpoint_display = self.project_relative_display(&checkpoint_dir);

        self.training_config.rllib_resume_from = resume_dir.to_string_lossy().to_string();
        self.export_config.rllib_checkpoint_path = checkpoint_dir.to_string_lossy().to_string();
        self.export_config.rllib_checkpoint_number = Some(checkpoint_number);
        if let Some(iteration) = sample.training_iteration() {
            self.metrics_resume_iteration = Some(iteration);
        } else {
            let freq = self.training_config.rllib_checkpoint_frequency as u64;
            if freq > 0 {
                self.metrics_resume_iteration = Some(checkpoint_number as u64 * freq);
            }
        }
        self.metrics_resume_label = Some(format!(
            "checkpoint #{checkpoint_number} ({checkpoint_display})"
        ));

        self.set_status(
            format!(
                "Resume config updated → run: {resume_display}, checkpoint: #{checkpoint_number} ({checkpoint_display})"
            ),
            StatusKind::Success,
        );

        Ok(())
    }

    pub fn toggle_current_setting(&mut self) {
        self.adjust_setting(1);
    }

    pub fn settings_field_value(&self, field: SettingsField) -> String {
        match field {
            SettingsField::AnimationsEnabled => {
                if self.controller_settings.animations_enabled() {
                    "On".to_string()
                } else {
                    "Off".to_string()
                }
            }
            SettingsField::AnimationSpeed => self
                .controller_settings
                .animation_speed()
                .label()
                .to_string(),
            SettingsField::AutoScrollTrainingLog => {
                if self.controller_settings.auto_scroll_training_log() {
                    "On".to_string()
                } else {
                    "Off".to_string()
                }
            }
        }
    }

    pub fn metrics_cycle_focus_next(&mut self) {
        self.metrics_focus = match self.metrics_focus {
            MetricsFocus::History => MetricsFocus::Policies,
            MetricsFocus::Policies => MetricsFocus::History,
            MetricsFocus::Chart => MetricsFocus::History,
            MetricsFocus::Summary => MetricsFocus::History,
        };

        // self.metrics_focus = match self.metrics_focus {
        //     MetricsFocus::History => MetricsFocus::Summary,
        //     MetricsFocus::Summary => MetricsFocus::Policies,
        //     MetricsFocus::Policies => MetricsFocus::Chart,
        //     MetricsFocus::Chart => MetricsFocus::History,
        // };
    }

    pub fn metrics_cycle_focus_previous(&mut self) {
        self.metrics_focus = match self.metrics_focus {
            MetricsFocus::History => MetricsFocus::Policies,
            MetricsFocus::Policies => MetricsFocus::History,
            MetricsFocus::Chart => MetricsFocus::History,
            MetricsFocus::Summary => MetricsFocus::History,
        };

        // self.metrics_focus = match self.metrics_focus {
        //     MetricsFocus::History => MetricsFocus::Chart,
        //     MetricsFocus::Chart => MetricsFocus::Policies,
        //     MetricsFocus::Policies => MetricsFocus::Summary,
        //     MetricsFocus::Summary => MetricsFocus::History,
        // };
    }

    pub fn metrics_summary_scroll(&self) -> usize {
        self.metrics_summary_scroll
    }

    pub fn metrics_policies_scroll(&self) -> usize {
        self.metrics_policies_scroll
    }

    pub fn clamp_all_metrics_scrolls(&mut self) {
        // This gets called from the event loop to ensure scroll values stay reasonable
        // Individual panels will still do their own clamping, but this prevents
        // scroll values from growing too large when content changes

        // We can't know the exact bounds without the UI context, but we can
        // at least ensure they don't grow unbounded. The UI will apply final clamping.

        // Summary scroll - arbitrary max to prevent extreme values
        if self.metrics_summary_scroll > 1000 {
            self.metrics_summary_scroll = 100;
        }

        // Policies scroll - same logic
        if self.metrics_policies_scroll > 1000 {
            self.metrics_policies_scroll = 100;
        }

        // Horizontal scroll - limit to reasonable policy count
        if let Some(sample) = self.training_metrics_history().last() {
            let num_policies = sample.policies().len();
            if num_policies > 0 {
                self.metrics_policies_horizontal_scroll = self
                    .metrics_policies_horizontal_scroll
                    .min(num_policies.saturating_sub(1));
            }
        }

        if self.simulator_event_scroll > self.simulator_event_log.len().saturating_sub(1) {
            self.simulator_event_scroll = self.simulator_event_log.len().saturating_sub(1);
        }

        if self.simulator_actions_scroll > self.simulator_actions.len().saturating_sub(1) {
            self.simulator_actions_scroll = self.simulator_actions.len().saturating_sub(1);
        }
    }

    pub fn metrics_scroll_up(&mut self, amount: usize) {
        match self.metrics_focus {
            MetricsFocus::History => {
                self.metrics_history_scroll = self.metrics_history_scroll.saturating_sub(amount);
            }
            MetricsFocus::Summary => {
                self.metrics_summary_scroll = self.metrics_summary_scroll.saturating_sub(amount);
            }
            MetricsFocus::Policies => {
                self.metrics_policies_scroll = self.metrics_policies_scroll.saturating_sub(amount);
            }
            MetricsFocus::Chart => {
                // Chart doesn't scroll, but we can use it for zoom or other controls later
            }
        }
    }

    pub fn metrics_scroll_down(&mut self, amount: usize) {
        match self.metrics_focus {
            MetricsFocus::History => {
                self.metrics_history_scroll = self.metrics_history_scroll.saturating_add(amount);
            }
            MetricsFocus::Summary => {
                self.metrics_summary_scroll = self.metrics_summary_scroll.saturating_add(amount);
            }
            MetricsFocus::Policies => {
                self.metrics_policies_scroll = self.metrics_policies_scroll.saturating_add(amount);
            }
            MetricsFocus::Chart => {
                // Chart doesn't scroll
            }
        }
    }

    pub fn metrics_policies_expanded(&self) -> bool {
        self.metrics_policies_expanded
    }

    pub fn metrics_toggle_policies_expanded(&mut self) {
        self.metrics_policies_expanded = !self.metrics_policies_expanded;
        // Reset scrolls when toggling
        if self.metrics_policies_expanded {
            self.metrics_policies_horizontal_scroll = 0;
            self.metrics_policies_scroll = 0;
        }
    }

    pub fn metrics_policies_horizontal_scroll(&self) -> usize {
        self.metrics_policies_horizontal_scroll
    }

    pub fn metrics_scroll_policies_left(&mut self) {
        self.metrics_policies_horizontal_scroll =
            self.metrics_policies_horizontal_scroll.saturating_sub(1);
    }

    pub fn metrics_scroll_policies_right(&mut self) {
        self.metrics_policies_horizontal_scroll =
            self.metrics_policies_horizontal_scroll.saturating_add(1);
    }

    pub fn export_output(&self) -> &[String] {
        &self.export_output
    }

    pub fn export_output_scroll(&self) -> usize {
        self.export_output_scroll
    }

    pub fn scroll_export_output_up(&mut self, lines: usize) {
        if self.export_output.is_empty() {
            self.export_output_scroll = 0;
            return;
        }
        let max_offset = self.export_output.len().saturating_sub(1);
        self.export_output_scroll = self
            .export_output_scroll
            .saturating_add(lines)
            .min(max_offset);
    }

    pub fn scroll_export_output_down(&mut self, lines: usize) {
        if lines >= self.export_output_scroll {
            self.export_output_scroll = 0;
        } else {
            self.export_output_scroll -= lines;
        }
    }

    pub fn reset_export_output_scroll(&mut self) {
        self.export_output_scroll = 0;
    }

    pub fn latest_training_metric(&self) -> Option<&MetricSample> {
        self.training_metrics_history().last()
    }

    pub fn is_training_running(&self) -> bool {
        self.training_running
    }

    pub fn is_export_running(&self) -> bool {
        self.export_running
    }

    pub fn training_config(&self) -> &TrainingConfig {
        &self.training_config
    }

    // pub fn training_config_mut(&mut self) -> &mut TrainingConfig {
    //     &mut self.training_config
    // }

    pub fn export_config(&self) -> &ExportConfig {
        &self.export_config
    }

    pub fn is_training_config_valid(&self) -> bool {
        self.training_config_valid
    }

    pub fn update_validation_status(&mut self) {
        self.advanced_validation_errors = self.collect_advanced_validation_errors();
        self.training_config_valid =
            self.validate_training_config().is_ok() && self.advanced_validation_errors.is_empty();
    }

    pub fn toggle_training_mode(&mut self) {
        self.training_config.mode = match self.training_config.mode {
            TrainingMode::SingleAgent => TrainingMode::MultiAgent,
            TrainingMode::MultiAgent => TrainingMode::SingleAgent,
        };
        if let Err(error) = self.persist_training_config() {
            self.set_status(
                format!("Failed to save training config: {error}"),
                StatusKind::Error,
            );
        }
        self.rebuild_advanced_fields();
        self.update_validation_status();
    }

    pub fn show_help(&mut self) {
        self.input_mode = InputMode::Help;
    }

    pub fn hide_help(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn is_help_visible(&self) -> bool {
        self.input_mode == InputMode::Help
    }

    pub fn request_quit(&mut self) {
        if self.training_running {
            self.set_status(
                "Cannot quit while training is running. Cancel training first.",
                StatusKind::Warning,
            );
        } else {
            self.input_mode = InputMode::ConfirmQuit;
        }
    }

    pub fn cancel_quit(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn confirm_quit(&mut self) {
        self.should_quit = true;
    }

    pub fn start_config_edit(&mut self, field: ConfigField) {
        if self.start_choice_menu_if_applicable(field) {
            return;
        }
        let origin_mode = self.input_mode;
        self.config_return_mode = Some(origin_mode);
        self.input_mode = match origin_mode {
            InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig => {
                InputMode::EditingAdvancedConfig
            }
            _ => InputMode::EditingConfig,
        };
        self.active_config_field = Some(field);
        self.config_edit_buffer = self.config_field_value(field);
    }

    pub fn cancel_config_edit(&mut self) {
        let return_mode = self.config_return_mode.take().unwrap_or(InputMode::Normal);
        self.input_mode = return_mode;
        self.active_config_field = None;
        self.config_edit_buffer.clear();
        if matches!(
            return_mode,
            InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig
        ) {
            self.rebuild_advanced_fields();
        }
    }

    pub fn push_config_char(&mut self, ch: char) {
        if self.config_edit_buffer.len() >= 256 {
            return;
        }
        self.config_edit_buffer.push(ch);
    }

    pub fn pop_config_char(&mut self) {
        self.config_edit_buffer.pop();
    }

    pub fn confirm_config_edit(&mut self) {
        let field = match self.active_config_field {
            Some(f) => f,
            None => return,
        };

        let value = self.config_edit_buffer.clone();
        match self
            .set_config_field_value(field, &value)
            .and_then(|_| self.persist_training_config())
        {
            Ok(_) => {
                self.set_status("Configuration updated", StatusKind::Success);
                self.cancel_config_edit();
                self.update_validation_status();
            }
            Err(e) => {
                self.set_status(format!("Failed to update config: {}", e), StatusKind::Error);
            }
        }
    }

    fn start_choice_menu_if_applicable(&mut self, field: ConfigField) -> bool {
        let Some(options) = self.build_config_choices(field) else {
            return false;
        };
        if options.is_empty() {
            return false;
        }
        let origin_mode = self.input_mode;
        self.config_return_mode = Some(origin_mode);
        self.input_mode = InputMode::SelectingConfigOption;
        self.active_config_field = Some(field);
        let current = self.config_field_value(field);
        let normalized = current.trim().to_lowercase();
        let mut selected = options
            .iter()
            .position(|choice| {
                choice.value.to_lowercase() == normalized
                    || choice.label.to_lowercase() == normalized
            })
            .unwrap_or(0);
        if !options.is_empty() && selected >= options.len() {
            selected = options.len() - 1;
        }
        self.choice_menu = Some(ChoiceMenuState {
            field,
            options,
            selected,
        });
        true
    }

    pub fn move_choice_selection(&mut self, delta: isize) {
        if let Some(menu) = self.choice_menu.as_mut() {
            if menu.options.is_empty() {
                return;
            }
            let len = menu.options.len() as isize;
            let mut idx = menu.selected as isize + delta;
            idx = ((idx % len) + len) % len;
            menu.selected = idx as usize;
        }
    }

    pub fn confirm_choice_selection(&mut self) -> Result<()> {
        let (field, value) = match self.choice_menu.as_ref().and_then(|menu| {
            menu.selected_choice()
                .map(|choice| (menu.field, choice.value.clone()))
        }) {
            Some(pair) => pair,
            None => return Ok(()),
        };

        match self
            .set_config_field_value(field, &value)
            .and_then(|_| self.persist_training_config())
        {
            Ok(_) => {
                self.set_status("Configuration updated", StatusKind::Success);
                self.exit_choice_menu();
                self.update_validation_status();
                Ok(())
            }
            Err(error) => {
                self.set_status(
                    format!("Failed to update config: {}", error),
                    StatusKind::Error,
                );
                Ok(())
            }
        }
    }

    pub fn cancel_choice_selection(&mut self) {
        self.exit_choice_menu();
    }

    fn exit_choice_menu(&mut self) {
        self.choice_menu = None;
        self.active_config_field = None;
        let return_mode = self.config_return_mode.take().unwrap_or(InputMode::Normal);
        self.input_mode = return_mode;
        if matches!(
            return_mode,
            InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig
        ) {
            self.rebuild_advanced_fields();
        }
    }

    fn build_config_choices(&self, field: ConfigField) -> Option<Vec<ConfigChoice>> {
        match field {
            ConfigField::Sb3PolicyType | ConfigField::RllibPolicyType => Some(
                POLICY_TYPE_LIST
                    .iter()
                    .map(|policy| {
                        ConfigChoice::new(policy.label(), policy.as_str(), policy.summary())
                    })
                    .collect(),
            ),
            ConfigField::RllibAlgorithm => Some(
                RLLIB_ALGORITHM_LIST
                    .iter()
                    .map(|algo| {
                        ConfigChoice::new(algo.trainer_name(), algo.as_str(), algo.summary())
                    })
                    .collect(),
            ),
            ConfigField::RllibStopMode => Some(
                RLLIB_STOP_MODE_CHOICES
                    .iter()
                    .map(|(value, label, desc)| ConfigChoice::new(*label, *value, *desc))
                    .collect(),
            ),
            ConfigField::RllibBatchMode => Some(
                RLLIB_BATCH_MODE_CHOICES
                    .iter()
                    .map(|(value, label, desc)| ConfigChoice::new(*label, *value, *desc))
                    .collect(),
            ),
            ConfigField::RllibFramework => Some(
                RLLIB_FRAMEWORK_CHOICES
                    .iter()
                    .map(|(value, label, desc)| ConfigChoice::new(*label, *value, *desc))
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn active_config_field(&self) -> Option<ConfigField> {
        self.active_config_field
    }

    pub fn config_edit_buffer(&self) -> &str {
        &self.config_edit_buffer
    }

    pub fn open_advanced_config(&mut self) {
        self.rebuild_advanced_fields();
        if self.advanced_fields.is_empty() {
            self.set_status(
                "No advanced settings available for this mode yet.",
                StatusKind::Info,
            );
            return;
        }
        if self.advanced_selection >= self.advanced_fields.len() {
            self.advanced_selection = 0;
        }
        self.input_mode = InputMode::AdvancedConfig;
    }

    pub fn close_advanced_config(&mut self) {
        self.input_mode = InputMode::Normal;
        self.config_return_mode = None;
        self.active_config_field = None;
        self.config_edit_buffer.clear();
        self.advanced_fields.clear();
        self.advanced_selection = 0;
    }

    pub fn advanced_fields(&self) -> &[ConfigField] {
        &self.advanced_fields
    }

    pub fn choice_menu(&self) -> Option<ChoiceMenuView<'_>> {
        self.choice_menu.as_ref().map(|menu| ChoiceMenuView {
            field: menu.field,
            options: &menu.options,
            selected: menu.selected,
        })
    }

    pub fn advanced_selection(&self) -> usize {
        self.advanced_selection
    }

    pub fn advanced_field_error(&self, field: ConfigField) -> Option<&str> {
        self.advanced_validation_errors
            .get(&field)
            .map(|s| s.as_str())
    }

    pub fn selected_advanced_field(&self) -> Option<ConfigField> {
        self.advanced_fields.get(self.advanced_selection).copied()
    }

    pub fn select_next_advanced_field(&mut self) {
        if !self.advanced_fields.is_empty() {
            self.advanced_selection = (self.advanced_selection + 1) % self.advanced_fields.len();
        }
    }

    pub fn select_previous_advanced_field(&mut self) {
        if !self.advanced_fields.is_empty() {
            if self.advanced_selection == 0 {
                self.advanced_selection = self.advanced_fields.len() - 1;
            } else {
                self.advanced_selection -= 1;
            }
        }
    }

    pub fn edit_selected_advanced_field(&mut self) {
        if let Some(field) = self.selected_advanced_field() {
            if field == ConfigField::RllibResumeFrom {
                self.start_config_file_browser(field);
            } else {
                self.start_config_edit(field);
            }
        }
    }

    pub fn clear_selected_advanced_field(&mut self) {
        if self.is_training_running() {
            self.set_status(
                "Cannot modify settings while training is running",
                StatusKind::Warning,
            );
            return;
        }

        if let Some(field) = self.selected_advanced_field() {
            match self.reset_config_field_to_default(field) {
                Ok(true) => {
                    self.set_status("Field reset to default", StatusKind::Success);
                }
                Ok(false) => {
                    self.set_status("Field already at default value", StatusKind::Info);
                }
                Err(error) => {
                    self.set_status(format!("Failed to reset field: {error}"), StatusKind::Error);
                }
            }
        }
    }

    pub fn export_mode(&self) -> ExportMode {
        self.export_mode
    }

    pub fn toggle_export_mode(&mut self) -> Result<()> {
        self.export_mode = match self.export_mode {
            ExportMode::StableBaselines3 => ExportMode::Rllib,
            ExportMode::Rllib => ExportMode::StableBaselines3,
        };
        self.rebuild_export_fields();
        self.persist_export_state()
    }

    pub fn export_focus(&self) -> ExportFocus {
        self.export_focus
    }

    pub fn toggle_export_focus(&mut self) {
        self.export_focus = match self.export_focus {
            ExportFocus::Fields => ExportFocus::Output,
            ExportFocus::Output => ExportFocus::Fields,
        };
    }

    pub fn export_fields(&self) -> &[ExportField] {
        &self.export_fields
    }

    pub fn export_selection(&self) -> usize {
        self.export_selection
    }

    pub fn selected_export_field(&self) -> Option<ExportField> {
        self.export_fields.get(self.export_selection).copied()
    }

    pub fn select_next_export_field(&mut self) {
        if !self.export_fields.is_empty() {
            self.export_selection = (self.export_selection + 1) % self.export_fields.len();
        }
    }

    pub fn select_previous_export_field(&mut self) {
        if !self.export_fields.is_empty() {
            if self.export_selection == 0 {
                self.export_selection = self.export_fields.len() - 1;
            } else {
                self.export_selection -= 1;
            }
        }
    }

    pub fn edit_selected_export_field(&mut self) {
        if let Some(field) = self.selected_export_field() {
            if field.is_toggle() {
                self.toggle_export_field(field);
            } else if field.uses_file_browser() {
                self.start_export_file_browser(field);
            } else {
                self.start_export_edit(field);
            }
        }
    }

    pub fn clear_selected_export_field(&mut self) {
        if self.export_focus != ExportFocus::Fields {
            return;
        }

        if self.is_export_running() {
            self.set_status(
                "Cannot edit export options during an active export",
                StatusKind::Warning,
            );
            return;
        }

        if let Some(field) = self.selected_export_field() {
            match self.reset_export_field_to_default(field) {
                Ok(true) => {
                    self.set_status("Export option reset to default", StatusKind::Success);
                }
                Ok(false) => {
                    self.set_status("Export option already at default value", StatusKind::Info);
                }
                Err(error) => {
                    self.set_status(
                        format!("Failed to reset export option: {error}"),
                        StatusKind::Error,
                    );
                }
            }
            self.rebuild_export_fields();
        }
    }

    pub fn active_export_field(&self) -> Option<ExportField> {
        self.active_export_field
    }

    pub fn export_edit_buffer(&self) -> &str {
        &self.export_edit_buffer
    }

    pub fn config_field_value(&self, field: ConfigField) -> String {
        match field {
            ConfigField::EnvPath => self.training_config.env_path.clone(),
            ConfigField::Timesteps => self.training_config.timesteps.to_string(),
            ConfigField::ExperimentName => self.training_config.experiment_name.clone(),
            ConfigField::Sb3PolicyType => self.training_config.sb3_policy_type.as_str().to_string(),
            ConfigField::Sb3Speedup => self.training_config.sb3_speedup.to_string(),
            ConfigField::Sb3NParallel => self.training_config.sb3_n_parallel.to_string(),
            ConfigField::Sb3Viz => if self.training_config.sb3_viz {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ConfigField::Sb3PolicyLayers => {
                format_usize_list(&self.training_config.sb3_policy_layers)
            }
            ConfigField::Sb3CnnChannels => {
                format_usize_list(&self.training_config.sb3_cnn_channels)
            }
            ConfigField::Sb3LstmHiddenSize => self.training_config.sb3_lstm_hidden_size.to_string(),
            ConfigField::Sb3LstmNumLayers => self.training_config.sb3_lstm_num_layers.to_string(),
            ConfigField::Sb3GrnHiddenSize => self.training_config.sb3_grn_hidden_size.to_string(),
            ConfigField::Sb3LearningRate => format_f64(self.training_config.sb3_learning_rate),
            ConfigField::Sb3BatchSize => self.training_config.sb3_batch_size.to_string(),
            ConfigField::Sb3NSteps => self.training_config.sb3_n_steps.to_string(),
            ConfigField::Sb3Gamma => format_f64(self.training_config.sb3_gamma),
            ConfigField::Sb3GaeLambda => format_f64(self.training_config.sb3_gae_lambda),
            ConfigField::Sb3EntCoef => format_f64(self.training_config.sb3_ent_coef),
            ConfigField::Sb3ClipRange => format_f64(self.training_config.sb3_clip_range),
            ConfigField::Sb3VfCoef => format_f64(self.training_config.sb3_vf_coef),
            ConfigField::Sb3MaxGradNorm => format_f64(self.training_config.sb3_max_grad_norm),
            ConfigField::RllibConfigFile => self.training_config.rllib_config_file.clone(),
            ConfigField::RllibShowWindow => if self.training_config.rllib_show_window {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ConfigField::RllibAlgorithm => self
                .training_config
                .rllib_algorithm
                .trainer_name()
                .to_string(),
            ConfigField::RllibEnvActionRepeat => {
                self.training_config.rllib_env_action_repeat.to_string()
            }
            ConfigField::RllibEnvSpeedup => self.training_config.rllib_env_speedup.to_string(),
            ConfigField::RllibNumWorkers => self.training_config.rllib_num_workers.to_string(),
            ConfigField::RllibNumEnvWorkers => {
                self.training_config.rllib_num_envs_per_worker.to_string()
            }
            ConfigField::RllibTrainBatchSize => {
                self.training_config.rllib_train_batch_size.to_string()
            }
            ConfigField::RllibSgdMinibatchSize => {
                self.training_config.rllib_sgd_minibatch_size.to_string()
            }
            ConfigField::RllibNumSgdIter => self.training_config.rllib_num_sgd_iter.to_string(),
            ConfigField::RllibLr => format_f64(self.training_config.rllib_lr),
            ConfigField::RllibGamma => format_f64(self.training_config.rllib_gamma),
            ConfigField::RllibLambda => format_f64(self.training_config.rllib_lambda),
            ConfigField::RllibClipParam => format_f64(self.training_config.rllib_clip_param),
            ConfigField::RllibEntropyCoeff => format_f64(self.training_config.rllib_entropy_coeff),
            ConfigField::RllibVfLossCoeff => format_f64(self.training_config.rllib_vf_loss_coeff),
            ConfigField::RllibGradClip => format_f64(self.training_config.rllib_grad_clip),
            ConfigField::RllibFramework => self.training_config.rllib_framework.clone(),
            ConfigField::RllibActivation => self.training_config.rllib_activation.clone(),
            ConfigField::RllibBatchMode => self.training_config.rllib_batch_mode.clone(),
            ConfigField::RllibRolloutFragmentLength => self
                .training_config
                .rllib_rollout_fragment_length
                .to_string(),
            ConfigField::RllibNumGpus => format_f64(self.training_config.rllib_num_gpus),
            ConfigField::RllibMaxSeqLen => self.training_config.rllib_max_seq_len.to_string(),
            ConfigField::RllibFcnetHiddens => {
                format_usize_list(&self.training_config.rllib_fcnet_hiddens)
            }
            ConfigField::RllibPolicyType => {
                self.training_config.rllib_policy_type.as_str().to_string()
            }
            ConfigField::RllibCnnChannels => {
                format_usize_list(&self.training_config.rllib_cnn_channels)
            }
            ConfigField::RllibLstmCellSize => self.training_config.rllib_lstm_cell_size.to_string(),
            ConfigField::RllibLstmNumLayers => {
                self.training_config.rllib_lstm_num_layers.to_string()
            }
            ConfigField::RllibLstmIncludePrevActions => {
                if self.training_config.rllib_lstm_include_prev_actions {
                    "true"
                } else {
                    "false"
                }
                .to_string()
            }
            ConfigField::RllibGrnHiddenSize => {
                self.training_config.rllib_grn_hidden_size.to_string()
            }
            ConfigField::RllibCheckpointFrequency => {
                self.training_config.rllib_checkpoint_frequency.to_string()
            }
            ConfigField::RllibResumeFrom => self.training_config.rllib_resume_from.clone(),
            ConfigField::RllibStopMode => match self.training_config.rllib_stop_mode {
                RllibStopMode::TimeSeconds => "time_seconds".to_string(),
                RllibStopMode::Timesteps => "timesteps".to_string(),
            },
            ConfigField::RllibStopTimeSeconds => {
                self.training_config.rllib_stop_time_seconds.to_string()
            }
        }
    }

    pub fn export_field_value(&self, field: ExportField) -> String {
        match field {
            ExportField::Sb3ModelPath => self.export_config.sb3_model_path.clone(),
            ExportField::Sb3OutputPath => self.export_config.sb3_output_path.clone(),
            ExportField::Sb3Algo => self.export_config.sb3_algo.clone(),
            ExportField::Sb3Opset => self.export_config.sb3_opset.to_string(),
            ExportField::Sb3IrVersion => self.export_config.sb3_ir_version.to_string(),
            ExportField::Sb3UseObsArray => if self.export_config.sb3_use_obs_array {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ExportField::Sb3SkipVerify => if self.export_config.sb3_skip_verify {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ExportField::RllibCheckpointPath => self.export_config.rllib_checkpoint_path.clone(),
            ExportField::RllibCheckpointNumber => self
                .export_config
                .rllib_checkpoint_number
                .map(|value| value.to_string())
                .unwrap_or_default(),
            ExportField::RllibOutputDir => self.export_config.rllib_output_dir.clone(),
            ExportField::RllibPolicyId => self.export_config.rllib_policy_id.clone(),
            ExportField::RllibOpset => self.export_config.rllib_opset.to_string(),
            ExportField::RllibIrVersion => self.export_config.rllib_ir_version.to_string(),
            ExportField::RllibMultiagent => if self.export_config.rllib_multiagent {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ExportField::RllibPrefix => self.export_config.rllib_prefix.clone(),
        }
    }

    pub fn toggle_export_field(&mut self, field: ExportField) {
        match field {
            ExportField::Sb3UseObsArray => {
                self.export_config.sb3_use_obs_array = !self.export_config.sb3_use_obs_array;
                let state = if self.export_config.sb3_use_obs_array {
                    "enabled"
                } else {
                    "disabled"
                };
                self.set_status(format!("SB3 obs-array export {state}"), StatusKind::Info);
            }
            ExportField::Sb3SkipVerify => {
                self.export_config.sb3_skip_verify = !self.export_config.sb3_skip_verify;
                let state = if self.export_config.sb3_skip_verify {
                    "Skipping verification"
                } else {
                    "Running verification"
                };
                self.set_status(state, StatusKind::Info);
            }
            ExportField::RllibMultiagent => {
                self.export_config.rllib_multiagent = !self.export_config.rllib_multiagent;
                let state = if self.export_config.rllib_multiagent {
                    "Multi-agent export"
                } else {
                    "Single-agent export"
                };
                self.set_status(state, StatusKind::Info);
            }
            _ => {}
        }
        if let Err(error) = self.persist_export_state() {
            self.set_status(
                format!("Failed to save export settings: {error}"),
                StatusKind::Error,
            );
        }
        self.rebuild_export_fields();
    }

    pub fn start_export_edit(&mut self, field: ExportField) {
        let origin_mode = self.input_mode;
        self.export_return_mode = Some(origin_mode);
        self.input_mode = InputMode::EditingExport;
        self.active_export_field = Some(field);
        self.export_edit_buffer = self.export_field_value(field);
    }

    pub fn cancel_export_edit(&mut self) {
        let return_mode = self.export_return_mode.take().unwrap_or(InputMode::Normal);
        self.input_mode = return_mode;
        self.active_export_field = None;
        self.export_edit_buffer.clear();
    }

    pub fn push_export_char(&mut self, ch: char) {
        if self.export_edit_buffer.len() >= 256 {
            return;
        }
        self.export_edit_buffer.push(ch);
    }

    pub fn pop_export_char(&mut self) {
        self.export_edit_buffer.pop();
    }

    pub fn confirm_export_edit(&mut self) {
        let field = match self.active_export_field {
            Some(field) => field,
            None => return,
        };
        let value = self.export_edit_buffer.clone();
        match self.set_export_field_value(field, &value) {
            Ok(()) => {
                self.set_status("Export option updated", StatusKind::Success);
                self.cancel_export_edit();
                self.rebuild_export_fields();
            }
            Err(error) => {
                self.set_status(
                    format!("Failed to update export option: {error}"),
                    StatusKind::Error,
                );
            }
        }
    }

    fn set_export_field_value(&mut self, field: ExportField, value: &str) -> Result<()> {
        let trimmed = value.trim();
        match field {
            ExportField::Sb3ModelPath => {
                self.export_config.sb3_model_path = trimmed.to_string();
            }
            ExportField::Sb3OutputPath => {
                self.export_config.sb3_output_path = trimmed.to_string();
            }
            ExportField::Sb3Algo => {
                self.export_config.sb3_algo = trimmed.to_string();
            }
            ExportField::Sb3Opset => {
                let opset: u32 = trimmed
                    .parse()
                    .wrap_err("Opset must be a positive integer")?;
                if opset == 0 {
                    bail!("Opset must be greater than 0");
                }
                self.export_config.sb3_opset = opset;
            }
            ExportField::Sb3IrVersion => {
                let ir: u32 = trimmed
                    .parse()
                    .wrap_err("IR version must be a positive integer")?;
                if ir == 0 {
                    bail!("IR version must be greater than 0");
                }
                self.export_config.sb3_ir_version = ir;
            }
            ExportField::Sb3UseObsArray | ExportField::Sb3SkipVerify => {
                bail!("Toggle values cannot be edited as text");
            }
            ExportField::RllibCheckpointPath => {
                if trimmed.is_empty() {
                    self.export_config.rllib_checkpoint_path.clear();
                    self.export_config.rllib_checkpoint_number = None;
                } else {
                    self.export_config.rllib_checkpoint_path = trimmed.to_string();
                    self.sync_checkpoint_number_from_path(trimmed);
                }
            }
            ExportField::RllibCheckpointNumber => {
                if trimmed.is_empty() {
                    self.export_config.rllib_checkpoint_number = None;
                } else {
                    let number: u32 = trimmed
                        .parse()
                        .wrap_err("Checkpoint number must be a non-negative integer")?;
                    self.export_config.rllib_checkpoint_number = Some(number);
                }
            }
            ExportField::RllibOutputDir => {
                self.export_config.rllib_output_dir = trimmed.to_string();
            }
            ExportField::RllibPolicyId => {
                self.export_config.rllib_policy_id = trimmed.to_string();
            }
            ExportField::RllibOpset => {
                let opset: u32 = trimmed
                    .parse()
                    .wrap_err("Opset must be a positive integer")?;
                if opset == 0 {
                    bail!("Opset must be greater than 0");
                }
                self.export_config.rllib_opset = opset;
            }
            ExportField::RllibIrVersion => {
                let ir: u32 = trimmed
                    .parse()
                    .wrap_err("IR version must be a positive integer")?;
                if ir == 0 {
                    bail!("IR version must be greater than 0");
                }
                self.export_config.rllib_ir_version = ir;
            }
            ExportField::RllibMultiagent => {
                bail!("Toggle values cannot be edited as text");
            }
            ExportField::RllibPrefix => {
                self.export_config.rllib_prefix = trimmed.to_string();
            }
        }
        self.persist_export_state()?;
        Ok(())
    }

    fn reset_export_field_to_default(&mut self, field: ExportField) -> Result<bool> {
        let defaults = ExportConfig::default();
        let changed = match field {
            ExportField::Sb3ModelPath => {
                if self.export_config.sb3_model_path == defaults.sb3_model_path {
                    false
                } else {
                    self.export_config.sb3_model_path = defaults.sb3_model_path.clone();
                    true
                }
            }
            ExportField::Sb3OutputPath => {
                if self.export_config.sb3_output_path == defaults.sb3_output_path {
                    false
                } else {
                    self.export_config.sb3_output_path = defaults.sb3_output_path.clone();
                    true
                }
            }
            ExportField::Sb3Algo => {
                if self.export_config.sb3_algo == defaults.sb3_algo {
                    false
                } else {
                    self.export_config.sb3_algo = defaults.sb3_algo.clone();
                    true
                }
            }
            ExportField::Sb3Opset => {
                if self.export_config.sb3_opset == defaults.sb3_opset {
                    false
                } else {
                    self.export_config.sb3_opset = defaults.sb3_opset;
                    true
                }
            }
            ExportField::Sb3IrVersion => {
                if self.export_config.sb3_ir_version == defaults.sb3_ir_version {
                    false
                } else {
                    self.export_config.sb3_ir_version = defaults.sb3_ir_version;
                    true
                }
            }
            ExportField::Sb3UseObsArray => {
                if self.export_config.sb3_use_obs_array == defaults.sb3_use_obs_array {
                    false
                } else {
                    self.export_config.sb3_use_obs_array = defaults.sb3_use_obs_array;
                    true
                }
            }
            ExportField::Sb3SkipVerify => {
                if self.export_config.sb3_skip_verify == defaults.sb3_skip_verify {
                    false
                } else {
                    self.export_config.sb3_skip_verify = defaults.sb3_skip_verify;
                    true
                }
            }
            ExportField::RllibCheckpointPath => {
                if self.export_config.rllib_checkpoint_path == defaults.rllib_checkpoint_path {
                    false
                } else {
                    self.export_config.rllib_checkpoint_path =
                        defaults.rllib_checkpoint_path.clone();
                    self.export_config.rllib_checkpoint_number = defaults.rllib_checkpoint_number;
                    true
                }
            }
            ExportField::RllibCheckpointNumber => {
                if self.export_config.rllib_checkpoint_number == defaults.rllib_checkpoint_number {
                    false
                } else {
                    self.export_config.rllib_checkpoint_number = defaults.rllib_checkpoint_number;
                    true
                }
            }
            ExportField::RllibOutputDir => {
                if self.export_config.rllib_output_dir == defaults.rllib_output_dir {
                    false
                } else {
                    self.export_config.rllib_output_dir = defaults.rllib_output_dir.clone();
                    true
                }
            }
            ExportField::RllibPolicyId => {
                if self.export_config.rllib_policy_id == defaults.rllib_policy_id {
                    false
                } else {
                    self.export_config.rllib_policy_id = defaults.rllib_policy_id.clone();
                    true
                }
            }
            ExportField::RllibOpset => {
                if self.export_config.rllib_opset == defaults.rllib_opset {
                    false
                } else {
                    self.export_config.rllib_opset = defaults.rllib_opset;
                    true
                }
            }
            ExportField::RllibIrVersion => {
                if self.export_config.rllib_ir_version == defaults.rllib_ir_version {
                    false
                } else {
                    self.export_config.rllib_ir_version = defaults.rllib_ir_version;
                    true
                }
            }
            ExportField::RllibMultiagent => {
                if self.export_config.rllib_multiagent == defaults.rllib_multiagent {
                    false
                } else {
                    self.export_config.rllib_multiagent = defaults.rllib_multiagent;
                    true
                }
            }
            ExportField::RllibPrefix => {
                if self.export_config.rllib_prefix == defaults.rllib_prefix {
                    false
                } else {
                    self.export_config.rllib_prefix = defaults.rllib_prefix.clone();
                    true
                }
            }
        };

        if changed {
            self.persist_export_state()?;
        }
        Ok(changed)
    }

    fn rebuild_export_fields(&mut self) {
        self.export_fields = self.build_export_fields();
        if self.export_fields.is_empty() {
            self.export_selection = 0;
        } else if self.export_selection >= self.export_fields.len() {
            self.export_selection = self.export_fields.len() - 1;
        }
    }

    fn build_export_fields(&self) -> Vec<ExportField> {
        match self.export_mode {
            ExportMode::StableBaselines3 => vec![
                ExportField::Sb3ModelPath,
                ExportField::Sb3OutputPath,
                ExportField::Sb3Algo,
                ExportField::Sb3Opset,
                ExportField::Sb3IrVersion,
                ExportField::Sb3UseObsArray,
                ExportField::Sb3SkipVerify,
            ],
            ExportMode::Rllib => vec![
                ExportField::RllibCheckpointPath,
                ExportField::RllibCheckpointNumber,
                ExportField::RllibOutputDir,
                ExportField::RllibPrefix,
                ExportField::RllibPolicyId,
                ExportField::RllibOpset,
                ExportField::RllibIrVersion,
                ExportField::RllibMultiagent,
            ],
        }
    }

    fn reset_config_field_to_default(&mut self, field: ConfigField) -> Result<bool> {
        let defaults = TrainingConfig::default();
        let changed = match field {
            ConfigField::EnvPath => {
                if self.training_config.env_path == defaults.env_path {
                    false
                } else {
                    self.training_config.env_path = defaults.env_path.clone();
                    true
                }
            }
            ConfigField::Timesteps => {
                if self.training_config.timesteps == defaults.timesteps {
                    false
                } else {
                    self.training_config.timesteps = defaults.timesteps;
                    true
                }
            }
            ConfigField::ExperimentName => {
                if self.training_config.experiment_name == defaults.experiment_name {
                    false
                } else {
                    self.training_config.experiment_name = defaults.experiment_name.clone();
                    true
                }
            }
            ConfigField::Sb3PolicyType => {
                if self.training_config.sb3_policy_type == defaults.sb3_policy_type {
                    false
                } else {
                    self.training_config.sb3_policy_type = defaults.sb3_policy_type;
                    self.rebuild_advanced_fields();
                    true
                }
            }
            ConfigField::Sb3Speedup => {
                if self.training_config.sb3_speedup == defaults.sb3_speedup {
                    false
                } else {
                    self.training_config.sb3_speedup = defaults.sb3_speedup;
                    true
                }
            }
            ConfigField::Sb3NParallel => {
                if self.training_config.sb3_n_parallel == defaults.sb3_n_parallel {
                    false
                } else {
                    self.training_config.sb3_n_parallel = defaults.sb3_n_parallel;
                    true
                }
            }
            ConfigField::Sb3Viz => {
                if self.training_config.sb3_viz == defaults.sb3_viz {
                    false
                } else {
                    self.training_config.sb3_viz = defaults.sb3_viz;
                    true
                }
            }
            ConfigField::Sb3PolicyLayers => {
                if self.training_config.sb3_policy_layers == defaults.sb3_policy_layers {
                    false
                } else {
                    self.training_config.sb3_policy_layers = defaults.sb3_policy_layers.clone();
                    true
                }
            }
            ConfigField::Sb3CnnChannels => {
                if self.training_config.sb3_cnn_channels == defaults.sb3_cnn_channels {
                    false
                } else {
                    self.training_config.sb3_cnn_channels = defaults.sb3_cnn_channels.clone();
                    true
                }
            }
            ConfigField::Sb3LstmHiddenSize => {
                if self.training_config.sb3_lstm_hidden_size == defaults.sb3_lstm_hidden_size {
                    false
                } else {
                    self.training_config.sb3_lstm_hidden_size = defaults.sb3_lstm_hidden_size;
                    true
                }
            }
            ConfigField::Sb3LstmNumLayers => {
                if self.training_config.sb3_lstm_num_layers == defaults.sb3_lstm_num_layers {
                    false
                } else {
                    self.training_config.sb3_lstm_num_layers = defaults.sb3_lstm_num_layers;
                    true
                }
            }
            ConfigField::Sb3GrnHiddenSize => {
                if self.training_config.sb3_grn_hidden_size == defaults.sb3_grn_hidden_size {
                    false
                } else {
                    self.training_config.sb3_grn_hidden_size = defaults.sb3_grn_hidden_size;
                    true
                }
            }
            ConfigField::Sb3LearningRate => {
                if (self.training_config.sb3_learning_rate - defaults.sb3_learning_rate).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_learning_rate = defaults.sb3_learning_rate;
                    true
                }
            }
            ConfigField::Sb3BatchSize => {
                if self.training_config.sb3_batch_size == defaults.sb3_batch_size {
                    false
                } else {
                    self.training_config.sb3_batch_size = defaults.sb3_batch_size;
                    true
                }
            }
            ConfigField::Sb3NSteps => {
                if self.training_config.sb3_n_steps == defaults.sb3_n_steps {
                    false
                } else {
                    self.training_config.sb3_n_steps = defaults.sb3_n_steps;
                    true
                }
            }
            ConfigField::Sb3Gamma => {
                if (self.training_config.sb3_gamma - defaults.sb3_gamma).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.sb3_gamma = defaults.sb3_gamma;
                    true
                }
            }
            ConfigField::Sb3GaeLambda => {
                if (self.training_config.sb3_gae_lambda - defaults.sb3_gae_lambda).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_gae_lambda = defaults.sb3_gae_lambda;
                    true
                }
            }
            ConfigField::Sb3EntCoef => {
                if (self.training_config.sb3_ent_coef - defaults.sb3_ent_coef).abs() < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_ent_coef = defaults.sb3_ent_coef;
                    true
                }
            }
            ConfigField::Sb3ClipRange => {
                if (self.training_config.sb3_clip_range - defaults.sb3_clip_range).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_clip_range = defaults.sb3_clip_range;
                    true
                }
            }
            ConfigField::Sb3VfCoef => {
                if (self.training_config.sb3_vf_coef - defaults.sb3_vf_coef).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.sb3_vf_coef = defaults.sb3_vf_coef;
                    true
                }
            }
            ConfigField::Sb3MaxGradNorm => {
                if (self.training_config.sb3_max_grad_norm - defaults.sb3_max_grad_norm).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_max_grad_norm = defaults.sb3_max_grad_norm;
                    true
                }
            }
            ConfigField::RllibConfigFile => {
                if self.training_config.rllib_config_file == defaults.rllib_config_file {
                    false
                } else {
                    self.training_config.rllib_config_file = defaults.rllib_config_file.clone();
                    true
                }
            }
            ConfigField::RllibShowWindow => {
                if self.training_config.rllib_show_window == defaults.rllib_show_window {
                    false
                } else {
                    self.training_config.rllib_show_window = defaults.rllib_show_window;
                    true
                }
            }
            ConfigField::RllibAlgorithm => {
                if self.training_config.rllib_algorithm == defaults.rllib_algorithm {
                    false
                } else {
                    self.training_config.rllib_algorithm = defaults.rllib_algorithm;
                    true
                }
            }
            ConfigField::RllibEnvActionRepeat => {
                if self.training_config.rllib_env_action_repeat == defaults.rllib_env_action_repeat
                {
                    false
                } else {
                    self.training_config.rllib_env_action_repeat = defaults.rllib_env_action_repeat;
                    true
                }
            }
            ConfigField::RllibEnvSpeedup => {
                if self.training_config.rllib_env_speedup == defaults.rllib_env_speedup {
                    false
                } else {
                    self.training_config.rllib_env_speedup = defaults.rllib_env_speedup;
                    true
                }
            }
            ConfigField::RllibNumWorkers => {
                if self.training_config.rllib_num_workers == defaults.rllib_num_workers {
                    false
                } else {
                    self.training_config.rllib_num_workers = defaults.rllib_num_workers;
                    true
                }
            }
            ConfigField::RllibNumEnvWorkers => {
                if self.training_config.rllib_num_envs_per_worker
                    == defaults.rllib_num_envs_per_worker
                {
                    false
                } else {
                    self.training_config.rllib_num_envs_per_worker =
                        defaults.rllib_num_envs_per_worker;
                    true
                }
            }
            ConfigField::RllibTrainBatchSize => {
                if self.training_config.rllib_train_batch_size == defaults.rllib_train_batch_size {
                    false
                } else {
                    self.training_config.rllib_train_batch_size = defaults.rllib_train_batch_size;
                    true
                }
            }
            ConfigField::RllibSgdMinibatchSize => {
                if self.training_config.rllib_sgd_minibatch_size
                    == defaults.rllib_sgd_minibatch_size
                {
                    false
                } else {
                    self.training_config.rllib_sgd_minibatch_size =
                        defaults.rllib_sgd_minibatch_size;
                    true
                }
            }
            ConfigField::RllibNumSgdIter => {
                if self.training_config.rllib_num_sgd_iter == defaults.rllib_num_sgd_iter {
                    false
                } else {
                    self.training_config.rllib_num_sgd_iter = defaults.rllib_num_sgd_iter;
                    true
                }
            }
            ConfigField::RllibLr => {
                if (self.training_config.rllib_lr - defaults.rllib_lr).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.rllib_lr = defaults.rllib_lr;
                    true
                }
            }
            ConfigField::RllibGamma => {
                if (self.training_config.rllib_gamma - defaults.rllib_gamma).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.rllib_gamma = defaults.rllib_gamma;
                    true
                }
            }
            ConfigField::RllibLambda => {
                if (self.training_config.rllib_lambda - defaults.rllib_lambda).abs() < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_lambda = defaults.rllib_lambda;
                    true
                }
            }
            ConfigField::RllibClipParam => {
                if (self.training_config.rllib_clip_param - defaults.rllib_clip_param).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_clip_param = defaults.rllib_clip_param;
                    true
                }
            }
            ConfigField::RllibEntropyCoeff => {
                if (self.training_config.rllib_entropy_coeff - defaults.rllib_entropy_coeff).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_entropy_coeff = defaults.rllib_entropy_coeff;
                    true
                }
            }
            ConfigField::RllibVfLossCoeff => {
                if (self.training_config.rllib_vf_loss_coeff - defaults.rllib_vf_loss_coeff).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_vf_loss_coeff = defaults.rllib_vf_loss_coeff;
                    true
                }
            }
            ConfigField::RllibGradClip => {
                if (self.training_config.rllib_grad_clip - defaults.rllib_grad_clip).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_grad_clip = defaults.rllib_grad_clip;
                    true
                }
            }
            ConfigField::RllibFramework => {
                if self.training_config.rllib_framework == defaults.rllib_framework {
                    false
                } else {
                    self.training_config.rllib_framework = defaults.rllib_framework.clone();
                    true
                }
            }
            ConfigField::RllibActivation => {
                if self.training_config.rllib_activation == defaults.rllib_activation {
                    false
                } else {
                    self.training_config.rllib_activation = defaults.rllib_activation.clone();
                    true
                }
            }
            ConfigField::RllibBatchMode => {
                if self.training_config.rllib_batch_mode == defaults.rllib_batch_mode {
                    false
                } else {
                    self.training_config.rllib_batch_mode = defaults.rllib_batch_mode.clone();
                    true
                }
            }
            ConfigField::RllibRolloutFragmentLength => {
                if self.training_config.rllib_rollout_fragment_length
                    == defaults.rllib_rollout_fragment_length
                {
                    false
                } else {
                    self.training_config.rllib_rollout_fragment_length =
                        defaults.rllib_rollout_fragment_length;
                    true
                }
            }
            ConfigField::RllibNumGpus => {
                if (self.training_config.rllib_num_gpus - defaults.rllib_num_gpus).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_num_gpus = defaults.rllib_num_gpus;
                    true
                }
            }
            ConfigField::RllibMaxSeqLen => {
                if self.training_config.rllib_max_seq_len == defaults.rllib_max_seq_len {
                    false
                } else {
                    self.training_config.rllib_max_seq_len = defaults.rllib_max_seq_len;
                    true
                }
            }
            ConfigField::RllibFcnetHiddens => {
                if self.training_config.rllib_fcnet_hiddens == defaults.rllib_fcnet_hiddens {
                    false
                } else {
                    self.training_config.rllib_fcnet_hiddens = defaults.rllib_fcnet_hiddens.clone();
                    true
                }
            }
            ConfigField::RllibPolicyType => {
                if self.training_config.rllib_policy_type == defaults.rllib_policy_type {
                    false
                } else {
                    self.training_config.rllib_policy_type = defaults.rllib_policy_type;
                    self.rebuild_advanced_fields();
                    true
                }
            }
            ConfigField::RllibCnnChannels => {
                if self.training_config.rllib_cnn_channels == defaults.rllib_cnn_channels {
                    false
                } else {
                    self.training_config.rllib_cnn_channels = defaults.rllib_cnn_channels.clone();
                    true
                }
            }
            ConfigField::RllibLstmCellSize => {
                if self.training_config.rllib_lstm_cell_size == defaults.rllib_lstm_cell_size {
                    false
                } else {
                    self.training_config.rllib_lstm_cell_size = defaults.rllib_lstm_cell_size;
                    true
                }
            }
            ConfigField::RllibLstmNumLayers => {
                if self.training_config.rllib_lstm_num_layers == defaults.rllib_lstm_num_layers {
                    false
                } else {
                    self.training_config.rllib_lstm_num_layers = defaults.rllib_lstm_num_layers;
                    true
                }
            }
            ConfigField::RllibLstmIncludePrevActions => {
                if self.training_config.rllib_lstm_include_prev_actions
                    == defaults.rllib_lstm_include_prev_actions
                {
                    false
                } else {
                    self.training_config.rllib_lstm_include_prev_actions =
                        defaults.rllib_lstm_include_prev_actions;
                    true
                }
            }
            ConfigField::RllibGrnHiddenSize => {
                if self.training_config.rllib_grn_hidden_size == defaults.rllib_grn_hidden_size {
                    false
                } else {
                    self.training_config.rllib_grn_hidden_size = defaults.rllib_grn_hidden_size;
                    true
                }
            }
            ConfigField::RllibCheckpointFrequency => {
                if self.training_config.rllib_checkpoint_frequency
                    == defaults.rllib_checkpoint_frequency
                {
                    false
                } else {
                    self.training_config.rllib_checkpoint_frequency =
                        defaults.rllib_checkpoint_frequency;
                    true
                }
            }
            ConfigField::RllibResumeFrom => {
                if self.training_config.rllib_resume_from == defaults.rllib_resume_from {
                    false
                } else {
                    self.training_config.rllib_resume_from = defaults.rllib_resume_from.clone();
                    true
                }
            }
            ConfigField::RllibStopMode => {
                if self.training_config.rllib_stop_mode == defaults.rllib_stop_mode {
                    false
                } else {
                    self.training_config.rllib_stop_mode = defaults.rllib_stop_mode;
                    true
                }
            }
            ConfigField::RllibStopTimeSeconds => {
                if self.training_config.rllib_stop_time_seconds == defaults.rllib_stop_time_seconds
                {
                    false
                } else {
                    self.training_config.rllib_stop_time_seconds = defaults.rllib_stop_time_seconds;
                    true
                }
            }
        };

        if changed {
            self.persist_training_config()?;
            self.update_validation_status();
        }
        Ok(changed)
    }

    fn set_config_field_value(&mut self, field: ConfigField, value: &str) -> Result<()> {
        let trimmed = value.trim();
        match field {
            ConfigField::EnvPath => {
                self.training_config.env_path = trimmed.to_string();
            }
            ConfigField::Timesteps => {
                self.training_config.timesteps = trimmed
                    .parse()
                    .wrap_err("Timesteps must be a positive number")?;
            }
            ConfigField::ExperimentName => {
                if trimmed.is_empty() {
                    bail!("Experiment name cannot be empty");
                }
                self.training_config.experiment_name = trimmed.to_string();
            }
            ConfigField::Sb3PolicyType => {
                let Some(policy) = PolicyType::from_str(trimmed) else {
                    bail!("Unknown SB3 policy type '{trimmed}'");
                };
                if self.training_config.sb3_policy_type != policy {
                    self.training_config.sb3_policy_type = policy;
                    self.rebuild_advanced_fields();
                }
            }
            ConfigField::Sb3Speedup => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Speedup must be a positive integer")?;
                if val == 0 {
                    bail!("Speedup must be at least 1");
                }
                self.training_config.sb3_speedup = val;
            }
            ConfigField::Sb3NParallel => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("N parallel must be a positive integer")?;
                if val == 0 {
                    bail!("N parallel must be at least 1");
                }
                self.training_config.sb3_n_parallel = val;
            }
            ConfigField::Sb3Viz => {
                self.training_config.sb3_viz =
                    matches!(trimmed.to_lowercase().as_str(), "true" | "yes" | "1" | "on");
            }
            ConfigField::Sb3PolicyLayers => {
                let layers = parse_usize_list(trimmed)?;
                self.training_config.sb3_policy_layers = layers;
            }
            ConfigField::Sb3CnnChannels => {
                let layers = parse_usize_list(trimmed)?;
                self.training_config.sb3_cnn_channels = layers;
            }
            ConfigField::Sb3LstmHiddenSize => {
                let val: usize = trimmed
                    .parse()
                    .wrap_err("Hidden size must be a positive integer")?;
                if val == 0 {
                    bail!("Hidden size must be greater than 0");
                }
                self.training_config.sb3_lstm_hidden_size = val;
            }
            ConfigField::Sb3LstmNumLayers => {
                let val: usize = trimmed
                    .parse()
                    .wrap_err("Number of layers must be a positive integer")?;
                if val == 0 {
                    bail!("Number of layers must be at least 1");
                }
                self.training_config.sb3_lstm_num_layers = val;
            }
            ConfigField::Sb3GrnHiddenSize => {
                let val: usize = trimmed
                    .parse()
                    .wrap_err("Hidden size must be a positive integer")?;
                if val == 0 {
                    bail!("Hidden size must be greater than 0");
                }
                self.training_config.sb3_grn_hidden_size = val;
            }
            ConfigField::Sb3LearningRate => {
                let val: f64 = trimmed.parse().wrap_err("Learning rate must be a number")?;
                if val <= 0.0 {
                    bail!("Learning rate must be greater than 0");
                }
                self.training_config.sb3_learning_rate = val;
            }
            ConfigField::Sb3BatchSize => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Batch size must be a positive integer")?;
                if val == 0 {
                    bail!("Batch size must be at least 1");
                }
                self.training_config.sb3_batch_size = val;
            }
            ConfigField::Sb3NSteps => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("n_steps must be a positive integer")?;
                if val == 0 {
                    bail!("n_steps must be at least 1");
                }
                self.training_config.sb3_n_steps = val;
            }
            ConfigField::Sb3Gamma => {
                let val: f64 = trimmed.parse().wrap_err("Gamma must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("Gamma must be between 0 and 1");
                }
                self.training_config.sb3_gamma = val;
            }
            ConfigField::Sb3GaeLambda => {
                let val: f64 = trimmed.parse().wrap_err("GAE Lambda must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("GAE Lambda must be between 0 and 1");
                }
                self.training_config.sb3_gae_lambda = val;
            }
            ConfigField::Sb3EntCoef => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Entropy coefficient must be a number")?;
                if val < 0.0 {
                    bail!("Entropy coefficient cannot be negative");
                }
                self.training_config.sb3_ent_coef = val;
            }
            ConfigField::Sb3ClipRange => {
                let val: f64 = trimmed.parse().wrap_err("Clip range must be a number")?;
                if val <= 0.0 {
                    bail!("Clip range must be greater than 0");
                }
                self.training_config.sb3_clip_range = val;
            }
            ConfigField::Sb3VfCoef => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Value function coefficient must be a number")?;
                if val < 0.0 {
                    bail!("Value function coefficient cannot be negative");
                }
                self.training_config.sb3_vf_coef = val;
            }
            ConfigField::Sb3MaxGradNorm => {
                let val: f64 = trimmed.parse().wrap_err("Max grad norm must be a number")?;
                if val <= 0.0 {
                    bail!("Max grad norm must be greater than 0");
                }
                self.training_config.sb3_max_grad_norm = val;
            }
            ConfigField::RllibConfigFile => {
                if trimmed.is_empty() {
                    bail!("Config file path cannot be empty");
                }
                self.training_config.rllib_config_file = normalize_rllib_config_value(trimmed);
            }
            ConfigField::RllibShowWindow => {
                self.training_config.rllib_show_window =
                    matches!(trimmed.to_lowercase().as_str(), "true" | "yes" | "1" | "on");
            }
            ConfigField::RllibAlgorithm => {
                let Some(algo) = RllibAlgorithm::from_str(trimmed) else {
                    bail!("Unknown RLlib algorithm '{trimmed}'");
                };
                self.training_config.rllib_algorithm = algo;
            }
            ConfigField::RllibEnvActionRepeat => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Action repeat must be a positive integer")?;
                if val == 0 {
                    bail!("Action repeat must be at least 1");
                }
                self.training_config.rllib_env_action_repeat = val;
            }
            ConfigField::RllibEnvSpeedup => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Speedup must be a positive integer")?;
                if val == 0 {
                    bail!("Speedup must be at least 1");
                }
                self.training_config.rllib_env_speedup = val;
            }
            ConfigField::RllibNumWorkers => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Number of workers must be a positive integer")?;
                if val == 0 {
                    bail!("Number of workers must be at least 1");
                }
                self.training_config.rllib_num_workers = val;
            }
            ConfigField::RllibNumEnvWorkers => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Environments per worker must be a positive integer")?;
                if val == 0 {
                    bail!("Environments per worker must be at least 1");
                }
                self.training_config.rllib_num_envs_per_worker = val;
            }
            ConfigField::RllibTrainBatchSize => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Train batch size must be a positive integer")?;
                if val == 0 {
                    bail!("Train batch size must be at least 1");
                }
                self.training_config.rllib_train_batch_size = val;
            }
            ConfigField::RllibSgdMinibatchSize => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("SGD minibatch size must be a positive integer")?;
                if val == 0 {
                    bail!("SGD minibatch size must be at least 1");
                }
                self.training_config.rllib_sgd_minibatch_size = val;
            }
            ConfigField::RllibNumSgdIter => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Number of SGD iterations must be a positive integer")?;
                if val == 0 {
                    bail!("Number of SGD iterations must be at least 1");
                }
                self.training_config.rllib_num_sgd_iter = val;
            }
            ConfigField::RllibLr => {
                let val: f64 = trimmed.parse().wrap_err("Learning rate must be a number")?;
                if val <= 0.0 {
                    bail!("Learning rate must be greater than 0");
                }
                self.training_config.rllib_lr = val;
            }
            ConfigField::RllibGamma => {
                let val: f64 = trimmed.parse().wrap_err("Gamma must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("Gamma must be between 0 and 1");
                }
                self.training_config.rllib_gamma = val;
            }
            ConfigField::RllibLambda => {
                let val: f64 = trimmed.parse().wrap_err("Lambda must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("Lambda must be between 0 and 1");
                }
                self.training_config.rllib_lambda = val;
            }
            ConfigField::RllibClipParam => {
                let val: f64 = trimmed.parse().wrap_err("Clip param must be a number")?;
                if val <= 0.0 {
                    bail!("Clip param must be greater than 0");
                }
                self.training_config.rllib_clip_param = val;
            }
            ConfigField::RllibEntropyCoeff => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Entropy coefficient must be a number")?;
                if val < 0.0 {
                    bail!("Entropy coefficient cannot be negative");
                }
                self.training_config.rllib_entropy_coeff = val;
            }
            ConfigField::RllibVfLossCoeff => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("VF loss coefficient must be a number")?;
                if val < 0.0 {
                    bail!("VF loss coefficient cannot be negative");
                }
                self.training_config.rllib_vf_loss_coeff = val;
            }
            ConfigField::RllibGradClip => {
                let val: f64 = trimmed.parse().wrap_err("Grad clip must be a number")?;
                if val <= 0.0 {
                    bail!("Grad clip must be greater than 0");
                }
                self.training_config.rllib_grad_clip = val;
            }
            ConfigField::RllibFramework => {
                if trimmed.is_empty() {
                    bail!("Framework cannot be empty");
                }
                self.training_config.rllib_framework = trimmed.to_string();
            }
            ConfigField::RllibActivation => {
                if trimmed.is_empty() {
                    bail!("Activation cannot be empty");
                }
                self.training_config.rllib_activation = trimmed.to_string();
            }
            ConfigField::RllibBatchMode => {
                if trimmed.is_empty() {
                    bail!("Batch mode cannot be empty");
                }
                let lower = trimmed.to_lowercase();
                if lower != "truncate_episodes" && lower != "complete_episodes" {
                    bail!("Batch mode must be 'truncate_episodes' or 'complete_episodes'");
                }
                self.training_config.rllib_batch_mode = lower;
            }
            ConfigField::RllibRolloutFragmentLength => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Rollout fragment length must be a positive integer")?;
                if val == 0 {
                    bail!("Rollout fragment length must be at least 1");
                }
                self.training_config.rllib_rollout_fragment_length = val;
            }
            ConfigField::RllibNumGpus => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Number of GPUs must be a number")?;
                if val < 0.0 {
                    bail!("Number of GPUs cannot be negative");
                }
                self.training_config.rllib_num_gpus = val;
            }
            ConfigField::RllibMaxSeqLen => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Max sequence length must be a positive integer")?;
                if val == 0 {
                    bail!("Max sequence length must be at least 1");
                }
                self.training_config.rllib_max_seq_len = val;
            }
            ConfigField::RllibFcnetHiddens => {
                let layers = parse_usize_list(trimmed)?;
                self.training_config.rllib_fcnet_hiddens = layers;
            }
            ConfigField::RllibPolicyType => {
                let Some(policy) = PolicyType::from_str(trimmed) else {
                    bail!("Unknown RLlib policy type '{trimmed}'");
                };
                if self.training_config.rllib_policy_type != policy {
                    self.training_config.rllib_policy_type = policy;
                    self.rebuild_advanced_fields();
                }
            }
            ConfigField::RllibCnnChannels => {
                let layers = parse_usize_list(trimmed)?;
                self.training_config.rllib_cnn_channels = layers;
            }
            ConfigField::RllibLstmCellSize => {
                let val: usize = trimmed
                    .parse()
                    .wrap_err("Hidden size must be a positive integer")?;
                if val == 0 {
                    bail!("Hidden size must be greater than 0");
                }
                self.training_config.rllib_lstm_cell_size = val;
            }
            ConfigField::RllibLstmNumLayers => {
                let val: usize = trimmed
                    .parse()
                    .wrap_err("Number of layers must be a positive integer")?;
                if val == 0 {
                    bail!("Number of layers must be at least 1");
                }
                self.training_config.rllib_lstm_num_layers = val;
            }
            ConfigField::RllibLstmIncludePrevActions => {
                self.training_config.rllib_lstm_include_prev_actions =
                    matches!(trimmed.to_lowercase().as_str(), "true" | "yes" | "1" | "on");
            }
            ConfigField::RllibGrnHiddenSize => {
                let val: usize = trimmed
                    .parse()
                    .wrap_err("Hidden size must be a positive integer")?;
                if val == 0 {
                    bail!("Hidden size must be greater than 0");
                }
                self.training_config.rllib_grn_hidden_size = val;
            }
            ConfigField::RllibCheckpointFrequency => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Checkpoint frequency must be a positive integer")?;
                if val == 0 {
                    bail!("Checkpoint frequency must be at least 1");
                }
                self.training_config.rllib_checkpoint_frequency = val;
            }
            ConfigField::RllibResumeFrom => {
                self.training_config.rllib_resume_from = trimmed.to_string();
            }
            ConfigField::RllibStopMode => {
                let mode = trimmed.to_lowercase();
                let stop_mode = match mode.as_str() {
                    "time" | "time_seconds" | "seconds" | "s" => RllibStopMode::TimeSeconds,
                    "timesteps" | "steps" | "timesteps_total" | "t" => RllibStopMode::Timesteps,
                    _ => bail!("Stop mode must be 'time' or 'timesteps'"),
                };
                self.training_config.rllib_stop_mode = stop_mode;
            }
            ConfigField::RllibStopTimeSeconds => {
                let val: u64 = trimmed
                    .parse()
                    .wrap_err("Time limit must be a positive integer of seconds")?;
                if val == 0 {
                    bail!("Time limit must be at least 1 second");
                }
                self.training_config.rllib_stop_time_seconds = val;
            }
        }
        Ok(())
    }

    fn persist_training_config(&mut self) -> Result<()> {
        if let Some(path) = self.training_config_path() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).wrap_err_with(|| {
                    format!("failed to create config directory {}", parent.display())
                })?;
            }
            let json = serde_json::to_string_pretty(&self.training_config).wrap_err_with(|| {
                format!("failed to serialize training config for {}", path.display())
            })?;
            fs::write(&path, json).wrap_err_with(|| {
                format!("failed to write training config to {}", path.display())
            })?;
        }
        Ok(())
    }

    fn training_config_path(&self) -> Option<PathBuf> {
        self.active_project.as_ref().map(|project| {
            project
                .root_path
                .join(PROJECT_CONFIG_DIR)
                .join(TRAINING_CONFIG_FILENAME)
        })
    }

    fn load_training_config_for_active_project(&mut self) -> bool {
        let mut had_error = false;
        if let Some(path) = self.training_config_path() {
            if path.exists() {
                match fs::read_to_string(&path) {
                    Ok(contents) => match serde_json::from_str::<TrainingConfig>(&contents) {
                        Ok(config) => {
                            self.training_config = config;
                            if let Some(previous) = self.normalize_rllib_config_file_setting() {
                                if let Some(project) = &self.active_project {
                                    if let Err(error) = self
                                        .migrate_rllib_config_file(&project.root_path, &previous)
                                    {
                                        self.set_status(
                                            format!(
                                                "Failed to move RLlib config into .rlcontroller: {}",
                                                error
                                            ),
                                            StatusKind::Warning,
                                        );
                                        had_error = true;
                                    }
                                }
                                if let Err(error) = self.persist_training_config() {
                                    self.set_status(
                                        format!("Failed to update RLlib config path: {}", error),
                                        StatusKind::Warning,
                                    );
                                    had_error = true;
                                }
                            }
                        }
                        Err(error) => {
                            self.training_config = TrainingConfig::default();
                            self.set_status(
                                format!("Invalid training config, using defaults: {}", error),
                                StatusKind::Warning,
                            );
                            had_error = true;
                        }
                    },
                    Err(error) => {
                        self.training_config = TrainingConfig::default();
                        self.set_status(
                            format!("Failed to read training config, using defaults: {}", error),
                            StatusKind::Warning,
                        );
                        had_error = true;
                    }
                }
            } else {
                self.training_config = TrainingConfig::default();
                if let Err(error) = self.persist_training_config() {
                    self.set_status(
                        format!("Failed to create training config: {error}"),
                        StatusKind::Error,
                    );
                    had_error = true;
                }
            }
        } else {
            self.training_config = TrainingConfig::default();
        }
        self.rebuild_advanced_fields();
        self.update_validation_status();
        had_error
    }

    fn persist_export_state(&mut self) -> Result<()> {
        if let Some(path) = self.export_config_path() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).wrap_err_with(|| {
                    format!(
                        "failed to create export config directory {}",
                        parent.display()
                    )
                })?;
            }
            let state = ExportState {
                mode: self.export_mode,
                config: self.export_config.clone(),
            };
            let json = serde_json::to_string_pretty(&state).wrap_err_with(|| {
                format!("failed to serialize export config for {}", path.display())
            })?;
            fs::write(&path, json)
                .wrap_err_with(|| format!("failed to write export config to {}", path.display()))?;
        }
        Ok(())
    }

    fn export_config_path(&self) -> Option<PathBuf> {
        self.active_project.as_ref().map(|project| {
            project
                .root_path
                .join(PROJECT_CONFIG_DIR)
                .join(EXPORT_CONFIG_FILENAME)
        })
    }

    fn load_export_state_for_active_project(&mut self) -> bool {
        let mut had_error = false;
        if let Some(path) = self.export_config_path() {
            if path.exists() {
                match fs::read_to_string(&path) {
                    Ok(contents) => {
                        let parsed_state =
                            serde_json::from_str::<ExportState>(&contents).or_else(|_| {
                                serde_json::from_str::<ExportConfig>(&contents).map(|config| {
                                    ExportState {
                                        mode: ExportMode::default(),
                                        config,
                                    }
                                })
                            });
                        match parsed_state {
                            Ok(state) => {
                                self.export_mode = state.mode;
                                self.export_config = state.config;
                                let checkpoint_path =
                                    self.export_config.rllib_checkpoint_path.clone();
                                self.sync_checkpoint_number_from_path(&checkpoint_path);
                            }
                            Err(error) => {
                                self.export_mode = ExportMode::default();
                                self.export_config = ExportConfig::default();
                                self.set_status(
                                    format!("Invalid export config, using defaults: {}", error),
                                    StatusKind::Warning,
                                );
                                had_error = true;
                            }
                        }
                    }
                    Err(error) => {
                        self.export_mode = ExportMode::default();
                        self.export_config = ExportConfig::default();
                        self.set_status(
                            format!("Failed to read export config, using defaults: {error}"),
                            StatusKind::Warning,
                        );
                        had_error = true;
                    }
                }
            } else {
                self.export_mode = ExportMode::default();
                self.export_config = ExportConfig::default();
                if let Err(error) = self.persist_export_state() {
                    self.set_status(
                        format!("Failed to create export config: {error}"),
                        StatusKind::Error,
                    );
                    had_error = true;
                }
            }
        } else {
            self.export_mode = ExportMode::default();
            self.export_config = ExportConfig::default();
        }
        self.rebuild_export_fields();
        had_error
    }

    fn rebuild_advanced_fields(&mut self) {
        self.advanced_fields = self.build_advanced_fields();
        if self.advanced_fields.is_empty() {
            self.advanced_selection = 0;
        } else if self.advanced_selection >= self.advanced_fields.len() {
            self.advanced_selection = self.advanced_fields.len() - 1;
        }
    }

    fn collect_advanced_validation_errors(&self) -> HashMap<ConfigField, String> {
        let mut errors = HashMap::new();

        if self.training_config.mode != TrainingMode::MultiAgent {
            return errors;
        }

        let cfg = &self.training_config;
        let total_envs = std::cmp::max(1, cfg.rllib_num_workers) * cfg.rllib_num_envs_per_worker;
        let expected_batch_size =
            cfg.rllib_rollout_fragment_length.saturating_mul(total_envs);

        fn check_range(
            errors: &mut HashMap<ConfigField, String>,
            field: ConfigField,
            value: f64,
            min: f64,
            max: f64,
            label: &str,
        ) {
            if value < min || value > max {
                errors.insert(field, format!("{label} should be between {min} and {max}"));
            }
        }

        if cfg.rllib_num_workers > 64 {
            errors.insert(
                ConfigField::RllibNumWorkers,
                "Workers should be between 0 and 64".to_string(),
            );
        }

        if !(1..=16).contains(&cfg.rllib_num_envs_per_worker) {
            errors.insert(
                ConfigField::RllibNumEnvWorkers,
                "Envs per worker should be between 1 and 16".to_string(),
            );
        }

        if !(32..=4096).contains(&cfg.rllib_rollout_fragment_length) {
            errors.insert(
                ConfigField::RllibRolloutFragmentLength,
                "Rollout fragment length should be between 32 and 4096".to_string(),
            );
        }

        if !(128..=200_000).contains(&cfg.rllib_train_batch_size) {
            errors.insert(
                ConfigField::RllibTrainBatchSize,
                "Train batch size should be between 128 and 200000".to_string(),
            );
        }

        if !(64..=1024).contains(&cfg.rllib_sgd_minibatch_size) {
            errors.insert(
                ConfigField::RllibSgdMinibatchSize,
                "Minibatch size should be between 64 and 1024".to_string(),
            );
        }

        if cfg.rllib_train_batch_size < expected_batch_size
            || cfg.rllib_train_batch_size % expected_batch_size != 0
        {
            let next_multiple = ((cfg.rllib_train_batch_size.max(expected_batch_size)
                + expected_batch_size
                - 1)
                / expected_batch_size)
                * expected_batch_size;
            errors.insert(
                ConfigField::RllibTrainBatchSize,
                format!(
                    "Train batch must be a multiple of rollout_fragment_length × workers × envs ({} × {} × {}) = {}. Currently {}. Try {} or {}.",
                    cfg.rllib_rollout_fragment_length,
                    std::cmp::max(1, cfg.rllib_num_workers),
                    cfg.rllib_num_envs_per_worker,
                    expected_batch_size,
                    cfg.rllib_train_batch_size,
                    next_multiple,
                    next_multiple + expected_batch_size
                ),
            );
        }

        if !(1..=50).contains(&cfg.rllib_num_sgd_iter) {
            errors.insert(
                ConfigField::RllibNumSgdIter,
                "SGD iterations should be between 1 and 50".to_string(),
            );
        }

        check_range(
            &mut errors,
            ConfigField::RllibLr,
            cfg.rllib_lr,
            0.00001,
            0.01,
            "Learning rate",
        );
        check_range(
            &mut errors,
            ConfigField::RllibGamma,
            cfg.rllib_gamma,
            0.9,
            0.9999,
            "Gamma",
        );
        check_range(
            &mut errors,
            ConfigField::RllibLambda,
            cfg.rllib_lambda,
            0.8,
            1.0,
            "Lambda",
        );
        check_range(
            &mut errors,
            ConfigField::RllibClipParam,
            cfg.rllib_clip_param,
            0.1,
            0.3,
            "Clip param",
        );
        check_range(
            &mut errors,
            ConfigField::RllibEntropyCoeff,
            cfg.rllib_entropy_coeff,
            0.0,
            0.1,
            "Entropy coeff",
        );
        check_range(
            &mut errors,
            ConfigField::RllibVfLossCoeff,
            cfg.rllib_vf_loss_coeff,
            0.1,
            1.0,
            "VF loss coeff",
        );
        check_range(
            &mut errors,
            ConfigField::RllibGradClip,
            cfg.rllib_grad_clip,
            0.1,
            10.0,
            "Grad clip",
        );

        if cfg.rllib_policy_type == PolicyType::Lstm
            && cfg.rllib_max_seq_len > cfg.rllib_rollout_fragment_length
        {
            errors.insert(
                ConfigField::RllibRolloutFragmentLength,
                format!(
                    "Rollout fragment length ({}) should be at least the LSTM max_seq_len ({})",
                    cfg.rllib_rollout_fragment_length, cfg.rllib_max_seq_len
                ),
            );
        }

        if cfg.rllib_policy_type == PolicyType::Lstm
            && cfg.rllib_sgd_minibatch_size <= cfg.rllib_max_seq_len
        {
            errors.insert(
                ConfigField::RllibSgdMinibatchSize,
                format!(
                    "Minibatch size ({}) must be greater than the LSTM max_seq_len ({})",
                    cfg.rllib_sgd_minibatch_size, cfg.rllib_max_seq_len
                ),
            );
        }

        let framework = cfg.rllib_framework.to_lowercase();
        if framework != "torch" && framework != "tf2" {
            errors.insert(
                ConfigField::RllibFramework,
                "Framework should be 'torch' or 'tf2'".to_string(),
            );
        }

        let activation = cfg.rllib_activation.to_lowercase();
        if activation != "relu" && activation != "tanh" && activation != "elu" {
            errors.insert(
                ConfigField::RllibActivation,
                "Activation should be relu, tanh, or elu".to_string(),
            );
        }

        if cfg.rllib_checkpoint_frequency > 1000 {
            errors.insert(
                ConfigField::RllibCheckpointFrequency,
                "Checkpoint frequency should be between 1 and 1000 (0 disables)".to_string(),
            );
        }

        if cfg.rllib_stop_mode == RllibStopMode::TimeSeconds && cfg.rllib_stop_time_seconds == 0 {
            errors.insert(
                ConfigField::RllibStopTimeSeconds,
                "Time limit must be at least 1 second".to_string(),
            );
        }

        errors
    }

    fn build_advanced_fields(&self) -> Vec<ConfigField> {
        match self.training_config.mode {
            TrainingMode::SingleAgent => {
                let mut fields = vec![ConfigField::Sb3PolicyType, ConfigField::Sb3PolicyLayers];
                match self.training_config.sb3_policy_type {
                    PolicyType::Cnn => fields.push(ConfigField::Sb3CnnChannels),
                    PolicyType::Lstm => {
                        fields.push(ConfigField::Sb3LstmHiddenSize);
                        fields.push(ConfigField::Sb3LstmNumLayers);
                    }
                    PolicyType::Grn => fields.push(ConfigField::Sb3GrnHiddenSize),
                    PolicyType::Mlp => {}
                }
                fields.extend_from_slice(&[
                    ConfigField::Sb3Speedup,
                    ConfigField::Sb3NParallel,
                    ConfigField::Sb3Viz,
                    ConfigField::Sb3LearningRate,
                    ConfigField::Sb3BatchSize,
                    ConfigField::Sb3NSteps,
                    ConfigField::Sb3Gamma,
                    ConfigField::Sb3GaeLambda,
                    ConfigField::Sb3EntCoef,
                    ConfigField::Sb3ClipRange,
                    ConfigField::Sb3VfCoef,
                    ConfigField::Sb3MaxGradNorm,
                ]);
                fields
            }
            TrainingMode::MultiAgent => {
                let mut fields = vec![
                    ConfigField::RllibConfigFile,
                    ConfigField::RllibStopMode,
                    ConfigField::RllibStopTimeSeconds,
                    ConfigField::RllibResumeFrom,
                    ConfigField::RllibShowWindow,
                    ConfigField::RllibAlgorithm,
                    ConfigField::RllibEnvActionRepeat,
                    ConfigField::RllibEnvSpeedup,
                    ConfigField::RllibPolicyType,
                    ConfigField::RllibFcnetHiddens,
                ];
                match self.training_config.rllib_policy_type {
                    PolicyType::Cnn => fields.push(ConfigField::RllibCnnChannels),
                    PolicyType::Lstm => {
                        fields.push(ConfigField::RllibLstmCellSize);
                        fields.push(ConfigField::RllibLstmNumLayers);
                        fields.push(ConfigField::RllibMaxSeqLen);
                        fields.push(ConfigField::RllibLstmIncludePrevActions);
                    }
                    PolicyType::Grn => fields.push(ConfigField::RllibGrnHiddenSize),
                    PolicyType::Mlp => {}
                }
                fields.extend_from_slice(&[
                    ConfigField::RllibNumGpus,
                    ConfigField::RllibNumWorkers,
                    ConfigField::RllibNumEnvWorkers,
                    ConfigField::RllibTrainBatchSize,
                    ConfigField::RllibSgdMinibatchSize,
                    ConfigField::RllibNumSgdIter,
                    ConfigField::RllibLr,
                    ConfigField::RllibGamma,
                    ConfigField::RllibLambda,
                    ConfigField::RllibClipParam,
                    ConfigField::RllibEntropyCoeff,
                    ConfigField::RllibVfLossCoeff,
                    ConfigField::RllibGradClip,
                    ConfigField::RllibFramework,
                    ConfigField::RllibActivation,
                    ConfigField::RllibBatchMode,
                    ConfigField::RllibRolloutFragmentLength,
                    ConfigField::RllibCheckpointFrequency,
                ]);
                fields
            }
        }
    }

    pub fn start_config_file_browser(&mut self, field: ConfigField) {
        let kind = match field {
            ConfigField::EnvPath => FileBrowserKind::ExistingFile {
                extensions: Vec::new(),
            },
            ConfigField::RllibConfigFile => FileBrowserKind::ExistingFile {
                extensions: vec!["yaml".into(), "yml".into(), "json".into()],
            },
            ConfigField::RllibResumeFrom => FileBrowserKind::Directory {
                allow_create: false,
                require_checkpoints: false,
            },
            _ => FileBrowserKind::ExistingFile {
                extensions: Vec::new(),
            },
        };
        self.start_file_browser(FileBrowserTarget::Config(field), kind, None);
    }

    pub fn start_export_file_browser(&mut self, field: ExportField) {
        let (kind, suggested_name) = match field {
            ExportField::Sb3ModelPath => (
                FileBrowserKind::ExistingFile {
                    extensions: vec!["zip".into()],
                },
                None,
            ),
            ExportField::Sb3OutputPath => (
                FileBrowserKind::OutputFile {
                    extension: Some("onnx".into()),
                },
                self.default_sb3_output_name(),
            ),
            ExportField::RllibCheckpointPath => (
                FileBrowserKind::Directory {
                    allow_create: false,
                    require_checkpoints: true,
                },
                None,
            ),
            ExportField::RllibOutputDir => (
                FileBrowserKind::Directory {
                    allow_create: true,
                    require_checkpoints: false,
                },
                None,
            ),
            _ => (
                FileBrowserKind::ExistingFile {
                    extensions: Vec::new(),
                },
                None,
            ),
        };
        self.start_file_browser(FileBrowserTarget::Export(field), kind, suggested_name);
    }

    fn default_sb3_output_name(&self) -> Option<String> {
        if !self.export_config.sb3_output_path.trim().is_empty() {
            let path = PathBuf::from(self.export_config.sb3_output_path.trim());
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                return Some(name.to_string());
            }
        }

        if !self.export_config.sb3_model_path.trim().is_empty() {
            let path = PathBuf::from(self.export_config.sb3_model_path.trim());
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                return Some(format!("{stem}.onnx"));
            }
        }

        Some(String::from("export.onnx"))
    }

    fn start_file_browser(
        &mut self,
        target: FileBrowserTarget,
        kind: FileBrowserKind,
        suggested_name: Option<String>,
    ) {
        self.input_mode = InputMode::BrowsingFiles;
        self.file_browser_target = Some(target);
        self.file_browser_kind = kind;
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_default_name = suggested_name;
        self.file_browser_input.clear();
        self.file_browser_selected = 0;

        self.file_browser_path = self.determine_browser_start_path(&target);
        self.refresh_file_browser();
    }

    fn determine_browser_start_path(&self, target: &FileBrowserTarget) -> PathBuf {
        let (project_root, project_logs) = if let Some(project) = &self.active_project {
            (project.root_path.clone(), project.logs_path.clone())
        } else {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
            (cwd.clone(), cwd.join("logs"))
        };

        match target {
            FileBrowserTarget::Config(ConfigField::EnvPath) => self
                .resolve_existing_path(&self.training_config.env_path)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Config(ConfigField::RllibConfigFile) => self
                .resolve_existing_path(&self.training_config.rllib_config_file)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Config(ConfigField::RllibResumeFrom) => self
                .resolve_existing_path(&self.training_config.rllib_resume_from)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Config(_) => project_root.clone(),
            FileBrowserTarget::Export(ExportField::Sb3ModelPath) => self
                .resolve_existing_path(&self.export_config.sb3_model_path)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Export(ExportField::Sb3OutputPath) => {
                let current = self
                    .resolve_existing_path(&self.export_config.sb3_output_path)
                    .and_then(|path| {
                        if path.is_file() {
                            path.parent().map(|p| p.to_path_buf())
                        } else {
                            Some(path)
                        }
                    });
                current
                    .or_else(|| {
                        self.resolve_existing_path(&self.export_config.sb3_model_path)
                            .and_then(|p| p.parent().map(|parent| parent.to_path_buf()))
                    })
                    .unwrap_or_else(|| project_root.clone())
            }
            FileBrowserTarget::Export(ExportField::RllibCheckpointPath) => self
                .resolve_existing_path(&self.export_config.rllib_checkpoint_path)
                .unwrap_or_else(|| project_logs.clone()),
            FileBrowserTarget::Export(ExportField::RllibOutputDir) => self
                .resolve_existing_path(&self.export_config.rllib_output_dir)
                .unwrap_or_else(|| project_root.join("onnx_exports")),
            FileBrowserTarget::Export(_) => project_root.clone(),
            FileBrowserTarget::SimulatorEnvPath => self
                .resolve_existing_path(&self.simulator_config.env_path)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::InterfaceAgentPath => self
                .resolve_existing_path(&self.interface_config.agent_path)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::ProjectLocation => {
                let raw = self.project_location_buffer.trim();
                let mut candidate = if raw.is_empty() {
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
                } else {
                    PathBuf::from(raw)
                };
                if !candidate.exists() {
                    if let Some(parent) = candidate.parent() {
                        candidate = parent.to_path_buf();
                    } else {
                        candidate = PathBuf::from("/");
                    }
                }
                candidate
            }
            FileBrowserTarget::SavedRun => self
                .active_project
                .as_ref()
                .map(|project| project.runs_dir())
                .unwrap_or_else(|| project_root.clone()),
        }
    }

    fn resolve_existing_path(&self, value: &str) -> Option<PathBuf> {
        if value.trim().is_empty() {
            return None;
        }
        let path = PathBuf::from(value.trim());
        if path.is_absolute() {
            Some(path)
        } else if let Some(project) = &self.active_project {
            Some(project.root_path.join(path))
        } else {
            std::env::current_dir().ok().map(|cwd| cwd.join(path))
        }
    }

    pub fn cancel_file_browser(&mut self) {
        self.input_mode = InputMode::Normal;
        self.file_browser_target = None;
        self.file_browser_entries.clear();
        self.file_browser_selected = 0;
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.file_browser_default_name = None;
    }

    pub fn file_browser_path(&self) -> &Path {
        &self.file_browser_path
    }

    pub fn file_browser_entries(&self) -> &[FileBrowserEntry] {
        &self.file_browser_entries
    }

    pub fn file_browser_selected(&self) -> usize {
        self.file_browser_selected
    }

    pub fn file_browser_state(&self) -> FileBrowserState {
        self.file_browser_state
    }

    pub fn file_browser_input(&self) -> &str {
        &self.file_browser_input
    }

    pub fn file_browser_kind(&self) -> &FileBrowserKind {
        &self.file_browser_kind
    }

    pub fn file_browser_select_next(&mut self) {
        if !self.file_browser_entries.is_empty() {
            self.file_browser_selected =
                (self.file_browser_selected + 1) % self.file_browser_entries.len();
        }
    }

    pub fn file_browser_select_previous(&mut self) {
        if !self.file_browser_entries.is_empty() {
            if self.file_browser_selected == 0 {
                self.file_browser_selected = self.file_browser_entries.len() - 1;
            } else {
                self.file_browser_selected -= 1;
            }
        }
    }

    pub fn file_browser_enter(&mut self) {
        if let Some(entry) = self
            .file_browser_entries
            .get(self.file_browser_selected)
            .cloned()
        {
            match entry {
                FileBrowserEntry::Parent(path) | FileBrowserEntry::Directory(path) => {
                    self.file_browser_path = path;
                    self.file_browser_selected = 0;
                    self.refresh_file_browser();
                }
                FileBrowserEntry::File(path) => {
                    if matches!(self.file_browser_kind, FileBrowserKind::ExistingFile { .. })
                        || matches!(self.file_browser_kind, FileBrowserKind::OutputFile { .. })
                    {
                        self.finalize_file_selection(path);
                    }
                }
            }
        }
    }

    pub fn file_browser_go_up(&mut self) {
        if let Some(parent) = self.file_browser_path.parent() {
            self.file_browser_path = parent.to_path_buf();
            self.file_browser_selected = 0;
            self.refresh_file_browser();
        }
    }

    pub fn file_browser_finalize_selection(&mut self) {
        match self.file_browser_kind.clone() {
            FileBrowserKind::Directory { .. } => {
                let target_path = self.file_browser_path.clone();
                self.finalize_directory_selection(target_path);
            }
            FileBrowserKind::ExistingFile { extensions } => {
                if let Some(entry) = self.file_browser_entries.get(self.file_browser_selected) {
                    if let FileBrowserEntry::File(path) = entry {
                        if self.extension_allowed(path, &extensions) {
                            self.finalize_file_selection(path.clone());
                            return;
                        }
                    }
                }
                self.set_status(
                    "Select a file that matches the required type",
                    StatusKind::Warning,
                );
            }
            FileBrowserKind::OutputFile { extension } => {
                if let Some(entry) = self.file_browser_entries.get(self.file_browser_selected) {
                    match entry {
                        FileBrowserEntry::File(path) => {
                            if extension
                                .as_ref()
                                .map(|ext| self.path_has_extension(path, ext))
                                .unwrap_or(true)
                            {
                                self.finalize_file_selection(path.clone());
                                return;
                            }
                        }
                        FileBrowserEntry::Directory(path) | FileBrowserEntry::Parent(path) => {
                            self.file_browser_path = path.clone();
                            self.file_browser_selected = 0;
                            self.refresh_file_browser();
                            self.begin_file_naming(extension);
                            return;
                        }
                    }
                }
                self.begin_file_naming(extension);
            }
        }
    }

    fn begin_file_naming(&mut self, extension: Option<String>) {
        self.file_browser_state = FileBrowserState::NamingFile;
        self.file_browser_input =
            self.file_browser_default_name
                .clone()
                .unwrap_or_else(|| match extension.as_deref() {
                    Some(ext) => format!("export.{ext}"),
                    None => String::from("export"),
                });
        if let Some(ext) = extension {
            if !self.file_browser_input.ends_with(&format!(".{ext}")) {
                self.file_browser_input.push('.');
                self.file_browser_input.push_str(&ext);
            }
        }
        self.set_status(
            "Type a file name and press Enter to confirm",
            StatusKind::Info,
        );
    }

    pub fn file_browser_begin_new_folder(&mut self) {
        self.file_browser_state = FileBrowserState::NamingFolder;
        self.file_browser_input.clear();
        self.set_status("Enter a name for the new folder", StatusKind::Info);
    }

    pub fn file_browser_push_char(&mut self, ch: char) {
        if matches!(
            self.file_browser_state,
            FileBrowserState::NamingFolder | FileBrowserState::NamingFile
        ) && !ch.is_control()
            && self.file_browser_input.len() < 128
        {
            self.file_browser_input.push(ch);
        }
    }

    pub fn file_browser_pop_char(&mut self) {
        self.file_browser_input.pop();
    }

    pub fn file_browser_cancel_input(&mut self) {
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.set_status("Selection cancelled", StatusKind::Info);
    }

    pub fn file_browser_confirm_input(&mut self) {
        match self.file_browser_state {
            FileBrowserState::NamingFolder => self.confirm_new_folder(),
            FileBrowserState::NamingFile => self.confirm_new_file(),
            FileBrowserState::Browsing => {}
        }
    }

    fn confirm_new_folder(&mut self) {
        let name = self.file_browser_input.trim();
        if name.is_empty() {
            self.set_status("Folder name cannot be empty", StatusKind::Warning);
            return;
        }
        if name.contains(std::path::MAIN_SEPARATOR) || name.contains('/') {
            self.set_status(
                "Folder name cannot contain path separators",
                StatusKind::Warning,
            );
            return;
        }
        let new_path = self.file_browser_path.join(name);
        if new_path.exists() {
            self.set_status(
                "A folder with that name already exists",
                StatusKind::Warning,
            );
            return;
        }
        if let Err(error) = fs::create_dir_all(&new_path) {
            self.set_status(
                format!("Failed to create folder: {error}"),
                StatusKind::Error,
            );
            return;
        }

        self.set_status("Folder created", StatusKind::Success);
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.refresh_file_browser();
        if let Some(index) = self.file_browser_entries.iter().position(
            |entry| matches!(entry, FileBrowserEntry::Directory(path) if path == &new_path),
        ) {
            self.file_browser_selected = index;
        }
    }

    fn confirm_new_file(&mut self) {
        let name = self.file_browser_input.trim();
        if name.is_empty() {
            self.set_status("File name cannot be empty", StatusKind::Warning);
            return;
        }
        let extension_check =
            if let FileBrowserKind::OutputFile { extension } = &self.file_browser_kind {
                if let Some(ext) = extension {
                    if !name.ends_with(&format!(".{ext}")) {
                        self.set_status(
                            format!("File name must end with .{ext}"),
                            StatusKind::Warning,
                        );
                        return;
                    }
                }
                true
            } else {
                true
            };

        if !extension_check {
            return;
        }

        let path = self.file_browser_path.join(name);
        if let Some(parent) = path.parent() {
            if let Err(error) = fs::create_dir_all(parent) {
                self.set_status(
                    format!("Failed to prepare directory: {error}"),
                    StatusKind::Error,
                );
                return;
            }
        }

        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.finalize_file_selection(path);
    }

    fn finalize_directory_selection(&mut self, path: PathBuf) {
        if let Some(FileBrowserTarget::Export(ExportField::RllibCheckpointPath)) =
            self.file_browser_target
        {
            if let Err(error) = self.validate_rllib_checkpoint_dir(&path) {
                self.set_status(
                    format!("Invalid RLlib checkpoint directory: {error}"),
                    StatusKind::Warning,
                );
                return;
            }
        }

        self.apply_browser_selection(path);
    }

    fn finalize_file_selection(&mut self, path: PathBuf) {
        match self.file_browser_kind {
            FileBrowserKind::ExistingFile { ref extensions } => {
                if !self.extension_allowed(&path, extensions) {
                    self.set_status("File type not allowed", StatusKind::Warning);
                    return;
                }
            }
            FileBrowserKind::OutputFile { ref extension } => {
                if let Some(ext) = extension {
                    if !self.path_has_extension(&path, ext) {
                        self.set_status(
                            format!("File must have .{ext} extension"),
                            StatusKind::Warning,
                        );
                        return;
                    }
                }
            }
            FileBrowserKind::Directory { .. } => {}
        }

        self.apply_browser_selection(path);
    }

    fn apply_browser_selection(&mut self, path: PathBuf) {
        match self.file_browser_target {
            Some(FileBrowserTarget::ProjectLocation) => {
                self.cancel_file_browser();
                self.input_mode = InputMode::CreatingProject;
                if let Err(error) = self.finish_project_location_selection(path) {
                    self.set_status(
                        format!("Failed to set project location: {error}"),
                        StatusKind::Error,
                    );
                }
                return;
            }
            Some(FileBrowserTarget::SavedRun) => {
                self.cancel_file_browser();
                if let Err(error) = self.load_run_overlay_from_path(path.clone()) {
                    self.set_status(format!("Failed to load run: {error}"), StatusKind::Error);
                }
                return;
            }
            Some(FileBrowserTarget::Config(field)) => {
                let stored_value = self.stringify_for_storage(&path);
                if let Err(error) = self.apply_config_browser_selection(field, stored_value) {
                    self.set_status(
                        format!("Failed to apply selection: {error}"),
                        StatusKind::Error,
                    );
                    return;
                }
                self.update_validation_status();
                self.set_status("Configuration updated", StatusKind::Success);
            }
            Some(FileBrowserTarget::Export(field)) => {
                let stored_value = self.stringify_for_export(&path);
                if let Err(error) = self.apply_export_browser_selection(field, stored_value) {
                    self.set_status(
                        format!("Failed to apply selection: {error}"),
                        StatusKind::Error,
                    );
                    return;
                }
                self.set_status("Export option updated", StatusKind::Success);
                self.rebuild_export_fields();
            }
            Some(FileBrowserTarget::SimulatorEnvPath) => {
                let stored_value = self.stringify_for_storage(&path);
                self.simulator_config.env_path = stored_value;
                self.set_status("Simulator environment updated", StatusKind::Success);
            }
            Some(FileBrowserTarget::InterfaceAgentPath) => {
                let stored_value = self.stringify_for_storage(&path);
                if self.interface_config.agent_type == AgentType::Rllib {
                    self.interface_config.rllib_checkpoint_number =
                        Self::checkpoint_number_from_path(&path);
                }
                self.interface_config.agent_path = stored_value;
                self.set_status("Interface agent updated", StatusKind::Success);
            }
            None => {}
        }

        self.cancel_file_browser();
    }

    fn apply_config_browser_selection(&mut self, field: ConfigField, value: String) -> Result<()> {
        match field {
            ConfigField::EnvPath => {
                self.training_config.env_path = value;
            }
            ConfigField::RllibConfigFile => {
                self.training_config.rllib_config_file = normalize_rllib_config_value(&value);
            }
            ConfigField::RllibResumeFrom => {
                self.training_config.rllib_resume_from = value;
            }
            _ => {}
        }
        self.persist_training_config()
    }

    fn apply_export_browser_selection(&mut self, field: ExportField, value: String) -> Result<()> {
        match field {
            ExportField::Sb3ModelPath => {
                if !value.to_lowercase().ends_with(".zip") {
                    bail!("SB3 model must be a .zip file");
                }
                self.export_config.sb3_model_path = value;
            }
            ExportField::Sb3OutputPath => {
                if !value.to_lowercase().ends_with(".onnx") {
                    bail!("SB3 output must end with .onnx");
                }
                self.export_config.sb3_output_path = value;
            }
            ExportField::RllibCheckpointPath => {
                self.validate_rllib_checkpoint_dir(&PathBuf::from(&value))?;
                self.sync_checkpoint_number_from_path(&value);
                self.export_config.rllib_checkpoint_path = value;
            }
            ExportField::RllibOutputDir => {
                self.export_config.rllib_output_dir = value;
            }
            ExportField::RllibPolicyId => {
                self.export_config.rllib_policy_id = value;
            }
            _ => {}
        }
        self.persist_export_state()?;
        Ok(())
    }

    fn stringify_for_storage(&self, path: &Path) -> String {
        if let Some(project) = &self.active_project {
            if let Ok(relative) = path.strip_prefix(&project.root_path) {
                return relative.to_string_lossy().to_string();
            }
        }
        path.to_string_lossy().to_string()
    }

    fn rllib_config_absolute_path(&self, project_root: &Path) -> PathBuf {
        let raw = self.training_config.rllib_config_file.trim();
        let value = if raw.is_empty() {
            default_rllib_config_file()
        } else {
            raw.to_string()
        };
        let path = PathBuf::from(&value);
        if path.is_absolute() {
            path
        } else {
            project_root.join(path)
        }
    }

    fn normalize_rllib_config_file_setting(&mut self) -> Option<String> {
        let normalized = normalize_rllib_config_value(&self.training_config.rllib_config_file);
        if normalized != self.training_config.rllib_config_file {
            let previous = self.training_config.rllib_config_file.clone();
            self.training_config.rllib_config_file = normalized;
            Some(previous)
        } else {
            None
        }
    }

    fn migrate_rllib_config_file(&self, project_root: &Path, previous_value: &str) -> Result<()> {
        let trimmed = previous_value.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let previous_path = PathBuf::from(trimmed);
        let previous_abs = if previous_path.is_absolute() {
            previous_path
        } else {
            project_root.join(previous_path)
        };
        let new_path = self.rllib_config_absolute_path(project_root);
        if previous_abs == new_path || !previous_abs.exists() || new_path.exists() {
            return Ok(());
        }
        if let Some(parent) = new_path.parent() {
            fs::create_dir_all(parent)
                .wrap_err_with(|| format!("failed to create directory {}", parent.display()))?;
        }
        if let Err(rename_err) = fs::rename(&previous_abs, &new_path) {
            fs::copy(&previous_abs, &new_path).wrap_err_with(|| {
                format!(
                    "failed to copy RLlib config from {} to {} after rename error {rename_err}",
                    previous_abs.display(),
                    new_path.display()
                )
            })?;
            fs::remove_file(&previous_abs).wrap_err_with(|| {
                format!(
                    "failed to remove old RLlib config {} after copy",
                    previous_abs.display()
                )
            })?;
        }
        Ok(())
    }

    fn stringify_for_export(&self, path: &Path) -> String {
        if path.is_absolute() {
            return path.to_string_lossy().to_string();
        }

        if let Some(project) = &self.active_project {
            return project.root_path.join(path).to_string_lossy().to_string();
        }

        if let Ok(cwd) = std::env::current_dir() {
            return cwd.join(path).to_string_lossy().to_string();
        }

        path.to_string_lossy().to_string()
    }

    fn extension_allowed(&self, path: &Path, extensions: &[String]) -> bool {
        if extensions.is_empty() {
            return true;
        }
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                extensions
                    .iter()
                    .any(|allowed| allowed.eq_ignore_ascii_case(ext))
            })
            .unwrap_or(false)
    }

    fn path_has_extension(&self, path: &Path, extension: &str) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
    }

    fn refresh_file_browser(&mut self) {
        self.file_browser_entries.clear();

        if let Some(parent) = self.file_browser_path.parent() {
            self.file_browser_entries
                .push(FileBrowserEntry::Parent(parent.to_path_buf()));
        }

        if let Ok(entries) = fs::read_dir(&self.file_browser_path) {
            let mut dirs = Vec::new();
            let mut files = Vec::new();

            for entry in entries.flatten() {
                let path = entry.path();
                match entry.file_type() {
                    Ok(ft) if ft.is_dir() => dirs.push(path),
                    Ok(ft) if ft.is_file() => files.push(path),
                    _ => {}
                }
            }

            dirs.sort();
            files.sort();

            for dir in dirs {
                self.file_browser_entries
                    .push(FileBrowserEntry::Directory(dir));
            }

            match self.file_browser_kind {
                FileBrowserKind::ExistingFile { .. }
                | FileBrowserKind::OutputFile { .. }
                | FileBrowserKind::Directory { .. } => {
                    for file in files {
                        self.file_browser_entries.push(FileBrowserEntry::File(file));
                    }
                }
            }
        }

        if self.file_browser_selected >= self.file_browser_entries.len() {
            self.file_browser_selected = self.file_browser_entries.len().saturating_sub(1);
        }
    }

    pub fn validate_training_config(&self) -> Result<()> {
        if self.active_project.is_none() {
            bail!("No project selected");
        }

        if self.training_config.env_path.trim().is_empty() {
            bail!("Environment path is required");
        }

        let project_root = &self.active_project.as_ref().unwrap().root_path;
        let env_path_input = self.training_config.env_path.trim();
        let env_path = PathBuf::from(env_path_input);
        let env_path = if env_path.is_absolute() {
            env_path
        } else {
            project_root.join(env_path)
        };
        if !env_path.exists() {
            bail!("Environment path does not exist: {}", env_path.display());
        }

        if !env_path.is_file() {
            bail!("Environment path must be a file: {}", env_path.display());
        }

        if self.training_config.timesteps == 0 {
            bail!("Timesteps must be greater than 0");
        }

        if self.training_config.experiment_name.trim().is_empty() {
            bail!("Experiment name cannot be empty");
        }

        if self.training_config.mode == TrainingMode::MultiAgent {
            let config_path = self.rllib_config_absolute_path(project_root);
            if !config_path.exists() {
                bail!(
                    "RLlib config file not found: {}. Use 'g' to generate it.",
                    config_path.display()
                );
            }

            match self.training_config.rllib_stop_mode {
                RllibStopMode::TimeSeconds => {
                    if self.training_config.rllib_stop_time_seconds == 0 {
                        bail!("RLlib time limit must be at least 1 second");
                    }
                }
                RllibStopMode::Timesteps => {
                    // Already verified timesteps > 0 above
                }
            }

            let resume_trimmed = self.training_config.rllib_resume_from.trim();
            if !resume_trimmed.is_empty() {
                let resume_input = PathBuf::from(resume_trimmed);
                let resume_path = if resume_input.is_absolute() {
                    resume_input
                } else {
                    project_root.join(resume_input)
                };
                if !resume_path.exists() {
                    bail!("Resume directory not found: {}", resume_path.display());
                }
                if !resume_path.is_dir() {
                    bail!("Resume path must be a directory: {}", resume_path.display());
                }
                let tuner_file = resume_path.join("tuner.pkl");
                let legacy_tune_file = resume_path.join("tune.pkl");
                if !tuner_file.is_file() && !legacy_tune_file.is_file() {
                    bail!(
                        "Resume directory is missing tuner.pkl (Ray AIR) or tune.pkl (legacy): {}",
                        tuner_file.display()
                    );
                }
            }
        }

        Ok(())
    }

    pub fn generate_rllib_config(&mut self) -> Result<()> {
        if let Some(project) = &self.active_project {
            let config_path = self.rllib_config_absolute_path(&project.root_path);
            let existed = config_path.exists();
            match write_rllib_config(&config_path, &self.training_config) {
                Ok(()) => {
                    let message = if existed {
                        format!("Updated config: {}", config_path.display())
                    } else {
                        format!("Created config: {}", config_path.display())
                    };
                    self.set_status(message, StatusKind::Success);
                }
                Err(error) => {
                    self.set_status(
                        format!("Failed to write config: {}", error),
                        StatusKind::Error,
                    );
                    return Ok(());
                }
            }
            self.update_validation_status();
        } else {
            self.set_status("No project selected", StatusKind::Warning);
        }
        Ok(())
    }

    pub fn set_status<S: Into<String>>(&mut self, text: S, kind: StatusKind) {
        self.status = Some(StatusMessage {
            text: text.into(),
            kind,
        });
    }

    pub fn clear_status(&mut self) {
        self.status = None;
    }

    pub fn start_project_creation(&mut self) {
        self.input_mode = InputMode::CreatingProject;
        self.project_creation_stage = ProjectCreationStage::Name;
        self.project_name_buffer.clear();
        self.project_location_buffer.clear();
        self.set_status("Enter a name for the new project", StatusKind::Info);
    }

    pub fn cancel_project_creation(&mut self) {
        self.input_mode = InputMode::Normal;
        self.project_name_buffer.clear();
        self.project_location_buffer.clear();
        self.project_creation_stage = ProjectCreationStage::Name;
        self.clear_status();
    }

    pub fn push_project_name_char(&mut self, ch: char) {
        match self.project_creation_stage {
            ProjectCreationStage::Name => {
                if self.project_name_buffer.len() >= 48 {
                    return;
                }
                if ch.is_control() {
                    return;
                }
                self.project_name_buffer.push(ch);
            }
            ProjectCreationStage::Location => self.push_project_location_char(ch),
        }
    }

    pub fn pop_project_name_char(&mut self) {
        match self.project_creation_stage {
            ProjectCreationStage::Name => {
                self.project_name_buffer.pop();
            }
            ProjectCreationStage::Location => {
                self.project_location_buffer.pop();
            }
        }
    }

    fn push_project_location_char(&mut self, ch: char) {
        if ch.is_control() {
            return;
        }
        if self.project_location_buffer.len() >= PROJECT_LOCATION_MAX_LEN {
            return;
        }
        self.project_location_buffer.push(ch);
    }

    pub fn confirm_project_creation(&mut self) -> Result<()> {
        match self.project_creation_stage {
            ProjectCreationStage::Name => {
                let name = self.project_name_buffer.trim().to_string();
                if name.is_empty() {
                    self.set_status("Project name cannot be empty", StatusKind::Warning);
                    return Ok(());
                }
                self.begin_project_location_selection(&name)?;
                Ok(())
            }
            ProjectCreationStage::Location => {
                let name = self.project_name_buffer.trim();
                if name.is_empty() {
                    self.set_status("Project name cannot be empty", StatusKind::Warning);
                    self.project_creation_stage = ProjectCreationStage::Name;
                    return Ok(());
                }
                let project_dir = match self.determine_project_directory(name) {
                    Ok(dir) => dir,
                    Err(error) => {
                        self.set_status(
                            format!("Failed to resolve project location: {error}"),
                            StatusKind::Error,
                        );
                        return Ok(());
                    }
                };
                match self
                    .project_manager
                    .register_project(name, project_dir.clone())
                {
                    Ok(info) => {
                        self.set_status(
                            format!(
                                "Project '{}' created (logs at {})",
                                info.name,
                                info.logs_path.display()
                            ),
                            StatusKind::Success,
                        );
                        self.input_mode = InputMode::Normal;
                        self.project_name_buffer.clear();
                        self.project_location_buffer = project_dir.to_string_lossy().to_string();
                        self.project_creation_stage = ProjectCreationStage::Name;
                        self.refresh_projects(Some(info.logs_path.clone()))?;
                        self.set_active_project_by_path(&info.logs_path)?;
                    }
                    Err(error) => {
                        self.set_status(
                            format!("Failed to create project: {error}"),
                            StatusKind::Error,
                        );
                    }
                }
                Ok(())
            }
        }
    }

    pub fn select_next_project(&mut self) {
        if self.projects.is_empty() {
            return;
        }
        self.selected_project = (self.selected_project + 1) % self.projects.len();
    }

    pub fn select_previous_project(&mut self) {
        if self.projects.is_empty() {
            return;
        }
        if self.selected_project == 0 {
            self.selected_project = self.projects.len() - 1;
        } else {
            self.selected_project -= 1;
        }
    }

    pub fn set_active_project(&mut self) -> Result<()> {
        if let Some(project) = self.selected_project().cloned() {
            self.set_active_project_inner(project)?;
        }
        Ok(())
    }

    fn begin_project_location_selection(&mut self, name: &str) -> Result<()> {
        if self.project_location_buffer.trim().is_empty() {
            self.project_location_buffer = self
                .default_project_directory_for_name(name)
                .to_string_lossy()
                .to_string();
        }
        self.project_creation_stage = ProjectCreationStage::Location;
        self.set_status(
            "Select the directory where logs should be stored",
            StatusKind::Info,
        );
        self.start_project_location_browser();
        Ok(())
    }

    fn start_project_location_browser(&mut self) {
        let kind = FileBrowserKind::Directory {
            allow_create: true,
            require_checkpoints: false,
        };
        self.start_file_browser(FileBrowserTarget::ProjectLocation, kind, None);
    }

    fn finish_project_location_selection(&mut self, path: PathBuf) -> Result<()> {
        self.project_location_buffer = path.to_string_lossy().to_string();
        self.project_creation_stage = ProjectCreationStage::Location;
        self.input_mode = InputMode::CreatingProject;
        self.confirm_project_creation()
    }

    fn load_run_overlay_from_path(&mut self, path: PathBuf) -> Result<()> {
        let saved_run = runs::load_saved_run(&path)?;
        self.push_overlay_from_run(&saved_run, path.clone())?;

        if self.should_activate_archived_run_view() {
            self.set_archived_run_view(saved_run, path);
        } else {
            let newly_added = self.saved_run_overlays.len().saturating_sub(1);
            self.selected_overlay_index = Some(newly_added);
            if let Some(overlay) = self.saved_run_overlays.get(newly_added) {
                self.set_status(
                    format!(
                        "Loaded overlay: {} — press 'o' to inspect or 'O' to switch runs",
                        overlay.label()
                    ),
                    StatusKind::Success,
                );
            }
        }
        Ok(())
    }

    fn push_overlay_from_run(&mut self, saved_run: &SavedRun, path: PathBuf) -> Result<()> {
        if self
            .saved_run_overlays
            .iter()
            .any(|overlay| overlay.id == saved_run.id || overlay.path == path)
        {
            bail!("Run already loaded as overlay");
        }

        let color = self.next_overlay_color();
        let label = format!(
            "{} [{}]",
            saved_run.experiment_name, saved_run.training_mode
        );

        if self.saved_run_overlays.len() >= MAX_RUN_OVERLAYS {
            if let Some(removed) = self.saved_run_overlays.get(0).cloned() {
                self.saved_run_overlays.remove(0);
                self.handle_overlay_evicted(&removed);
            }
        }

        self.saved_run_overlays.push(RunOverlay {
            id: saved_run.id.clone(),
            label: label.clone(),
            color,
            path,
            run: saved_run.clone(),
        });
        Ok(())
    }

    pub fn overlay_chart_series(
        &self,
        option: &ChartMetricOption,
        max_points: usize,
    ) -> Vec<ChartOverlaySeries> {
        match option.kind() {
            ChartMetricKind::AllPoliciesRewardMean
            | ChartMetricKind::AllPoliciesEpisodeLenMean
            | ChartMetricKind::AllPoliciesLearnerStat(_) => return Vec::new(),
            _ => {}
        }

        let mut overlays = Vec::new();
        for overlay in &self.saved_run_overlays {
            let metrics = overlay.metrics();
            let len = metrics.len();
            let start = len.saturating_sub(max_points);
            let mut points = Vec::new();
            for (idx, sample) in metrics.iter().enumerate().skip(start) {
                if let Some(value) = App::chart_value_for_sample(sample, option) {
                    let x = sample
                        .training_iteration()
                        .map(|iter| iter as f64)
                        .unwrap_or_else(|| idx as f64);
                    points.push((x, value));
                }
            }
            if !points.is_empty() {
                overlays.push(ChartOverlaySeries {
                    label: overlay.label.clone(),
                    color: overlay.color,
                    points,
                });
            }
        }
        overlays
    }

    fn next_overlay_color(&mut self) -> Color {
        let color = OVERLAY_COLORS[self.overlay_color_cursor % OVERLAY_COLORS.len()];
        self.overlay_color_cursor = (self.overlay_color_cursor + 1) % OVERLAY_COLORS.len();
        color
    }

    fn set_active_project_by_path(&mut self, path: &PathBuf) -> Result<()> {
        if let Some(project) = self
            .projects
            .iter()
            .find(|info| &info.logs_path == path)
            .cloned()
        {
            self.set_active_project_inner(project)?;
        }
        Ok(())
    }

    fn set_active_project_inner(&mut self, project: ProjectInfo) -> Result<()> {
        self.project_manager.mark_as_used(&project)?;
        if let Err(err) = std::env::set_current_dir(&project.root_path) {
            self.set_status(
                format!(
                    "Failed to switch to project directory {}: {err}",
                    project.root_path.display()
                ),
                StatusKind::Warning,
            );
        }
        self.saved_run_overlays.clear();
        self.selected_overlay_index = None;
        self.overlay_color_cursor = 0;
        self.drop_archived_run_view();
        self.active_project = Some(project.clone());
        let training_error = self.load_training_config_for_active_project();
        let export_error = self.load_export_state_for_active_project();
        self.refresh_projects(Some(project.logs_path.clone()))?;
        if !training_error && !export_error {
            self.set_status(
                format!("Active project: {}", project.name),
                StatusKind::Success,
            );
        }
        Ok(())
    }

    pub fn force_refresh_projects(&mut self) -> Result<()> {
        self.refresh_projects(self.active_project.as_ref().map(|p| p.logs_path.clone()))
    }

    pub fn start_training(&mut self) -> Result<()> {
        if self.training_running {
            self.set_status(
                "Training already running. Please wait for it to finish.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        // Validate configuration
        if let Err(e) = self.validate_training_config() {
            self.set_status(format!("Validation failed: {}", e), StatusKind::Error);
            return Ok(());
        }

        let resuming_multi =
            self.training_config.mode == TrainingMode::MultiAgent
                && !self.training_config.rllib_resume_from.trim().is_empty();
        let resume_baseline = if resuming_multi {
            let mut data = self.training_metrics_history().to_vec();
            if data.len() > TRAINING_METRIC_HISTORY_LIMIT {
                let excess = data.len() - TRAINING_METRIC_HISTORY_LIMIT;
                data.drain(0..excess);
            }
            data
        } else {
            Vec::new()
        };
        self.drop_archived_run_view();
        self.training_receiver = None;
        self.training_cancel = None;
        self.training_running = true;
        if resuming_multi {
            // Keep a copy of the pre-resume metrics so the chart can show the full timeline.
            self.metrics_timeline = resume_baseline;
        } else {
            self.metrics_timeline.clear();
            self.metrics_resume_iteration = None;
            self.metrics_resume_label = None;
        }
        self.training_metrics.clear();
        self.current_run_start = Some(SystemTime::now());
        self.metrics_history_index = 0;
        self.metrics_chart_index = 0;
        let now = Instant::now();
        self.metric_timer_start = Some(now);
        self.metric_last_sample_time = Some(now);
        self.reset_training_output_scroll();

        let mode_name = match self.training_config.mode {
            TrainingMode::SingleAgent => "Single-Agent (SB3)",
            TrainingMode::MultiAgent => "Multi-Agent (RLlib)",
        };
        self.set_status(
            format!("Starting {} training...", mode_name),
            StatusKind::Info,
        );

        let (tx, rx) = mpsc::channel();
        self.training_receiver = Some(rx);
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.training_cancel = Some(cancel_tx);

        let command = determine_python_command();
        let project = self.active_project.as_ref().unwrap();
        let cwd = project.root_path.clone();

        let (script_path, args) = match self.training_config.mode {
            TrainingMode::SingleAgent => {
                let script = find_script(&cwd, "stable_baselines3_training_script.py")?;
                let sb3_logs = project.logs_path.join("sb3");
                fs::create_dir_all(&sb3_logs).wrap_err_with(|| {
                    format!("failed to create SB3 log directory {}", sb3_logs.display())
                })?;
                let mut args = vec![
                    format!("--env_path={}", self.training_config.env_path),
                    format!("--experiment_dir={}", sb3_logs.to_string_lossy()),
                    format!("--experiment_name={}", self.training_config.experiment_name),
                    format!("--timesteps={}", self.training_config.timesteps),
                    format!("--speedup={}", self.training_config.sb3_speedup),
                    format!("--n_parallel={}", self.training_config.sb3_n_parallel),
                    format!(
                        "--policy-type={}",
                        self.training_config.sb3_policy_type.as_str()
                    ),
                    format!(
                        "--policy-hidden-layers={}",
                        format_usize_list_compact(&self.training_config.sb3_policy_layers)
                    ),
                ];
                match self.training_config.sb3_policy_type {
                    PolicyType::Cnn => args.push(format!(
                        "--cnn-channels={}",
                        format_usize_list_compact(&self.training_config.sb3_cnn_channels)
                    )),
                    PolicyType::Lstm => {
                        args.push(format!(
                            "--lstm-hidden-size={}",
                            self.training_config.sb3_lstm_hidden_size
                        ));
                        args.push(format!(
                            "--lstm-num-layers={}",
                            self.training_config.sb3_lstm_num_layers
                        ));
                    }
                    PolicyType::Grn => args.push(format!(
                        "--grn-hidden-size={}",
                        self.training_config.sb3_grn_hidden_size
                    )),
                    PolicyType::Mlp => {}
                }
                if self.training_config.sb3_viz {
                    args.push("--viz".to_string());
                }
                (script, args)
            }
            TrainingMode::MultiAgent => {
                let script = find_script(&cwd, "rllib_training_script.py")?;
                let rllib_logs = project.logs_path.join("rllib");
                fs::create_dir_all(&rllib_logs).wrap_err_with(|| {
                    format!(
                        "failed to create RLlib log directory {}",
                        rllib_logs.display()
                    )
                })?;
                let mut args = vec![
                    format!("--config_file={}", self.training_config.rllib_config_file),
                    format!("--experiment_dir={}", rllib_logs.to_string_lossy()),
                ];
                let resume_path = if resuming_multi {
                    let trimmed = self.training_config.rllib_resume_from.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(self.resolve_project_path(project, trimmed))
                    }
                } else {
                    None
                };

                if let Some(ref path) = resume_path {
                    if let Some(backup_path) = self.backup_resume_directory(path)? {
                        let display = self.project_relative_display(&backup_path);
                        self.set_status(
                            format!("Backed up resume run to {}", display),
                            StatusKind::Info,
                        );
                    }
                    if self.metrics_resume_iteration.is_none() {
                        if let Some(number) = self.export_config.rllib_checkpoint_number {
                            let freq = self.training_config.rllib_checkpoint_frequency as u64;
                            if freq > 0 {
                                self.metrics_resume_iteration = Some(number as u64 * freq);
                            } else if let Some(iter) = self
                                .metrics_timeline
                                .last()
                                .and_then(|s| s.training_iteration())
                            {
                                self.metrics_resume_iteration = Some(iter);
                            }
                        } else if let Some(iter) = self
                            .metrics_timeline
                            .last()
                            .and_then(|s| s.training_iteration())
                        {
                            self.metrics_resume_iteration = Some(iter);
                        }
                    }
                    let display = self.project_relative_display(path);
                    self.metrics_resume_label =
                        Some(format!("Resume baseline from {}", display));
                    args.push(format!("--resume={}", path.to_string_lossy()));
                }
                (script, args)
            }
        };

        let display_cmd = format!(
            "{} -u {} {}",
            command,
            script_path.display(),
            args.join(" ")
        );

        if !self.training_output.is_empty() {
            self.append_training_line(String::new());
        }
        self.append_training_line(format!("$ {}", display_cmd));

        spawn_training_task(tx, command, script_path, args, cwd, cancel_rx);

        Ok(())
    }

    pub fn start_demo_training(&mut self) -> Result<()> {
        if self.training_running {
            self.set_status(
                "Training already running. Please wait for it to finish.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        self.training_receiver = None;
        self.training_cancel = None;
        self.training_running = true;
        self.metrics_timeline.clear();
        self.metrics_resume_iteration = None;
        self.metrics_resume_label = None;
        self.training_metrics.clear();
        self.current_run_start = Some(SystemTime::now());
        self.metrics_history_index = 0;
        self.metrics_chart_index = 0;
        let now = Instant::now();
        self.metric_timer_start = Some(now);
        self.metric_last_sample_time = Some(now);
        self.reset_training_output_scroll();
        self.set_status("Starting demo training run...", StatusKind::Info);

        let (tx, rx) = mpsc::channel();
        self.training_receiver = Some(rx);
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.training_cancel = Some(cancel_tx);

        let command = determine_python_command();
        let cwd = std::env::current_dir().wrap_err("failed to determine current directory")?;
        // use this script path: /home/kasper/GameProjects/agents/demo.py
        let mut script_path: std::path::PathBuf = "/home/kasper/GameProjects/agents/demo.py".into();
        let mut workdir = cwd.clone();

        if !script_path.exists() {
            if let Some(parent) = cwd.parent() {
                let candidate = parent.join("demo.py");
                if candidate.exists() {
                    script_path = candidate;
                    workdir = parent.to_path_buf();
                }
            }
        }

        let script_path = script_path;
        let workdir = workdir;
        let args = Vec::new(); // No args for demo
        let display_cmd = format!("{} -u {}", command, script_path.display());

        if !self.training_output.is_empty() {
            self.append_training_line(String::new());
        }
        self.append_training_line(format!("$ {}", display_cmd));

        spawn_training_task(tx, command, script_path, args, workdir, cancel_rx);

        Ok(())
    }

    pub fn cancel_training(&mut self) {
        if let Some(cancel) = self.training_cancel.take() {
            let _ = cancel.send(());
            self.append_training_line("! Cancellation requested by user");
            self.set_status("Stopping training...", StatusKind::Info);
        }
    }

    pub fn start_simulator(&mut self) -> Result<()> {
        if self.simulator_running {
            self.set_status(
                "Simulator already running. Stop it before starting a new session.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        let project_root = match self.active_project.as_ref() {
            Some(project) => project.root_path.clone(),
            None => {
                self.set_status("Select a project first", StatusKind::Warning);
                return Ok(());
            }
        };

        let command = determine_python_command();
        let script_path = find_script(&project_root, "simulator.py")?;

        let mut args = vec![format!("--mode={}", self.simulator_config.mode.arg())];

        if let Some(path) = self.resolve_existing_path(&self.simulator_config.env_path) {
            args.push(format!("--env-path={}", path.to_string_lossy()));
        } else if !self.simulator_config.env_path.trim().is_empty() {
            args.push(format!(
                "--env-path={}",
                self.simulator_config.env_path.trim()
            ));
        }

        if self.simulator_config.show_window {
            args.push("--show-window".to_string());
        } else {
            args.push("--headless".to_string());
        }

        if self.simulator_config.step_delay > 0.0 {
            args.push(format!(
                "--step-delay={:.4}",
                self.simulator_config.step_delay.max(0.0)
            ));
        }

        args.push(format!(
            "--restart-delay={:.4}",
            self.simulator_config.restart_delay.max(0.0)
        ));

        if let Some(max_steps) = self.simulator_config.max_steps {
            if max_steps > 0 {
                args.push(format!("--max-steps={max_steps}"));
            }
        }

        if let Some(max_episodes) = self.simulator_config.max_episodes {
            if max_episodes > 0 {
                args.push(format!("--max-episodes={max_episodes}"));
            }
        }

        if !self.simulator_config.auto_restart {
            args.push("--no-auto-restart".to_string());
        }

        if self.simulator_config.log_tracebacks {
            args.push("--log-tracebacks".to_string());
        }

        let (tx, rx) = mpsc::channel();
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.simulator_receiver = Some(rx);
        self.simulator_cancel = Some(cancel_tx);
        self.simulator_running = true;
        self.simulator_focus = SimulatorFocus::Events;
        self.simulator_event_log.clear();
        self.simulator_event_scroll = 0;
        self.simulator_actions.clear();
        self.simulator_actions_scroll = 0;
        self.simulator_action_meta = None;
        self.simulator_status_line = Some("Launching simulator...".to_string());
        if !self.simulator_compact_user_override {
            self.simulator_compact_view = false;
        }

        let display_cmd = format!(
            "{} -u {} {}",
            command,
            script_path.display(),
            args.join(" ")
        );

        self.append_simulator_event_entry(SimulatorEventEntry {
            timestamp: None,
            kind: "command".into(),
            message: format!("$ {display_cmd}"),
            severity: SimulatorEventSeverity::Info,
        });

        spawn_simulator_task(tx, command, script_path, args, project_root, cancel_rx);

        self.set_status("Simulator starting...", StatusKind::Info);
        Ok(())
    }

    pub fn cancel_simulator(&mut self) {
        if let Some(cancel) = self.simulator_cancel.take() {
            let _ = cancel.send(());
            self.append_simulator_event_entry(SimulatorEventEntry {
                timestamp: None,
                kind: "status".into(),
                message: "Cancellation requested...".to_string(),
                severity: SimulatorEventSeverity::Warning,
            });
            self.set_status("Stopping simulator...", StatusKind::Info);
        }
    }

    pub fn start_simulator_file_browser(&mut self) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before changing the environment path.",
                StatusKind::Warning,
            );
            return;
        }
        let kind = FileBrowserKind::ExistingFile {
            extensions: Vec::new(),
        };
        self.start_file_browser(FileBrowserTarget::SimulatorEnvPath, kind, None);
    }

    pub fn simulator_use_training_env_path(&mut self) {
        if self.training_config.env_path.trim().is_empty() {
            self.set_status("Training environment path is empty.", StatusKind::Warning);
            return;
        }
        self.simulator_config.env_path = self.training_config.env_path.clone();
        self.set_status(
            "Simulator environment path synced with training config.",
            StatusKind::Success,
        );
    }

    pub fn toggle_simulator_mode(&mut self) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before changing the mode.",
                StatusKind::Warning,
            );
            return;
        }
        self.simulator_config.mode.toggle();
        self.set_status(
            format!("Simulator mode: {}", self.simulator_config.mode.label()),
            StatusKind::Info,
        );
    }

    pub fn toggle_simulator_show_window(&mut self) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before toggling the window.",
                StatusKind::Warning,
            );
            return;
        }
        self.simulator_config.show_window = !self.simulator_config.show_window;
        self.set_status(
            format!(
                "Simulator window: {}",
                if self.simulator_config.show_window {
                    "Visible"
                } else {
                    "Headless"
                }
            ),
            StatusKind::Info,
        );
    }

    pub fn toggle_simulator_auto_restart(&mut self) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before toggling auto-restart.",
                StatusKind::Warning,
            );
            return;
        }
        self.simulator_config.auto_restart = !self.simulator_config.auto_restart;
        self.set_status(
            format!(
                "Simulator auto-restart: {}",
                if self.simulator_config.auto_restart {
                    "Enabled"
                } else {
                    "Disabled"
                }
            ),
            StatusKind::Info,
        );
    }

    pub fn toggle_simulator_tracebacks(&mut self) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before toggling tracebacks.",
                StatusKind::Warning,
            );
            return;
        }
        self.simulator_config.log_tracebacks = !self.simulator_config.log_tracebacks;
        self.set_status(
            format!(
                "Simulator tracebacks: {}",
                if self.simulator_config.log_tracebacks {
                    "Verbose"
                } else {
                    "Hidden"
                }
            ),
            StatusKind::Info,
        );
    }

    pub fn adjust_simulator_step_delay(&mut self, delta: f64) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before adjusting delays.",
                StatusKind::Warning,
            );
            return;
        }
        self.simulator_config.step_delay = (self.simulator_config.step_delay + delta).max(0.0);
        self.set_status(
            format!("Step delay: {:.2}s", self.simulator_config.step_delay),
            StatusKind::Info,
        );
    }

    pub fn adjust_simulator_restart_delay(&mut self, delta: f64) {
        if self.simulator_running {
            self.set_status(
                "Stop the simulator before adjusting delays.",
                StatusKind::Warning,
            );
            return;
        }
        self.simulator_config.restart_delay =
            (self.simulator_config.restart_delay + delta).max(0.0);
        self.set_status(
            format!("Restart delay: {:.2}s", self.simulator_config.restart_delay),
            StatusKind::Info,
        );
    }

    pub fn cycle_simulator_focus(&mut self) {
        self.simulator_focus = match self.simulator_focus {
            SimulatorFocus::Events => SimulatorFocus::Actions,
            SimulatorFocus::Actions => SimulatorFocus::Events,
        };
    }

    pub fn toggle_simulator_compact_view(&mut self) {
        self.simulator_compact_view = !self.simulator_compact_view;
        self.simulator_compact_user_override = true;
    }

    pub fn simulator_scroll_up(&mut self, amount: usize) {
        match self.simulator_focus {
            SimulatorFocus::Events => {
                self.simulator_event_scroll = self.simulator_event_scroll.saturating_add(amount);
            }
            SimulatorFocus::Actions => {
                self.simulator_actions_scroll =
                    self.simulator_actions_scroll.saturating_add(amount);
            }
        }
    }

    pub fn simulator_scroll_down(&mut self, amount: usize) {
        match self.simulator_focus {
            SimulatorFocus::Events => {
                self.simulator_event_scroll = self.simulator_event_scroll.saturating_sub(amount);
            }
            SimulatorFocus::Actions => {
                self.simulator_actions_scroll =
                    self.simulator_actions_scroll.saturating_sub(amount);
            }
        }
    }

    // Interface methods (similar to simulator)
    pub fn start_interface(&mut self) -> Result<()> {
        if self.interface_running {
            self.set_status(
                "Interface already running. Stop it before starting a new session.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        let project_root = match self.active_project.as_ref() {
            Some(project) => project.root_path.clone(),
            None => {
                self.set_status("Select a project first", StatusKind::Warning);
                return Ok(());
            }
        };

        // Validate agent path
        if self.interface_config.agent_path.trim().is_empty() {
            self.set_status(
                "Agent path is required. Use 'p' to select an agent.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        let command = determine_python_command();
        let script_path = find_script(&project_root, "interface.py")?;

        let mut args = vec![
            self.interface_config.agent_type.arg().to_string(),
            self.interface_config.agent_path.clone(),
        ];

        args.push(format!("--mode={}", self.interface_config.mode.arg()));

        if self.interface_config.show_window {
            args.push("--show-window".to_string());
        } else {
            args.push("--headless".to_string());
        }

        if self.interface_config.step_delay > 0.0 {
            args.push(format!(
                "--step-delay={:.4}",
                self.interface_config.step_delay.max(0.0)
            ));
        }

        args.push(format!(
            "--restart-delay={:.4}",
            self.interface_config.restart_delay.max(0.0)
        ));

        if !self.interface_config.auto_restart {
            args.push("--no-auto-restart".to_string());
        }

        if self.interface_config.log_tracebacks {
            args.push("--log-tracebacks".to_string());
        }

        // Add RLlib-specific args
        if self.interface_config.agent_type == AgentType::Rllib {
            if let Some(num) = self.interface_config.rllib_checkpoint_number {
                args.push(format!("--checkpoint-number={num}"));
            }
            if !self.interface_config.rllib_policy_id.trim().is_empty() {
                args.push(format!(
                    "--policy={}",
                    self.interface_config.rllib_policy_id
                ));
            }
        }

        // Add SB3-specific args
        if self.interface_config.agent_type == AgentType::StableBaselines3 {
            if !self.interface_config.sb3_algo.trim().is_empty() {
                args.push(format!("--algo={}", self.interface_config.sb3_algo));
            }
        }

        let (tx, rx) = mpsc::channel();
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.interface_receiver = Some(rx);
        self.interface_cancel = Some(cancel_tx);
        self.interface_running = true;
        self.interface_focus = InterfaceFocus::Events;
        self.interface_event_log.clear();
        self.interface_event_scroll = 0;
        self.interface_actions.clear();
        self.interface_actions_scroll = 0;
        self.interface_action_meta = None;
        self.interface_status_line = Some("Launching interface...".to_string());
        if !self.interface_compact_user_override {
            self.interface_compact_view = false;
        }

        let display_cmd = format!(
            "{} -u {} {}",
            command,
            script_path.display(),
            args.join(" ")
        );

        self.append_interface_event_entry(InterfaceEventEntry {
            timestamp: None,
            kind: "command".into(),
            message: format!("$ {display_cmd}"),
            severity: SimulatorEventSeverity::Info,
        });

        spawn_interface_task(tx, command, script_path, args, project_root, cancel_rx);

        self.set_status("Interface starting...", StatusKind::Info);
        Ok(())
    }

    pub fn cancel_interface(&mut self) {
        if let Some(cancel) = self.interface_cancel.take() {
            let _ = cancel.send(());
            self.append_interface_event_entry(InterfaceEventEntry {
                timestamp: None,
                kind: "status".into(),
                message: "Cancellation requested...".to_string(),
                severity: SimulatorEventSeverity::Warning,
            });
            self.set_status("Stopping interface...", StatusKind::Info);
        }
    }

    pub fn start_interface_agent_browser(&mut self) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before changing the agent path.",
                StatusKind::Warning,
            );
            return;
        }

        let extensions = match self.interface_config.agent_type {
            AgentType::StableBaselines3 => vec!["zip".into()],
            AgentType::Rllib => Vec::new(), // Directories for RLlib
        };

        let kind = if self.interface_config.agent_type == AgentType::Rllib {
            FileBrowserKind::Directory {
                allow_create: false,
                require_checkpoints: true,
            }
        } else {
            FileBrowserKind::ExistingFile { extensions }
        };

        self.start_file_browser(FileBrowserTarget::InterfaceAgentPath, kind, None);
    }

    pub fn toggle_interface_agent_type(&mut self) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before changing the agent type.",
                StatusKind::Warning,
            );
            return;
        }
        self.interface_config.agent_type.toggle();
        self.set_status(
            format!("Agent type: {}", self.interface_config.agent_type.label()),
            StatusKind::Info,
        );
    }

    pub fn interface_use_export_agent_path(&mut self) {
        match self.interface_config.agent_type {
            AgentType::StableBaselines3 => {
                let sb3_path = self.export_config.sb3_model_path.trim();
                if sb3_path.is_empty() {
                    self.set_status(
                        "SB3 model path is empty in the Export tab.",
                        StatusKind::Warning,
                    );
                    return;
                }
                self.interface_config.agent_path = sb3_path.to_string();
                if !self.export_config.sb3_algo.trim().is_empty() {
                    self.interface_config.sb3_algo = self.export_config.sb3_algo.clone();
                }
                self.set_status(
                    "Interface agent path synced with SB3 export config.",
                    StatusKind::Success,
                );
            }
            AgentType::Rllib => {
                let checkpoint_path = self.export_config.rllib_checkpoint_path.trim();
                if checkpoint_path.is_empty() {
                    self.set_status(
                        "RLlib checkpoint path is empty in the Export tab.",
                        StatusKind::Warning,
                    );
                    return;
                }
                self.interface_config.agent_path = checkpoint_path.to_string();
                self.interface_config.rllib_checkpoint_number =
                    self.export_config.rllib_checkpoint_number;
                self.interface_config.rllib_policy_id = self.export_config.rllib_policy_id.clone();
                self.set_status(
                    "Interface checkpoint synced with Export config.",
                    StatusKind::Success,
                );
            }
        }
    }

    pub fn toggle_interface_mode(&mut self) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before changing mode.",
                StatusKind::Warning,
            );
            return;
        }
        self.interface_config.mode.toggle();
        self.set_status(
            format!("Interface mode: {}", self.interface_config.mode.label()),
            StatusKind::Info,
        );
    }

    pub fn toggle_interface_window(&mut self) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before changing window settings.",
                StatusKind::Warning,
            );
            return;
        }
        self.interface_config.show_window = !self.interface_config.show_window;
        let state = if self.interface_config.show_window {
            "visible"
        } else {
            "hidden"
        };
        self.set_status(format!("Godot window: {state}"), StatusKind::Info);
    }

    pub fn toggle_interface_auto_restart(&mut self) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before changing auto-restart.",
                StatusKind::Warning,
            );
            return;
        }
        self.interface_config.auto_restart = !self.interface_config.auto_restart;
        let state = if self.interface_config.auto_restart {
            "enabled"
        } else {
            "disabled"
        };
        self.set_status(format!("Auto-restart: {state}"), StatusKind::Info);
    }

    pub fn toggle_interface_tracebacks(&mut self) {
        self.interface_config.log_tracebacks = !self.interface_config.log_tracebacks;
        let state = if self.interface_config.log_tracebacks {
            "enabled"
        } else {
            "disabled"
        };
        self.set_status(format!("Tracebacks: {state}"), StatusKind::Info);
    }

    pub fn adjust_interface_step_delay(&mut self, delta: f64) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before adjusting delays.",
                StatusKind::Warning,
            );
            return;
        }
        self.interface_config.step_delay = (self.interface_config.step_delay + delta).max(0.0);
        self.set_status(
            format!("Step delay: {:.2}s", self.interface_config.step_delay),
            StatusKind::Info,
        );
    }

    pub fn adjust_interface_restart_delay(&mut self, delta: f64) {
        if self.interface_running {
            self.set_status(
                "Stop the interface before adjusting delays.",
                StatusKind::Warning,
            );
            return;
        }
        self.interface_config.restart_delay =
            (self.interface_config.restart_delay + delta).max(0.0);
        self.set_status(
            format!("Restart delay: {:.2}s", self.interface_config.restart_delay),
            StatusKind::Info,
        );
    }

    pub fn cycle_interface_focus(&mut self) {
        self.interface_focus = match self.interface_focus {
            InterfaceFocus::Events => InterfaceFocus::Actions,
            InterfaceFocus::Actions => InterfaceFocus::Events,
        };
    }

    pub fn toggle_interface_compact_view(&mut self) {
        self.interface_compact_view = !self.interface_compact_view;
        self.interface_compact_user_override = true;
    }

    pub fn interface_scroll_up(&mut self, amount: usize) {
        match self.interface_focus {
            InterfaceFocus::Events => {
                self.interface_event_scroll = self.interface_event_scroll.saturating_add(amount);
            }
            InterfaceFocus::Actions => {
                self.interface_actions_scroll =
                    self.interface_actions_scroll.saturating_add(amount);
            }
        }
    }

    pub fn interface_scroll_down(&mut self, amount: usize) {
        match self.interface_focus {
            InterfaceFocus::Events => {
                self.interface_event_scroll = self.interface_event_scroll.saturating_sub(amount);
            }
            InterfaceFocus::Actions => {
                self.interface_actions_scroll =
                    self.interface_actions_scroll.saturating_sub(amount);
            }
        }
    }

    fn append_interface_event_entry(&mut self, entry: InterfaceEventEntry) {
        self.interface_status_line = Some(entry.message.clone());
        self.interface_event_log.push(entry);
        if self.interface_event_log.len() > SIM_EVENT_BUFFER_LIMIT {
            let overflow = self.interface_event_log.len() - SIM_EVENT_BUFFER_LIMIT;
            self.interface_event_log.drain(0..overflow);
        }
        if self.controller_settings.auto_scroll_training_log() {
            self.interface_event_scroll = 0;
        } else {
            self.clamp_interface_event_scroll();
        }
    }

    fn clamp_interface_event_scroll(&mut self) {
        if self.interface_event_log.is_empty() {
            self.interface_event_scroll = 0;
            return;
        }
        let max_offset = self.interface_event_log.len().saturating_sub(1);
        if self.interface_event_scroll > max_offset {
            self.interface_event_scroll = max_offset;
        }
    }

    pub fn start_run_overlay_browser(&mut self) -> Result<()> {
        let project = match self.active_project.as_ref() {
            Some(project) => project,
            None => {
                self.set_status("Select a project first", StatusKind::Warning);
                return Ok(());
            }
        };

        fs::create_dir_all(project.runs_dir()).wrap_err_with(|| {
            format!(
                "failed to prepare runs directory {}",
                project.runs_dir().display()
            )
        })?;

        self.start_file_browser(
            FileBrowserTarget::SavedRun,
            FileBrowserKind::ExistingFile {
                extensions: vec!["json".into()],
            },
            None,
        );
        self.set_status("Select a saved run to inspect or overlay", StatusKind::Info);
        Ok(())
    }

    pub fn clear_run_overlays(&mut self) {
        if self.saved_run_overlays.is_empty() {
            self.set_status("No overlays to clear", StatusKind::Info);
            return;
        }
        let had_view = self.drop_archived_run_view();
        self.saved_run_overlays.clear();
        self.selected_overlay_index = None;
        self.overlay_color_cursor = 0;
        let message = if had_view {
            "Cleared run overlays and returned to live metrics"
        } else {
            "Cleared run overlays"
        };
        self.set_status(message.to_string(), StatusKind::Info);
    }

    pub fn start_export(&mut self) -> Result<()> {
        if self.export_running {
            self.set_status(
                "Export already running. Please wait for it to finish.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        let project = match self.active_project.as_ref() {
            Some(project) => project.clone(),
            None => {
                self.set_status("No project selected", StatusKind::Warning);
                return Ok(());
            }
        };

        self.export_receiver = None;
        self.export_cancel = None;
        self.export_running = true;
        self.reset_export_output_scroll();
        self.export_output.clear();

        let command = determine_python_command();
        let workdir = project.root_path.clone();
        let (tx, rx) = mpsc::channel();
        self.export_receiver = Some(rx);
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.export_cancel = Some(cancel_tx);

        let (script_path, args) = match self.export_mode {
            ExportMode::StableBaselines3 => self.prepare_sb3_export_args(&project)?,
            ExportMode::Rllib => self.prepare_rllib_export_args(&project)?,
        };

        let display_cmd = format!(
            "{} -u {} {}",
            command,
            script_path.display(),
            args.join(" ")
        );

        self.append_export_line(format!("$ {display_cmd}"));
        spawn_export_task(tx, command, script_path, args, workdir, cancel_rx);
        self.set_status("Export started...", StatusKind::Info);

        Ok(())
    }

    pub fn cancel_export(&mut self) {
        if let Some(cancel) = self.export_cancel.take() {
            let _ = cancel.send(());
            self.append_export_line("! Cancellation requested by user");
            self.set_status("Stopping export...", StatusKind::Info);
        }
    }

    fn prepare_sb3_export_args(&self, project: &ProjectInfo) -> Result<(PathBuf, Vec<String>)> {
        let model_path_str = self.export_config.sb3_model_path.trim();
        if model_path_str.is_empty() {
            bail!("SB3 model path is required");
        }
        let model_path = self.resolve_project_path(project, model_path_str);
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }

        let output_path = if self.export_config.sb3_output_path.trim().is_empty() {
            model_path.with_extension("onnx")
        } else {
            self.resolve_project_path(project, self.export_config.sb3_output_path.trim())
        };

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).wrap_err_with(|| {
                format!("failed to create export directory {}", parent.display())
            })?;
        }

        let script_path = find_script(&project.root_path, "convert_sb3_to_onnx.py")?;

        let mut args = Vec::new();
        args.push(model_path.to_string_lossy().to_string());
        args.push("--output".to_string());
        args.push(output_path.to_string_lossy().to_string());
        if !self.export_config.sb3_algo.trim().is_empty() {
            args.push("--algo".to_string());
            args.push(self.export_config.sb3_algo.trim().to_string());
        }
        args.push("--opset".to_string());
        args.push(self.export_config.sb3_opset.to_string());
        args.push("--ir-version".to_string());
        args.push(self.export_config.sb3_ir_version.to_string());
        if self.export_config.sb3_use_obs_array {
            args.push("--use-obs-array".to_string());
        }
        if self.export_config.sb3_skip_verify {
            args.push("--no-verify".to_string());
        }

        Ok((script_path, args))
    }

    fn prepare_rllib_export_args(&self, project: &ProjectInfo) -> Result<(PathBuf, Vec<String>)> {
        let checkpoint_path_str = self.export_config.rllib_checkpoint_path.trim();
        if checkpoint_path_str.is_empty() {
            bail!("RLlib checkpoint path is required");
        }
        let checkpoint_path = self.resolve_project_path(project, checkpoint_path_str);
        if !checkpoint_path.exists() {
            bail!("Checkpoint path not found: {}", checkpoint_path.display());
        }

        let is_direct_checkpoint = checkpoint_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("checkpoint_"))
            .unwrap_or(false);

        if !is_direct_checkpoint {
            self.validate_rllib_checkpoint_dir(&checkpoint_path)?;
            if let Some(number) = self.export_config.rllib_checkpoint_number {
                self.ensure_rllib_checkpoint_number_exists(&checkpoint_path, number)?;
            }
        }

        let output_dir = if self.export_config.rllib_output_dir.trim().is_empty() {
            project.root_path.join("onnx_exports")
        } else {
            self.resolve_project_path(project, self.export_config.rllib_output_dir.trim())
        };
        fs::create_dir_all(&output_dir).wrap_err_with(|| {
            format!("failed to create export directory {}", output_dir.display())
        })?;

        let script_path = find_script(&project.root_path, "convert_rllib_to_onnx.py")?;

        let mut args = Vec::new();
        args.push(checkpoint_path.to_string_lossy().to_string());
        args.push("--output".to_string());
        args.push(output_dir.to_string_lossy().to_string());
        args.push("--opset".to_string());
        args.push(self.export_config.rllib_opset.to_string());
        args.push("--ir-version".to_string());
        args.push(self.export_config.rllib_ir_version.to_string());
        if !is_direct_checkpoint {
            if let Some(number) = self.export_config.rllib_checkpoint_number {
                args.push("--checkpoint-number".to_string());
                args.push(number.to_string());
            }
        }
        if !self.export_config.rllib_multiagent {
            args.push("--no-multiagent".to_string());
        }
        if !self.export_config.rllib_policy_id.trim().is_empty() {
            args.push("--policy".to_string());
            args.push(self.export_config.rllib_policy_id.trim().to_string());
        }
        if !self.export_config.rllib_prefix.trim().is_empty() {
            args.push("--prefix".to_string());
            args.push(self.export_config.rllib_prefix.trim().to_string());
        }

        Ok((script_path, args))
    }

    fn validate_rllib_checkpoint_dir(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            bail!("Path does not exist: {}", path.display());
        }
        if !path.is_dir() {
            bail!("Path is not a directory: {}", path.display());
        }

        let is_direct_checkpoint = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("checkpoint_"))
            .unwrap_or(false);

        if is_direct_checkpoint {
            return Ok(());
        }

        let mut checkpoint_dirs = 0;
        let entries = fs::read_dir(path)
            .wrap_err_with(|| format!("failed to read checkpoint directory {}", path.display()))?;

        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("checkpoint_") {
                        checkpoint_dirs += 1;
                    }
                }
            }
        }

        if checkpoint_dirs == 0 {
            bail!("No checkpoint_* directories found in {}", path.display());
        }

        Ok(())
    }

    fn determine_project_directory(&self, name: &str) -> Result<PathBuf> {
        let trimmed = self.project_location_buffer.trim();
        let mut path = if trimmed.is_empty() {
            self.default_project_directory_for_name(name)
        } else {
            PathBuf::from(trimmed)
        };
        if !path.is_absolute() {
            let cwd = std::env::current_dir().wrap_err("failed to determine current directory")?;
            path = cwd.join(path);
        }
        Ok(path)
    }

    fn default_project_directory_for_name(&self, name: &str) -> PathBuf {
        self.project_manager.default_project_dir_for(name)
    }

    fn ensure_rllib_checkpoint_number_exists(&self, path: &Path, number: u32) -> Result<()> {
        let padded_name = format!("checkpoint_{number:06}");
        let padded_path = path.join(&padded_name);
        if padded_path.is_dir() {
            return Ok(());
        }

        let simple_name = format!("checkpoint_{number}");
        let simple_path = path.join(&simple_name);
        if simple_path.is_dir() {
            return Ok(());
        }

        let entries = fs::read_dir(path)
            .wrap_err_with(|| format!("failed to read checkpoint directory {}", path.display()))?;

        for entry in entries.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }
            let name_os = entry.file_name();
            let Some(name) = name_os.to_str() else {
                continue;
            };
            if let Some(suffix) = name.strip_prefix("checkpoint_") {
                if suffix.parse::<u32>().ok() == Some(number) {
                    return Ok(());
                }
            }
        }

        bail!(
            "Checkpoint checkpoint_{:06} not found in {}",
            number,
            path.display()
        );
    }

    fn checkpoint_number_from_path(path: &Path) -> Option<u32> {
        let name = path.file_name()?.to_str()?;
        let suffix = name.strip_prefix("checkpoint_")?;
        suffix.parse::<u32>().ok()
    }

    fn sync_checkpoint_number_from_path(&mut self, value: &str) {
        self.export_config.rllib_checkpoint_number =
            Self::checkpoint_number_from_path(Path::new(value));
    }

    fn resolve_project_path(&self, project: &ProjectInfo, value: &str) -> PathBuf {
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            project.root_path.join(path)
        }
    }

    fn backup_resume_directory(&self, resume_path: &Path) -> Result<Option<PathBuf>> {
        let Some(name) = resume_path.file_name().and_then(|n| n.to_str()) else {
            return Ok(None);
        };
        let Some(parent) = resume_path.parent() else {
            return Ok(None);
        };

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let backup_path = parent.join(format!("{name}_backup_{stamp}"));
        if backup_path.exists() {
            return Ok(None);
        }

        self.copy_dir_recursive(resume_path, &backup_path)?;
        Ok(Some(backup_path))
    }

    fn copy_dir_recursive(&self, src: &Path, dst: &Path) -> Result<()> {
        fs::create_dir_all(dst)
            .wrap_err_with(|| format!("failed to create backup dir {}", dst.display()))?;
        for entry in fs::read_dir(src)
            .wrap_err_with(|| format!("failed to read directory {}", src.display()))?
        {
            let entry = entry.wrap_err_with(|| format!("failed to read entry in {}", src.display()))?;
            let path = entry.path();
            let target = dst.join(entry.file_name());
            let file_type = entry
                .file_type()
                .wrap_err_with(|| format!("failed to read file type for {}", path.display()))?;
            if file_type.is_dir() {
                self.copy_dir_recursive(&path, &target)?;
            } else if file_type.is_file() {
                fs::copy(&path, &target).wrap_err_with(|| {
                    format!("failed to copy {} to {}", path.display(), target.display())
                })?;
            }
        }
        Ok(())
    }

    fn persist_completed_run(&mut self) {
        let Some(project) = self.active_project.as_ref() else {
            return;
        };
        if self.metrics_timeline.is_empty() {
            return;
        }

        let runs_dir = project.runs_dir();
        if let Err(error) = fs::create_dir_all(&runs_dir) {
            self.set_status(
                format!("Failed to prepare runs directory: {error}"),
                StatusKind::Error,
            );
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let slug = slugify_name(&self.training_config.experiment_name);
        let mut file_name = format!("{timestamp}_{slug}.json");
        let mut counter = 1;
        let mut path = runs_dir.join(&file_name);
        while path.exists() {
            file_name = format!("{timestamp}_{slug}_{counter}.json");
            path = runs_dir.join(&file_name);
            counter += 1;
        }

        let duration_seconds = self
            .current_run_start
            .and_then(|start| SystemTime::now().duration_since(start).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let run = SavedRun::new(
            file_name.clone(),
            project.name.clone(),
            self.training_config.experiment_name.clone(),
            match self.training_config.mode {
                TrainingMode::SingleAgent => "SB3".to_string(),
                TrainingMode::MultiAgent => "RLlib".to_string(),
            },
            timestamp,
            duration_seconds,
            self.metrics_timeline.clone(),
            self.training_output.clone(),
        );

        match runs::save_saved_run(&path, &run) {
            Ok(()) => {
                self.set_status(
                    format!("Saved run metrics to {}", path.display()),
                    StatusKind::Success,
                );
            }
            Err(error) => {
                self.set_status(format!("Failed to save run: {error}"), StatusKind::Error);
            }
        }

        self.current_run_start = None;
    }

    fn append_export_line(&mut self, line: impl Into<String>) {
        self.export_output.push(line.into());
        if self.export_output.len() > EXPORT_BUFFER_LIMIT {
            let overflow = self.export_output.len() - EXPORT_BUFFER_LIMIT;
            self.export_output.drain(0..overflow);
        }
        self.clamp_export_output_scroll();
    }

    fn clamp_export_output_scroll(&mut self) {
        if self.export_output.is_empty() {
            self.export_output_scroll = 0;
            return;
        }
        let max_offset = self.export_output.len().saturating_sub(1);
        if self.export_output_scroll > max_offset {
            self.export_output_scroll = max_offset;
        }
    }

    pub fn process_background_tasks(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        let mut simulator_events = Vec::new();
        let mut simulator_disconnected = false;
        let mut interface_events = Vec::new();
        let mut interface_disconnected = false;
        let mut export_events = Vec::new();
        let mut export_disconnected = false;

        if let Some(rx) = self.training_receiver.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(TrainingEvent::Finished(code)) => {
                        events.push(TrainingEvent::Finished(code));
                        break;
                    }
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        if let Some(rx) = self.simulator_receiver.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(SimulatorEvent::Finished(code)) => {
                        simulator_events.push(SimulatorEvent::Finished(code));
                        break;
                    }
                    Ok(event) => simulator_events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        simulator_disconnected = true;
                        break;
                    }
                }
            }
        }

        if let Some(rx) = self.interface_receiver.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(InterfaceEvent::Finished(code)) => {
                        interface_events.push(InterfaceEvent::Finished(code));
                        break;
                    }
                    Ok(event) => interface_events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        interface_disconnected = true;
                        break;
                    }
                }
            }
        }

        if let Some(rx) = self.export_receiver.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(ExportEvent::Finished(code)) => {
                        export_events.push(ExportEvent::Finished(code));
                        break;
                    }
                    Ok(event) => export_events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        export_disconnected = true;
                        break;
                    }
                }
            }
        }

        let mut finished = false;
        for event in events {
            match event {
                TrainingEvent::Line(line) => self.handle_training_line(line),
                TrainingEvent::Error(message) => {
                    self.append_training_line(format!("! {}", message));
                    self.set_status(message, StatusKind::Error);
                }
                TrainingEvent::Finished(code) => {
                    finished = true;
                    let message = match code {
                        Some(0) => "Training completed successfully.".to_string(),
                        Some(code) => format!("Training finished with exit code {code}."),
                        None => "Training finished.".to_string(),
                    };
                    self.set_status(message.clone(), StatusKind::Info);
                    self.append_training_line(message);
                }
            }
        }
        if finished || disconnected {
            self.training_running = false;
            self.training_receiver = None;
            self.training_cancel = None;
            if finished && !disconnected {
                self.persist_completed_run();
            }
            if disconnected {
                self.set_status(
                    "Training task disconnected unexpectedly.",
                    StatusKind::Warning,
                );
            }
        }

        let mut simulator_finished = false;
        for event in simulator_events {
            match event {
                SimulatorEvent::Line(line) => self.handle_simulator_line(line),
                SimulatorEvent::Error(message) => {
                    self.append_simulator_event_entry(SimulatorEventEntry {
                        timestamp: None,
                        kind: "error".into(),
                        message: message.clone(),
                        severity: SimulatorEventSeverity::Error,
                    });
                    self.set_status(message, StatusKind::Error);
                }
                SimulatorEvent::Finished(code) => {
                    simulator_finished = true;
                    let message = match code {
                        Some(0) => "Simulator finished cleanly.".to_string(),
                        Some(code) => format!("Simulator exited with code {code}."),
                        None => "Simulator finished.".to_string(),
                    };
                    self.append_simulator_event_entry(SimulatorEventEntry {
                        timestamp: None,
                        kind: "status".into(),
                        message: message.clone(),
                        severity: SimulatorEventSeverity::Info,
                    });
                    self.set_status(message, StatusKind::Info);
                }
            }
        }
        if simulator_finished || simulator_disconnected {
            self.simulator_running = false;
            self.simulator_receiver = None;
            self.simulator_cancel = None;
            if simulator_disconnected {
                self.append_simulator_event_entry(SimulatorEventEntry {
                    timestamp: None,
                    kind: "warning".into(),
                    message: "Simulator task disconnected unexpectedly.".to_string(),
                    severity: SimulatorEventSeverity::Warning,
                });
                self.set_status(
                    "Simulator task disconnected unexpectedly.",
                    StatusKind::Warning,
                );
            }
            if simulator_finished && !simulator_disconnected {
                self.simulator_status_line = Some("Simulator idle.".to_string());
            }
        }

        let mut interface_finished = false;
        for event in interface_events {
            match event {
                InterfaceEvent::Line(line) => self.handle_interface_line(line),
                InterfaceEvent::Error(message) => {
                    self.append_interface_event_entry(InterfaceEventEntry {
                        timestamp: None,
                        kind: "error".into(),
                        message: message.clone(),
                        severity: SimulatorEventSeverity::Error,
                    });
                    self.set_status(message, StatusKind::Error);
                }
                InterfaceEvent::Finished(code) => {
                    interface_finished = true;
                    let message = match code {
                        Some(0) => "Interface finished cleanly.".to_string(),
                        Some(code) => format!("Interface exited with code {code}."),
                        None => "Interface finished.".to_string(),
                    };
                    self.append_interface_event_entry(InterfaceEventEntry {
                        timestamp: None,
                        kind: "status".into(),
                        message: message.clone(),
                        severity: SimulatorEventSeverity::Info,
                    });
                    self.set_status(message, StatusKind::Info);
                }
            }
        }
        if interface_finished || interface_disconnected {
            self.interface_running = false;
            self.interface_receiver = None;
            self.interface_cancel = None;
            if interface_disconnected {
                self.append_interface_event_entry(InterfaceEventEntry {
                    timestamp: None,
                    kind: "warning".into(),
                    message: "Interface task disconnected unexpectedly.".to_string(),
                    severity: SimulatorEventSeverity::Warning,
                });
                self.set_status(
                    "Interface task disconnected unexpectedly.",
                    StatusKind::Warning,
                );
            }
            if interface_finished && !interface_disconnected {
                self.interface_status_line = Some("Interface idle.".to_string());
            }
        }

        let mut export_finished = false;
        for event in export_events {
            match event {
                ExportEvent::Line(line) => self.append_export_line(line),
                ExportEvent::Error(message) => {
                    self.append_export_line(format!("! {message}"));
                    self.set_status(message, StatusKind::Error);
                }
                ExportEvent::Finished(code) => {
                    export_finished = true;
                    let message = match code {
                        Some(0) => "Export completed successfully.".to_string(),
                        Some(code) => format!("Export finished with exit code {code}."),
                        None => "Export finished.".to_string(),
                    };
                    self.append_export_line(message.clone());
                    self.set_status(message, StatusKind::Info);
                }
            }
        }
        if export_finished || export_disconnected {
            self.export_running = false;
            self.export_receiver = None;
            self.export_cancel = None;
            if export_disconnected {
                self.set_status(
                    "Export task disconnected unexpectedly.",
                    StatusKind::Warning,
                );
            }
        }
    }

    fn refresh_projects(&mut self, prefer_path: Option<PathBuf>) -> Result<()> {
        self.projects = self.project_manager.list_projects()?;
        if let Some(path) = prefer_path {
            if let Some(index) = self.projects.iter().position(|info| info.logs_path == path) {
                self.selected_project = index;
            }
        }
        self.ensure_selection_valid();
        Ok(())
    }

    fn ensure_selection_valid(&mut self) {
        if self.projects.is_empty() {
            self.selected_project = 0;
        } else if self.selected_project >= self.projects.len() {
            self.selected_project = self.projects.len() - 1;
        }
    }

    fn append_training_line(&mut self, line: impl Into<String>) {
        self.training_output.push(line.into());
        if self.training_output.len() > TRAINING_BUFFER_LIMIT {
            let overflow = self.training_output.len() - TRAINING_BUFFER_LIMIT;
            self.training_output.drain(0..overflow);
        }
        self.clamp_training_output_scroll();
    }

    fn clamp_training_output_scroll(&mut self) {
        if self.training_output.is_empty() {
            self.training_output_scroll = 0;
            return;
        }
        if self.controller_settings.auto_scroll_training_log() {
            self.training_output_scroll = 0;
            return;
        }
        let max_offset = self.training_output.len().saturating_sub(1);
        if self.training_output_scroll > max_offset {
            self.training_output_scroll = max_offset;
        }
    }

    fn handle_training_line(&mut self, line: String) {
        if self.try_parse_metric_line(&line) {
            return;
        }
        let sanitized = sanitize_console_line(&line);
        self.append_training_line(sanitized);
    }

    fn try_parse_metric_line(&mut self, line: &str) -> bool {
        let payload = match line.strip_prefix(METRIC_PREFIX) {
            Some(payload) => payload,
            None => return false,
        };

        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            if let Some(sample) = MetricSample::from_value(
                &value,
                self.training_config.rllib_checkpoint_frequency as u64,
            ) {
                self.record_metric_sample(sample);
            }
        }
        true
    }

    fn record_metric_sample(&mut self, mut sample: MetricSample) {
        let now = Instant::now();
        let start = if let Some(start) = self.metric_timer_start {
            start
        } else {
            self.metric_timer_start = Some(now);
            now
        };

        let total_duration = now.saturating_duration_since(start).as_secs_f64().max(0.0);
        let iter_duration = if let Some(previous) = self.metric_last_sample_time {
            now.saturating_duration_since(previous)
                .as_secs_f64()
                .max(0.0)
        } else {
            total_duration
        };

        sample.set_time_total_s(total_duration);
        sample.set_time_this_iter_s(iter_duration);

        self.metric_last_sample_time = Some(now);
        let previous_offset = self.metrics_history_index;
        self.training_metrics.push(sample.clone());
        self.metrics_timeline.push(sample);

        if self.training_metrics.len() > TRAINING_METRIC_HISTORY_LIMIT {
            let excess = self.training_metrics.len() - TRAINING_METRIC_HISTORY_LIMIT;
            self.training_metrics.drain(0..excess);
        }
        if self.metrics_timeline.len() > TRAINING_METRIC_HISTORY_LIMIT {
            let excess = self.metrics_timeline.len() - TRAINING_METRIC_HISTORY_LIMIT;
            self.metrics_timeline.drain(0..excess);
        }
        if previous_offset != 0 {
            self.metrics_history_index = previous_offset.saturating_add(1);
        }
        let history_len = self.training_metrics_history().len();
        if history_len == 0 {
            self.metrics_history_index = 0;
        } else if self.metrics_history_index >= history_len {
            self.metrics_history_index = history_len.saturating_sub(1);
        }
        self.ensure_chart_metric_index();
    }

    fn append_simulator_event_entry(&mut self, entry: SimulatorEventEntry) {
        self.simulator_status_line = Some(entry.message.clone());
        self.simulator_event_log.push(entry);
        if self.simulator_event_log.len() > SIM_EVENT_BUFFER_LIMIT {
            let overflow = self.simulator_event_log.len() - SIM_EVENT_BUFFER_LIMIT;
            self.simulator_event_log.drain(0..overflow);
        }
        if self.controller_settings.auto_scroll_training_log() {
            self.simulator_event_scroll = 0;
        } else {
            self.clamp_simulator_event_scroll();
        }
    }

    fn clamp_simulator_event_scroll(&mut self) {
        if self.simulator_event_log.is_empty() {
            self.simulator_event_scroll = 0;
            return;
        }
        let max_offset = self.simulator_event_log.len().saturating_sub(1);
        if self.simulator_event_scroll > max_offset {
            self.simulator_event_scroll = max_offset;
        }
    }

    fn handle_simulator_line(&mut self, line: String) {
        if self.try_parse_simulator_action(&line) {
            return;
        }
        if self.try_parse_simulator_event(&line) {
            return;
        }
        self.append_simulator_event_entry(SimulatorEventEntry {
            timestamp: None,
            kind: "stdout".into(),
            message: line,
            severity: SimulatorEventSeverity::Info,
        });
    }

    fn try_parse_simulator_event(&mut self, line: &str) -> bool {
        let payload = match line.strip_prefix(SIM_EVENT_PREFIX) {
            Some(payload) => payload,
            None => return false,
        };

        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            let timestamp = value.get("timestamp").and_then(|ts| match ts {
                Value::String(s) => Some(s.clone()),
                Value::Number(n) => Some(n.to_string()),
                _ => None,
            });
            let kind = value
                .get("kind")
                .and_then(|k| k.as_str())
                .unwrap_or("event")
                .to_string();
            let severity = match kind.as_str() {
                "error" => SimulatorEventSeverity::Error,
                "warning" => SimulatorEventSeverity::Warning,
                _ => SimulatorEventSeverity::Info,
            };
            let message = value
                .get("message")
                .and_then(|msg| msg.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    if kind == "connected" {
                        let attempt = value
                            .get("attempt")
                            .and_then(|a| a.as_u64())
                            .map(|a| format!("attempt {a}"))
                            .unwrap_or_else(|| "attempt unknown".to_string());
                        let mode = value
                            .get("mode")
                            .and_then(|m| m.as_str())
                            .unwrap_or("single");
                        Some(format!("Connected ({attempt}, mode: {mode})"))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| payload.to_string());
            self.append_simulator_event_entry(SimulatorEventEntry {
                timestamp,
                kind,
                message,
                severity,
            });
        }
        true
    }

    fn try_parse_simulator_action(&mut self, line: &str) -> bool {
        let payload = match line.strip_prefix(SIM_ACTION_PREFIX) {
            Some(payload) => payload,
            None => return false,
        };

        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            self.update_simulator_actions_from_payload(&value);
        }
        true
    }

    fn update_simulator_actions_from_payload(&mut self, value: &Value) {
        let episode = value.get("episode").and_then(|v| v.as_u64());
        let step = value.get("step").and_then(|v| v.as_u64());
        let mode = value
            .get("mode")
            .and_then(|m| m.as_str())
            .map(|m| {
                if m.eq_ignore_ascii_case("multi") {
                    SimulatorMode::Multi
                } else {
                    SimulatorMode::Single
                }
            })
            .unwrap_or(self.simulator_config.mode);

        let mut rows = Vec::new();
        if let Some(agents) = value.get("agents").and_then(|a| a.as_array()) {
            for agent in agents {
                let agent_id = agent
                    .get("agent")
                    .and_then(|a| a.as_str())
                    .unwrap_or("agent")
                    .to_string();
                let policy = agent
                    .get("policy")
                    .and_then(|p| p.as_str())
                    .map(|p| p.to_string());
                let action = format_action_value(
                    agent.get("action").unwrap_or(&Value::Null),
                    SIM_ACTION_VALUE_MAX_LEN,
                );
                let reward = agent.get("reward").and_then(|r| r.as_f64());
                let terminated = agent
                    .get("terminated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let truncated = agent
                    .get("truncated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let info = agent
                    .get("info")
                    .filter(|info| !info.is_null())
                    .map(|info| truncate_display_value(info, SIM_INFO_VALUE_MAX_LEN));
                rows.push(SimulatorAgentRow {
                    episode,
                    step,
                    agent_id,
                    policy,
                    action,
                    reward,
                    terminated,
                    truncated,
                    info,
                });
            }
        }

        rows.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));
        let new_count = rows.len();
        self.simulator_actions.extend(rows);
        if self.simulator_actions.len() > SIM_ACTION_HISTORY_LIMIT {
            let overflow = self.simulator_actions.len() - SIM_ACTION_HISTORY_LIMIT;
            self.simulator_actions.drain(0..overflow);
        }

        if !self.simulator_compact_user_override {
            if new_count > SIM_ACTION_AUTO_COMPACT_THRESHOLD {
                self.simulator_compact_view = true;
            } else {
                self.simulator_compact_view = false;
            }
        }

        if self.controller_settings.auto_scroll_training_log() {
            self.simulator_actions_scroll = 0;
        } else {
            let max_offset = self.simulator_actions.len();
            if self.simulator_actions_scroll > max_offset {
                self.simulator_actions_scroll = max_offset;
            }
        }

        self.simulator_action_meta = Some(SimulatorActionMeta {
            episode,
            step,
            mode,
            total_agents: new_count,
        });

        if let Some(meta) = self.simulator_action_meta {
            let summary = match (meta.episode(), meta.step()) {
                (Some(ep), Some(step)) => format!(
                    "{} episode {} step {} ({} agents)",
                    meta.mode().label(),
                    ep,
                    step,
                    meta.total_agents()
                ),
                _ => format!(
                    "{} action update ({} agents)",
                    meta.mode().label(),
                    meta.total_agents()
                ),
            };
            self.simulator_status_line = Some(summary);
        }
    }

    fn handle_interface_line(&mut self, line: String) {
        if self.try_parse_interface_action(&line) {
            return;
        }
        if self.try_parse_interface_event(&line) {
            return;
        }
        self.append_interface_event_entry(InterfaceEventEntry {
            timestamp: None,
            kind: "stdout".into(),
            message: line,
            severity: SimulatorEventSeverity::Info,
        });
    }

    fn try_parse_interface_event(&mut self, line: &str) -> bool {
        let payload = match line.strip_prefix(INTERFACE_EVENT_PREFIX) {
            Some(payload) => payload,
            None => return false,
        };

        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            let timestamp = value.get("timestamp").and_then(|ts| match ts {
                Value::String(s) => Some(s.clone()),
                Value::Number(n) => Some(n.to_string()),
                _ => None,
            });
            let kind = value
                .get("kind")
                .and_then(|k| k.as_str())
                .unwrap_or("event")
                .to_string();
            let severity = match kind.as_str() {
                "error" => SimulatorEventSeverity::Error,
                "warning" => SimulatorEventSeverity::Warning,
                _ => SimulatorEventSeverity::Info,
            };
            let message = value
                .get("message")
                .and_then(|msg| msg.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    if kind == "connected" {
                        let attempt = value
                            .get("attempt")
                            .and_then(|a| a.as_u64())
                            .map(|a| format!("attempt {a}"))
                            .unwrap_or_else(|| "attempt unknown".to_string());
                        let mode = value
                            .get("mode")
                            .and_then(|m| m.as_str())
                            .unwrap_or("single");
                        Some(format!("Connected ({attempt}, mode: {mode})"))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| payload.to_string());
            self.append_interface_event_entry(InterfaceEventEntry {
                timestamp,
                kind,
                message,
                severity,
            });
        }
        true
    }

    fn try_parse_interface_action(&mut self, line: &str) -> bool {
        let payload = match line.strip_prefix(INTERFACE_ACTION_PREFIX) {
            Some(payload) => payload,
            None => return false,
        };

        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            self.update_interface_actions_from_payload(&value);
        }
        true
    }

    fn update_interface_actions_from_payload(&mut self, value: &Value) {
        let episode = value.get("episode").and_then(|v| v.as_u64());
        let step = value.get("step").and_then(|v| v.as_u64());
        let mode = value
            .get("mode")
            .and_then(|m| m.as_str())
            .map(|m| {
                if m.eq_ignore_ascii_case("multi") {
                    SimulatorMode::Multi
                } else {
                    SimulatorMode::Single
                }
            })
            .unwrap_or(self.interface_config.mode);

        let mut rows = Vec::new();
        if let Some(agents) = value.get("agents").and_then(|a| a.as_array()) {
            for agent in agents {
                let agent_id = agent
                    .get("agent")
                    .and_then(|a| a.as_str())
                    .unwrap_or("agent")
                    .to_string();
                let policy = agent
                    .get("policy")
                    .and_then(|p| p.as_str())
                    .map(|p| p.to_string());
                let observation = agent
                    .get("observation")
                    .map(|obs| truncate_display_value(obs, SIM_ACTION_VALUE_MAX_LEN));
                let action = format_action_value(
                    agent.get("action").unwrap_or(&Value::Null),
                    SIM_ACTION_VALUE_MAX_LEN,
                );
                let reward = agent.get("reward").and_then(|r| r.as_f64());
                let terminated = agent
                    .get("terminated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let truncated = agent
                    .get("truncated")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let info = agent
                    .get("info")
                    .filter(|info| !info.is_null())
                    .map(|info| truncate_display_value(info, SIM_INFO_VALUE_MAX_LEN));
                rows.push(InterfaceAgentRow {
                    episode,
                    step,
                    agent_id,
                    policy,
                    observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    info,
                });
            }
        }

        rows.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));
        let new_count = rows.len();
        self.interface_actions.extend(rows);
        if self.interface_actions.len() > SIM_ACTION_HISTORY_LIMIT {
            let overflow = self.interface_actions.len() - SIM_ACTION_HISTORY_LIMIT;
            self.interface_actions.drain(0..overflow);
        }

        if !self.interface_compact_user_override {
            if new_count > SIM_ACTION_AUTO_COMPACT_THRESHOLD {
                self.interface_compact_view = true;
            } else {
                self.interface_compact_view = false;
            }
        }

        if self.controller_settings.auto_scroll_training_log() {
            self.interface_actions_scroll = 0;
        } else {
            let max_offset = self.interface_actions.len();
            if self.interface_actions_scroll > max_offset {
                self.interface_actions_scroll = max_offset;
            }
        }

        self.interface_action_meta = Some(InterfaceActionMeta {
            episode,
            step,
            mode,
            total_agents: new_count,
        });

        if let Some(meta) = self.interface_action_meta {
            let summary = match (meta.episode(), meta.step()) {
                (Some(ep), Some(step)) => format!(
                    "{} episode {} step {} ({} agents)",
                    meta.mode().label(),
                    ep,
                    step,
                    meta.total_agents()
                ),
                _ => format!(
                    "{} action update ({} agents)",
                    meta.mode().label(),
                    meta.total_agents()
                ),
            };
            self.interface_status_line = Some(summary);
        }
    }
}

fn sanitize_console_line(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if matches!(chars.peek(), Some('[')) {
                chars.next();
                while let Some(next) = chars.next() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
                continue;
            }
        }
        match ch {
            '\r' | '\x07' | '\x0b' | '\x0c' => continue,
            _ => output.push(ch),
        }
    }
    output
}

fn format_f64(value: f64) -> String {
    let mut string = format!("{value:.6}");
    while string.contains('.') && string.ends_with('0') {
        string.pop();
    }
    if string.ends_with('.') {
        string.pop();
    }
    if string.is_empty() {
        string.push('0');
    }
    string
}

fn truncate_display_value(value: &Value, limit: usize) -> String {
    let raw = match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(num) => num
            .as_f64()
            .map(format_number_short)
            .unwrap_or_else(|| num.to_string()),
        Value::String(s) => s.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string()),
    };
    truncate_string(raw, limit)
}

fn format_action_value(value: &Value, limit: usize) -> String {
    fn render(value: &Value) -> String {
        match value {
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(num) => num
                .as_f64()
                .map(format_number_short)
                .unwrap_or_else(|| num.to_string()),
            Value::String(s) => format!("\"{}\"", s),
            Value::Array(items) => {
                let parts = items.iter().map(render).collect::<Vec<_>>();
                format!("[{}]", parts.join(", "))
            }
            Value::Object(map) => {
                let parts = map
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, render(v)))
                    .collect::<Vec<_>>();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }

    let rendered = render(value);
    truncate_string(rendered, limit)
}

fn truncate_string(mut text: String, limit: usize) -> String {
    if limit == 0 {
        return String::new();
    }
    if text.len() > limit {
        let cutoff = if limit > 3 { limit - 3 } else { limit };
        text.truncate(cutoff);
        text.push_str("...");
    }
    text
}

fn escape_single_quotes(input: &str) -> String {
    input.replace('\'', "''")
}

fn format_number_short(value: f64) -> String {
    let mut string = format!("{value:.4}");
    while string.contains('.') && string.ends_with('0') {
        string.pop();
    }
    if string.ends_with('.') {
        string.pop();
    }
    if string.is_empty() {
        string.push('0');
    }
    string
}

fn format_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_usize_list_compact(values: &[usize]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_usize_list(input: &str) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    for token in input.split(|c| matches!(c, ',' | ';' | ' ' | '\t')) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value: usize = token
            .parse()
            .wrap_err_with(|| format!("Invalid layer size '{token}'"))?;
        if value == 0 {
            bail!("Layer sizes must be greater than zero");
        }
        values.push(value);
    }
    if values.is_empty() {
        bail!("At least one layer size is required");
    }
    Ok(values)
}

impl Index<TabId> for App {
    type Output = Tab;

    fn index(&self, index: TabId) -> &Self::Output {
        self.tabs
            .iter()
            .find(|tab| tab.id == index)
            .expect("tab exists")
    }
}

fn default_projects_root() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("CONTROLLER_PROJECTS_ROOT") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return Ok(PathBuf::from(trimmed));
        }
    }

    if let Ok(xdg_data_home) = std::env::var("XDG_DATA_HOME") {
        let mut root = PathBuf::from(xdg_data_home);
        root.push("godot_rl_controller");
        root.push("projects");
        return Ok(root);
    }

    let home = std::env::var("HOME").wrap_err("HOME not set; cannot determine projects root")?;
    let mut root = PathBuf::from(home);
    root.push(".local");
    root.push("share");
    root.push("godot_rl_controller");
    root.push("projects");
    Ok(root)
}

fn determine_python_command() -> String {
    if let Ok(cmd) = std::env::var("CONTROLLER_PYTHON_BIN") {
        if !cmd.trim().is_empty() {
            return cmd;
        }
    }
    if let Some(cmd) = EMBEDDED_PYTHON_BIN {
        if !cmd.trim().is_empty() {
            return cmd.to_string();
        }
    }

    std::env::var("PYTHON")
        .or_else(|_| std::env::var("PYTHON3"))
        .unwrap_or_else(|_| "python3".to_string())
}

fn slugify_name(name: &str) -> String {
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
        "run".to_string()
    } else {
        slug.trim_matches('-').to_string()
    }
}

fn controller_scripts_root() -> Option<PathBuf> {
    if let Ok(value) = std::env::var("CONTROLLER_SCRIPTS_ROOT") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    EMBEDDED_SCRIPT_ROOT
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
}

fn find_script(base_dir: &Path, script_name: &str) -> Result<PathBuf> {
    // First check in the project directory
    let project_script = base_dir.join(script_name);
    if project_script.exists() {
        return Ok(project_script);
    }

    // Then check in the controller root (parent of projects dir)
    if let Some(parent) = base_dir.parent() {
        if let Some(grandparent) = parent.parent() {
            let root_script = grandparent.join(script_name);
            if root_script.exists() {
                return Ok(root_script);
            }
        }
    }

    if let Some(root) = controller_scripts_root() {
        let embedded = root.join(script_name);
        if embedded.exists() {
            return Ok(embedded);
        }
    }

    // Finally check current working directory
    let cwd = std::env::current_dir().wrap_err("failed to determine current directory")?;
    let cwd_script = cwd.join(script_name);
    if cwd_script.exists() {
        return Ok(cwd_script);
    }

    bail!("Training script '{}' not found", script_name)
}

fn spawn_export_task(
    tx: Sender<ExportEvent>,
    command: String,
    script_path: PathBuf,
    args: Vec<String>,
    workdir: PathBuf,
    cancel_rx: Receiver<()>,
) {
    thread::spawn(move || {
        if !script_path.exists() {
            let _ = tx.send(ExportEvent::Error(format!(
                "Script not found: {}",
                script_path.display()
            )));
            let _ = tx.send(ExportEvent::Finished(None));
            return;
        }

        let mut cmd = Command::new(&command);
        cmd.arg("-u")
            .arg(&script_path)
            .args(&args)
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(ExportEvent::Error(format!(
                    "Failed to start export command: {error}"
                )));
                let _ = tx.send(ExportEvent::Finished(None));
                return;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stdout) = stdout {
            let tx_stdout = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if tx_stdout.send(ExportEvent::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = stderr {
            let tx_stderr = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let formatted = format!("! {line}");
                    if tx_stderr.send(ExportEvent::Line(formatted)).is_err() {
                        break;
                    }
                }
            });
        }

        let cancel_thread = thread::spawn(move || {
            if cancel_rx.recv().is_ok() {
                #[cfg(unix)]
                {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                }
                #[cfg(not(unix))]
                {
                    // Fallback for non-Unix systems; the process will be terminated below.
                }
            }
        });

        let status = child.wait();

        drop(cancel_thread);

        match status {
            Ok(status) => {
                let code = status.code();
                let message = match code {
                    Some(code) => format!("Process exited with code {code}."),
                    None => "Process terminated by signal.".to_string(),
                };
                let _ = tx.send(ExportEvent::Line(message));
                let _ = tx.send(ExportEvent::Finished(code));
            }
            Err(error) => {
                let _ = tx.send(ExportEvent::Error(format!(
                    "Failed to wait for process: {error}"
                )));
                let _ = tx.send(ExportEvent::Finished(None));
            }
        }
    });
}

fn spawn_training_task(
    tx: Sender<TrainingEvent>,
    command: String,
    script_path: PathBuf,
    args: Vec<String>,
    workdir: PathBuf,
    cancel_rx: Receiver<()>,
) {
    thread::spawn(move || {
        if !script_path.exists() {
            let _ = tx.send(TrainingEvent::Error(format!(
                "Script not found: {}",
                script_path.display()
            )));
            let _ = tx.send(TrainingEvent::Finished(None));
            return;
        }

        // Use -u flag to force unbuffered output from Python
        let mut cmd = Command::new(&command);
        cmd.arg("-u") // Force Python to use unbuffered mode
            .arg(&script_path)
            .args(&args)
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("CONTROLLER_METRICS", "1"); // Enable metrics output

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(TrainingEvent::Error(format!(
                    "Failed to start training command: {error}"
                )));
                let _ = tx.send(TrainingEvent::Finished(None));
                return;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stdout) = stdout {
            let tx_stdout = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if tx_stdout.send(TrainingEvent::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = stderr {
            let tx_stderr = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let formatted = format!("! {}", line);
                    if tx_stderr.send(TrainingEvent::Line(formatted)).is_err() {
                        break;
                    }
                }
            });
        }

        // Handle cancellation with SIGINT (like Ctrl+C)
        let cancel_thread = thread::spawn(move || {
            if cancel_rx.recv().is_ok() {
                // On Unix, send SIGINT to emulate keyboard interrupt
                #[cfg(unix)]
                {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                }
                // On Windows, we have to use kill
                #[cfg(not(unix))]
                {
                    // This is a fallback for non-Unix systems
                    // Windows doesn't have SIGINT, so we use terminate
                }
            }
        });

        let status = child.wait();

        // Make sure cancel thread completes
        drop(cancel_thread);

        match status {
            Ok(status) => {
                let code = status.code();
                let message = match code {
                    Some(code) => format!("Process exited with code {code}."),
                    None => "Process terminated by signal.".to_string(),
                };
                let _ = tx.send(TrainingEvent::Line(message));
                let _ = tx.send(TrainingEvent::Finished(code));
            }
            Err(error) => {
                let _ = tx.send(TrainingEvent::Error(format!(
                    "Failed to wait for process: {error}"
                )));
                let _ = tx.send(TrainingEvent::Finished(None));
            }
        }
    });
}

fn spawn_simulator_task(
    tx: Sender<SimulatorEvent>,
    command: String,
    script_path: PathBuf,
    args: Vec<String>,
    workdir: PathBuf,
    cancel_rx: Receiver<()>,
) {
    thread::spawn(move || {
        if !script_path.exists() {
            let _ = tx.send(SimulatorEvent::Error(format!(
                "Script not found: {}",
                script_path.display()
            )));
            let _ = tx.send(SimulatorEvent::Finished(None));
            return;
        }

        let mut cmd = Command::new(&command);
        cmd.arg("-u")
            .arg(&script_path)
            .args(&args)
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(SimulatorEvent::Error(format!(
                    "Failed to start simulator command: {error}"
                )));
                let _ = tx.send(SimulatorEvent::Finished(None));
                return;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stdout) = stdout {
            let tx_stdout = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if tx_stdout.send(SimulatorEvent::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = stderr {
            let tx_stderr = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let formatted = format!("! {}", line);
                    if tx_stderr.send(SimulatorEvent::Line(formatted)).is_err() {
                        break;
                    }
                }
            });
        }

        let mut cancel_requested = false;
        let mut kill_deadline: Option<Instant> = None;

        let status_result = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => {}
                Err(error) => break Err(error),
            }

            if !cancel_requested {
                match cancel_rx.try_recv() {
                    Ok(_) => {
                        cancel_requested = true;
                        kill_deadline = Some(Instant::now() + Duration::from_secs(10));
                        let _ = tx.send(SimulatorEvent::Line(
                            "Cancellation requested (sending Ctrl+C)...".into(),
                        ));
                        #[cfg(unix)]
                        {
                            use nix::sys::signal::{kill, Signal};
                            use nix::unistd::Pid;
                            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                        }
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {}
                }
            } else if let Some(deadline) = kill_deadline {
                if Instant::now() >= deadline {
                    let _ = tx.send(SimulatorEvent::Line(
                        "Simulator unresponsive; forcing termination...".into(),
                    ));
                    if let Err(error) = child.kill() {
                        let _ = tx.send(SimulatorEvent::Error(format!(
                            "Failed to terminate simulator: {error}"
                        )));
                        break Err(error);
                    } else {
                        break child.wait();
                    }
                }
            }

            thread::sleep(Duration::from_millis(100));
        };

        match status_result {
            Ok(status) => {
                let code = status.code();
                let message = match code {
                    Some(code) => format!("Process exited with code {code}."),
                    None => "Process terminated by signal.".to_string(),
                };
                let _ = tx.send(SimulatorEvent::Line(message));
                let _ = tx.send(SimulatorEvent::Finished(code));
            }
            Err(error) => {
                let _ = tx.send(SimulatorEvent::Error(format!(
                    "Failed to wait for process: {error}"
                )));
                let _ = tx.send(SimulatorEvent::Finished(None));
            }
        }
    });
}

fn spawn_interface_task(
    tx: Sender<InterfaceEvent>,
    command: String,
    script_path: PathBuf,
    args: Vec<String>,
    workdir: PathBuf,
    cancel_rx: Receiver<()>,
) {
    thread::spawn(move || {
        if !script_path.exists() {
            let _ = tx.send(InterfaceEvent::Error(format!(
                "Script not found: {}",
                script_path.display()
            )));
            let _ = tx.send(InterfaceEvent::Finished(None));
            return;
        }

        let mut cmd = Command::new(&command);
        cmd.arg("-u")
            .arg(&script_path)
            .args(&args)
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(InterfaceEvent::Error(format!(
                    "Failed to start interface command: {error}"
                )));
                let _ = tx.send(InterfaceEvent::Finished(None));
                return;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stdout) = stdout {
            let tx_stdout = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if tx_stdout.send(InterfaceEvent::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = stderr {
            let tx_stderr = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let formatted = format!("! {}", line);
                    if tx_stderr.send(InterfaceEvent::Line(formatted)).is_err() {
                        break;
                    }
                }
            });
        }

        let mut cancel_requested = false;
        let mut kill_deadline: Option<Instant> = None;

        let status_result = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => {}
                Err(error) => break Err(error),
            }

            if !cancel_requested {
                match cancel_rx.try_recv() {
                    Ok(_) => {
                        cancel_requested = true;
                        kill_deadline = Some(Instant::now() + Duration::from_secs(10));
                        let _ = tx.send(InterfaceEvent::Line(
                            "Cancellation requested (sending Ctrl+C)...".into(),
                        ));
                        #[cfg(unix)]
                        {
                            use nix::sys::signal::{kill, Signal};
                            use nix::unistd::Pid;
                            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                        }
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {}
                }
            } else if let Some(deadline) = kill_deadline {
                if Instant::now() >= deadline {
                    let _ = tx.send(InterfaceEvent::Line(
                        "Interface unresponsive; forcing termination...".into(),
                    ));
                    if let Err(error) = child.kill() {
                        let _ = tx.send(InterfaceEvent::Error(format!(
                            "Failed to terminate interface: {error}"
                        )));
                        break Err(error);
                    } else {
                        break child.wait();
                    }
                }
            }

            thread::sleep(Duration::from_millis(100));
        };

        match status_result {
            Ok(status) => {
                let code = status.code();
                let message = match code {
                    Some(code) => format!("Process exited with code {code}."),
                    None => "Process terminated by signal.".to_string(),
                };
                let _ = tx.send(InterfaceEvent::Line(message));
                let _ = tx.send(InterfaceEvent::Finished(code));
            }
            Err(error) => {
                let _ = tx.send(InterfaceEvent::Error(format!(
                    "Failed to wait for process: {error}"
                )));
                let _ = tx.send(InterfaceEvent::Finished(None));
            }
        }
    });
}

fn normalize_rllib_config_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default_rllib_config_file();
    }
    let path = Path::new(trimmed);
    if path.is_absolute() || path_starts_with_config_dir(path) {
        trimmed.to_string()
    } else {
        Path::new(PROJECT_CONFIG_DIR)
            .join(path)
            .to_string_lossy()
            .to_string()
    }
}

fn path_starts_with_config_dir(path: &Path) -> bool {
    for component in path.components() {
        match component {
            Component::CurDir => continue,
            Component::Normal(name) => return name == OsStr::new(PROJECT_CONFIG_DIR),
            _ => return false,
        }
    }
    false
}

fn write_rllib_config(path: &Path, config: &TrainingConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create directory {}", parent.display()))?;
    }

    let env_path = config.env_path.trim();
    let escaped_env_path = if env_path.is_empty() {
        String::from("''")
    } else {
        format!("'{}'", escape_single_quotes(env_path))
    };

    let show_window = if config.rllib_show_window {
        "true"
    } else {
        "false"
    };
    let algorithm = config.rllib_algorithm.trainer_name();
    let action_repeat = config.rllib_env_action_repeat;
    let speedup = config.rllib_env_speedup;
    let lr = format_f64(config.rllib_lr);
    let lambda = format_f64(config.rllib_lambda);
    let gamma = format_f64(config.rllib_gamma);
    let vf_loss_coeff = format_f64(config.rllib_vf_loss_coeff);
    let entropy_coeff = format_f64(config.rllib_entropy_coeff);
    let clip_param_comment = format_f64(config.rllib_clip_param);
    let grad_clip_comment = format_f64(config.rllib_grad_clip);
    let rollout_fragment_length = config.rllib_rollout_fragment_length;
    let sgd_minibatch_size = config.rllib_sgd_minibatch_size;
    let num_workers = config.rllib_num_workers;
    let num_envs_per_worker = config.rllib_num_envs_per_worker;
    let train_batch_size = config.rllib_train_batch_size;
    let num_sgd_iter = config.rllib_num_sgd_iter;
    let batch_mode = &config.rllib_batch_mode;
    let framework = &config.rllib_framework;
    let checkpoint_frequency = config.rllib_checkpoint_frequency;
    let num_gpus = format_f64(config.rllib_num_gpus);
    let stop_line = match config.rllib_stop_mode {
        RllibStopMode::TimeSeconds => {
            format!("    time_total_s: {}", config.rllib_stop_time_seconds)
        }
        RllibStopMode::Timesteps => format!("    timesteps_total: {}", config.timesteps),
    };

    let format_list = |values: &[usize]| {
        if values.is_empty() {
            String::from("[]")
        } else {
            format!(
                "[{}]",
                values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    };

    let fcnet_hiddens = format_list(&config.rllib_fcnet_hiddens);
    let cnn_channels = format_list(&config.rllib_cnn_channels);
    let include_prev_actions = if config.rllib_lstm_include_prev_actions {
        "true"
    } else {
        "false"
    };
    let model_block = match config.rllib_policy_type {
        PolicyType::Mlp => format!(
            "    model:\n        vf_share_layers: False\n        fcnet_hiddens: {fcnet_hiddens}\n"
        ),
        PolicyType::Cnn => format!(
            "    model:\n        vf_share_layers: False\n        custom_model: tui_cnn\n        custom_model_config:\n            channels: {cnn_channels}\n            fcnet_hiddens: {fcnet_hiddens}\n"
        ),
        PolicyType::Lstm => format!(
            "    _disable_action_flattening: true\n    model:\n        vf_share_layers: False\n        use_lstm: true\n        max_seq_len: {max_seq_len}\n        lstm_cell_size: {hidden}\n        lstm_use_prev_action: {include_prev}\n        lstm_use_prev_reward: false\n        fcnet_hiddens: {fcnet_hiddens}\n",
            hidden = config.rllib_lstm_cell_size,
            max_seq_len = config.rllib_max_seq_len,
            include_prev = include_prev_actions
        ),
        PolicyType::Grn => format!(
            "    model:\n        vf_share_layers: False\n        custom_model: tui_grn\n        custom_model_config:\n            hidden_size: {hidden}\n            fcnet_hiddens: {fcnet_hiddens}\n",
            hidden = config.rllib_grn_hidden_size
        ),
    };

    let content = format!(
        "algorithm: {algorithm}\n\n# Multi-agent-env setting:\n# If true:\n# - Any AIController with done = true will receive zeroes as action values until all AIControllers are done, an episode ends at that point.\n# - ai_controller.needs_reset will also be set to true every time a new episode begins (but you can ignore it in your env if needed).\n# If false:\n# - AIControllers auto-reset in Godot and will receive actions after setting done = true.\n# - Each AIController has its own episodes that can end/reset at any point.\n# Set to false if you have a single policy name for all agents set in AIControllers\nenv_is_multiagent: true\n\ncheckpoint_frequency: {checkpoint_frequency}\n\n# You can set one or more stopping criteria\nstop:\n    #episode_reward_mean: 0\n    #training_iteration: 1000\n    #timesteps_total: 10000\n{stop_line}\n\nconfig:\n    env: godot\n    env_config:\n      env_path: {escaped_env_path} # Set your env path here (exported executable from Godot) - e.g. env_path: 'env_path.exe' on Windows\n      action_repeat: {action_repeat} # Doesn't need to be set here, you can set this in sync node in Godot editor as well\n      show_window: {show_window} # Displays game window while training. Might be faster when false in some cases, turning off also reduces GPU usage if you don't need rendering.\n      speedup: {speedup} # Speeds up Godot physics\n\n    framework: {framework} # ONNX models exported with torch are compatible with the current Godot RL Agents Plugin\n\n    lr: {lr}\n    lambda: {lambda}\n    gamma: {gamma}\n\n    vf_loss_coeff: {vf_loss_coeff}\n    vf_clip_param: .inf\n    #clip_param: {clip_param_comment}\n    entropy_coeff: {entropy_coeff}\n    entropy_coeff_schedule: null\n    #grad_clip: {grad_clip_comment}\n\n    normalize_actions: False\n    clip_actions: True # During onnx inference we simply clip the actions to [-1.0, 1.0] range, set here to match\n\n    rollout_fragment_length: {rollout_fragment_length}\n    sgd_minibatch_size: {sgd_minibatch_size}\n    minibatch_size: {sgd_minibatch_size}\n    num_workers: {num_workers}\n    num_envs_per_worker: {num_envs_per_worker} # This will be set automatically if not multi-agent. If multi-agent, changing this changes how many envs to launch per worker.\n    sample_timeout_s: 120\n    train_batch_size: {train_batch_size}\n\n    num_sgd_iter: {num_sgd_iter}\n    batch_mode: {batch_mode}\n\n    num_gpus: {num_gpus}\n{model_block}"
    );

    fs::write(path, content)
        .wrap_err_with(|| format!("failed to write RLlib config to {}", path.display()))?;

    Ok(())
}
