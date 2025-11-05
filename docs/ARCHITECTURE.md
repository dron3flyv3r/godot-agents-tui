# Architecture Documentation

This document describes the technical architecture of the Godot RL Training Controller.

## Overview

The controller is a Rust-based TUI (Terminal User Interface) application that orchestrates Python-based RL training workflows. It provides a user-friendly interface for managing complex training configurations while maintaining the flexibility and power of command-line tools.

## System Components

### 1. Rust Controller (TUI Application)

**Technology Stack:**
- **Language**: Rust 2021 edition
- **TUI Framework**: [Ratatui](https://github.com/ratatui-org/ratatui) (v0.29.0)
- **Terminal Backend**: [Crossterm](https://github.com/crossterm-rs/crossterm) (v0.28.1)
- **Error Handling**: [color-eyre](https://github.com/eyre-rs/eyre) (v0.6.3)

**Core Modules:**

#### `main.rs`
- Application entry point
- Event loop management
- Keyboard input routing
- Terminal setup and teardown
- Cross-platform signal handling

#### `app.rs` (5364 lines)
- Central application state
- Project management logic
- Training process orchestration
- Metrics collection and parsing
- Export functionality
- Configuration management

**Key Structures:**
```rust
pub struct App {
    // Tab navigation
    tabs: Vec<TabInfo>,
    active_tab_index: usize,
    
    // Project management
    project_manager: ProjectManager,
    active_project: Option<ProjectInfo>,
    
    // Training state
    training_process: Option<TrainingProcess>,
    training_output: Vec<String>,
    training_metrics: Vec<MetricSample>,
    
    // Export state
    export_process: Option<ExportProcess>,
    export_output: Vec<String>,
    
    // UI state
    input_mode: InputMode,
    status_message: Option<StatusMessage>,
    // ... more fields
}
```

#### `ui.rs` (2729 lines)
- All rendering logic
- Widget composition
- Layout management
- Color schemes and styling
- Chart rendering for metrics

**Key Features:**
- Responsive layouts adapting to terminal size
- Real-time chart updates
- Syntax highlighting for configuration
- Alphanumeric sorting for policy names

#### `project.rs`
- Project data structures
- Project file I/O
- Configuration serialization
- Project indexing

### 2. Python Training Scripts

#### `stable_baselines3_training_script.py`
**Purpose**: Single-agent training orchestration

**Key Components:**
- PPO algorithm implementation from SB3
- Godot environment wrapper integration
- Custom callback system for metrics
- Checkpoint management
- ONNX export integration

**Callback System:**
```python
class ControllerMetricsCallback(BaseCallback):
    """Emits structured metrics to stdout for controller parsing"""
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._emit_metrics()
        return True
```

**Metrics Format:**
```
@METRIC {"kind": "iteration", "training_iteration": 10, ...}
```

#### `rllib_training_script.py`
**Purpose**: Multi-agent training orchestration

**Key Components:**
- RLlib algorithm configuration
- Multi-agent policy mapping
- PettingZoo wrapper for multi-agent
- Custom callbacks for metric emission
- Checkpoint frequency control

**Callback System:**
```python
class ControllerMetricsCallback(TrainCallbackBase, tune.Callback):
    """Emits structured metrics for multi-agent training"""
    
    def on_trial_result(self, iteration, trials, trial, result, **info):
        self._emit_result(result)
```

### 3. ONNX Export Scripts

#### `convert_sb3_to_onnx.py`
**Purpose**: Convert SB3 models to ONNX format

**Process:**
1. Load SB3 model from `.zip` file
2. Extract policy network
3. Wrap in `OnnxablePolicy` for proper I/O signatures
4. Export using `torch.onnx.export()`
5. Verify ONNX model
6. Convert to target IR version

**Key Functions:**
```python
def load_model(model_path, algo_name=None) -> BaseAlgorithm
def export_model_to_onnx(model, export_path, **options) -> None
```

#### `convert_rllib_to_onnx.py`
**Purpose**: Convert RLlib checkpoints to ONNX format

**Process:**
1. Locate checkpoint directory
2. Load RLlib algorithm and policies
3. Extract policy models
4. Wrap in `RLlibOnnxablePolicy`
5. Export each policy separately
6. Handle multi-agent configurations

**Key Functions:**
```python
def find_checkpoint(base_path, checkpoint_number) -> str
def export_rllib_policy_to_onnx(policy, export_path, **options) -> None
```

## Data Flow

### Training Workflow

```
User Input (TUI)
    ↓
App State Update (app.rs)
    ↓
Configuration Serialization (project.rs)
    ↓
Python Process Spawn (std::process::Command)
    ↓
Training Script Execution
    ↓
Stdout Stream (with @METRIC lines)
    ↓
Metric Parsing (app.rs)
    ↓
UI Update (ui.rs)
    ↓
Chart Rendering (ratatui)
```

### Process Communication

**Controller → Python:**
- Command-line arguments
- Configuration files (JSON, YAML)
- Environment variables (`CONTROLLER_METRICS`)

**Python → Controller:**
- Stdout stream (training logs)
- Stderr stream (error messages)
- Structured metrics (JSON prefixed with `@METRIC`)
- Exit codes

## Metric System

### Metric Collection

**Sources:**
1. **SB3 Callback**: Emits metrics every N steps
2. **RLlib Callback**: Emits metrics every iteration
3. **Custom Metrics**: From Godot environment

### Metric Format

```json
{
  "kind": "iteration",
  "timestamp": "2025-11-05T14:30:00Z",
  "training_iteration": 10,
  "timesteps_total": 100000,
  "episodes_total": 500,
  "episode_reward_mean": 45.2,
  "episode_reward_min": 10.0,
  "episode_reward_max": 98.5,
  "episode_len_mean": 200.0,
  "policies": {
    "policy_1": {
      "reward_mean": 45.2,
      "custom_metrics": {...}
    }
  }
}
```

### Metric Parsing

```rust
impl MetricSample {
    fn from_value(value: &Value, checkpoint_frequency: u64) -> Option<Self> {
        // Extract iteration metrics
        // Parse policy-specific metrics
        // Calculate checkpoints
        // Handle custom metrics
    }
}
```

### Metric Storage

- **Buffer**: Last 2000 samples retained
- **Structure**: `Vec<MetricSample>`
- **Access**: O(1) for latest, O(n) for history

## Configuration Management

### Configuration Layers

1. **Project Config** (`project.json`)
   - Basic metadata
   - Creation timestamp
   - Project name

2. **Training Config** (`training_config.json`)
   - Environment path
   - Timesteps
   - All hyperparameters for SB3 and RLlib
   - Mode selection

3. **RLlib Config** (`rllib_config.yaml`)
   - Environment factory
   - Policy mappings
   - Algorithm-specific settings

4. **Export Config** (`export_config.json`)
   - Checkpoint paths
   - Output directories
   - ONNX settings

### Configuration Loading

```rust
pub struct ProjectInfo {
    pub name: String,
    pub path: PathBuf,
    pub created: u64,
    pub training_config: TrainingConfig,
    pub export_config: Option<ExportConfig>,
}

impl ProjectInfo {
    pub fn load(path: &Path) -> Result<Self> {
        // Load project.json
        // Load training_config.json
        // Load export_config.json (optional)
        // Merge configurations
    }
}
```

## UI Architecture

### Tab System

**Structure:**
```rust
pub struct TabInfo {
    pub id: TabId,
    pub title: &'static str,
}

pub enum TabId {
    Home,      // Project management
    Train,     // Training orchestration
    Metrics,   // Real-time visualization
    Export,    // ONNX export
}
```

**Navigation:**
- Number keys (1-4) for direct access
- Arrow keys for sequential navigation
- Tab-specific keyboard shortcuts

### Input Modes

```rust
pub enum InputMode {
    Normal,                    // Standard navigation
    CreatingProject,           // Text input for new project
    EditingConfig,            // Config field editing
    EditingAdvancedConfig,    // Advanced config editing
    AdvancedConfig,           // Advanced config navigation
    BrowsingFiles,            // File browser
    Help,                     // Help overlay
    ConfirmQuit,              // Quit confirmation
    EditingExport,            // Export config editing
}
```

**Mode Transitions:**
- User input determines mode changes
- Each mode has specific key handlers
- ESC generally returns to Normal mode

### Rendering Pipeline

```rust
pub fn render(frame: &mut Frame, app: &App) {
    // Check for overlays (help, quit confirm)
    if app.is_help_visible() {
        render_help_overlay(frame, app);
        return;
    }
    
    // Split screen layout
    let [header, content] = split_layout(frame.area());
    
    // Render tabs
    render_tabs(frame, header, app);
    
    // Render active tab content
    match app.active_tab().id {
        TabId::Home => render_home(frame, content, app),
        TabId::Train => render_train(frame, content, app),
        TabId::Metrics => render_metrics(frame, content, app),
        TabId::Export => render_export(frame, content, app),
    }
}
```

### Chart Rendering

**Technology**: Ratatui's `Chart` widget with custom `Dataset`

**Features:**
- Real-time updates (no redraw flicker)
- Auto-scaling Y-axis
- Time-series X-axis
- Multiple datasets (policies)
- Color coding for different metrics

**Implementation:**
```rust
fn render_reward_chart(frame: &mut Frame, area: Rect, metrics: &[MetricSample]) {
    let datasets = vec![
        Dataset::default()
            .name("Mean Reward")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&reward_data),
    ];
    
    let chart = Chart::new(datasets)
        .block(Block::default().title("Rewards").borders(Borders::ALL))
        .x_axis(Axis::default().bounds([min_x, max_x]))
        .y_axis(Axis::default().bounds([min_y, max_y]));
    
    frame.render_widget(chart, area);
}
```

## Process Management

### Training Process

```rust
pub struct TrainingProcess {
    child: Child,              // std::process::Child
    stdout_thread: JoinHandle, // Background thread for stdout
    stderr_thread: JoinHandle, // Background thread for stderr
    output_receiver: Receiver<String>,
    error_receiver: Receiver<String>,
}
```

**Lifecycle:**
1. **Spawn**: Create Python process with pipes
2. **Monitor**: Background threads read stdout/stderr
3. **Parse**: Main thread receives lines, parses metrics
4. **Cancel**: Send SIGINT on user request
5. **Cleanup**: Wait for process exit, join threads

**Signal Handling:**
```rust
fn cancel_training(&mut self) -> Result<()> {
    if let Some(process) = &mut self.training_process {
        nix::sys::signal::kill(
            Pid::from_raw(process.child.id() as i32),
            Signal::SIGINT
        )?;
    }
    Ok(())
}
```

### Export Process

Similar structure to training process, but typically shorter-lived.

## File System Organization

```
projects/
├── index.json                      # Project registry
└── <project-name>/
    ├── project.json                # Metadata
    ├── training_config.json        # Training params
    ├── rllib_config.yaml           # RLlib specific
    ├── export_config.json          # Export settings
    ├── logs/
    │   ├── sb3/                   # SB3 logs
    │   │   └── <experiment>/
    │   │       ├── model.zip
    │   │       └── tensorboard/
    │   └── rllib/                 # RLlib checkpoints
    │       └── PPO_<timestamp>/
    │           └── PPO_<trial>/
    │               └── checkpoint_N/
    └── onnx_exports/              # Exported models
        ├── policy_1.onnx
        └── policy_2.onnx
```

## Performance Considerations

### Optimizations

1. **Release Build**:
   - LTO enabled (`lto = "fat"`)
   - Single codegen unit
   - Native CPU target
   - Panic = abort

2. **Buffering**:
   - Training output: 500 lines max
   - Metrics history: 2000 samples max
   - Prevents memory growth

3. **Parsing**:
   - Lazy JSON parsing
   - Only parse `@METRIC` lines
   - Skip non-metric output

4. **Rendering**:
   - No unnecessary redraws
   - Efficient diff-based rendering (Ratatui)
   - Chart data downsampling for large histories

### Scalability

- **Multi-agent**: Handles 100+ policies
- **Long training**: Maintains performance over days
- **Large logs**: Circular buffers prevent OOM
- **Parallel environments**: No controller overhead

## Error Handling

### Strategy

**Rust Side:**
- `color-eyre` for rich error context
- `Result<T>` for all fallible operations
- Graceful degradation where possible

**Python Side:**
- Exception logging to stderr
- Structured error messages
- Controller parses and displays

### Error Display

```rust
pub enum StatusKind {
    Info,
    Success,
    Warning,
    Error,
}

pub struct StatusMessage {
    kind: StatusKind,
    text: String,
    timestamp: Instant,
}
```

Status messages appear at bottom of screen with color coding.

## Testing Strategy

### Unit Tests
- Configuration parsing
- Metric parsing
- Project management
- File I/O

### Integration Tests
- Demo script (`demo.py`) for output streaming
- Python environment checker (`check_py_env.py`)
- Manual TUI testing

### Future Improvements
- Automated UI testing with snapshot tests
- Mock training processes
- Configuration validation tests

## Dependencies

### Rust Dependencies
```toml
[dependencies]
color-eyre = "0.6.3"      # Error handling
crossterm = "0.28.1"      # Terminal control
nix = "0.30.1"            # POSIX signals
ratatui = "0.29.0"        # TUI framework
serde = "1.0"             # Serialization
serde_json = "1.0"        # JSON support
```

### Python Dependencies
- `stable-baselines3` - RL algorithms
- `ray[rllib]` - Multi-agent RL
- `godot-rl` - Godot integration
- `torch` - Deep learning
- `onnx` - Model export
- `pyyaml` - Config parsing

## Security Considerations

1. **No sensitive data in configs**: Projects are stored as plain text
2. **Process isolation**: Training runs in separate processes
3. **File permissions**: Respects system permissions
4. **Input validation**: All user input validated before use

## Future Architecture Plans

1. **Plugin System**: Allow custom training frameworks
2. **Remote Training**: SSH-based remote execution
3. **Database**: SQLite for better project management
4. **Web UI**: Optional web interface
5. **Distributed Training**: Multi-machine coordination
6. **Cloud Integration**: AWS/GCP training support

## Contributing to Architecture

When making architectural changes:

1. **Maintain separation**: Keep Rust/Python boundaries clean
2. **Document protocols**: Any new IPC must be documented
3. **Preserve backwards compatibility**: Old configs should still load
4. **Add tests**: All new components need tests
5. **Update this doc**: Keep architecture doc in sync

---

**Version**: 0.1.0  
**Last Updated**: November 5, 2025
