use std::collections::BTreeMap;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Cell, Chart, Clear, Dataset, GraphType, List, ListItem,
    ListState, Paragraph, Row, Table, Wrap,
};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{
    config::RLLIB_ALGORITHM_LIST, AgentType, App, ChartExportOptionField, ChartLegendPosition,
    ConfigField, FileBrowserEntry, FileBrowserKind, FileBrowserState, InputMode, InterfaceFocus,
    MetricSample, MetricsFocus, MetricsSettingField, MetricsSettingsPanel, PoliciesSortMode,
    SimulatorFocus, StatusKind, SummaryVerbosity, TabId,
};
use ratatui::widgets::LegendPosition;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::utils::alphanumeric_sort_key;

fn focused_border_color(app: &App, focused: bool) -> Color {
    if !focused {
        return Color::DarkGray;
    }
    if !app.animations_enabled() {
        return Color::Cyan;
    }
    if app.animation_phase() % 2 == 0 {
        Color::Cyan
    } else {
        Color::LightCyan
    }
}

fn format_axis_value(value: f64, prefer_integers: bool) -> String {
    let (scaled, suffix) = {
        let abs = value.abs();
        if abs >= 1_000_000_000.0 {
            (value / 1_000_000_000.0, "b")
        } else if abs >= 1_000_000.0 {
            (value / 1_000_000.0, "m")
        } else if abs >= 1_000.0 {
            (value / 1_000.0, "k")
        } else {
            (value, "")
        }
    };

    if prefer_integers || scaled.abs() >= 10.0 {
        format!("{scaled:.0}{suffix}")
    } else if scaled.abs() >= 1.0 {
        format!("{scaled:.1}{suffix}")
    } else {
        format!("{scaled:.2}{suffix}")
    }
}

fn compute_tick_step(min: f64, max: f64, target: usize) -> f64 {
    if !min.is_finite() || !max.is_finite() || target == 0 {
        return 1.0;
    }
    let range = (max - min).abs().max(1e-9);
    let rough = range / target as f64;
    if !rough.is_finite() {
        return 1.0;
    }
    let power = rough.log10().floor();
    let base = 10f64.powf(power);
    for factor in [1.0, 2.0, 5.0, 10.0] {
        let candidate = base * factor;
        if candidate >= rough {
            return candidate;
        }
    }
    base
}

fn build_axis_labels(
    min: f64,
    max: f64,
    slots: usize,
    prefer_integers: bool,
) -> Vec<Span<'static>> {
    if !min.is_finite() || !max.is_finite() || slots == 0 {
        return Vec::new();
    }

    let target = slots.clamp(3, 7);
    let step = compute_tick_step(min, max, target);
    let start = (min / step).floor() * step;
    let end = (max / step).ceil() * step;

    let mut labels = Vec::new();
    let mut value = start;
    let max_labels = target + 2;
    while value <= end + step * 0.25 && labels.len() < max_labels {
        labels.push(Span::raw(format_axis_value(value, prefer_integers)));
        value += step;
    }

    labels
}

fn format_duration_short(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3_600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3_600)
    } else {
        format!("{:.1}d ago", secs as f64 / 86_400.0)
    }
}

fn format_system_time_relative(time: SystemTime) -> String {
    match SystemTime::now().duration_since(time) {
        Ok(delta) => format_duration_short(delta),
        Err(_) => "just now".to_string(),
    }
}

fn format_unix_relative(timestamp: u64) -> String {
    let time = UNIX_EPOCH + Duration::from_secs(timestamp);
    format_system_time_relative(time)
}

pub fn render_home(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(3)])
        .split(area);

    let main = vertical[0];
    let status = vertical[1];

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(main);

    render_project_list(frame, columns[0], app);
    render_project_overview(frame, columns[1], app);
    render_home_status(frame, status, app);
}

fn render_project_overview(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if app.projects().is_empty() {
        let placeholder = Paragraph::new("No projects yet. Press 'n' to create one.")
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Project Details"),
            )
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let constraints: Vec<Constraint> = if area.height >= 12 {
        vec![Constraint::Min(7), Constraint::Length(5)]
    } else {
        vec![Constraint::Min(5)]
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let project = app
        .selected_project()
        .or_else(|| app.active_project())
        .or_else(|| app.projects().first());

    let mut lines = Vec::new();
    let title = "Project Details";

    if let Some(project) = project {
        let mut header = project.name.clone();
        if project.read_only {
            header.push_str(" [RO]");
        }
        if project.source_archive.is_some() {
            header.push_str(" [Linked]");
        }
        lines.push(header);
        lines.push(format!("Root: {}", project.root_path.display()));
        lines.push(format!("Logs: {}", project.logs_path.display()));
        lines.push(format!(
            "Last used: {}",
            format_system_time_relative(project.last_used)
        ));
        let session_count = app.session_count_for_project(project);
        let agent_label = app
            .training_mode_label_for(project)
            .or_else(|| app.training_mode_label())
            .unwrap_or_else(|| "Unknown".to_string());
        lines.push(format!(
            "Sessions: {} • Agent: {}",
            session_count, agent_label
        ));
        if let Some((name, ts, runs)) = app.project_last_session_info(project) {
            lines.push(format!(
                "Latest session: {} • {} • runs: {}",
                name,
                format_unix_relative(ts),
                runs
            ));
        } else {
            lines.push("Latest session: None yet".to_string());
        }
        lines.push("Stats are cached on project scan.".to_string());
    } else {
        lines.push("No project selected".to_string());
    }

    let detail = lines.join("\n");
    let block = Block::default().borders(Borders::ALL).title(title);
    let paragraph = Paragraph::new(detail)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::White));

    frame.render_widget(paragraph, chunks[0]);

    if chunks.len() > 1 {
        render_python_environment(frame, chunks[1], app);
    }
}

fn render_python_environment(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Python Environment");

    let mut lines = Vec::new();

    // Check Stable Baselines 3
    let sb3_status = match app.python_sb3_available() {
        Some(true) => ("✓", Color::LightGreen),
        Some(false) => ("✗", Color::Red),
        None => ("?", Color::Yellow),
    };
    lines.push(Line::from(vec![
        Span::styled(
            sb3_status.0,
            Style::default()
                .fg(sb3_status.1)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled("Stable Baselines 3", Style::default().fg(Color::White)),
    ]));

    // Check Ray/RLlib
    let ray_status = match app.python_ray_available() {
        Some(true) => ("✓", Color::LightGreen),
        Some(false) => ("✗", Color::Red),
        None => ("?", Color::Yellow),
    };
    lines.push(Line::from(vec![
        Span::styled(
            ray_status.0,
            Style::default()
                .fg(ray_status.1)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled("Ray/RLlib", Style::default().fg(Color::White)),
    ]));

    if app.python_check_hint_visible() {
        lines.push(Line::from(Span::styled(
            "Press 'p' to run the library check",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left);

    frame.render_widget(paragraph, area);
}

fn render_project_list(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title("Projects");

    if app.projects().is_empty() {
        let placeholder = Paragraph::new("No projects yet. Press 'n' to create one.")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let active_path = app.active_project().map(|p| &p.logs_path);
    let items: Vec<ListItem> = app
        .projects()
        .iter()
        .map(|project| {
            let mut spans = Vec::new();
            let is_active = active_path.map_or(false, |path| path == &project.logs_path);
            let session_count = app.session_count_for_project(project);
            let agent_label = app
                .training_mode_label_for(project)
                .unwrap_or_else(|| "Unknown".to_string());
            spans.push(if is_active {
                Span::styled("● ", Style::default().fg(Color::LightGreen))
            } else {
                Span::raw("  ")
            });
            spans.push(Span::styled(
                project.name.clone(),
                Style::default().fg(Color::White),
            ));
            if project.read_only {
                spans.push(Span::styled(" [RO]", Style::default().fg(Color::LightRed)));
            }
            if project.source_archive.is_some() {
                spans.push(Span::styled(" [Linked]", Style::default().fg(Color::Cyan)));
            }
            spans.push(Span::styled(
                format!(
                    "  • used {}",
                    format_system_time_relative(project.last_used)
                ),
                Style::default().fg(Color::DarkGray),
            ));
            spans.push(Span::styled(
                format!("  • sessions: {} • agent: {}", session_count, agent_label),
                Style::default().fg(Color::DarkGray),
            ));
            ListItem::new(Line::from(spans))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");

    let mut state = ListState::default();
    state.select(app.selected_project_index());
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_home_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title("Status");

    let mut lines: Vec<Line> = Vec::new();

    match app.input_mode() {
        InputMode::CreatingProject => {
            lines.push(Line::from(vec![
                Span::styled("Project name: ", Style::default().fg(Color::LightBlue)),
                Span::styled(
                    format!("{}_", app.project_name_buffer()),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            lines.push(Line::from(Span::styled(
                "Enter to choose logs directory • Esc to cancel",
                Style::default().fg(Color::DarkGray),
            )));
        }
        InputMode::EditingConfig | InputMode::EditingAdvancedConfig => {
            let field_name = app
                .active_config_field()
                .map(|f| f.label())
                .unwrap_or("Unknown");
            let mut spans = Vec::new();
            spans.push(Span::styled(
                format!("Editing {}: ", field_name),
                Style::default().fg(Color::LightBlue),
            ));
            spans.push(Span::styled(
                format!("{}_", app.config_edit_buffer()),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(spans));
            if let Some((ok, err, normalized)) = app.config_edit_validation() {
                let (prefix, msg, color) = if ok {
                    (
                        "✓ ",
                        normalized.unwrap_or_else(|| "OK".to_string()),
                        Color::LightGreen,
                    )
                } else {
                    ("⚠ ", err, Color::LightRed)
                };
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{msg}"),
                    Style::default().fg(color),
                )));
            }
            lines.push(Line::from(Span::styled(
                "Enter to confirm • Esc to cancel",
                Style::default().fg(Color::DarkGray),
            )));
        }
        InputMode::SelectingConfigOption => {
            lines.push(Line::from(Span::styled(
                "Select an option from the menu...",
                Style::default().fg(Color::DarkGray),
            )));
        }
        InputMode::EditingExport => {
            let field_name = app
                .active_export_field()
                .map(|f| f.label())
                .unwrap_or("Unknown");
            let mut spans = Vec::new();
            spans.push(Span::styled(
                format!("Editing {}: ", field_name),
                Style::default().fg(Color::LightBlue),
            ));
            spans.push(Span::styled(
                format!("{}_", app.export_edit_buffer()),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(spans));
            lines.push(Line::from(Span::styled(
                "Enter to confirm • Esc to cancel",
                Style::default().fg(Color::DarkGray),
            )));
        }
        InputMode::Normal => {
            if let Some(status) = app.status() {
                let style = status_style(status.kind);
                let mut text = status.text.clone();
                if app.is_training_running()
                    || app.is_export_running()
                    || app.is_project_archive_running()
                {
                    if let Some(spinner) = app.spinner_char() {
                        text = format!("{spinner} {text}");
                    }
                }
                lines.push(Line::from(Span::styled(text, style)));
            }

            let instruction = if area.width < 48 {
                "n: new • Enter: activate • ↑/↓: navigate"
            } else {
                "Press 'n' to create a project • Enter to activate • ↑/↓ to navigate"
            };
            lines.push(Line::from(Span::styled(
                instruction,
                Style::default().fg(Color::DarkGray),
            )));
        }
        InputMode::BrowsingFiles
        | InputMode::Help
        | InputMode::ConfirmQuit
        | InputMode::ConfirmAction
        | InputMode::AdvancedConfig
        | InputMode::ChartExportOptions
        | InputMode::EditingChartExportOption
        | InputMode::MetricsSettings
        | InputMode::EditingMetricsSetting
        | InputMode::ConfirmProjectImport => {
            // These modes have their own full-screen renders
        }
        InputMode::EditingProjectArchive => todo!(),
    }

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

pub fn render_train(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let constraints = if area.height >= 16 {
        [
            Constraint::Length(8),
            Constraint::Min(6),
            Constraint::Length(3),
        ]
    } else if area.height >= 12 {
        [
            Constraint::Length(6),
            Constraint::Min(4),
            Constraint::Length(3),
        ]
    } else {
        [
            Constraint::Length(4),
            Constraint::Min(3),
            Constraint::Length(2),
        ]
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_training_config(frame, chunks[0], app);
    render_training_output(frame, chunks[1], app);
    render_training_status(frame, chunks[2], app);
}

fn render_training_config(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if app.is_experimental() {
        render_mars_training_config(frame, area, app);
        return;
    }
    let config = app.training_config();
    let input_mode = app.input_mode();
    let active_field = app.active_config_field();
    let editing_basic = matches!(input_mode, InputMode::EditingConfig);

    let mode_text = match config.mode {
        crate::app::TrainingMode::SingleAgent => "Single-Agent (Stable Baselines 3)",
        crate::app::TrainingMode::MultiAgent => "Multi-Agent (RLlib)",
    };

    // Use cached validation result instead of calling expensive validation
    let validation_icon = if app.is_training_config_valid() {
        Span::styled("✓ ", Style::default().fg(Color::Green))
    } else {
        Span::styled("⚠ ", Style::default().fg(Color::Red))
    };

    let mut lines = vec![
        Line::from(vec![
            validation_icon,
            Span::styled("Mode: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                mode_text,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" (m:toggle)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Env: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                {
                    let is_editing = editing_basic && active_field == Some(ConfigField::EnvPath);
                    if is_editing {
                        format!("{}_", app.config_edit_buffer())
                    } else if config.env_path.is_empty() {
                        "<not set>".to_string()
                    } else {
                        config.env_path.clone()
                    }
                },
                {
                    let is_editing = editing_basic && active_field == Some(ConfigField::EnvPath);
                    let display_empty = if is_editing {
                        app.config_edit_buffer().trim().is_empty()
                    } else {
                        config.env_path.is_empty()
                    };
                    let base_color = if display_empty {
                        Color::Red
                    } else {
                        Color::White
                    };
                    let mut style = Style::default().fg(base_color);
                    if is_editing {
                        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    style
                },
            ),
            Span::styled(" (p:edit, b:browse)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Steps: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                {
                    let is_editing = editing_basic && active_field == Some(ConfigField::Timesteps);
                    if is_editing {
                        format!("{}_", app.config_edit_buffer())
                    } else {
                        format!("{}", config.timesteps)
                    }
                },
                {
                    let mut style = Style::default().fg(Color::White);
                    if editing_basic && active_field == Some(ConfigField::Timesteps) {
                        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    style
                },
            ),
            Span::styled(" (s:edit)", Style::default().fg(Color::DarkGray)),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Name: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                {
                    let is_editing =
                        editing_basic && active_field == Some(ConfigField::ExperimentName);
                    if is_editing {
                        format!("{}_", app.config_edit_buffer())
                    } else {
                        config.experiment_name.clone()
                    }
                },
                {
                    let mut style = Style::default().fg(Color::White);
                    if editing_basic && active_field == Some(ConfigField::ExperimentName) {
                        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    style
                },
            ),
            Span::styled(" (n:edit)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Press ", Style::default().fg(Color::DarkGray)),
            Span::styled("p/s/n", Style::default().fg(Color::Yellow)),
            Span::styled(" to edit basics, ", Style::default().fg(Color::DarkGray)),
            Span::styled("a", Style::default().fg(Color::Yellow)),
            Span::styled(
                " for advanced settings, ",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("g", Style::default().fg(Color::Yellow)),
            Span::styled(
                " to generate RLlib config, ",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("h", Style::default().fg(Color::Yellow)),
            Span::styled(" for help", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    if editing_basic {
        if let Some(field) = active_field {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Editing ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    field.label(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(": {}_", app.config_edit_buffer()),
                    Style::default().fg(Color::White),
                ),
            ]));
            if let Some((ok, error, normalized)) = app.config_edit_validation() {
                let (prefix, msg, color) = if ok {
                    (
                        "✓ ",
                        normalized.unwrap_or_else(|| "OK".to_string()),
                        Color::LightGreen,
                    )
                } else {
                    ("⚠ ", error, Color::LightRed)
                };
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{msg}"),
                    Style::default().fg(color),
                )));
            }
        }
    }

    if matches!(
        input_mode,
        InputMode::EditingConfig | InputMode::EditingAdvancedConfig
    ) {
        if let Some(field) = active_field {
            let field_name = field.label();
            lines.push(Line::from(vec![
                Span::styled("Editing ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    field_name,
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(": {}", app.config_edit_buffer()),
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(Span::styled(
                "Enter: save • Esc: cancel • Backspace: delete",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title("Training Configuration");

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn wrap_plain_line(line: &str, width: usize, trim: bool) -> Vec<String> {
    use std::collections::VecDeque;

    if width == 0 {
        return vec![String::new()];
    }
    let max_width = width as u16;

    let mut wrapped_lines: Vec<String> = Vec::new();
    let mut pending_line = String::new();
    let mut pending_word = String::new();
    let mut pending_whitespace: VecDeque<char> = VecDeque::new();

    let mut line_width: u16 = 0;
    let mut word_width: u16 = 0;
    let mut whitespace_width: u16 = 0;
    let mut non_whitespace_previous = false;

    for ch in line.chars() {
        let is_whitespace = ch.is_whitespace();
        let symbol_width = UnicodeWidthChar::width(ch).unwrap_or(0) as u16;

        if symbol_width > max_width {
            continue;
        }

        let word_found = non_whitespace_previous && is_whitespace;
        let trimmed_overflow =
            pending_line.is_empty() && trim && word_width + symbol_width > max_width;
        let whitespace_overflow =
            pending_line.is_empty() && trim && whitespace_width + symbol_width > max_width;
        let untrimmed_overflow = pending_line.is_empty()
            && !trim
            && word_width + whitespace_width + symbol_width > max_width;

        if word_found || trimmed_overflow || whitespace_overflow || untrimmed_overflow {
            if !pending_line.is_empty() || !trim {
                while let Some(ws_ch) = pending_whitespace.pop_front() {
                    pending_line.push(ws_ch);
                }
                line_width += whitespace_width;
            }

            pending_line.push_str(&pending_word);
            line_width += word_width;

            pending_word.clear();
            word_width = 0;
            pending_whitespace.clear();
            whitespace_width = 0;
        }

        let line_full = line_width >= max_width;
        let pending_word_overflow = symbol_width > 0
            && line_width + whitespace_width + word_width >= max_width;

        if line_full || pending_word_overflow {
            let mut remaining_width = max_width.saturating_sub(line_width);
            wrapped_lines.push(std::mem::take(&mut pending_line));
            line_width = 0;

            while let Some(&ws_ch) = pending_whitespace.front() {
                let ws_w = UnicodeWidthChar::width(ws_ch).unwrap_or(0) as u16;
                if ws_w > remaining_width {
                    break;
                }
                whitespace_width = whitespace_width.saturating_sub(ws_w);
                remaining_width = remaining_width.saturating_sub(ws_w);
                pending_whitespace.pop_front();
            }

            if is_whitespace && pending_whitespace.is_empty() {
                non_whitespace_previous = false;
                continue;
            }
        }

        if is_whitespace {
            whitespace_width += symbol_width;
            pending_whitespace.push_back(ch);
        } else {
            word_width += symbol_width;
            pending_word.push(ch);
        }

        non_whitespace_previous = !is_whitespace;
    }

    if pending_line.is_empty() && pending_word.is_empty() && !pending_whitespace.is_empty() {
        wrapped_lines.push(String::new());
    }
    if !pending_line.is_empty() || !trim {
        while let Some(ws_ch) = pending_whitespace.pop_front() {
            pending_line.push(ws_ch);
        }
    }
    pending_line.push_str(&pending_word);

    if !pending_line.is_empty() {
        wrapped_lines.push(pending_line);
    }
    if wrapped_lines.is_empty() {
        wrapped_lines.push(String::new());
    }
    wrapped_lines
}

fn wrap_output_lines(
    raw_lines: &[String],
    inner_width: u16,
    trim: bool,
) -> (Vec<Line<'static>>, Vec<usize>) {
    let width = inner_width.max(1) as usize;
    let mut wrapped: Vec<Line<'static>> = Vec::new();
    let mut heights: Vec<usize> = Vec::with_capacity(raw_lines.len());

    for raw in raw_lines {
        let parts = wrap_plain_line(raw, width, trim);
        heights.push(parts.len());
        for part in parts {
            wrapped.push(Line::raw(part));
        }
    }

    if wrapped.is_empty() {
        wrapped.push(Line::raw(String::new()));
    }

    (wrapped, heights)
}

fn render_training_output(frame: &mut Frame<'_>, area: Rect, app: &App) {
    frame.render_widget(Clear, area);
    let output_block = Block::default()
        .borders(Borders::ALL)
        .title("Training Output");

    if app.training_output().is_empty() {
        let output_text =
            "No output yet. Configure training and press 't' to start.".to_string();
        let paragraph = Paragraph::new(output_text)
            .block(output_block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(Color::White));
        frame.render_widget(paragraph, area);
        return;
    }

    let raw_lines = app.training_output();
    let inner_width = area.width.saturating_sub(2);
    let (wrapped_lines, per_raw_heights) = wrap_output_lines(raw_lines, inner_width, true);

    let mut paragraph = Paragraph::new(wrapped_lines)
        .block(output_block)
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::White));

    let visible_height = area.height.saturating_sub(2).max(1) as usize;
    let offset_from_bottom_raw = app
        .training_output_scroll()
        .min(raw_lines.len().saturating_sub(1));
    let bottom_raw_idx = raw_lines.len() - 1 - offset_from_bottom_raw;

    let mut prefix_visual = Vec::with_capacity(per_raw_heights.len() + 1);
    prefix_visual.push(0usize);
    for height in per_raw_heights {
        let next = prefix_visual.last().copied().unwrap_or(0).saturating_add(height);
        prefix_visual.push(next);
    }
    let end_visual_exclusive = prefix_visual[bottom_raw_idx + 1];
    let start_visual = end_visual_exclusive.saturating_sub(visible_height);
    let scroll_y = start_visual.min(u16::MAX as usize) as u16;
    paragraph = paragraph.scroll((scroll_y, 0));

    frame.render_widget(paragraph, area);
}

pub fn render_metrics(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if app.metrics_history_total_len() == 0 {
        let placeholder =
            Paragraph::new("No metrics yet. Start a training run to see live updates.")
                .block(Block::default().borders(Borders::ALL).title("Metrics"))
                .alignment(Alignment::Center)
                .style(Style::default().fg(Color::DarkGray))
                .wrap(Wrap { trim: true });
        frame.render_widget(placeholder, area);
        return;
    }

    let width = area.width;
    let height = area.height;

    // Dynamic layout based on terminal size
    if width < 80 || height < 20 {
        // Very small terminal - single column stacked layout
        render_metrics_compact(frame, area, app);
    } else if width < 120 {
        // Medium terminal - two column layout with chart on top
        render_metrics_medium(frame, area, app);
    } else {
        // Large terminal - three column layout with chart on top
        render_metrics_large(frame, area, app);
    }
}

fn render_mars_training_config(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let cfg = app.mars_config();
    let input_mode = app.input_mode();
    let active_field = app.active_config_field();

    let validation_icon = if app.is_training_config_valid() {
        Span::styled("✓ ", Style::default().fg(Color::Green))
    } else {
        Span::styled("⚠ ", Style::default().fg(Color::Red))
    };

    let mut lines = vec![
        Line::from(vec![
            validation_icon,
            Span::styled("Mode: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "MARS (experimental)",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Godot env: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if cfg.env_path.is_empty() {
                    "<not set>"
                } else {
                    &cfg.env_path
                },
                {
                    let base_color = if cfg.env_path.is_empty() {
                        Color::Red
                    } else {
                        Color::White
                    };
                    let mut style = Style::default().fg(base_color);
                    if input_mode == InputMode::EditingConfig
                        && active_field == Some(ConfigField::MarsEnvPath)
                    {
                        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    style
                },
            ),
            Span::styled(" (p:edit, b:browse)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Method: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&cfg.method, {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::SelectingConfigOption
                    && active_field == Some(ConfigField::MarsMethod)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Algorithm: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&cfg.algorithm, {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::SelectingConfigOption
                    && active_field == Some(ConfigField::MarsAlgorithm)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled(" (m:method, o:algo)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Episodes: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cfg.max_episodes), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsMaxEpisodes)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Steps/ep: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cfg.max_steps_per_episode), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsMaxStepsPerEpisode)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled(
                " (e:episodes, s:steps)",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Batch: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cfg.batch_size), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsBatchSize)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("LR: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cfg.learning_rate), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsLearningRate)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Seed: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cfg.seed), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsSeed)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled(
                " (edit via advanced settings for more)",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Save id: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&cfg.save_id, {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsSaveId)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Log interval: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cfg.log_interval), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::MarsLogInterval)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled(" (advanced to edit)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Press ", Style::default().fg(Color::DarkGray)),
            Span::styled("p/m/o/e/s", Style::default().fg(Color::Yellow)),
            Span::styled(" to edit basics, ", Style::default().fg(Color::DarkGray)),
            Span::styled("a", Style::default().fg(Color::Yellow)),
            Span::styled(
                " for advanced settings, ",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("h", Style::default().fg(Color::Yellow)),
            Span::styled(" for help", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    if matches!(
        input_mode,
        InputMode::EditingConfig | InputMode::EditingAdvancedConfig
    ) {
        if let Some(field) = active_field {
            let field_name = field.label();
            lines.push(Line::from(vec![
                Span::styled("Editing ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    field_name,
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(": {}", app.config_edit_buffer()),
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(Span::styled(
                "Enter: save • Esc: cancel • Backspace: delete",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title("Training (MARS)");

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

pub fn render_simulator(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let status_height = if area.height >= 16 { 7 } else { 5 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(status_height), Constraint::Min(4)])
        .split(area);

    render_simulator_status(frame, chunks[0], app);

    let feeds = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    render_simulator_events(frame, feeds[0], app);
    render_simulator_actions(frame, feeds[1], app);
}

fn render_simulator_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let config = app.simulator_config();
    let label_style = Style::default().fg(Color::DarkGray);
    let running = app.is_simulator_running();
    let mut lines = Vec::new();

    let status_color = if running {
        Color::LightGreen
    } else {
        Color::DarkGray
    };

    lines.push(Line::from(vec![
        Span::styled("Status: ", label_style),
        Span::styled(
            if running { "Running" } else { "Idle" },
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Mode: ", label_style),
        Span::styled(
            config.mode.label(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Window: ", label_style),
        Span::styled(
            if config.show_window {
                "Visible"
            } else {
                "Headless"
            },
            Style::default().fg(Color::Yellow),
        ),
        Span::styled("  Auto-Restart: ", label_style),
        Span::styled(
            if config.auto_restart { "On" } else { "Off" },
            Style::default().fg(if config.auto_restart {
                Color::LightGreen
            } else {
                Color::LightRed
            }),
        ),
        Span::styled("  Tracebacks: ", label_style),
        Span::styled(
            if config.log_tracebacks {
                "Verbose"
            } else {
                "Hidden"
            },
            Style::default().fg(Color::Gray),
        ),
    ]));

    let env_display = if config.env_path.trim().is_empty() {
        "<not set>"
    } else {
        config.env_path.trim()
    };

    lines.push(Line::from(vec![
        Span::styled("Env: ", label_style),
        Span::styled(
            env_display,
            Style::default().fg(if config.env_path.trim().is_empty() {
                Color::LightRed
            } else {
                Color::White
            }),
        ),
        Span::styled("  Step Delay: ", label_style),
        Span::styled(
            format!("{:.2}s", config.step_delay),
            Style::default().fg(Color::Gray),
        ),
        Span::styled("  Restart Delay: ", label_style),
        Span::styled(
            format!("{:.1}s", config.restart_delay),
            Style::default().fg(Color::Gray),
        ),
        Span::styled("  Focus: ", label_style),
        Span::styled(
            match app.simulator_focus() {
                SimulatorFocus::Events => "Events",
                SimulatorFocus::Actions => "Actions",
            },
            Style::default().fg(Color::LightCyan),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("Controls: ", label_style),
        Span::styled("s", Style::default().fg(Color::Yellow)),
        Span::styled("tart ", label_style),
        Span::styled("c", Style::default().fg(Color::Yellow)),
        Span::styled("ancel ", label_style),
        Span::styled("m", Style::default().fg(Color::Yellow)),
        Span::styled("ode ", label_style),
        Span::styled("w", Style::default().fg(Color::Yellow)),
        Span::styled("indow ", label_style),
        Span::styled("a", Style::default().fg(Color::Yellow)),
        Span::styled("uto ", label_style),
        Span::styled("t", Style::default().fg(Color::Yellow)),
        Span::styled("racebacks  ", label_style),
        Span::styled("p", Style::default().fg(Color::Yellow)),
        Span::styled("ick env  ", label_style),
        Span::styled("y", Style::default().fg(Color::Yellow)),
        Span::styled(" sync  ", label_style),
        Span::styled("f", Style::default().fg(Color::Yellow)),
        Span::styled("/Tab focus  ", label_style),
        Span::styled("v", Style::default().fg(Color::Yellow)),
        Span::styled(" compact", label_style),
    ]));

    lines.push(Line::from(vec![
        Span::styled("Delays: ", label_style),
        Span::styled("[ / ]", Style::default().fg(Color::Yellow)),
        Span::styled(" step  ", label_style),
        Span::styled("- / =", Style::default().fg(Color::Yellow)),
        Span::styled(" restart", label_style),
    ]));

    if let Some(latest) = app.simulator_status_line() {
        lines.push(Line::from(vec![
            Span::styled("Latest: ", label_style),
            Span::styled(latest, Style::default().fg(Color::White)),
        ]));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title("Simulator Overview");
    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn render_simulator_events(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let focus = matches!(app.simulator_focus(), SimulatorFocus::Events);
    let border_style = Style::default().fg(focused_border_color(app, focus));
    let title = if focus { "Events (focus)" } else { "Events" };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style);

    if app.simulator_events().is_empty() {
        let placeholder = Paragraph::new(if app.is_simulator_running() {
            "Awaiting simulator output..."
        } else {
            "Press 's' to launch the simulator."
        })
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let mut lines = Vec::with_capacity(app.simulator_events().len());
    for entry in app.simulator_events() {
        let timestamp = entry.timestamp.as_deref().unwrap_or("-");
        let severity_color = match entry.severity {
            crate::app::SimulatorEventSeverity::Error => Color::LightRed,
            crate::app::SimulatorEventSeverity::Warning => Color::Yellow,
            crate::app::SimulatorEventSeverity::Info => Color::White,
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("[{timestamp}] "),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("{:<8}", entry.kind),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(entry.message.clone(), Style::default().fg(severity_color)),
        ]));
    }

    let mut paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    let total_lines = app.simulator_events().len();
    let visible_height = area.height.saturating_sub(2).max(1) as usize;
    let max_offset = total_lines.saturating_sub(visible_height);
    let offset = app.simulator_event_scroll().min(max_offset);
    let first_line = total_lines.saturating_sub(visible_height + offset) as u16;
    paragraph = paragraph.scroll((first_line, 0));

    frame.render_widget(paragraph, area);
}

fn render_simulator_actions(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let focus = matches!(app.simulator_focus(), SimulatorFocus::Actions);
    let border_style = Style::default().fg(focused_border_color(app, focus));
    let rows = app.simulator_actions();

    let meta = app.simulator_action_meta();

    if rows.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Actions")
            .border_style(border_style);
        let placeholder = Paragraph::new(if app.is_simulator_running() {
            "Waiting for agents to report actions..."
        } else {
            "Start the simulator to stream actions."
        })
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let visible_height = area.height.saturating_sub(3).max(1) as usize;
    let total_rows = rows.len();
    let max_offset = total_rows.saturating_sub(visible_height);
    let offset_from_bottom = app.simulator_actions_scroll().min(max_offset);
    let mut visible_rows = Vec::new();
    let mut idx = total_rows.saturating_sub(visible_height + offset_from_bottom);
    while idx < total_rows {
        visible_rows.push(&rows[idx]);
        idx += 1;
    }
    visible_rows.reverse();

    let mut table_rows = Vec::new();
    for row in visible_rows.iter() {
        let reward = row
            .reward
            .map(|r| format!("{r:.2}"))
            .unwrap_or_else(|| "-".into());
        let done = match (row.terminated, row.truncated) {
            (true, true) => "term+trunc",
            (true, false) => "term",
            (false, true) => "trunc",
            _ => "",
        };

        let mut cells = Vec::new();
        let episode_text = row
            .episode
            .map(|ep| ep.to_string())
            .unwrap_or_else(|| "-".into());
        let step_text = row
            .step
            .map(|st| st.to_string())
            .unwrap_or_else(|| "-".into());

        cells.push(Cell::from(episode_text));
        cells.push(Cell::from(step_text));
        cells.push(Cell::from(row.agent_id.clone()));
        if !app.simulator_compact_view() {
            cells.push(Cell::from(row.policy.clone().unwrap_or_else(|| "-".into())));
        }
        cells.push(Cell::from(row.action.clone()));
        cells.push(Cell::from(reward));
        cells.push(Cell::from(done));
        if !app.simulator_compact_view() {
            cells.push(Cell::from(row.info.clone().unwrap_or_else(|| "".into())));
        }

        table_rows.push(Row::new(cells).style(Style::default().fg(Color::White)));
    }

    let header = if app.simulator_compact_view() {
        Row::new(vec!["Ep", "Step", "Agent", "Action", "Reward", "Done"])
    } else {
        Row::new(vec![
            "Ep", "Step", "Agent", "Policy", "Action", "Reward", "Done", "Info",
        ])
    }
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );

    let widths: Vec<Constraint> = if app.simulator_compact_view() {
        vec![
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(14),
            Constraint::Min(18),
            Constraint::Length(10),
            Constraint::Length(10),
        ]
    } else {
        vec![
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Min(18),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(18),
        ]
    };

    let mut title = String::from(if focus { "Actions (focus)" } else { "Actions" });
    title.push_str(&format!(
        " • displaying {} of {} rows",
        table_rows.len(),
        rows.len()
    ));
    if let Some(meta) = meta {
        title.push_str(&format!(" • {}", meta.mode().label()));
        if let (Some(ep), Some(step)) = (meta.episode(), meta.step()) {
            title.push_str(&format!(" • episode {ep} step {step}"));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style);

    let table = Table::new(table_rows, widths).header(header).block(block);

    frame.render_widget(table, area);
}

pub fn render_interface(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let status_height = if area.height >= 18 { 9 } else { 7 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(status_height), Constraint::Min(4)])
        .split(area);

    render_interface_status(frame, chunks[0], app);

    let feeds = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[1]);

    render_interface_events(frame, feeds[0], app);
    render_interface_actions(frame, feeds[1], app);
}

fn render_interface_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let config = app.interface_config();
    let running = app.is_interface_running();
    let label_style = Style::default().fg(Color::DarkGray);
    let status_color = if running {
        Color::LightGreen
    } else {
        Color::DarkGray
    };

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let mut summary = Vec::new();
    summary.push(Line::from(vec![
        Span::styled("Run: ", label_style),
        Span::styled(
            if running { "Running" } else { "Idle" },
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Mode: ", label_style),
        Span::styled(
            config.mode.label(),
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Agent: ", label_style),
        Span::styled(
            config.agent_type.label(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Artifact: ", label_style),
        Span::styled(
            config.model_format.label(),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled("  Python: workspace", label_style),
    ]));

    let agent_display = if config.agent_path.trim().is_empty() {
        "<agent path not set>"
    } else {
        config.agent_path.trim()
    };
    summary.push(Line::from(vec![
        Span::styled("Model Path: ", label_style),
        Span::styled(
            agent_display,
            Style::default().fg(if config.agent_path.trim().is_empty() {
                Color::LightRed
            } else {
                Color::White
            }),
        ),
    ]));

    match config.agent_type {
        AgentType::StableBaselines3 => {
            let algo_display = if config.sb3_algo.trim().is_empty() {
                "auto"
            } else {
                config.sb3_algo.trim()
            };
            summary.push(Line::from(vec![
                Span::styled("SB3 Algo: ", label_style),
                Span::styled(algo_display, Style::default().fg(Color::White)),
            ]));
        }
        AgentType::Rllib => {
            let checkpoint = config
                .rllib_checkpoint_number
                .map(|n| format!("checkpoint_{n:06}"))
                .unwrap_or_else(|| "latest".into());
            summary.push(Line::from(vec![
                Span::styled("RLlib Policy: ", label_style),
                Span::styled(
                    if config.rllib_policy_id.trim().is_empty() {
                        "<default>"
                    } else {
                        config.rllib_policy_id.trim()
                    },
                    Style::default().fg(Color::White),
                ),
                Span::styled("  Checkpoint: ", label_style),
                Span::styled(checkpoint, Style::default().fg(Color::White)),
            ]));
        }
    }

    summary.push(Line::from(vec![
        Span::styled("Auto-Restart: ", label_style),
        Span::styled(
            if config.auto_restart { "On" } else { "Off" },
            Style::default().fg(if config.auto_restart {
                Color::LightGreen
            } else {
                Color::LightRed
            }),
        ),
        Span::styled("  Tracebacks: ", label_style),
        Span::styled(
            if config.log_tracebacks {
                "Verbose"
            } else {
                "Hidden"
            },
            Style::default().fg(Color::Gray),
        ),
    ]));

    summary.push(Line::from(vec![
        Span::styled("Delays: ", label_style),
        Span::styled(
            format!("step {:.2}s", config.step_delay),
            Style::default().fg(Color::Gray),
        ),
        Span::raw("  "),
        Span::styled(
            format!("restart {:.1}s", config.restart_delay),
            Style::default().fg(Color::Gray),
        ),
        Span::styled("  Focus: ", label_style),
        Span::styled(
            match app.interface_focus() {
                InterfaceFocus::Events => "Events",
                InterfaceFocus::Actions => "Actions",
            },
            Style::default().fg(Color::LightCyan),
        ),
    ]));

    if let Some(latest) = app.interface_status_line() {
        summary.push(Line::from(vec![
            Span::styled("Latest: ", label_style),
            Span::styled(latest, Style::default().fg(Color::White)),
        ]));
    }

    let summary_block = Block::default()
        .borders(Borders::ALL)
        .title("Interface Overview");
    let summary_paragraph = Paragraph::new(summary)
        .block(summary_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(summary_paragraph, cols[0]);

    let mut controls = Vec::new();
    controls.push(Line::from(vec![
        Span::styled("s", Style::default().fg(Color::Yellow)),
        Span::raw(" start   "),
        Span::styled("c", Style::default().fg(Color::Yellow)),
        Span::raw(" cancel   "),
        Span::styled("b", Style::default().fg(Color::Yellow)),
        Span::raw(" pick model   "),
        Span::styled("y", Style::default().fg(Color::Yellow)),
        Span::raw(" sync export"),
    ]));
    controls.push(Line::from(vec![
        Span::styled("t", Style::default().fg(Color::Yellow)),
        Span::raw(" agent type   "),
        Span::styled("r", Style::default().fg(Color::Yellow)),
        Span::raw(" artifact   "),
        Span::styled("m", Style::default().fg(Color::Yellow)),
        Span::raw(" mode"),
    ]));
    controls.push(Line::from(vec![
        Span::styled("a", Style::default().fg(Color::Yellow)),
        Span::raw(" auto-restart   "),
        Span::styled("T", Style::default().fg(Color::Yellow)),
        Span::raw(" tracebacks   "),
        Span::styled("[/]", Style::default().fg(Color::Yellow)),
        Span::raw(" step delay   "),
        Span::styled("-/=", Style::default().fg(Color::Yellow)),
        Span::raw(" restart delay"),
    ]));
    controls.push(Line::from(vec![
        Span::styled("f/Tab", Style::default().fg(Color::Yellow)),
        Span::raw(" focus   "),
        Span::styled("v", Style::default().fg(Color::Yellow)),
        Span::raw(" compact view"),
    ]));

    let controls_block = Block::default()
        .borders(Borders::ALL)
        .title("Quick Controls");
    let controls_paragraph = Paragraph::new(controls)
        .block(controls_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(controls_paragraph, cols[1]);
}

fn render_interface_events(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let focus = matches!(app.interface_focus(), InterfaceFocus::Events);
    let border_style = Style::default().fg(focused_border_color(app, focus));
    let title = if focus { "Events (focus)" } else { "Events" };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style);

    if app.interface_events().is_empty() {
        let placeholder = Paragraph::new(if app.is_interface_running() {
            "Awaiting interface output..."
        } else {
            "Press 's' to launch the interface."
        })
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let mut lines = Vec::with_capacity(app.interface_events().len());
    for entry in app.interface_events() {
        let timestamp = entry.timestamp.as_deref().unwrap_or("-");
        let severity_color = match entry.severity {
            crate::app::SimulatorEventSeverity::Error => Color::LightRed,
            crate::app::SimulatorEventSeverity::Warning => Color::Yellow,
            crate::app::SimulatorEventSeverity::Info => Color::White,
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("[{timestamp}] "),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("{:<8}", entry.kind),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(entry.message.clone(), Style::default().fg(severity_color)),
        ]));
    }

    let mut paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    let total_lines = app.interface_events().len();
    let visible_height = area.height.saturating_sub(2).max(1) as usize;
    let max_offset = total_lines.saturating_sub(visible_height);
    let offset = app.interface_event_scroll().min(max_offset);
    let first_line = total_lines.saturating_sub(visible_height + offset) as u16;
    paragraph = paragraph.scroll((first_line, 0));

    frame.render_widget(paragraph, area);
}

fn render_interface_actions(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let focus = matches!(app.interface_focus(), InterfaceFocus::Actions);
    let border_style = Style::default().fg(focused_border_color(app, focus));
    let rows = app.interface_actions();
    let meta = app.interface_action_meta();

    if rows.is_empty() {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Actions")
            .border_style(border_style);
        let placeholder = Paragraph::new(if app.is_interface_running() {
            "Waiting for agent actions..."
        } else {
            "Start the interface to stream actions."
        })
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let visible_height = area.height.saturating_sub(3).max(1) as usize;
    let total_rows = rows.len();
    let max_offset = total_rows.saturating_sub(visible_height);
    let offset_from_bottom = app.interface_actions_scroll().min(max_offset);
    let mut visible_rows = Vec::new();
    let mut idx = total_rows.saturating_sub(visible_height + offset_from_bottom);
    while idx < total_rows {
        visible_rows.push(&rows[idx]);
        idx += 1;
    }
    visible_rows.reverse();

    let mut table_rows = Vec::new();
    for row in visible_rows.iter() {
        let reward = row
            .reward
            .map(|r| format!("{r:.2}"))
            .unwrap_or_else(|| "-".into());
        let done = match (row.terminated, row.truncated) {
            (true, true) => "term+trunc",
            (true, false) => "term",
            (false, true) => "trunc",
            _ => "",
        };

        let mut cells = Vec::new();
        let episode_text = row
            .episode
            .map(|ep| ep.to_string())
            .unwrap_or_else(|| "-".into());
        let step_text = row
            .step
            .map(|st| st.to_string())
            .unwrap_or_else(|| "-".into());

        cells.push(Cell::from(episode_text));
        cells.push(Cell::from(step_text));
        cells.push(Cell::from(row.agent_id.clone()));
        if !app.interface_compact_view() {
            cells.push(Cell::from(row.policy.clone().unwrap_or_else(|| "-".into())));
        }
        if !app.interface_compact_view() {
            cells.push(Cell::from(
                row.observation.clone().unwrap_or_else(|| "-".into()),
            ));
        }
        cells.push(Cell::from(row.action.clone()));
        cells.push(Cell::from(reward));
        cells.push(Cell::from(done));
        if !app.interface_compact_view() {
            cells.push(Cell::from(row.info.clone().unwrap_or_else(|| "".into())));
        }

        table_rows.push(Row::new(cells).style(Style::default().fg(Color::White)));
    }

    let header = if app.interface_compact_view() {
        Row::new(vec!["Ep", "Step", "Agent", "Action", "Reward", "Done"])
    } else {
        Row::new(vec![
            "Ep",
            "Step",
            "Agent",
            "Policy",
            "Observation",
            "Action",
            "Reward",
            "Done",
            "Info",
        ])
    }
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );

    let widths: Vec<Constraint> = if app.interface_compact_view() {
        vec![
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(14),
            Constraint::Min(18),
            Constraint::Length(10),
            Constraint::Length(10),
        ]
    } else {
        vec![
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Length(18),
            Constraint::Min(18),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(18),
        ]
    };

    let mut title = String::from(if focus { "Actions (focus)" } else { "Actions" });
    title.push_str(&format!(
        " • displaying {} of {} rows",
        table_rows.len(),
        rows.len()
    ));
    if let Some(meta) = meta {
        title.push_str(&format!(" • {}", meta.mode().label()));
        if let (Some(ep), Some(step)) = (meta.episode(), meta.step()) {
            title.push_str(&format!(" • episode {ep} step {step}"));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style);
    let table = Table::new(table_rows, widths).header(header).block(block);
    frame.render_widget(table, area);
}

fn render_metrics_compact(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // Single column: Chart/Policies (swap based on expanded), then summary, then history
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(30),
            Constraint::Percentage(20),
        ])
        .split(area);

    // Swap chart and policies positions based on expanded state
    if app.metrics_policies_expanded() {
        render_metrics_policies_expanded(frame, chunks[0], app);
    } else {
        render_metrics_chart_section(frame, chunks[0], app);
    }
    render_metrics_summary(frame, chunks[1], app);
    render_metrics_history(frame, chunks[2], app);
}

fn render_metrics_medium(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // Chart/Policies on top (swap based on expanded), two columns below
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    // Swap chart and policies positions based on expanded state
    if app.metrics_policies_expanded() {
        render_metrics_policies_expanded(frame, rows[0], app);
    } else {
        render_metrics_chart_section(frame, rows[0], app);
    }

    // Bottom: Two columns - Summary and History+Chart/Policies stacked
    let bottom_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    render_metrics_summary(frame, bottom_columns[0], app);

    // Right side: History on top, Chart or Policies below (opposite of top)
    let right_stack = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(bottom_columns[1]);

    render_metrics_history(frame, right_stack[0], app);

    // Show the opposite in the bottom-right
    if app.metrics_policies_expanded() {
        // Policies are expanded at top, show chart at bottom
        render_metrics_chart(frame, right_stack[1], app);
    } else {
        // Chart is at top, show policies at bottom
        render_metrics_policies(frame, right_stack[1], app);
    }
}

fn render_metrics_large(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // Original three-column layout for large screens
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Swap chart and policies positions based on expanded state
    if app.metrics_policies_expanded() {
        render_metrics_policies_expanded(frame, rows[0], app);
    } else {
        render_metrics_chart_section(frame, rows[0], app);
    }

    // Bottom: Three columns for details
    let bottom_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30),
            Constraint::Percentage(40),
            Constraint::Percentage(30),
        ])
        .split(rows[1]);

    render_metrics_history(frame, bottom_columns[0], app);
    render_metrics_summary(frame, bottom_columns[1], app);

    // Show the opposite in the bottom-right
    if app.metrics_policies_expanded() {
        // Policies are expanded at top, show chart at bottom
        render_metrics_chart(frame, bottom_columns[2], app);
    } else {
        // Chart is at top, show policies at bottom
        render_metrics_policies(frame, bottom_columns[2], app);
    }
}

fn render_metrics_chart_section(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(area);

    render_metrics_chart(frame, chunks[0], app);
    render_metrics_chart_info(frame, chunks[1], app);
}

fn render_metrics_summary(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let sample = app
        .selected_metric_sample()
        .or_else(|| app.latest_training_metric());
    let summary_settings = app.metrics_summary_settings();

    if let Some(sample) = sample {
        let selection_index = app.metrics_history_selected_index();
        let position_text = if selection_index == 0 {
            "latest".to_string()
        } else {
            format!("{selection_index} back")
        };

        let mut lines = Vec::new();
        let available_width = area.width.saturating_sub(4) as usize; // Account for borders and padding
        let available_height = area.height.saturating_sub(2) as usize; // Account for borders

        // Header with position - always on one line - Selected: <position>  Iter: <iteration>  Checkpoint: <number>
        // Get the target from app.checkpoint_hint_display() if available
        let checkpoint_display = if let Some(hint) = app.checkpoint_hint_display() {
            format!("{}", hint.target)
        } else {
            "-".to_string()
        };

        lines.push(Line::from(vec![
            Span::styled("Selected: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                position_text,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Iter: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_u64(sample.training_iteration()),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw("  "),
            Span::styled("Checkpoint: ", Style::default().fg(Color::DarkGray)),
            Span::styled(checkpoint_display, Style::default().fg(Color::Cyan)),
        ]));

        // Timesteps and Episodes - smart wrap based on width
        if available_width >= 60 {
            // Wide enough - one line
            lines.push(Line::from(vec![
                Span::styled("Timesteps: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(sample.timesteps_total()),
                    Style::default().fg(Color::LightGreen),
                ),
                Span::raw("  "),
                Span::styled("Episodes: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(sample.episodes_total()),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(" ("),
                Span::styled(
                    format_option_u64(sample.episodes_this_iter()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" this iter)"),
            ]));
        } else {
            // Narrow - split into two lines
            lines.push(Line::from(vec![
                Span::styled("Timesteps: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(sample.timesteps_total()),
                    Style::default().fg(Color::LightGreen),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Episodes: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(sample.episodes_total()),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(" ("),
                Span::styled(
                    format_option_u64(sample.episodes_this_iter()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" this iter)"),
            ]));
        }

        // Rewards - always split for clarity
        lines.push(Line::from(vec![
            Span::styled("Reward: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_f64(sample.episode_reward_mean()),
                Style::default()
                    .fg(Color::LightMagenta)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  ["),
            Span::styled(
                format_option_f64(sample.episode_reward_min()),
                Style::default().fg(Color::White),
            ),
            Span::raw(" to "),
            Span::styled(
                format_option_f64(sample.episode_reward_max()),
                Style::default().fg(Color::White),
            ),
            Span::raw("]"),
        ]));

        // Episode length
        lines.push(Line::from(vec![
            Span::styled("Episode len: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_f64(sample.episode_len_mean()),
                Style::default().fg(Color::LightBlue),
            ),
        ]));

        // Throughput and time - smart wrap
        if summary_settings.show_throughput_rows {
            if available_width >= 50 {
                lines.push(Line::from(vec![
                    Span::styled("Throughput: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format_option_rate(sample.env_throughput(), " steps/s"),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw("  "),
                    Span::styled("Time: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format_option_duration(sample.time_total_s()),
                        Style::default().fg(Color::White),
                    ),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::styled("Throughput: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format_option_rate(sample.env_throughput(), " steps/s"),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("Time: ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format_option_duration(sample.time_total_s()),
                        Style::default().fg(Color::White),
                    ),
                ]));
            }
        }

        if let Some(hint) = app.checkpoint_hint_display() {
            lines.push(Line::from(vec![
                Span::styled("Local Run: ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!(
                    "local={}, offset={}, freq={}, Δ={}, start={}",
                    hint.local, hint.offset, hint.freq, hint.delta, hint.start
                )),
            ]));
        }

        // Only show detailed steps if we have enough vertical space
        if summary_settings.verbosity == SummaryVerbosity::Detailed && available_height >= 12 {
            // Environment steps - split into two lines for clarity
            lines.push(Line::from(vec![Span::styled(
                "Env steps: ",
                Style::default().fg(Color::DarkGray),
            )]));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format_option_u64(sample.num_env_steps_sampled()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" sampled, "),
                Span::styled(
                    format_option_u64(sample.num_env_steps_trained()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" trained"),
            ]));

            // Agent steps - split into two lines for clarity
            lines.push(Line::from(vec![Span::styled(
                "Agent steps: ",
                Style::default().fg(Color::DarkGray),
            )]));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format_option_u64(sample.num_agent_steps_sampled()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" sampled, "),
                Span::styled(
                    format_option_u64(sample.num_agent_steps_trained()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" trained"),
            ]));
        } else if available_height >= 8 {
            // Show compact version - just totals
            lines.push(Line::from(vec![
                Span::styled("Steps: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(sample.num_env_steps_sampled()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" env, "),
                Span::styled(
                    format_option_u64(sample.num_agent_steps_sampled()),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" agent"),
            ]));
        }

        // Custom metrics if any - only if we have space
        if available_height >= 10
            && !sample.custom_metrics().is_empty()
            && summary_settings.max_custom_metrics > 0
        {
            if let Some(custom_line) = summarize_custom_metrics(
                sample.custom_metrics(),
                summary_settings.max_custom_metrics,
            ) {
                lines.push(custom_line);
            }
        }

        if let Some(logs) = app.metrics_log_lines() {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                "Saved Run Logs (↑/↓ to scroll, press 'v' to return to live)",
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            for log_line in logs {
                lines.push(Line::from(vec![Span::styled(
                    format!("  {log_line}"),
                    Style::default().fg(Color::Gray),
                )]));
            }
        }

        // Apply scroll offset with bounds checking
        let total_lines = lines.len();
        let visible_lines = available_height;
        let scroll_offset = app
            .metrics_summary_scroll()
            .min(total_lines.saturating_sub(visible_lines));
        let lines_to_show: Vec<_> = lines.into_iter().skip(scroll_offset).collect();

        // Highlight border if this panel is focused
        let is_focused = app.metrics_focus() == MetricsFocus::Summary;
        let border_color = if is_focused {
            Color::Cyan
        } else {
            Color::DarkGray
        };
        let title = if is_focused {
            "Key Metrics [FOCUSED - ↑/↓ to scroll, Tab to switch]"
        } else {
            "Key Metrics"
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(border_color));
        frame.render_widget(
            Paragraph::new(lines_to_show)
                .block(block)
                .alignment(Alignment::Left)
                .wrap(Wrap { trim: true }),
            area,
        );
    } else {
        let placeholder = Paragraph::new("No metrics captured yet.")
            .block(Block::default().borders(Borders::ALL).title("Key Metrics"))
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
    }
}
fn render_metrics_history(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let settings = app.metrics_history_settings();
    let available_rows = area.height.saturating_sub(2) as usize;
    if available_rows == 0 {
        return;
    }

    let total = app.metrics_history_total_len();
    if total == 0 {
        return;
    }
    let selected_offset_from_latest = app
        .metrics_history_selected_index()
        .min(total.saturating_sub(1));
    let selected_display_index = if settings.sort_newest_first {
        selected_offset_from_latest
    } else {
        total
            .saturating_sub(1)
            .saturating_sub(selected_offset_from_latest)
    };

    let max_start = total.saturating_sub(available_rows);
    let start = selected_display_index
        .saturating_sub(available_rows / 2)
        .min(max_start);
    let end = (start + available_rows).min(total);

    let mut items = Vec::new();
    let max_width = area.width.saturating_sub(4) as usize;
    let window = app.metrics_history_window(start, end, settings.sort_newest_first);
    for sample in window.iter() {
        items.push(ListItem::new(metric_history_line(
            sample,
            max_width,
            settings.show_timestamp,
            settings.show_env_steps,
            settings.show_wall_clock,
        )));
    }

    let mut state = ListState::default();
    state.select(Some(selected_display_index.saturating_sub(start)));

    // Highlight border if this panel is focused
    let is_focused = app.metrics_focus() == MetricsFocus::History;
    let border_color = focused_border_color(app, is_focused);
    let mut title = if is_focused {
        if settings.sort_newest_first {
            "History (newest first) [FOCUSED - Tab to switch]".to_string()
        } else {
            "History (oldest first) [FOCUSED - Tab to switch]".to_string()
        }
    } else if settings.sort_newest_first {
        "History (newest first)".to_string()
    } else {
        "History (oldest first)".to_string()
    };
    if app.is_viewing_saved_run() {
        title.push_str(" • saved run");
    }

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(border_color)),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_stateful_widget(list, area, &mut state);
}

fn render_metrics_chart(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // Highlight border if this panel is focused
    let is_focused = app.metrics_focus() == MetricsFocus::Chart;
    let chart_settings = app.metrics_chart_settings();
    let border_color = focused_border_color(app, is_focused);
    let mut base_title = if is_focused {
        "Metric Chart [FOCUSED - Enter to view policies, x to export, Tab to switch]".to_string()
    } else {
        "Metric Chart [Enter to view policies]".to_string()
    };
    if let Some(label) = app.viewed_run_label() {
        if chart_settings.show_caption {
            base_title.push_str(&format!(" • Saved run: {label}"));
        }
    }
    if chart_settings.show_caption && chart_settings.show_resume {
        if let Some(iter) = app.resume_marker_iteration() {
            let metric_opt = app.current_chart_metric();
            let pre = metric_opt
                .as_ref()
                .and_then(|m| app.resume_marker_value(m))
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let post = metric_opt
                .as_ref()
                .and_then(|m| app.latest_chart_value(m))
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".to_string());
            base_title.push_str(&format!(" • Resume @ {iter} (pre {pre} → live {post})"));
        }
    }

    let base_block = Block::default()
        .borders(Borders::ALL)
        .title(base_title.clone())
        .border_style(Style::default().fg(border_color));

    let Some(metric_option) = app.current_chart_metric() else {
        let placeholder = Paragraph::new("No metric selected.")
            .block(base_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    };

    use crate::app::ChartMetricKind;
    match metric_option.kind() {
        ChartMetricKind::AllPoliciesRewardMean
        | ChartMetricKind::AllPoliciesEpisodeLenMean
        | ChartMetricKind::AllPoliciesLearnerStat(_) => {
            render_multi_series_chart(frame, area, app, &metric_option, base_block);
            return;
        }
        _ => {}
    }

    let max_points = chart_settings
        .max_points
        .unwrap_or_else(|| (area.width as usize).saturating_mul(4).max(1));
    let chart_data = app.chart_data(max_points, chart_settings.smoothing);
    let overlay_series = app.overlay_chart_series(
        &metric_option,
        max_points,
        chart_settings.smoothing,
        chart_settings.align_overlays_to_start,
    );

    if chart_data.is_none() && overlay_series.is_empty() {
        let placeholder = Paragraph::new("No data for the selected metric yet.")
            .block(base_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    let mut datasets = Vec::new();
    let mut y_min_primary = f64::INFINITY;
    let mut y_max_primary = f64::NEG_INFINITY;
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;

    let primary_color = app.chart_primary_color();
    let selection_color = app.chart_selection_color();
    let resume_before_color = app.chart_resume_before_color();
    let resume_after_color = app.chart_resume_after_color();
    let resume_markers = if chart_settings.show_resume {
        app.resume_markers_for_metric(&metric_option)
    } else {
        Vec::new()
    };
    let mut resume_boundaries: Vec<f64> = resume_markers.iter().map(|(x, _, _, _)| *x).collect();
    resume_boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut owned_segments: Vec<(String, Vec<(f64, f64)>, Color)> = Vec::new();
    if let Some(chart_data) = &chart_data {
        if resume_boundaries.is_empty() {
            owned_segments.push((
                chart_data.label.clone(),
                chart_data.points.clone(),
                primary_color,
            ));
        } else {
            let mut current_points: Vec<(f64, f64)> = Vec::new();
            let mut boundary_iter = resume_boundaries.iter().peekable();
            let mut current_color = resume_before_color;
            let mut part_idx = 1;

            for (x, y) in &chart_data.points {
                while let Some(boundary) = boundary_iter.peek() {
                    if x >= *boundary {
                        if !current_points.is_empty() {
                            owned_segments.push((
                                format!("{} part {}", chart_data.label, part_idx),
                                std::mem::take(&mut current_points),
                                current_color,
                            ));
                            part_idx += 1;
                        }
                        current_color = resume_after_color;
                        boundary_iter.next();
                    } else {
                        break;
                    }
                }
                current_points.push((*x, *y));
            }
            if !current_points.is_empty() {
                owned_segments.push((
                    format!("{} part {}", chart_data.label, part_idx),
                    current_points,
                    current_color,
                ));
            }
        }
    }

    for (label, points, color) in &owned_segments {
        for &(x, y) in points {
            if y.is_finite() {
                y_min_primary = y_min_primary.min(y);
                y_max_primary = y_max_primary.max(y);
            }
            x_min = x_min.min(x);
            x_max = x_max.max(x);
        }
        datasets.push(
            Dataset::default()
                .name(label.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(*color))
                .data(points),
        );
    }

    for series in &overlay_series {
        for &(x, _) in &series.points {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
        }
    }

    // Live-priority scaling: while training, let the primary series control the Y axis.
    // If the primary series has no points yet, fall back to full scaling so the chart isn't empty.
    let live_priority_scaling = app.is_training_running();
    let live_scale_active =
        live_priority_scaling && y_min_primary.is_finite() && y_max_primary.is_finite();
    let mut y_min = y_min_primary;
    let mut y_max = y_max_primary;
    if !live_scale_active {
        for series in &overlay_series {
            for &(_, y) in &series.points {
                if y.is_finite() {
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }
    }

    let mut overlay_clipped_above = false;
    let mut overlay_clipped_below = false;
    if live_scale_active && y_min.is_finite() && y_max.is_finite() {
        for series in &overlay_series {
            for &(_, y) in &series.points {
                if !y.is_finite() {
                    continue;
                }
                if y > y_max {
                    overlay_clipped_above = true;
                } else if y < y_min {
                    overlay_clipped_below = true;
                }
                if overlay_clipped_above && overlay_clipped_below {
                    break;
                }
            }
            if overlay_clipped_above && overlay_clipped_below {
                break;
            }
        }
    }

    for series in &overlay_series {
        let overlay_style = if live_scale_active {
            Style::default()
                .fg(series.color)
                .add_modifier(Modifier::DIM)
        } else {
            Style::default().fg(series.color)
        };
        datasets.push(
            Dataset::default()
                .name(series.label.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(overlay_style)
                .data(&series.points),
        );
    }

    let selected_x = app
        .selected_metric_sample()
        .and_then(|sample| sample.training_iteration().map(|iter| iter as f64))
        .or_else(|| Some(app.metrics_history_selected_index() as f64));
    let selected_y = app.selected_chart_value();
    let mut selection_line = Vec::new();
    let mut selection_marker = Vec::new();
    let mut overlay_clip_markers: Vec<(f64, f64)> = Vec::new();

    for (x, y, _, _) in &resume_markers {
        if x.is_finite() {
            x_min = x_min.min(*x);
            x_max = x_max.max(*x);
        }
        if !live_scale_active {
            if let Some(val) = y {
                if val.is_finite() {
                    y_min = y_min.min(*val);
                    y_max = y_max.max(*val);
                }
            }
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    if (y_max - y_min).abs() < 1e-6 {
        let delta = (y_max.abs().max(1.0)) * 0.1;
        y_min -= delta;
        y_max += delta;
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = 0.0;
        x_max = 1.0;
    }
    if (x_max - x_min).abs() < 1e-3 {
        x_max = x_min + 1.0;
    }

    if live_scale_active && (overlay_clipped_above || overlay_clipped_below) {
        if overlay_clipped_above {
            overlay_clip_markers.push((x_max, y_max));
        }
        if overlay_clipped_below {
            overlay_clip_markers.push((x_max, y_min));
        }
    }

    if !overlay_clip_markers.is_empty() {
        datasets.push(
            Dataset::default()
                .name(String::new())
                .marker(Marker::Block)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(Color::DarkGray))
                .data(&overlay_clip_markers),
        );
    }

    if chart_settings.show_selection {
        if let Some(x) = selected_x {
            if x.is_finite() && x >= x_min && x <= x_max {
                selection_line.push((x, y_min));
                selection_line.push((x, y_max));
                if let Some(y) = selected_y {
                    if y.is_finite() {
                        selection_marker.push((x, y));
                    }
                }
            }
        }
    }

    if !selection_line.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Selected")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(selection_color))
                .data(&selection_line),
        );
    }

    if !selection_marker.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Here")
                .marker(Marker::Block)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(selection_color))
                .data(&selection_marker),
        );
    }
    let mut resume_lines: Vec<(String, Vec<(f64, f64)>, Color)> = Vec::new();
    let mut resume_point_sets: Vec<(String, Vec<(f64, f64)>, Color)> = Vec::new();
    for (idx, (x, y, color, label)) in resume_markers.iter().enumerate() {
        if !x.is_finite() || *x < x_min || *x > x_max {
            continue;
        }
        let name = if label.is_empty() {
            format!("Resume #{} @ {}", idx + 1, *x as u64)
        } else {
            format!("{} (iter {})", label, *x as u64)
        };
        resume_lines.push((name.clone(), vec![(*x, y_min), (*x, y_max)], *color));
        if let Some(val) = y {
            if val.is_finite() {
                resume_point_sets.push((format!("{name} value"), vec![(*x, *val)], *color));
            }
        }
    }
    for (name, points, color) in &resume_lines {
        datasets.push(
            Dataset::default()
                .name(name.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(*color))
                .data(points),
        );
    }
    for (name, points, color) in &resume_point_sets {
        datasets.push(
            Dataset::default()
                .name(name.clone())
                .marker(Marker::Block)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(*color))
                .data(points),
        );
    }

    let x_labels = build_axis_labels(
        x_min,
        x_max,
        area.width.saturating_sub(6) as usize / 12,
        true,
    );
    let y_labels = build_axis_labels(
        y_min,
        y_max,
        area.height.saturating_sub(4) as usize / 6,
        false,
    );

    let x_axis = Axis::default()
        .title(Span::styled(
            chart_settings.x_label.as_str(),
            Style::default().fg(Color::DarkGray),
        ))
        .labels(x_labels)
        .bounds([x_min, x_max]);

    let y_axis = Axis::default()
        .title(Span::styled(
            if chart_settings.y_label.is_empty() || chart_settings.y_label == "Value" {
                metric_option.label()
            } else {
                chart_settings.y_label.as_str()
            },
            Style::default().fg(Color::DarkGray),
        ))
        .labels(y_labels)
        .bounds([y_min, y_max]);

    let mut chart = Chart::new(datasets)
        .block({
            let mut title = base_title.clone();
            if live_scale_active {
                title.push_str(" • live scale");
                if overlay_series.is_empty() {
                    // no-op
                } else if overlay_clipped_above || overlay_clipped_below {
                    let arrows = match (overlay_clipped_above, overlay_clipped_below) {
                        (true, true) => "↑↓",
                        (true, false) => "↑",
                        (false, true) => "↓",
                        _ => "",
                    };
                    title.push_str(&format!(" (overlays clipped {arrows})"));
                } else {
                    title.push_str(" (overlays muted)");
                }
            }
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(border_color))
        })
        .x_axis(x_axis)
        .y_axis(y_axis);

    if chart_settings.show_legend {
        let legend_pos = match chart_settings.legend_position {
            ChartLegendPosition::Auto => LegendPosition::Top,
            ChartLegendPosition::UpperLeft => LegendPosition::Top,
            ChartLegendPosition::UpperRight => LegendPosition::Top,
            ChartLegendPosition::LowerLeft => LegendPosition::Bottom,
            ChartLegendPosition::LowerRight => LegendPosition::Bottom,
            ChartLegendPosition::None => LegendPosition::Top,
        };
        chart = chart.legend_position(Some(legend_pos));
    } else {
        chart = chart.hidden_legend_constraints((Constraint::Length(0), Constraint::Length(0)));
    }

    frame.render_widget(chart, area);
}

fn render_multi_series_chart(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &App,
    metric_option: &crate::app::ChartMetricOption,
    block: Block<'_>,
) {
    let chart_settings = app.metrics_chart_settings();
    let selection_color = app.chart_selection_color();
    let resume_markers = if chart_settings.show_resume {
        app.resume_markers_for_metric(metric_option)
    } else {
        Vec::new()
    };
    let max_points = chart_settings
        .max_points
        .unwrap_or_else(|| (area.width as usize).saturating_mul(4).max(1));
    let multi_data =
        app.chart_multi_series_data(metric_option, max_points, chart_settings.smoothing);

    if multi_data.is_empty() {
        let placeholder = Paragraph::new("No data for overlay chart yet.")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
        return;
    }

    // Calculate bounds across all series
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;

    for entry in &multi_data {
        for &(x, y) in &entry.points {
            if x.is_finite() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
            if y.is_finite() {
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }
    }
    for (x, y, _, _) in &resume_markers {
        if x.is_finite() {
            x_min = x_min.min(*x);
            x_max = x_max.max(*x);
        }
        if let Some(val) = y {
            if val.is_finite() {
                y_min = y_min.min(*val);
                y_max = y_max.max(*val);
            }
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = 0.0;
        x_max = 1.0;
    }

    if (y_max - y_min).abs() < 1e-6 {
        let delta = (y_max.abs().max(1.0)) * 0.1;
        y_min -= delta;
        y_max += delta;
    }
    if (x_max - x_min).abs() < 1e-3 {
        x_max = x_min + 1.0;
    }

    // Ghost series first (drawn under primaries), then primaries.
    let ghost_palette = [
        Color::DarkGray,
        Color::Gray,
        Color::Rgb(180, 180, 180),
        Color::White,
    ];
    let mut datasets: Vec<Dataset> = Vec::new();
    for entry in multi_data.iter().filter(|e| e.is_ghost) {
        let color = ghost_palette[entry.ghost_index % ghost_palette.len()];
        datasets.push(
            Dataset::default()
                // Blank name keeps ghosts out of the legend while still rendering.
                .name(String::new())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(color))
                .data(&entry.points),
        );
    }
    for (idx, entry) in multi_data.iter().filter(|e| !e.is_ghost).enumerate() {
        let color = app.policy_color(&entry.policy_id, idx);
        datasets.push(
            Dataset::default()
                .name(entry.label.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(color))
                .data(&entry.points),
        );
    }

    let selected_x = app
        .selected_metric_sample()
        .and_then(|sample| sample.training_iteration().map(|iter| iter as f64))
        .or_else(|| Some(app.metrics_history_selected_index() as f64));
    let mut selection_line = Vec::new();
    if chart_settings.show_selection {
        if let Some(x) = selected_x {
            if x.is_finite() && x >= x_min && x <= x_max {
                selection_line.push((x, y_min));
                selection_line.push((x, y_max));
            }
        }
    }

    if !selection_line.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Selected")
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(selection_color))
                .data(&selection_line),
        );
    }
    let mut resume_lines: Vec<(String, Vec<(f64, f64)>, Color)> = Vec::new();
    let mut resume_point_sets: Vec<(String, Vec<(f64, f64)>, Color)> = Vec::new();
    for (idx, (x, y, color, label)) in resume_markers.iter().enumerate() {
        if !x.is_finite() || *x < x_min || *x > x_max {
            continue;
        }
        let name = if label.is_empty() {
            format!("Resume #{} @ {}", idx + 1, *x as u64)
        } else {
            format!("{} (iter {})", label, *x as u64)
        };
        resume_lines.push((name.clone(), vec![(*x, y_min), (*x, y_max)], *color));
        if let Some(val) = y {
            if val.is_finite() {
                resume_point_sets.push((format!("{name} value"), vec![(*x, *val)], *color));
            }
        }
    }
    for (name, points, color) in &resume_lines {
        datasets.push(
            Dataset::default()
                .name(name.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(*color))
                .data(points),
        );
    }
    for (name, points, color) in &resume_point_sets {
        datasets.push(
            Dataset::default()
                .name(name.clone())
                .marker(Marker::Block)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(*color))
                .data(points),
        );
    }

    let x_labels = build_axis_labels(
        x_min,
        x_max,
        area.width.saturating_sub(6) as usize / 12,
        true,
    );
    let y_labels = build_axis_labels(
        y_min,
        y_max,
        area.height.saturating_sub(4) as usize / 6,
        false,
    );

    let x_axis = Axis::default()
        .title(Span::styled(
            chart_settings.x_label.as_str(),
            Style::default().fg(Color::DarkGray),
        ))
        .labels(x_labels)
        .bounds([x_min, x_max]);

    // Get metric label from option
    use crate::app::ChartMetricKind;
    let y_label = match metric_option.kind() {
        ChartMetricKind::AllPoliciesRewardMean => "Reward",
        ChartMetricKind::AllPoliciesEpisodeLenMean => "Episode Length",
        ChartMetricKind::AllPoliciesLearnerStat(key) => key.as_str(),
        _ => "Value",
    };

    let y_axis = Axis::default()
        .title(Span::styled(
            if chart_settings.y_label.is_empty() || chart_settings.y_label == "Value" {
                y_label
            } else {
                chart_settings.y_label.as_str()
            },
            Style::default().fg(Color::DarkGray),
        ))
        .labels(y_labels)
        .bounds([y_min, y_max]);

    let mut chart = Chart::new(datasets)
        .block(block)
        .x_axis(x_axis)
        .y_axis(y_axis);

    if chart_settings.show_legend {
        let legend_pos = match chart_settings.legend_position {
            ChartLegendPosition::Auto => LegendPosition::Top,
            ChartLegendPosition::UpperLeft => LegendPosition::TopLeft,
            ChartLegendPosition::UpperRight => LegendPosition::TopRight,
            ChartLegendPosition::LowerLeft => LegendPosition::BottomLeft,
            ChartLegendPosition::LowerRight => LegendPosition::BottomRight,
            ChartLegendPosition::None => LegendPosition::Top,
        };
        chart = chart.legend_position(Some(legend_pos));
    } else {
        chart = chart.hidden_legend_constraints((Constraint::Length(0), Constraint::Length(0)));
    }

    frame.render_widget(chart, area);
}

fn render_metrics_chart_info(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let info_settings = app.metrics_info_settings();
    let metric = app.current_chart_metric();
    let metric_label = metric
        .as_ref()
        .map(|m| m.label().to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let policy_label = metric
        .as_ref()
        .and_then(|m| m.policy_id().map(|p| p.to_string()))
        .unwrap_or_else(|| "—".to_string());
    let value = app
        .selected_chart_value()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "—".to_string());

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let available_width = columns[0].width.saturating_sub(4) as usize;
    let mut lines = Vec::new();

    if available_width >= 60 {
        lines.push(Line::from(vec![
            Span::styled("Metric: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                metric_label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Value: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                value,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Policy: ", Style::default().fg(Color::DarkGray)),
            Span::styled(policy_label, Style::default().fg(Color::Yellow)),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("Metric: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                metric_label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Value: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                value,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Policy: ", Style::default().fg(Color::DarkGray)),
            Span::styled(policy_label, Style::default().fg(Color::Yellow)),
        ]));
    }

    if info_settings.show_hints {
        if available_width >= 90 {
            lines.push(Line::from(vec![Span::styled(
                "↑/↓: navigate history  •  PgUp/PgDn: jump  •  Home/End: edges  •  , / . : cycle metric",
                Style::default().fg(Color::DarkGray),
            )]));
        } else if available_width >= 60 {
            lines.push(Line::from(vec![Span::styled(
                "↑/↓: history  •  PgUp/Dn: jump marker  •  Home/End: edges  •  ,/.: metric",
                Style::default().fg(Color::DarkGray),
            )]));
        } else {
            lines.push(Line::from(vec![Span::styled(
                "↑/↓: history  •  PgUp/Dn: jump marker  •  ,/.: metric",
                Style::default().fg(Color::DarkGray),
            )]));
        }

        lines.push(Line::from(vec![Span::styled(
            "r: use selected checkpoint directory to prefill resume + export paths",
            Style::default().fg(Color::LightGreen),
        )]));

        if app.has_saved_run_overlays() {
            if let Some(label) = app.selected_overlay_label() {
                lines.push(Line::from(vec![Span::styled(
                    format!("Selected run: {label} — 'o' view / 'O' cycle"),
                    Style::default().fg(Color::LightCyan),
                )]));
            }
        }

        if !app.discovered_runs().is_empty() {
            let hint_label = app
                .selected_discovered_label()
                .unwrap_or_else(|| "latest detected".to_string());
            lines.push(Line::from(vec![Span::styled(
                format!("Detected RLlib runs: {hint_label} — 'f' pick/run menu, 'g' quick load latest, 'G' cycle+load"),
                Style::default().fg(Color::LightYellow),
            )]));
        } else {
            lines.push(Line::from(vec![Span::styled(
                "Press 'f' to scan logs/rllib, pick a run, or choose a file manually",
                Style::default().fg(Color::DarkGray),
            )]));
        }

        if let Some(hint) = app.metrics_source_hint() {
            lines.push(Line::from(vec![Span::styled(
                hint,
                Style::default().fg(Color::LightMagenta),
            )]));
        }
    }

    let is_focused = app.metrics_focus() == MetricsFocus::Chart;
    let border_color = focused_border_color(app, is_focused);
    let mut title = if is_focused {
        "Chart Info [FOCUSED - ↑/↓ to navigate]".to_string()
    } else {
        "Chart Info".to_string()
    };
    if let Some(source) = app.metrics_source_hint() {
        title.push_str(&format!(" • {source}"));
    }

    let info_block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(border_color));
    frame.render_widget(
        Paragraph::new(lines)
            .block(info_block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true }),
        columns[0],
    );

    let stats_block = Block::default()
        .borders(Borders::ALL)
        .title("Marker Stats")
        .border_style(Style::default().fg(border_color));
    let stats_content = if info_settings.show_marker_stats {
        Paragraph::new(marker_stats_lines(app))
    } else {
        Paragraph::new(Line::from(Span::styled(
            "Marker stats hidden (toggle in settings)",
            Style::default().fg(Color::DarkGray),
        )))
    };
    frame.render_widget(
        stats_content
            .block(stats_block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true }),
        columns[1],
    );
}

fn marker_stats_lines(app: &App) -> Vec<Line<'static>> {
    let sample = match app.selected_metric_sample() {
        Some(s) => s,
        None => {
            return vec![Line::from(Span::styled(
                "No data yet.",
                Style::default().fg(Color::DarkGray),
            ))]
        }
    };

    let mut lines = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("Iter: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_option_u64(sample.training_iteration()),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw("  "),
        Span::styled("Steps: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_option_u64(sample.timesteps_total()),
            Style::default().fg(Color::LightGreen),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("Reward: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_option_f64(sample.episode_reward_mean()),
            Style::default().fg(Color::LightMagenta),
        ),
        Span::raw("  "),
        Span::styled("Len: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_option_f64(sample.episode_len_mean()),
            Style::default().fg(Color::LightBlue),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("Throughput: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_option_rate(sample.env_throughput(), " steps/s"),
            Style::default().fg(Color::LightYellow),
        ),
    ]));

    lines.push(Line::from(vec![
        Span::styled("Episodes: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format_option_u64(sample.episodes_total()),
            Style::default().fg(Color::White),
        ),
    ]));

    if let Some(metric) = app.current_chart_metric() {
        if let Some(policy_id) = metric.policy_id() {
            if let Some(policy) = sample.policies().get(policy_id) {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{policy_id} reward: "),
                        Style::default()
                            .fg(Color::LightCyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format_option_f64(policy.reward_mean()),
                        Style::default().fg(Color::White),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{policy_id} len: "),
                        Style::default()
                            .fg(Color::LightCyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format_option_f64(policy.episode_len_mean()),
                        Style::default().fg(Color::White),
                    ),
                ]));
            }
        }
    }

    lines
}

fn render_metrics_policies(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Policies")
        .border_style(Style::default().fg(Color::DarkGray));

    let sample = if let Some(sample) = app.selected_metric_sample() {
        sample
    } else {
        let placeholder = Paragraph::new("Policy metrics will appear once available.")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray))
            .wrap(Wrap { trim: true });
        frame.render_widget(placeholder, area);
        return;
    };

    if sample.policies().is_empty() {
        let placeholder = Paragraph::new("Policy metrics will appear once available.")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray))
            .wrap(Wrap { trim: true });
        frame.render_widget(placeholder, area);
        return;
    }

    let settings = app.metrics_policies_settings();
    let highlighted_policy = app
        .current_chart_metric()
        .and_then(|metric| metric.policy_id().map(|p| p.to_string()));

    let available_width = area.width.saturating_sub(4) as usize; // Account for borders
    let available_height = area.height.saturating_sub(2) as usize; // Account for borders

    // Collect and sort policies using configured mode
    let mut policies: Vec<_> = sample.policies().iter().collect();
    policies.sort_by(|a, b| match settings.sort {
        PoliciesSortMode::Alphanumeric => {
            let key_a = alphanumeric_sort_key(a.0);
            let key_b = alphanumeric_sort_key(b.0);
            key_a.cmp(&key_b)
        }
        PoliciesSortMode::RewardDescending => {
            let a_reward = a.1.reward_mean().unwrap_or(f64::MIN);
            let b_reward = b.1.reward_mean().unwrap_or(f64::MIN);
            b_reward
                .partial_cmp(&a_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    let mut items = Vec::new();

    for (policy_id, metrics) in policies {
        let comparison = app.policy_comparison(policy_id);
        let mut lines = Vec::new();
        let header_style = if highlighted_policy
            .as_deref()
            .map(|p| p == policy_id)
            .unwrap_or(false)
        {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        };
        lines.push(Line::from(vec![Span::styled(
            policy_id.clone(),
            header_style,
        )]));

        // Reward mean on its own line
        lines.push(Line::from(vec![
            Span::styled("Reward μ: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_f64(metrics.reward_mean()),
                Style::default()
                    .fg(Color::LightMagenta)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        // Min/max on second line with indentation
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("min: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_f64(metrics.reward_min()),
                Style::default().fg(Color::White),
            ),
            Span::raw("  "),
            Span::styled("max: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_f64(metrics.reward_max()),
                Style::default().fg(Color::White),
            ),
        ]));

        // Episode length and completed - smart wrap
        if available_width >= 35 {
            lines.push(Line::from(vec![
                Span::styled("Ep len: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_f64(metrics.episode_len_mean()),
                    Style::default().fg(Color::LightBlue),
                ),
                Span::raw("  "),
                Span::styled("Done: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(metrics.completed_episodes()),
                    Style::default().fg(Color::White),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("Ep len: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_f64(metrics.episode_len_mean()),
                    Style::default().fg(Color::LightBlue),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Done: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format_option_u64(metrics.completed_episodes()),
                    Style::default().fg(Color::White),
                ),
            ]));
        }

        if settings.show_overlay_deltas {
            if let Some(comp) = comparison.as_ref() {
                if let Some(line) = render_compact_policy_delta("Δ Reward μ: ", comp.reward_mean)
                {
                    lines.push(line);
                }
                if let Some(line) = render_compact_policy_delta("Δ Ep len: ", comp.episode_len_mean)
                {
                    lines.push(line);
                }
            }
        }

        // Learner stats - show key metrics, wrap intelligently
        // Limit stats shown based on available height and setting
        let auto_limit = if available_height < 8 {
            2 // Very limited space, show only 2 stats per policy
        } else if available_height < 12 {
            3 // Limited space, show 3 stats per policy
        } else {
            5 // Normal space, show up to 5 stats per policy
        };
        let max_learner_stats = auto_limit.min(settings.max_learner_stats);

        if !metrics.learner_stats().is_empty() {
            // Show most important learner stats on separate lines for readability
            let important_stats = ["total_loss", "policy_loss", "vf_loss", "kl", "entropy"];
            let mut shown = 0;

            for stat_name in &important_stats {
                if shown >= max_learner_stats {
                    break;
                }
                if let Some(value) = metrics.learner_stats().get(*stat_name) {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(
                            format!("{}: ", stat_name),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(format!("{:.4}", value), Style::default().fg(Color::White)),
                    ]));
                    shown += 1;
                    if shown >= 3 && available_width < 30 {
                        // Limited width space, show fewer stats
                        break;
                    }
                }
            }

            // Show count of remaining stats if any
            let remaining = metrics.learner_stats().len().saturating_sub(shown);
            if remaining > 0 {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        format!("(+{} more stats)", remaining),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
        }

        // Custom metrics - only show if we have enough height
        if available_height >= 10 && !metrics.custom_metrics().is_empty() {
            if let Some(custom_line) = summarize_custom_metrics(metrics.custom_metrics(), 4) {
                lines.push(custom_line);
            }
        }

        items.push(ListItem::new(lines));
    }

    // Apply scroll offset with bounds checking
    let total_items = items.len();
    let visible_items = available_height.saturating_sub(2); // Account for borders
    let scroll_offset = app
        .metrics_policies_scroll()
        .min(total_items.saturating_sub(visible_items.max(1)));
    let items_to_show: Vec<_> = items.into_iter().skip(scroll_offset).collect();

    // Highlight border if this panel is focused
    let is_focused = app.metrics_focus() == MetricsFocus::Policies;
    let border_color = if is_focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let title = if is_focused {
        "Policies [FOCUSED - ↑/↓ to scroll, Enter to expand, Tab to switch]"
    } else {
        "Policies [Enter to expand]"
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(border_color));

    let list = List::new(items_to_show).block(block);
    frame.render_widget(list, area);
}

fn render_metrics_policies_expanded(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let sample = if let Some(sample) = app.selected_metric_sample() {
        sample
    } else {
        let placeholder = Paragraph::new(
            "Policy metrics will appear once available.\n\nPress Enter to return to chart view.",
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Expanded Policies View")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: true });
        frame.render_widget(placeholder, area);
        return;
    };

    if sample.policies().is_empty() {
        let placeholder = Paragraph::new(
            "Policy metrics will appear once available.\n\nPress Enter to return to chart view.",
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Expanded Policies View")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: true });
        frame.render_widget(placeholder, area);
        return;
    }

    let settings = app.metrics_policies_settings();
    // Collect and sort policies using configured mode
    let mut policies: Vec<_> = sample.policies().iter().collect();
    policies.sort_by(|a, b| match settings.sort {
        PoliciesSortMode::Alphanumeric => {
            let key_a = alphanumeric_sort_key(a.0);
            let key_b = alphanumeric_sort_key(b.0);
            key_a.cmp(&key_b)
        }
        PoliciesSortMode::RewardDescending => {
            let a_reward = a.1.reward_mean().unwrap_or(f64::MIN);
            let b_reward = b.1.reward_mean().unwrap_or(f64::MIN);
            b_reward
                .partial_cmp(&a_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    let num_policies = policies.len();
    let h_scroll = app.metrics_policies_horizontal_scroll();
    let v_scroll = app.metrics_policies_scroll();

    // Calculate how many policies we can show side-by-side
    let available_width = area.width.saturating_sub(2); // Account for outer borders
    let min_policy_width = 35; // Minimum width per policy column
    let policies_per_screen = (available_width / min_policy_width).max(1) as usize;

    // Determine which policies to show based on horizontal scroll (with bounds checking)
    let start_idx = h_scroll.min(num_policies.saturating_sub(1));
    let end_idx = (start_idx + policies_per_screen).min(num_policies);
    let visible_policies = &policies[start_idx..end_idx];

    // Calculate equal width for each visible policy
    let policy_width = if visible_policies.len() > 0 {
        available_width / visible_policies.len() as u16
    } else {
        available_width
    };

    // Create title with navigation hints
    let title = format!(
        "Expanded Policies View [{}/{} policies] [←/→ or h/l to scroll, ↑/↓ or k/j to scroll content, Enter to close]",
        start_idx + 1,
        num_policies
    );

    let outer_block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(Color::Cyan));

    let inner_area = outer_block.inner(area);
    frame.render_widget(outer_block, area);

    // Create horizontal layout for visible policies
    let mut constraints = vec![Constraint::Length(policy_width); visible_policies.len()];
    if let Some(last) = constraints.last_mut() {
        *last = Constraint::Min(policy_width); // Last one takes remaining space
    }

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(inner_area);

    // Render each policy in its own column
    for (idx, ((policy_id, metrics), chunk)) in
        visible_policies.iter().zip(chunks.iter()).enumerate()
    {
        render_single_policy_detailed(frame, *chunk, policy_id, metrics, v_scroll, idx == 0, app);
    }
}

fn render_single_policy_detailed(
    frame: &mut Frame<'_>,
    area: Rect,
    policy_id: &str,
    metrics: &crate::app::PolicyMetrics,
    scroll_offset: usize,
    is_first: bool,
    app: &App,
) {
    let settings = app.metrics_policies_settings();
    let mut lines = Vec::new();

    // Policy name as header
    lines.push(Line::from(vec![Span::styled(
        policy_id.to_string(),
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
    )]));
    lines.push(Line::from(""));

    // Reward statistics
    lines.push(Line::from(vec![Span::styled(
        "Rewards:",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )]));
    lines.push(Line::from(vec![
        Span::raw("  Mean: "),
        Span::styled(
            format_option_f64(metrics.reward_mean()),
            Style::default()
                .fg(Color::LightMagenta)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  Min:  "),
        Span::styled(
            format_option_f64(metrics.reward_min()),
            Style::default().fg(Color::White),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  Max:  "),
        Span::styled(
            format_option_f64(metrics.reward_max()),
            Style::default().fg(Color::White),
        ),
    ]));
    lines.push(Line::from(""));

    // Episode statistics
    lines.push(Line::from(vec![Span::styled(
        "Episodes:",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )]));
    lines.push(Line::from(vec![
        Span::raw("  Len μ: "),
        Span::styled(
            format_option_f64(metrics.episode_len_mean()),
            Style::default().fg(Color::LightBlue),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  Done:  "),
        Span::styled(
            format_option_u64(metrics.completed_episodes()),
            Style::default().fg(Color::White),
        ),
    ]));
    lines.push(Line::from(""));

    // Learner stats
    if !metrics.learner_stats().is_empty() && settings.max_learner_stats > 0 {
        lines.push(Line::from(vec![Span::styled(
            "Learner Stats:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]));

        for (idx, (key, value)) in metrics.learner_stats().iter().enumerate() {
            if idx >= settings.max_learner_stats {
                break;
            }
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{}: ", key), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.6}", value), Style::default().fg(Color::White)),
            ]));
        }
        lines.push(Line::from(""));
    }

    // Custom metrics
    if settings.show_custom_metrics && !metrics.custom_metrics().is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "Custom Metrics:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]));

        for (key, value) in metrics.custom_metrics() {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{}: ", key), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.4}", value), Style::default().fg(Color::White)),
            ]));
        }
    }

    if settings.show_overlay_deltas {
        if let Some(comp) = app.policy_comparison(policy_id) {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                format!("Comparison vs {}", comp.baseline_label),
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            )]));
            if let Some(line) = render_expanded_policy_delta("Reward μ", comp.reward_mean) {
                lines.push(line);
            }
            if let Some(line) = render_expanded_policy_delta("Reward min", comp.reward_min) {
                lines.push(line);
            }
            if let Some(line) = render_expanded_policy_delta("Reward max", comp.reward_max) {
                lines.push(line);
            }
            if let Some(line) = render_expanded_policy_delta("Ep len μ", comp.episode_len_mean) {
                lines.push(line);
            }
            if let Some(line) =
                render_expanded_policy_delta_u64("Completed", comp.completed_episodes)
            {
                lines.push(line);
            }
        }
    }

    // Apply vertical scroll with bounds checking
    let total_lines = lines.len();
    let visible_lines = area.height.saturating_sub(2) as usize; // Account for borders
    let clamped_scroll = scroll_offset.min(total_lines.saturating_sub(visible_lines));
    let lines_to_show: Vec<_> = lines.into_iter().skip(clamped_scroll).collect();

    // Only show left border for non-first columns to avoid double borders
    let borders = if is_first {
        Borders::ALL
    } else {
        Borders::TOP | Borders::RIGHT | Borders::BOTTOM
    };

    let block = Block::default()
        .borders(borders)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(lines_to_show)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn metric_history_line(
    sample: &MetricSample,
    max_width: usize,
    show_timestamp: bool,
    show_env_steps: bool,
    wall_clock: bool,
) -> Line<'static> {
    let timestamp = if !show_timestamp {
        String::new()
    } else if wall_clock {
        sample.timestamp().unwrap_or("—").to_string()
    } else {
        let ts = sample.timestamp().unwrap_or("—").to_string();
        let ts = ts
            .split('T')
            .nth(1)
            .unwrap_or("—")
            .split('Z')
            .nth(0)
            .unwrap_or("—");
        let ts = ts.split(':').take(3).collect::<Vec<_>>().join(":");
        ts.split('.').nth(0).unwrap_or("—").to_string()
    };

    let mut segments: Vec<(String, Style)> = Vec::new();
    if show_timestamp {
        segments.push((
            format!("{timestamp} "),
            Style::default().fg(Color::DarkGray),
        ));
        segments.push(("  ".to_string(), Style::default()));
    }
    segments.push((
        format!("iter {}", format_option_u64(sample.training_iteration())),
        Style::default().fg(Color::Cyan),
    ));
    segments.push(("  ".to_string(), Style::default()));
    segments.push((
        format!("steps {}", format_option_u64(sample.timesteps_total())),
        Style::default().fg(Color::LightGreen),
    ));
    segments.push(("  ".to_string(), Style::default()));
    segments.push((
        format!("reward {}", format_option_f64(sample.episode_reward_mean())),
        Style::default().fg(Color::LightMagenta),
    ));
    segments.push(("  ".to_string(), Style::default()));
    segments.push((
        format!("len {}", format_option_f64(sample.episode_len_mean())),
        Style::default().fg(Color::LightBlue),
    ));
    segments.push(("  ".to_string(), Style::default()));
    if show_env_steps {
        if let Some(env_steps) = sample.env_steps_this_iter() {
            segments.push((
                format!("env {}", env_steps),
                Style::default().fg(Color::LightYellow),
            ));
        }
    }

    let spans = truncate_segments(segments, max_width);
    Line::from(spans)
}

fn truncate_segments(segments: Vec<(String, Style)>, max_width: usize) -> Vec<Span<'static>> {
    if max_width == 0 {
        return Vec::new();
    }
    let mut collected = Vec::new();
    let mut width = 0usize;
    for (text, style) in segments {
        let remaining = max_width.saturating_sub(width);
        if remaining == 0 {
            break;
        }
        let full_width = UnicodeWidthStr::width(text.as_str());
        if full_width <= remaining {
            collected.push(Span::styled(text, style));
            width += full_width;
        } else {
            let mut partial = String::new();
            let mut partial_width = 0usize;
            for ch in text.chars() {
                let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
                if partial_width + ch_width > remaining {
                    break;
                }
                partial.push(ch);
                partial_width += ch_width;
            }
            if !partial.is_empty() {
                collected.push(Span::styled(partial, style));
            }
            break;
        }
    }
    collected
}

fn metrics_setting_label(field: MetricsSettingField) -> &'static str {
    match field {
        MetricsSettingField::ChartShowLegend => "Show legend",
        MetricsSettingField::ChartLegendPosition => "Legend position",
        MetricsSettingField::ChartShowResumeMarker => "Resume marker",
        MetricsSettingField::ChartShowSelectionMarker => "Selection marker",
        MetricsSettingField::ChartShowCaption => "Caption/title",
        MetricsSettingField::ChartShowGhostOverlays => "Show ghost overlays",
        MetricsSettingField::ChartGhostSpillLimit => "Ghost spill limit (Auto)",
        MetricsSettingField::ChartXAxisLabel => "X axis title",
        MetricsSettingField::ChartYAxisLabel => "Y axis title",
        MetricsSettingField::ChartAlignOverlaysToStart => "Align overlays to start",
        MetricsSettingField::ChartMaxPoints => "Max points (Auto)",
        MetricsSettingField::ChartSmoothing => "Smoothing",
        MetricsSettingField::ChartPrimaryColor => "Series color",
        MetricsSettingField::ChartSelectionColor => "Selection color",
        MetricsSettingField::ChartResumeBeforeColor => "Resume (before) color",
        MetricsSettingField::ChartResumeAfterColor => "Resume (after) color",
        MetricsSettingField::ChartResumeMarkerColor => "Resume marker color",
        MetricsSettingField::ChartPaletteName => "Palette",
        MetricsSettingField::HistorySortNewestFirst => "Newest first",
        MetricsSettingField::HistoryAutoFollow => "Auto-follow latest",
        MetricsSettingField::HistoryPageStep => "Page step",
        MetricsSettingField::HistoryShowTimestamp => "Show timestamp",
        MetricsSettingField::HistoryShowEnvSteps => "Show env steps",
        MetricsSettingField::HistoryShowWallClock => "Wall-clock time",
        MetricsSettingField::SummaryVerbosity => "Verbosity",
        MetricsSettingField::SummaryMaxCustom => "Max custom metrics",
        MetricsSettingField::SummaryShowOverlayDeltas => "Overlay deltas",
        MetricsSettingField::SummaryShowThroughput => "Throughput/time rows",
        MetricsSettingField::PoliciesDefaultView => "Default view",
        MetricsSettingField::PoliciesSort => "Sort policies by",
        MetricsSettingField::PoliciesMaxLearnerStats => "Max learner stats",
        MetricsSettingField::PoliciesShowCustomMetrics => "Show custom metrics",
        MetricsSettingField::PoliciesShowOverlayDeltas => "Overlay deltas",
        MetricsSettingField::PoliciesStartExpanded => "Start expanded",
        MetricsSettingField::PoliciesColorMode => "Policy color mode",
        MetricsSettingField::PoliciesColorOverride => "Policy color override",
        MetricsSettingField::InfoShowHints => "Show hints",
        MetricsSettingField::InfoShowMarkerStats => "Show marker stats",
    }
}

fn format_option_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "—".to_string())
}

fn format_option_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "—".to_string())
}

fn format_option_rate(value: Option<f64>, suffix: &str) -> String {
    value
        .map(|v| format!("{v:.2}{suffix}"))
        .unwrap_or_else(|| format!("—{suffix}"))
}

fn format_option_duration(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|seconds| format_duration(seconds.max(0.0)))
        .unwrap_or_else(|| "—".to_string())
}

fn render_compact_policy_delta(
    label: &'static str,
    pair: Option<(f64, f64)>,
) -> Option<Line<'static>> {
    pair.map(move |(live, baseline)| {
        Line::from(vec![
            Span::styled(label, Style::default().fg(Color::DarkGray)),
            delta_span(live - baseline),
        ])
    })
}

fn render_expanded_policy_delta(
    label: &'static str,
    pair: Option<(f64, f64)>,
) -> Option<Line<'static>> {
    pair.map(move |(live, baseline)| {
        Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{label}: "), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{baseline:.3}"), Style::default().fg(Color::Gray)),
            Span::raw(" → "),
            Span::styled(format!("{live:.3}"), Style::default().fg(Color::LightGreen)),
            Span::raw(" (Δ "),
            delta_span(live - baseline),
            Span::raw(")"),
        ])
    })
}

fn render_expanded_policy_delta_u64(
    label: &'static str,
    pair: Option<(u64, u64)>,
) -> Option<Line<'static>> {
    pair.map(move |(live, baseline)| {
        let delta = live as i64 - baseline as i64;
        Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{label}: "), Style::default().fg(Color::DarkGray)),
            Span::styled(baseline.to_string(), Style::default().fg(Color::Gray)),
            Span::raw(" → "),
            Span::styled(live.to_string(), Style::default().fg(Color::LightGreen)),
            Span::raw(" (Δ "),
            delta_span_u64(delta),
            Span::raw(")"),
        ])
    })
}

fn delta_span(delta: f64) -> Span<'static> {
    let color = if delta > 1e-6 {
        Color::LightGreen
    } else if delta < -1e-6 {
        Color::LightRed
    } else {
        Color::Gray
    };
    Span::styled(format!("{delta:+.3}"), Style::default().fg(color))
}

fn delta_span_u64(delta: i64) -> Span<'static> {
    let color = if delta > 0 {
        Color::LightGreen
    } else if delta < 0 {
        Color::LightRed
    } else {
        Color::Gray
    };
    Span::styled(format!("{delta:+}"), Style::default().fg(color))
}

fn format_duration(seconds: f64) -> String {
    if seconds >= 3600.0 {
        let hours = (seconds / 3600.0).floor() as u64;
        let minutes = ((seconds - hours as f64 * 3600.0) / 60.0).floor() as u64;
        let remaining = seconds - hours as f64 * 3600.0 - minutes as f64 * 60.0;
        if minutes > 0 {
            format!("{hours}h {minutes}m {remaining:.0}s")
        } else {
            format!("{hours}h {remaining:.0}s")
        }
    } else if seconds >= 60.0 {
        let minutes = (seconds / 60.0).floor() as u64;
        let remaining = seconds - minutes as f64 * 60.0;
        if remaining >= 10.0 {
            format!("{minutes}m {remaining:.0}s")
        } else {
            format!("{minutes}m {remaining:.1}s")
        }
    } else if seconds >= 10.0 {
        format!("{seconds:.1}s")
    } else if seconds >= 1.0 {
        format!("{seconds:.2}s")
    } else {
        format!("{seconds:.3}s")
    }
}

fn summarize_custom_metrics(
    metrics: &BTreeMap<String, f64>,
    max_items: usize,
) -> Option<Line<'static>> {
    if metrics.is_empty() {
        return None;
    }
    let mut spans = Vec::new();
    for (index, (key, value)) in metrics.iter().take(max_items).enumerate() {
        if index > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            format!("{key}: {value:.2}"),
            Style::default().fg(Color::Gray),
        ));
    }
    let remaining = metrics.len().saturating_sub(max_items);
    if remaining > 0 {
        spans.push(Span::styled(
            format!(" +{remaining} more"),
            Style::default().fg(Color::DarkGray),
        ));
    }
    Some(Line::from(spans))
}

fn render_training_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let instructions = if app.is_training_running() {
        "Training running… press 'c' to cancel (confirm)"
    } else {
        match app.input_mode() {
            InputMode::EditingConfig | InputMode::EditingAdvancedConfig => {
                "Editing config: type to update • Enter: save • Esc: cancel"
            }
            InputMode::AdvancedConfig => {
                "Advanced config: ↑/↓ navigate • Enter: edit • a/Esc: close"
            }
            _ => {
                "t: start training • m: toggle mode • d: run demo • g: write RLlib config • p/s/n: edit basics • a: advanced panel • b: browse • l: clear log • ↑/↓: scroll • PgUp/PgDn: fast scroll"
            }
        }
    };

    let mut status_text = app
        .status()
        .map(|s| s.text.clone())
        .unwrap_or_else(|| "Ready".to_string());
    if app.is_training_running() || app.is_export_running() {
        if let Some(spinner) = app.spinner_char() {
            status_text = format!("{spinner} {status_text}");
        }
    }
    let status_style = app
        .status()
        .map(|s| status_style(s.kind))
        .unwrap_or_else(|| Style::default().fg(Color::DarkGray));

    let lines = vec![
        Line::from(Span::styled(status_text, status_style)),
        Line::from(Span::styled(
            instructions,
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let status_block = Block::default().borders(Borders::ALL).title("Status");
    let status_paragraph = Paragraph::new(lines)
        .block(status_block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    frame.render_widget(status_paragraph, area);
}

pub fn render_export(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let constraints = if area.height >= 16 {
        [
            Constraint::Length(7),
            Constraint::Min(6),
            Constraint::Length(3),
        ]
    } else if area.height >= 12 {
        [
            Constraint::Length(6),
            Constraint::Min(4),
            Constraint::Length(3),
        ]
    } else {
        [
            Constraint::Length(5),
            Constraint::Min(3),
            Constraint::Length(2),
        ]
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_export_config(frame, chunks[0], app);
    render_export_output(frame, chunks[1], app);
    render_export_status(frame, chunks[2], app);
}

pub fn render_projects(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(6),
            Constraint::Min(4),
            Constraint::Length(5),
        ])
        .split(area);

    // Header with active project info
    let project_line = if let Some(project) = app.active_project() {
        let mut line = format!("Active project: {}", project.name);
        if project.read_only {
            line.push_str(" [read-only]");
        }
        line
    } else {
        "No active project selected".to_string()
    };
    let header = Paragraph::new(Line::from(project_line))
        .block(Block::default().borders(Borders::ALL).title("Project"))
        .alignment(Alignment::Left);
    frame.render_widget(header, chunks[0]);

    // Options + sessions columns
    let body_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Settings list
    let fields = app.project_archive_fields();
    let mut items = Vec::new();
    for field in fields {
        let label = App::project_archive_field_label(*field);
        let editing_name =
            app.input_mode() == InputMode::EditingProjectArchive && *field == crate::app::ProjectArchiveField::Name;
        let value = if editing_name {
            app.project_archive_edit_buffer().to_string()
        } else {
            app.project_archive_field_value(*field)
        };
        let value_style = if editing_name {
            Style::default().fg(Color::LightGreen)
        } else {
            Style::default().fg(Color::Cyan)
        };
        let content = Line::from(vec![
            Span::styled(label, Style::default().fg(Color::Yellow)),
            Span::raw(": "),
            Span::styled(value, value_style),
        ]);
        items.push(ListItem::new(content));
    }
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(app.project_archive_selection().min(items.len() - 1)));
    }
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Archive options (Enter to toggle/edit)"),
        )
        .highlight_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(
            if matches!(
                app.project_archive_focus(),
                crate::app::ProjectArchiveFocus::Options
            ) {
                "> "
            } else {
                "  "
            },
        );
    frame.render_stateful_widget(list, body_cols[0], &mut state);

    // Session picker
    let mut session_items = Vec::new();
    let sessions = app.project_archive_sessions();
    let mut selected_sessions_count = 0usize;
    for session in sessions.iter() {
        let selected = app
            .project_archive_selected_sessions()
            .iter()
            .any(|id| id == &session.id);
        if selected {
            selected_sessions_count += 1;
        }
        let marker = if selected { "[x]" } else { "[ ]" };
        let created = format_unix_relative(session.created_at);
        let runs = session.runs.len();
        let label = format!("{marker} {} • {created} • runs: {runs}", session.name);
        session_items.push(ListItem::new(Line::from(label)));
    }
    let mut session_state = ListState::default();
    if !session_items.is_empty() {
        session_state.select(Some(
            app.project_archive_session_selection()
                .min(session_items.len().saturating_sub(1)),
        ));
    }
    let session_block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            "Sessions ({}/{}) (Tab to focus, Enter to toggle)",
            selected_sessions_count,
            sessions.len()
        ));
    let session_list = List::new(session_items)
        .block(session_block)
        .highlight_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(
            if matches!(
                app.project_archive_focus(),
                crate::app::ProjectArchiveFocus::Sessions
            ) {
                "> "
            } else {
                "  "
            },
        );
    frame.render_stateful_widget(session_list, body_cols[1], &mut session_state);

    render_project_archive_output(frame, chunks[2], app);

    // Instructions / status
    let mut instructions = Vec::new();
    instructions.push(Line::from(
        "Up/Down: select • Enter/Space: toggle/edit • Tab: switch panel • x: export • i: import • I: preview",
    ));
    instructions.push(Line::from(
        "m/d/l: toggle include models/runs/logs • s: toggle scope • r: archive read-only • p: pick output path",
    ));
    instructions.push(Line::from("PgUp/PgDn: scroll archive log • c: cancel running archive task"));
    if app.input_mode() == InputMode::EditingProjectArchive {
        instructions.push(Line::from(Span::styled(
            "Editing archive name • Enter to save • Esc to cancel",
            Style::default().fg(Color::LightGreen),
        )));
    }
    let status_line = app
        .status()
        .map(|s| s.text.clone())
        .unwrap_or_else(|| "Ready".to_string());
    instructions.push(Line::from(Span::styled(
        format!("Status: {}", status_line),
        Style::default().fg(Color::DarkGray),
    )));

    let footer = Paragraph::new(instructions)
        .block(Block::default().borders(Borders::ALL).title("Controls"))
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    frame.render_widget(footer, chunks[3]);
}

fn render_project_archive_output(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let title = if app.is_project_archive_running() {
        "Archive Log (running)"
    } else {
        "Archive Log"
    };
    let block = Block::default().borders(Borders::ALL).title(title);

    if app.project_archive_output().is_empty() {
        let placeholder = Paragraph::new(if app.is_project_archive_running() {
            "Working…"
        } else {
            "No archive activity yet."
        })
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray))
        .wrap(Wrap { trim: true });
        frame.render_widget(placeholder, area);
        return;
    }

    let raw_lines = app.project_archive_output();
    let inner_width = area.width.saturating_sub(2);
    let (wrapped_lines, per_raw_heights) = wrap_output_lines(raw_lines, inner_width, false);

    let mut paragraph = Paragraph::new(wrapped_lines)
        .block(block)
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::White));

    let visible_height = area.height.saturating_sub(2).max(1) as usize;
    let offset_from_bottom_raw = app
        .project_archive_output_scroll()
        .min(raw_lines.len().saturating_sub(1));
    let bottom_raw_idx = raw_lines.len() - 1 - offset_from_bottom_raw;

    let mut prefix_visual = Vec::with_capacity(per_raw_heights.len() + 1);
    prefix_visual.push(0usize);
    for height in per_raw_heights {
        let next = prefix_visual.last().copied().unwrap_or(0).saturating_add(height);
        prefix_visual.push(next);
    }
    let end_visual_exclusive = prefix_visual[bottom_raw_idx + 1];
    let start_visual = end_visual_exclusive.saturating_sub(visible_height);
    let scroll_y = start_visual.min(u16::MAX as usize) as u16;
    paragraph = paragraph.scroll((scroll_y, 0));

    frame.render_widget(paragraph, area);
}

fn render_export_config(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let export_mode = app.export_mode();
    let focus = app.export_focus();
    let input_mode = app.input_mode();
    let active_field = app.active_export_field();

    let summary_height = if matches!(input_mode, InputMode::EditingExport) {
        6
    } else {
        4
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(summary_height), Constraint::Min(0)])
        .split(area);
    let summary_area = layout[0];
    let list_area = layout[1];

    let summary_lines = vec![
        Line::from(vec![
            Span::styled("Mode: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                export_mode.label(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  (m: toggle mode)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Focus: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                match focus {
                    crate::app::ExportFocus::Fields => "Fields",
                    crate::app::ExportFocus::Output => "Output",
                },
                Style::default().fg(Color::White),
            ),
            Span::styled(
                "  (Tab/o: switch focus)",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    let mut summary = summary_lines;

    if matches!(input_mode, InputMode::EditingExport) {
        if let Some(field) = active_field {
            summary.push(Line::from(vec![
                Span::styled("Editing ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    field.label(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(": {}_", app.export_edit_buffer()),
                    Style::default().fg(Color::White),
                ),
            ]));
            summary.push(Line::from(Span::styled(
                "Enter: save • Esc: cancel • Backspace: delete",
                Style::default().fg(Color::DarkGray),
            )));
        }
    } else {
        let hint = if app.is_export_running() {
            "Export running… press 'c' to cancel"
        } else {
            "Enter: toggle/edit/browse • x: start export"
        };
        summary.push(Line::from(Span::styled(
            hint,
            Style::default().fg(Color::DarkGray),
        )));
    }

    let summary_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title("Export Overview");
    frame.render_widget(
        Paragraph::new(summary)
            .block(summary_block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true }),
        summary_area,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .title("Export Options")
        .border_style(if focus == crate::app::ExportFocus::Fields {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    if app.export_fields().is_empty() {
        let placeholder = Paragraph::new("No export options available for this mode")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, list_area);
        return;
    }

    let config = app.export_config();
    let items: Vec<ListItem> = app
        .export_fields()
        .iter()
        .map(|field| {
            let (value, style) = match field {
                crate::app::ExportField::Sb3UseObsArray => {
                    if config.sb3_use_obs_array {
                        ("On".to_string(), Style::default().fg(Color::LightGreen))
                    } else {
                        ("Off".to_string(), Style::default().fg(Color::DarkGray))
                    }
                }
                crate::app::ExportField::Sb3SkipVerify => {
                    if config.sb3_skip_verify {
                        ("Skip".to_string(), Style::default().fg(Color::Yellow))
                    } else {
                        ("Verify".to_string(), Style::default().fg(Color::LightBlue))
                    }
                }
                crate::app::ExportField::RllibMultiagent => {
                    if config.rllib_multiagent {
                        (
                            "Multi-agent".to_string(),
                            Style::default().fg(Color::LightGreen),
                        )
                    } else {
                        (
                            "Single-agent".to_string(),
                            Style::default().fg(Color::LightBlue),
                        )
                    }
                }
                _ => (
                    app.export_field_value(*field),
                    Style::default().fg(Color::White),
                ),
            };

            let spans = vec![
                Span::styled(field.label(), Style::default().fg(Color::DarkGray)),
                Span::raw(": "),
                Span::styled(value, style),
            ];

            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.export_selection()));

    let highlight_style = if focus == crate::app::ExportFocus::Fields {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC)
    };

    let list = List::new(items)
        .block(block)
        .highlight_style(highlight_style)
        .highlight_symbol("› ");

    frame.render_stateful_widget(list, list_area, &mut state);
}

fn render_export_output(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let focus = app.export_focus();
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Export Output")
        .border_style(if focus == crate::app::ExportFocus::Output {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });

    if app.export_output().is_empty() {
        let output_text =
            "No export activity yet. Configure options and press 'x' to export.".to_string();
        let paragraph = Paragraph::new(output_text)
            .block(block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::White));
        frame.render_widget(paragraph, area);
        return;
    }

    let raw_lines = app.export_output();
    let inner_width = area.width.saturating_sub(2);
    let (wrapped_lines, per_raw_heights) = wrap_output_lines(raw_lines, inner_width, false);

    let mut paragraph = Paragraph::new(wrapped_lines)
        .block(block)
        .alignment(Alignment::Left)
        .style(Style::default().fg(Color::White));

    let visible_height = area.height.saturating_sub(2).max(1) as usize;
    let offset_from_bottom_raw = app
        .export_output_scroll()
        .min(raw_lines.len().saturating_sub(1));
    let bottom_raw_idx = raw_lines.len() - 1 - offset_from_bottom_raw;

    let mut prefix_visual = Vec::with_capacity(per_raw_heights.len() + 1);
    prefix_visual.push(0usize);
    for height in per_raw_heights {
        let next = prefix_visual.last().copied().unwrap_or(0).saturating_add(height);
        prefix_visual.push(next);
    }
    let end_visual_exclusive = prefix_visual[bottom_raw_idx + 1];
    let start_visual = end_visual_exclusive.saturating_sub(visible_height);
    let scroll_y = start_visual.min(u16::MAX as usize) as u16;
    paragraph = paragraph.scroll((scroll_y, 0));

    frame.render_widget(paragraph, area);
}

fn render_export_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let instructions = if app.is_export_running() {
        "Export running… press 'c' to cancel • Tab/o: focus output • ↑/↓: scroll"
    } else {
        match app.input_mode() {
            InputMode::EditingExport => {
                "Typing export value • Enter: save • Esc: cancel • Backspace: delete"
            }
            _ => {
                "x: start export • c: cancel • m: toggle mode • Enter: edit/toggle • Tab/o: switch focus • ↑/↓: move or scroll • PgUp/PgDn: fast scroll logs"
            }
        }
    };

    let status_text = app.status().map(|s| s.text.as_str()).unwrap_or("Ready");
    let status_style = app
        .status()
        .map(|s| status_style(s.kind))
        .unwrap_or_else(|| Style::default().fg(Color::DarkGray));

    let lines = vec![
        Line::from(Span::styled(status_text.to_string(), status_style)),
        Line::from(Span::styled(
            instructions,
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::default().borders(Borders::ALL).title("Status");
    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}
pub fn render_settings(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(5),
            Constraint::Length(3),
        ])
        .split(area);

    let items: Vec<ListItem> = app
        .settings_fields()
        .iter()
        .map(|field| {
            ListItem::new(vec![
                Line::from(vec![
                    Span::styled(
                        field.label(),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        app.settings_field_value(*field),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    field.description(),
                    Style::default().fg(Color::DarkGray),
                )),
            ])
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.settings_selection_index()));
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Controller Settings"),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");
    frame.render_stateful_widget(list, layout[0], &mut state);

    let selected_field = app.current_settings_field();
    let desc = Paragraph::new(vec![Line::from(Span::styled(
        selected_field.description(),
        Style::default().fg(Color::Gray),
    ))])
    .block(Block::default().borders(Borders::ALL).title("Details"))
    .wrap(Wrap { trim: true });
    frame.render_widget(desc, layout[1]);

    let instructions = Paragraph::new(vec![Line::from(Span::styled(
        "↑/↓ select • ←/→ adjust • Space/Enter toggle",
        Style::default().fg(Color::DarkGray),
    ))])
    .block(Block::default().borders(Borders::ALL).title("Controls"));
    frame.render_widget(instructions, layout[2]);
}

pub fn render_placeholder(frame: &mut Frame<'_>, area: Rect, tab: TabId, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(app[tab].title);

    let message = match tab {
        TabId::Home => unreachable!(),
        TabId::Train => "Training configuration pending",
        TabId::Metrics => "Metrics view pending",
        TabId::Simulator => "Simulator view pending",
        TabId::Interface => "Interface view pending",
        TabId::ExportModel => unreachable!(),
        TabId::Projects => "Projects view pending",
        TabId::Settings => "Settings pending",
    };

    let placeholder = Paragraph::new(Line::from(message))
        .block(block)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(placeholder, area);
}

fn status_style(kind: StatusKind) -> Style {
    match kind {
        StatusKind::Info => Style::default().fg(Color::LightBlue),
        StatusKind::Success => Style::default().fg(Color::LightGreen),
        StatusKind::Warning => Style::default().fg(Color::Yellow),
        StatusKind::Error => Style::default().fg(Color::LightRed),
    }
}

pub fn render_help_overlay(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();

    // Create a centered box
    let vertical_margin = area.height / 10;
    let horizontal_margin = area.width / 10;
    let help_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let mut help_text = Vec::new();
    help_text.push(Line::from(Span::styled(
        "KEYBOARD SHORTCUTS",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )));
    help_text.push(Line::from(""));

    for (title, entries) in build_help_sections(app) {
        help_text.push(Line::from(Span::styled(
            title,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )));
        for entry in entries {
            help_text.push(Line::from(entry));
        }
        help_text.push(Line::from(""));
    }

    help_text.push(Line::from(Span::styled(
        "Press ? / h / F1 / Esc to close",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    )));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Help ")
        .title_alignment(Alignment::Center);

    let paragraph = Paragraph::new(help_text)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, help_area);
}

fn build_help_sections(app: &App) -> Vec<(String, Vec<String>)> {
    let mut sections = Vec::new();
    let mut context_lines = Vec::new();
    context_lines.push(format!(
        "Tab: {}  •  Input: {}",
        app.active_tab().title,
        input_mode_label(app.input_mode())
    ));
    let extra = match app.active_tab().id {
        TabId::Train => format!(
            "Training: {}",
            if app.is_training_running() { "running" } else { "idle" }
        ),
        TabId::Metrics => format!("Metrics focus: {}", metrics_focus_label(app.metrics_focus())),
        TabId::Simulator => format!(
            "Simulator focus: {}",
            match app.simulator_focus() {
                SimulatorFocus::Events => "Events",
                SimulatorFocus::Actions => "Actions",
            }
        ),
        TabId::Interface => format!(
            "Interface focus: {}",
            match app.interface_focus() {
                InterfaceFocus::Events => "Events",
                InterfaceFocus::Actions => "Actions",
            }
        ),
        _ => String::new(),
    };
    if !extra.is_empty() {
        context_lines.push(extra);
    }
    sections.push(("Context".to_string(), context_lines));
    sections.push((
        "Global".to_string(),
        vec![
            "? / h / F1   Toggle this help overlay".to_string(),
            "q / Esc  Quit (when idle) or back out of dialogs".to_string(),
            "1-8      Jump to Home, Train, Metrics, Simulator, Interface, Export Model, Projects, Settings"
                .to_string(),
            "Home/End Jump to Home or Settings tab".to_string(),
            "y/n      Confirm or cancel prompts".to_string(),
        ],
    ));

    let tab_section = match app.active_tab().id {
        TabId::Home => ("Home tab".to_string(), home_help_lines()),
        TabId::Train => ("Train tab".to_string(), train_help_lines()),
        TabId::Metrics => ("Metrics tab".to_string(), metrics_help_lines(app)),
        TabId::Simulator => ("Simulator tab".to_string(), simulator_help_lines()),
        TabId::Interface => ("Interface tab".to_string(), interface_help_lines()),
        TabId::ExportModel => ("Export Model tab".to_string(), export_help_lines()),
        TabId::Projects => ("Projects tab".to_string(), projects_help_lines()),
        TabId::Settings => ("Settings tab".to_string(), settings_help_lines()),
    };
    sections.push(tab_section);

    sections.push((
        "File browser (when open)".to_string(),
        file_browser_help_lines(),
    ));

    sections
}

fn input_mode_label(mode: InputMode) -> &'static str {
    match mode {
        InputMode::Normal => "Normal",
        InputMode::CreatingProject => "Creating project",
        InputMode::EditingConfig => "Editing config",
        InputMode::EditingAdvancedConfig => "Editing advanced config",
        InputMode::SelectingConfigOption => "Selecting option",
        InputMode::AdvancedConfig => "Advanced config",
        InputMode::BrowsingFiles => "Browsing files",
        InputMode::Help => "Help",
        InputMode::ConfirmQuit => "Confirm quit",
        InputMode::ConfirmAction => "Confirm action",
        InputMode::EditingExport => "Editing export",
        InputMode::ChartExportOptions => "Chart export options",
        InputMode::EditingChartExportOption => "Editing chart export option",
        InputMode::MetricsSettings => "Metrics settings",
        InputMode::EditingMetricsSetting => "Editing metrics setting",
        InputMode::EditingProjectArchive => "Editing project archive",
        InputMode::ConfirmProjectImport => "Confirm project import",
    }
}

fn metrics_focus_label(focus: MetricsFocus) -> &'static str {
    match focus {
        MetricsFocus::History => "History",
        MetricsFocus::Summary => "Summary",
        MetricsFocus::Policies => "Policies",
        MetricsFocus::Chart => "Chart",
    }
}

fn home_help_lines() -> Vec<String> {
    vec![
        "Up/Down / j/k  Select a project".to_string(),
        "Enter          Activate the highlighted project".to_string(),
        "n              Create a new project".to_string(),
        "r              Refresh the project list".to_string(),
        "p              Check Python dependencies".to_string(),
    ]
}

fn train_help_lines() -> Vec<String> {
    vec![
        "t              Start training (after validation)".to_string(),
        "d              Launch demo training".to_string(),
        "c              Cancel the running training process (asks to confirm)".to_string(),
        "m              Toggle Single-Agent / Multi-Agent mode".to_string(),
        "g              Generate an RLlib config template".to_string(),
        "p / s / n      Edit env path, timesteps, or experiment name".to_string(),
        "a              Open advanced settings (Esc to close)".to_string(),
        "b              Browse for the environment path".to_string(),
        "l              Clear the training output log (asks to confirm)".to_string(),
        "Up/Down / j/k  Scroll training output".to_string(),
        "PgUp/PgDn      Fast-scroll training output".to_string(),
    ]
}

fn metrics_help_lines(app: &App) -> Vec<String> {
    let mut lines = vec![
        "Tab / Shift+Tab  Move focus between History, Summary, Policies, Chart".to_string(),
        "Enter            Expand/collapse the policies panel".to_string(),
        "Up/Down / j/k    Scroll the focused panel (History navigates entries)".to_string(),
        "PgUp/PgDn        Fast scroll or jump 10 items (also moves chart marker)".to_string(),
        "Home / End       Jump to newest / oldest metric".to_string(),
        "Left / Right     Scroll expanded policies, or move chart marker when chart is focused"
            .to_string(),
        ", / .            Cycle the active chart metric".to_string(),
        "x (chart focus)  Export chart as PNG (adjust options after choosing path)".to_string(),
        "l                Load a saved run as the primary view (no overlays)".to_string(),
        "c                Load a saved run overlay from disk".to_string(),
        "f                Pick a detected RLlib run (or manual)".to_string(),
        "C                Clear all overlays (asks to confirm)".to_string(),
        "o                Toggle viewing the selected saved run".to_string(),
        "O                Cycle through loaded runs".to_string(),
        "v                Return to live metrics when viewing a run".to_string(),
        "r                Apply the selected checkpoint directory to the training config"
            .to_string(),
    ];
    if app.has_saved_run_overlays() {
        if let Some(label) = app.selected_overlay_label() {
            lines.push(format!("Selected saved run: {label}"));
        }
    }
    lines
}

fn simulator_help_lines() -> Vec<String> {
    vec![
        "s / c          Start or cancel the simulator script".to_string(),
        "m / w / a      Toggle mode, window visibility, auto-restart".to_string(),
        "t              Toggle verbose Python tracebacks".to_string(),
        "p / y          Pick an environment binary or copy the training path".to_string(),
        "f / Tab        Switch focus between Events and Actions panes".to_string(),
        "v              Toggle compact agent table layout".to_string(),
        "[ / ]          Decrease / increase the step delay".to_string(),
        "- / =          Decrease / increase the restart delay".to_string(),
        "Up/Down        Scroll the focused pane (PgUp/PgDn for speed)".to_string(),
    ]
}

fn interface_help_lines() -> Vec<String> {
    vec![
        "s / c          Start or cancel the agent interface".to_string(),
        "m / a          Toggle mode or auto-restart".to_string(),
        "t / r / T      Toggle agent type, artifact, or tracebacks".to_string(),
        "b / y          Pick agent file or sync path from Export tab".to_string(),
        "f / Tab        Switch focus between Events and Actions panes".to_string(),
        "v              Toggle compact agent table layout".to_string(),
        "[ / ]          Decrease / increase the step delay".to_string(),
        "- / =          Decrease / increase the restart delay".to_string(),
        "Up/Down        Scroll the focused pane (PgUp/PgDn for speed)".to_string(),
    ]
}

fn export_help_lines() -> Vec<String> {
    vec![
        "x              Start the export process".to_string(),
        "c              Cancel the running export".to_string(),
        "m              Toggle export mode (SB3 / RLlib)".to_string(),
        "Tab / Shift+Tab / o / O  Switch focus between options and output log".to_string(),
        "Enter          Edit or toggle the selected option".to_string(),
        "Up/Down / j/k  Navigate options or scroll output".to_string(),
        "PgUp/PgDn      Fast-scroll the export log".to_string(),
        "Backspace      Clear the selected option (when options are focused)".to_string(),
    ]
}

fn projects_help_lines() -> Vec<String> {
    vec![
        "x              Export project/session archive".to_string(),
        "i              Import archive".to_string(),
        "I              Preview archive".to_string(),
        "m/d/l          Toggle include models/runs/logs".to_string(),
        "s              Toggle export scope".to_string(),
        "Enter / Space  Toggle selected option or session".to_string(),
        "r              Toggle archive read-only flag".to_string(),
        "p              Pick export output directory".to_string(),
        "Tab            Switch focus between options/sessions".to_string(),
        "PgUp/PgDn      Scroll archive log".to_string(),
        "c              Cancel running archive task".to_string(),
    ]
}

fn settings_help_lines() -> Vec<String> {
    vec![
        "Up/Down / j/k  Move between settings".to_string(),
        "Left/Right     Adjust the highlighted setting".to_string(),
        "Enter / Space  Toggle the selected option".to_string(),
    ]
}

fn file_browser_help_lines() -> Vec<String> {
    vec![
        "Up/Down / j/k  Highlight entries".to_string(),
        "Enter          Select file or descend into directory".to_string(),
        "Backspace / h  Go up to the parent directory".to_string(),
        "f              Finalize the current selection".to_string(),
        "n              Create a new folder (when allowed)".to_string(),
        "Esc            Cancel file selection".to_string(),
    ]
}

pub fn render_project_import_prompt(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();

    let vertical_margin = area.height / 6;
    let horizontal_margin = area.width / 8;
    let dialog_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let Some(view) = app.project_import_prompt_view() else {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Project Import");
        let placeholder = Paragraph::new("No archive selected.")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(Clear, dialog_area);
        frame.render_widget(placeholder, dialog_area);
        return;
    };

    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        "PROJECT ARCHIVE",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(format!(
        "Archive: {}",
        view.archive_path.display()
    )));
    lines.push(Line::from(format!("Name: {}", view.name)));
    lines.push(Line::from(format!("Scope: {}", view.scope)));
    lines.push(Line::from(format!(
        "Read-only in manifest: {}",
        if view.read_only { "yes" } else { "no" }
    )));
    lines.push(Line::from(format!(
        "Sessions in archive: {}",
        view.selected_sessions
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(format!(
        "Includes: models={} runs={} logs={} configs={} scripts={}",
        if view.include_models { "yes" } else { "no" },
        if view.include_runs { "yes" } else { "no" },
        if view.include_logs { "yes" } else { "no" },
        if view.include_configs { "yes" } else { "no" },
        if view.include_scripts { "yes" } else { "no" },
    )));
    lines.push(Line::from(""));
    let default_action = if view.default_action_is_preview {
        "Preview"
    } else {
        "Import"
    };
    lines.push(Line::from(Span::styled(
        format!("Default action: {default_action}"),
        Style::default().fg(Color::Yellow),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Enter: run default • i: import • p: preview • Esc: cancel",
        Style::default().fg(Color::LightGreen),
    )));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Project Import ")
        .title_alignment(Alignment::Center);

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false });

    frame.render_widget(Clear, dialog_area);
    frame.render_widget(paragraph, dialog_area);
}

pub fn render_confirm_quit(frame: &mut Frame<'_>, _app: &App) {
    let area = frame.area();

    let dialog_width = 50;
    let dialog_height = 7;
    let dialog_area = Rect {
        x: (area.width.saturating_sub(dialog_width)) / 2,
        y: (area.height.saturating_sub(dialog_height)) / 2,
        width: dialog_width,
        height: dialog_height,
    };

    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Are you sure you want to quit?",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Press 'y' to confirm, 'n' or Esc to cancel",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" Confirm Quit ")
        .title_alignment(Alignment::Center);

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, dialog_area);
}

pub fn render_confirm_action(frame: &mut Frame<'_>, app: &App) {
    let Some((title, body)) = app.confirm_action_prompt() else {
        return;
    };
    let area = frame.area();

    let dialog_width = 56;
    let dialog_height = 8;
    let dialog_area = Rect {
        x: (area.width.saturating_sub(dialog_width)) / 2,
        y: (area.height.saturating_sub(dialog_height)) / 2,
        width: dialog_width.min(area.width),
        height: dialog_height.min(area.height),
    };

    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            body,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Press 'y' to confirm, 'n' or Esc to cancel",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(format!(" {title} "))
        .title_alignment(Alignment::Center);

    let paragraph = Paragraph::new(text)
        .block(block)
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });

    frame.render_widget(Clear, dialog_area);
    frame.render_widget(paragraph, dialog_area);
}

pub fn render_file_browser(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();

    let vertical_margin = area.height / 8;
    let horizontal_margin = area.width / 8;
    let browser_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let status = app.status();
    let mut constraints = vec![Constraint::Length(3), Constraint::Min(5)];
    if status.is_some() {
        constraints.push(Constraint::Length(3));
    }
    constraints.push(Constraint::Length(3));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(browser_area);

    let path_chunk = chunks[0];
    let list_chunk = chunks[1];
    let mut next_chunk = 2;
    let status_chunk = if status.is_some() {
        let area = chunks[next_chunk];
        next_chunk += 1;
        Some(area)
    } else {
        None
    };
    let instructions_chunk = chunks[next_chunk];

    // Path display
    let path_text = format!(" Current: {} ", app.file_browser_path().display());
    let path_block = Block::default()
        .borders(Borders::ALL)
        .title(" File Browser ")
        .title_alignment(Alignment::Center)
        .border_style(Style::default().fg(Color::Cyan));
    let path_para = Paragraph::new(path_text)
        .block(path_block)
        .style(Style::default().fg(Color::White));
    frame.render_widget(path_para, path_chunk);

    // File list
    let items: Vec<ListItem> = app
        .file_browser_entries()
        .iter()
        .map(|entry| {
            let (prefix, style) = match entry {
                FileBrowserEntry::Parent(_) => (
                    "↥ ",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                ),
                FileBrowserEntry::Directory(_) => (
                    "📁 ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                FileBrowserEntry::File(_) => ("� ", Style::default().fg(Color::White)),
            };
            let mut spans = Vec::new();
            spans.push(Span::styled(prefix, style));
            spans.push(Span::styled(entry.display_name(), style));
            ListItem::new(Line::from(spans))
        })
        .collect();

    let list_block = Block::default().borders(Borders::ALL);
    let list = List::new(items)
        .block(list_block)
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");

    let mut state = ListState::default();
    if app.file_browser_state() == FileBrowserState::Browsing
        && !app.file_browser_entries().is_empty()
    {
        state.select(Some(app.file_browser_selected()));
    }
    frame.render_stateful_widget(list, list_chunk, &mut state);

    if let (Some(status_area), Some(status)) = (status_chunk, status) {
        let (label, color) = match status.kind {
            StatusKind::Info => ("INFO", Color::Cyan),
            StatusKind::Success => ("OK", Color::Green),
            StatusKind::Warning => ("WARN", Color::Yellow),
            StatusKind::Error => ("ERR", Color::Red),
        };

        let status_text = vec![Line::from(vec![
            Span::styled(
                format!("{label}: "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(status.text.as_str(), Style::default().fg(color)),
        ])];

        let status_block = Block::default()
            .borders(Borders::ALL)
            .title("Status")
            .title_alignment(Alignment::Center)
            .border_style(Style::default().fg(color));
        frame.render_widget(
            Paragraph::new(status_text)
                .block(status_block)
                .alignment(Alignment::Left),
            status_area,
        );
    }

    // Instructions
    let mut instructions: Vec<Line> = Vec::new();
    match app.file_browser_state() {
        FileBrowserState::Browsing => match app.file_browser_kind() {
            FileBrowserKind::Directory {
                allow_create,
                require_checkpoints,
            } => {
                let mut text = String::from("↑/↓ navigate • Enter: open • f: choose here");
                if *allow_create {
                    text.push_str(" • n: new folder");
                }
                text.push_str(" • Backspace: up • Esc: cancel");
                instructions.push(Line::from(Span::styled(
                    text,
                    Style::default().fg(Color::DarkGray),
                )));

                if *require_checkpoints {
                    instructions.push(Line::from(Span::styled(
                        "Folder must contain RLlib checkpoint_*/policies files",
                        Style::default().fg(Color::DarkGray),
                    )));
                }
            }
            FileBrowserKind::ExistingFile { .. } => {
                instructions.push(Line::from(Span::styled(
                    "↑/↓ navigate • Enter: open • f: select file • Backspace: up • Esc: cancel",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            FileBrowserKind::OutputFile { .. } => {
                instructions.push(Line::from(Span::styled(
                        "↑/↓ navigate • Enter: open • f: choose name • n: new folder • Backspace: up • Esc: cancel",
                        Style::default().fg(Color::DarkGray),
                    )));
            }
        },
        FileBrowserState::NamingFolder => {
            instructions.push(Line::from(vec![
                Span::styled("New folder name: ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("{}_", app.file_browser_input()),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            instructions.push(Line::from(Span::styled(
                "Enter: create • Esc: cancel",
                Style::default().fg(Color::DarkGray),
            )));
        }
        FileBrowserState::NamingFile => {
            instructions.push(Line::from(vec![
                Span::styled("File name: ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("{}_", app.file_browser_input()),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            instructions.push(Line::from(Span::styled(
                "Enter: confirm • Backspace: edit • Esc: cancel",
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    if let Some(status) = app.status() {
        instructions.push(Line::default());
        let (label, color) = match status.kind {
            StatusKind::Info => ("INFO", Color::Cyan),
            StatusKind::Success => ("OK", Color::Green),
            StatusKind::Warning => ("WARN", Color::Yellow),
            StatusKind::Error => ("ERR", Color::Red),
        };
        instructions.push(Line::from(vec![
            Span::styled(
                format!("{label}: "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(status.text.as_str(), Style::default().fg(color)),
        ]));
    }

    let inst_block = Block::default().borders(Borders::ALL);
    let inst_para = Paragraph::new(instructions)
        .block(inst_block)
        .alignment(Alignment::Left);
    frame.render_widget(inst_para, instructions_chunk);
}

pub fn render_chart_export_options(frame: &mut Frame<'_>, app: &App) {
    let Some(opts) = app.chart_export_options() else {
        return;
    };

    let area = frame.area();
    let vertical_margin = area.height / 8;
    let horizontal_margin = area.width / 8;
    let overlay_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(6),
            Constraint::Length(4),
        ])
        .split(overlay_area);

    let header_lines = vec![
        Line::from(Span::styled(
            "Chart Export",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("Destination: {}", opts.path.display()),
            Style::default().fg(Color::White),
        )),
    ];
    let header_block = Block::default()
        .borders(Borders::ALL)
        .title(" Export Options ")
        .title_alignment(Alignment::Center)
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(
        Paragraph::new(header_lines)
            .block(header_block)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true }),
        layout[0],
    );

    let fields = app.chart_export_fields();
    let items: Vec<ListItem> = fields
        .iter()
        .map(|field| {
            let label = match field {
                ChartExportOptionField::FileName => "File name",
                ChartExportOptionField::Theme => "Theme",
                ChartExportOptionField::ShowLegend => "Legend",
                ChartExportOptionField::LegendPosition => "Legend position",
                ChartExportOptionField::ShowResumeMarker => "Resume marker",
                ChartExportOptionField::ShowSelectionMarker => "Selection marker",
                ChartExportOptionField::ShowStatsBox => "Stats box",
                ChartExportOptionField::ShowCaption => "Caption",
                ChartExportOptionField::ShowGrid => "Grid",
                ChartExportOptionField::XAxisTitle => "X axis title",
                ChartExportOptionField::YAxisTitle => "Y axis title",
                ChartExportOptionField::Smoothing => "Smoothing",
            };
            let value = app
                .chart_export_option_value(*field)
                .unwrap_or_else(|| "-".to_string());
            let value_style = Style::default().fg(Color::Yellow);
            let spans = vec![
                Span::styled(
                    format!("{label}: "),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(value, value_style),
            ];
            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(app.chart_export_selection()));
    let list_block = Block::default().borders(Borders::ALL);
    let list = List::new(items)
        .block(list_block)
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");

    frame.render_stateful_widget(list, layout[1], &mut list_state);

    let instructions = if matches!(app.input_mode(), InputMode::EditingChartExportOption) {
        vec![
            Line::from(vec![
                Span::styled("Editing: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("{}_", app.chart_export_edit_buffer()),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                "Enter: save • Esc: cancel • Backspace: delete",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    } else {
        vec![
            Line::from(Span::styled(
                "↑/↓: navigate • Enter/Space: toggle or edit • s: save PNG • Esc: cancel",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(
                "Theme toggles dark/light; edit axis titles or file name before saving.",
                Style::default().fg(Color::DarkGray),
            )),
        ]
    };

    let footer_block = Block::default()
        .borders(Borders::ALL)
        .title(" Instructions ")
        .title_alignment(Alignment::Center);
    frame.render_widget(
        Paragraph::new(instructions)
            .block(footer_block)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        layout[2],
    );
}

pub fn render_metrics_settings(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let vertical_margin = area.height / 6;
    let horizontal_margin = area.width / 6;
    let overlay_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(overlay_area);

    let panel_label = match app.metrics_settings_panel() {
        MetricsSettingsPanel::Chart => "Chart Settings",
        MetricsSettingsPanel::History => "History Settings",
        MetricsSettingsPanel::Summary => "Summary Settings",
        MetricsSettingsPanel::Policies => "Policies Settings",
        MetricsSettingsPanel::Info => "Info Panel Settings",
    };

    let header_block = Block::default()
        .borders(Borders::ALL)
        .title(panel_label)
        .title_alignment(Alignment::Center)
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Adjust metrics panel behavior (live only)",
            Style::default().fg(Color::DarkGray),
        )))
        .block(header_block)
        .alignment(Alignment::Center),
        layout[0],
    );

    let fields = app.metrics_settings_fields();
    let items: Vec<ListItem> = fields
        .iter()
        .map(|field| {
            let label = metrics_setting_label(*field);
            let value = app.metrics_setting_value(*field);
            let spans = vec![
                Span::styled(
                    format!("{label}: "),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(value, Style::default().fg(Color::Yellow)),
            ];
            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    let selected = app
        .metrics_settings_selection()
        .min(fields.len().saturating_sub(1));
    state.select(Some(selected));

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title("Options")
                .title_alignment(Alignment::Center),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");

    frame.render_stateful_widget(list, layout[1], &mut state);

    let editing = matches!(app.input_mode(), InputMode::EditingMetricsSetting);
    let footer_text = if editing {
        format!(
            "Editing: {} — type, Enter to save, Esc to cancel",
            app.active_metrics_setting_field()
                .map(metrics_setting_label)
                .unwrap_or("value")
        )
    } else {
        "↑/↓ select • Enter/Space toggle/edit • s or Esc to close".to_string()
    };

    let footer_spans = if editing {
        vec![
            Span::styled(footer_text, Style::default().fg(Color::LightCyan)),
            Span::raw("  "),
            Span::styled(
                format!("Input: {}", app.metrics_settings_edit_buffer()),
                Style::default().fg(Color::Yellow),
            ),
        ]
    } else {
        vec![Span::styled(
            footer_text,
            Style::default().fg(Color::LightCyan),
        )]
    };

    frame.render_widget(
        Paragraph::new(Line::from(footer_spans))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Controls")
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true }),
        layout[2],
    );
}

pub fn render_advanced_config(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();

    let vertical_margin = area.height / 8;
    let horizontal_margin = area.width / 10;
    let overlay_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(5),
            Constraint::Length(5),
        ])
        .split(overlay_area);

    let mode_line = match app.training_config().mode {
        crate::app::TrainingMode::SingleAgent => "Stable Baselines 3 (single-agent)",
        crate::app::TrainingMode::MultiAgent => "RLlib (multi-agent)",
    };

    let header_lines = vec![
        Line::from(Span::styled(
            "Advanced Training Settings",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            mode_line,
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::ITALIC),
        )),
        Line::from(Span::styled(
            "Saved per project when you press Enter",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let header_block = Block::default()
        .borders(Borders::ALL)
        .title(" Advanced Settings ")
        .title_alignment(Alignment::Center)
        .border_style(Style::default().fg(Color::Cyan));

    let header = Paragraph::new(header_lines)
        .block(header_block)
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
    frame.render_widget(header, layout[0]);

    let fields = app.advanced_fields();
    if fields.is_empty() {
        let empty_block = Block::default().borders(Borders::ALL);
        let empty_text = Paragraph::new("No advanced settings available for this mode.")
            .block(empty_block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(empty_text, layout[1]);
    } else {
        let items: Vec<ListItem> = fields
            .iter()
            .map(|field| {
                let mut value = app.config_field_value(*field);
                if value.trim().is_empty() {
                    value = "<empty>".to_string();
                }
                let error = app.advanced_field_error(*field);
                let value_style = if error.is_some() {
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                let spans = vec![
                    Span::styled(
                        field.label(),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(": ", Style::default().fg(Color::DarkGray)),
                    Span::styled(value, value_style),
                    error
                        .map(|msg| {
                            Span::styled(
                                format!("  ⚠ {msg}"),
                                Style::default()
                                    .fg(Color::Red)
                                    .add_modifier(Modifier::ITALIC),
                            )
                        })
                        .unwrap_or_else(|| Span::raw("")),
                ];
                ListItem::new(Line::from(spans))
            })
            .collect();

        let list_block = Block::default().borders(Borders::ALL);
        let list = List::new(items)
            .block(list_block)
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("› ");

        let mut state = ListState::default();
        state.select(Some(app.advanced_selection()));
        frame.render_stateful_widget(list, layout[1], &mut state);
    }

    let description_block = Block::default()
        .borders(Borders::ALL)
        .title(" Field Description ");
    let description_lines = match app.selected_advanced_field() {
        Some(ConfigField::RllibAlgorithm) => algorithm_description_lines(app),
        Some(field) => vec![
            Line::from(Span::styled(
                field.label(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                field.description(),
                Style::default().fg(Color::White),
            )),
            app.advanced_field_error(field).map_or_else(
                || Line::from(Span::raw("")),
                |msg| {
                    Line::from(Span::styled(
                        format!("⚠ {msg}"),
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ))
                },
            ),
        ],
        None => vec![Line::from(Span::styled(
            "Select a field to view its details.",
            Style::default().fg(Color::DarkGray),
        ))],
    };
    let description_para = Paragraph::new(description_lines)
        .block(description_block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    frame.render_widget(description_para, layout[2]);

    let (instructions_lines, border_title) =
        if matches!(app.input_mode(), InputMode::EditingAdvancedConfig) {
            let field_label = app
                .active_config_field()
                .map(|f| f.label())
                .unwrap_or("Field");
            let mut first_line_spans = Vec::new();
            first_line_spans.push(Span::styled("Editing ", Style::default().fg(Color::Yellow)));
            first_line_spans.push(Span::styled(
                field_label,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            first_line_spans.push(Span::styled(
                format!(": {}_", app.config_edit_buffer()),
                Style::default().fg(Color::White),
            ));

            let validation_line = app
                .config_edit_validation()
                .map(|(ok, err, normalized)| {
                    if ok {
                        Line::from(Span::styled(
                            format!(
                                "✓ {}",
                                normalized.unwrap_or_else(|| "OK".to_string())
                            ),
                            Style::default().fg(Color::LightGreen),
                        ))
                    } else {
                        Line::from(Span::styled(
                            format!("⚠ {err}"),
                            Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD),
                        ))
                    }
                })
                .unwrap_or_else(|| Line::from(Span::raw("")));

            (
                vec![
                    Line::from(first_line_spans),
                    validation_line,
                    Line::from(Span::styled(
                        "Enter: save • Esc: cancel • Backspace: delete",
                        Style::default().fg(Color::DarkGray),
                    )),
                ],
                " Editing ",
            )
        } else {
            (
                vec![Line::from(Span::styled(
                    "↑/↓: Navigate • Enter: Edit • a/Esc/q: Close",
                    Style::default().fg(Color::DarkGray),
                ))],
                " Instructions ",
            )
        };

    let footer_block = Block::default()
        .borders(Borders::ALL)
        .title(border_title)
        .title_alignment(Alignment::Center);
    let footer = Paragraph::new(instructions_lines)
        .block(footer_block)
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
    frame.render_widget(footer, layout[3]);
}

pub fn render_choice_menu(frame: &mut Frame<'_>, app: &App) {
    let Some(menu) = app.choice_menu() else {
        return;
    };
    let field_label = menu.label;

    let area = frame.area();
    let vertical_margin = area.height / 8;
    let horizontal_margin = area.width / 10;
    let overlay_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    frame.render_widget(Clear, overlay_area);

    let outer_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(overlay_area);

    let header_block = Block::default()
        .borders(Borders::ALL)
        .title(" Choose Configuration Option ")
        .title_alignment(Alignment::Center)
        .border_style(Style::default().fg(Color::Cyan));
    let header_lines = vec![
        Line::from(Span::styled(
            format!("Select {}", field_label),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Use ↑/↓ to browse options. Enter to confirm, Esc to cancel.",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let header = Paragraph::new(header_lines)
        .block(header_block)
        .alignment(Alignment::Center);
    frame.render_widget(header, outer_layout[0]);

    let body_split = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(outer_layout[1]);

    let list_items: Vec<ListItem> = menu
        .options
        .iter()
        .map(|choice| {
            let spans = vec![
                Span::styled(
                    choice.label.as_str(),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  ({})", choice.value),
                    Style::default().fg(Color::DarkGray),
                ),
            ];
            ListItem::new(Line::from(spans))
        })
        .collect();
    let list = List::new(list_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Options ")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");
    let mut list_state = ListState::default();
    list_state.select(Some(menu.selected));
    frame.render_stateful_widget(list, body_split[0], &mut list_state);

    let description_block = Block::default()
        .borders(Borders::ALL)
        .title(" Description ");
    let description_text = menu
        .options
        .get(menu.selected)
        .map(|choice| choice.description.as_str())
        .unwrap_or("Select an option to view details.");
    let description_lines = vec![
        Line::from(Span::styled(
            menu.options
                .get(menu.selected)
                .map(|choice| choice.label.as_str())
                .unwrap_or("No selection"),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::raw("")),
        Line::from(Span::styled(
            description_text,
            Style::default().fg(Color::White),
        )),
    ];
    let description = Paragraph::new(description_lines)
        .block(description_block)
        .wrap(Wrap { trim: true })
        .alignment(Alignment::Left);
    frame.render_widget(description, body_split[1]);

    let footer_block = Block::default()
        .borders(Borders::ALL)
        .title(" Instructions ")
        .title_alignment(Alignment::Center);
    let footer = Paragraph::new(vec![Line::from(Span::styled(
        "↑/↓: Navigate • Enter: Select • Esc: Cancel",
        Style::default().fg(Color::DarkGray),
    ))])
    .block(footer_block)
    .alignment(Alignment::Center);
    frame.render_widget(footer, outer_layout[2]);
}

fn algorithm_description_lines(app: &App) -> Vec<Line<'static>> {
    let config = app.training_config();
    let current = config.rllib_algorithm;
    let available = RLLIB_ALGORITHM_LIST
        .iter()
        .map(|algo| algo.trainer_name())
        .collect::<Vec<_>>()
        .join(", ");

    vec![
        Line::from(Span::styled(
            "RLlib Algorithm",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled(
                "Algorithms available: ",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
            Span::styled(available, Style::default().fg(Color::White)),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled(
                format!("{}: ", current.trainer_name()),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(current.summary(), Style::default().fg(Color::White)),
        ]),
    ]
}
