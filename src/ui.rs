use std::collections::BTreeMap;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, ListState, Paragraph, Tabs,
    Wrap,
};
use ratatui::Frame;

use crate::app::{
    App, ConfigField, FileBrowserEntry, FileBrowserKind, FileBrowserState, InputMode, MetricSample,
    MetricsFocus, StatusKind, TabId,
};

/// Smart alphanumeric sorting for policy names like "policy_1", "policy_10", "policy_2"
/// Splits strings into alphabetic and numeric parts, then sorts accordingly
pub fn alphanumeric_sort_key(s: &str) -> Vec<(String, u64)> {
    let mut parts = Vec::new();
    let mut current_alpha = String::new();
    let mut current_num = String::new();
    
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            if !current_alpha.is_empty() {
                parts.push((current_alpha.clone(), 0));
                current_alpha.clear();
            }
            current_num.push(ch);
        } else {
            if !current_num.is_empty() {
                let num: u64 = current_num.parse().unwrap_or(0);
                parts.push((String::new(), num));
                current_num.clear();
            }
            current_alpha.push(ch);
        }
    }
    
    // Push remaining parts
    if !current_alpha.is_empty() {
        parts.push((current_alpha, 0));
    }
    if !current_num.is_empty() {
        let num: u64 = current_num.parse().unwrap_or(0);
        parts.push((String::new(), num));
    }
    
    parts
}

pub fn render(frame: &mut Frame<'_>, app: &App) {
    // Render help overlay if visible
    if app.is_help_visible() {
        render_help_overlay(frame, app);
        return;
    }

    // Render confirm quit dialog
    if app.input_mode() == InputMode::ConfirmQuit {
        render_confirm_quit(frame, app);
        return;
    }

    if matches!(
        app.input_mode(),
        InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig
    ) {
        render_advanced_config(frame, app);
        return;
    }

    // Render file browser if active
    if app.input_mode() == InputMode::BrowsingFiles {
        render_file_browser(frame, app);
        return;
    }

    let [header_area, content_area] = split_layout(frame.area());
    render_tabs(frame, header_area, app);
    render_content(frame, content_area, app);
}

fn split_layout(area: Rect) -> [Rect; 2] {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
        .split(area);
    [layout[0], layout[1]]
}

fn render_tabs(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let titles = app
        .tabs()
        .iter()
        .map(|tab| {
            Line::from(Span::styled(
                format!(" {} ", tab.title),
                Style::default().fg(Color::Cyan),
            ))
        })
        .collect::<Vec<_>>();

    let tabs = Tabs::new(titles)
        .highlight_style(
            Style::default()
                .fg(Color::White)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .select(app.active_index());

    frame.render_widget(tabs, area);
}

fn render_content(frame: &mut Frame<'_>, area: Rect, app: &App) {
    match app.active_tab().id {
        TabId::Home => render_home(frame, area, app),
        TabId::Train => render_train(frame, area, app),
        TabId::Metrics => render_metrics(frame, area, app),
        TabId::Export => render_export(frame, area, app),
        tab => render_placeholder(frame, area, tab, app),
    }
}

fn render_home(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let constraints = if area.height >= 16 {
        [
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Min(5),
            Constraint::Length(3),
        ]
    } else if area.height >= 12 {
        [
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(2),
        ]
    } else {
        [
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(2),
            Constraint::Length(2),
        ]
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_active_project(frame, chunks[0], app);
    render_python_environment(frame, chunks[1], app);
    render_project_list(frame, chunks[2], app);
    render_home_status(frame, chunks[3], app);
}

fn render_active_project(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let (title, detail) = if let Some(project) = app.active_project() {
        (
            "Active Project",
            format!("{}\n{}", project.name, project.path.display()),
        )
    } else {
        ("Active Project", "No project selected".to_string())
    };

    let block = Block::default().borders(Borders::ALL).title(title);
    let paragraph = Paragraph::new(detail)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::White));

    frame.render_widget(paragraph, area);
}

fn render_python_environment(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title("Python Environment");

    let mut lines = Vec::new();
    
    // Check Stable Baselines 3
    let sb3_status = match app.python_sb3_available() {
        Some(true) => ("✓", Color::LightGreen),
        Some(false) => ("✗", Color::Red),
        None => ("?", Color::Yellow),
    };
    lines.push(Line::from(vec![
        Span::styled(sb3_status.0, Style::default().fg(sb3_status.1).add_modifier(Modifier::BOLD)),
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
        Span::styled(ray_status.0, Style::default().fg(ray_status.1).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled("Ray/RLlib", Style::default().fg(Color::White)),
    ]));

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

    let active_path = app.active_project().map(|p| &p.path);
    let items: Vec<ListItem> = app
        .projects()
        .iter()
        .map(|project| {
            let mut spans = Vec::new();
            if active_path.map_or(false, |path| path == &project.path) {
                spans.push(Span::styled("● ", Style::default().fg(Color::LightGreen)));
            } else {
                spans.push(Span::raw("  "));
            }
            spans.push(Span::styled(
                project.name.clone(),
                Style::default().fg(Color::White),
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
            let mut spans = Vec::new();
            spans.push(Span::styled(
                "New project name: ",
                Style::default().fg(Color::LightBlue),
            ));
            spans.push(Span::styled(
                format!("{}_", app.project_name_buffer()),
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
            lines.push(Line::from(Span::styled(
                "Enter to confirm • Esc to cancel",
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
                lines.push(Line::from(Span::styled(status.text.clone(), style)));
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
        | InputMode::AdvancedConfig => {
            // These modes have their own full-screen renders
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn render_train(frame: &mut Frame<'_>, area: Rect, app: &App) {
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
    let config = app.training_config();
    let input_mode = app.input_mode();
    let active_field = app.active_config_field();

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
                if config.env_path.is_empty() {
                    "<not set>"
                } else {
                    &config.env_path
                },
                {
                    let base_color = if config.env_path.is_empty() {
                        Color::Red
                    } else {
                        Color::White
                    };
                    let mut style = Style::default().fg(base_color);
                    if input_mode == InputMode::EditingConfig
                        && active_field == Some(ConfigField::EnvPath)
                    {
                        style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                    }
                    style
                },
            ),
            Span::styled(" (p:edit, b:browse)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Steps: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", config.timesteps), {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::Timesteps)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
            Span::styled(" (s:edit)", Style::default().fg(Color::DarkGray)),
            Span::styled("  •  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Name: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&config.experiment_name, {
                let mut style = Style::default().fg(Color::White);
                if input_mode == InputMode::EditingConfig
                    && active_field == Some(ConfigField::ExperimentName)
                {
                    style = style.add_modifier(Modifier::BOLD | Modifier::UNDERLINED);
                }
                style
            }),
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

fn render_training_output(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let output_block = Block::default()
        .borders(Borders::ALL)
        .title("Training Output");
    let output_text = if app.training_output().is_empty() {
        "No output yet. Configure training and press 't' to start.".to_string()
    } else {
        app.training_output().join("\n")
    };

    let mut paragraph = Paragraph::new(output_text)
        .block(output_block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::White));

    if !app.training_output().is_empty() {
        let total_lines = app.training_output().len();
        let visible_height = area.height.saturating_sub(2).max(1) as usize;
        let max_offset = total_lines.saturating_sub(visible_height);
        let offset_from_bottom = app.training_output_scroll().min(max_offset);
        let first_line = total_lines.saturating_sub(visible_height + offset_from_bottom) as u16;
        paragraph = paragraph.scroll((first_line, 0));
    }

    frame.render_widget(paragraph, area);
}

fn render_metrics(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if app.training_metrics_history().is_empty() {
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
        .constraints([
            Constraint::Percentage(55),
            Constraint::Percentage(45),
        ])
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
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(rows[1]);

    render_metrics_summary(frame, bottom_columns[0], app);

    // Right side: History on top, Chart or Policies below (opposite of top)
    let right_stack = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
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
        .constraints([
            Constraint::Percentage(60),
            Constraint::Percentage(40),
        ])
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
        lines.push(Line::from(vec![
            Span::styled("Selected: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                position_text,
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Iter: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format_option_u64(sample.training_iteration()),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw("  "),
            Span::styled("Checkpoint: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                sample.checkpoints().unwrap_or_default().to_string(),
                Style::default().fg(Color::Cyan),
            ),
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
                Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD),
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

        // Only show detailed steps if we have enough vertical space
        if available_height >= 12 {
            // Environment steps - split into two lines for clarity
            lines.push(Line::from(vec![
                Span::styled("Env steps: ", Style::default().fg(Color::DarkGray)),
            ]));
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
            lines.push(Line::from(vec![
                Span::styled("Agent steps: ", Style::default().fg(Color::DarkGray)),
            ]));
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
        if available_height >= 10 && !sample.custom_metrics().is_empty() {
            if let Some(custom_line) = summarize_custom_metrics(sample.custom_metrics(), 4) {
                lines.push(custom_line);
            }
        }

        // Apply scroll offset with bounds checking
        let total_lines = lines.len();
        let visible_lines = available_height;
        let scroll_offset = app.metrics_summary_scroll().min(total_lines.saturating_sub(visible_lines));
        let lines_to_show: Vec<_> = lines.into_iter().skip(scroll_offset).collect();

        // Highlight border if this panel is focused
        let is_focused = app.metrics_focus() == MetricsFocus::Summary;
        let border_color = if is_focused { Color::Cyan } else { Color::DarkGray };
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
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Key Metrics"),
            )
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
    }
}fn render_metrics_history(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let history = app.training_metrics_history();
    let available_rows = area.height.saturating_sub(2) as usize;
    if available_rows == 0 {
        return;
    }

    let total = history.len();
    let selected = app
        .metrics_history_selected_index()
        .min(total.saturating_sub(1));
    let max_start = total.saturating_sub(available_rows);
    let start = selected.saturating_sub(available_rows / 2).min(max_start);
    let end = (start + available_rows).min(total);

    let mut items = Vec::new();
    for sample in history.iter().rev().skip(start).take(end - start) {
        items.push(ListItem::new(metric_history_line(sample)));
    }

    let mut state = ListState::default();
    state.select(Some(selected - start));

    // Highlight border if this panel is focused
    let is_focused = app.metrics_focus() == MetricsFocus::History;
    let border_color = if is_focused { Color::Cyan } else { Color::DarkGray };
    let title = if is_focused {
        "History (newest first) [FOCUSED - Tab to switch]"
    } else {
        "History (newest first)"
    };

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
    let border_color = if is_focused { Color::Cyan } else { Color::DarkGray };
    let title = if is_focused {
        "Metric Chart [FOCUSED - Enter to view policies, Tab to switch]"
    } else {
        "Metric Chart [Enter to view policies]"
    };
    
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(border_color));

    // Check if this is a multi-series chart
    if let Some(metric_option) = app.current_chart_metric() {
        use crate::app::ChartMetricKind;
        match metric_option.kind() {
            ChartMetricKind::AllPoliciesRewardMean
            | ChartMetricKind::AllPoliciesEpisodeLenMean
            | ChartMetricKind::AllPoliciesLearnerStat(_) => {
                // Render multi-series overlay chart
                render_multi_series_chart(frame, area, app, &metric_option, block);
                return;
            }
            _ => {
                // Continue with single-series chart rendering below
            }
        }
    }

    // Single-series chart rendering (existing code)
    let chart_data = app.chart_data((area.width as usize).saturating_mul(4));

    if let Some(chart_data) = chart_data {
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for &(_, value) in &chart_data.points {
            if value.is_finite() {
                y_min = y_min.min(value);
                y_max = y_max.max(value);
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

        let x_min = chart_data.points.first().map(|(x, _)| *x).unwrap_or(0.0);
        let mut x_max = chart_data
            .points
            .last()
            .map(|(x, _)| *x)
            .unwrap_or(x_min + 1.0);
        if (x_max - x_min).abs() < 1e-3 {
            x_max = x_min + 1.0;
        }

        let dataset = Dataset::default()
            .name(chart_data.label.clone())
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&chart_data.points);

        let x_axis = Axis::default()
            .title(Span::styled(
                "Iteration",
                Style::default().fg(Color::DarkGray),
            ))
            .bounds([x_min, x_max]);

        let y_axis = Axis::default()
            .title(Span::styled(
                chart_data.label.clone(),
                Style::default().fg(Color::DarkGray),
            ))
            .bounds([y_min, y_max]);

        let chart = Chart::new(vec![dataset])
            .block(block)
            .x_axis(x_axis)
            .y_axis(y_axis);

        frame.render_widget(chart, area);
    } else {
        let placeholder = Paragraph::new("No data for the selected metric yet.")
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(placeholder, area);
    }
}

fn render_multi_series_chart(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &App,
    metric_option: &crate::app::ChartMetricOption,
    block: Block<'_>,
) {
    let multi_data = app.chart_multi_series_data(metric_option);
    
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

    for (_, points) in &multi_data {
        for &(x, y) in points {
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

    // Define colors for different policies
    let colors = [
        Color::Cyan,
        Color::Yellow,
        Color::Magenta,
        Color::Green,
        Color::Red,
        Color::Blue,
        Color::LightCyan,
        Color::LightYellow,
        Color::LightMagenta,
        Color::LightGreen,
        Color::LightRed,
        Color::LightBlue,
    ];

    // Create datasets for each policy
    let datasets: Vec<Dataset> = multi_data
        .iter()
        .enumerate()
        .map(|(idx, (policy_id, points))| {
            let color = colors[idx % colors.len()];
            Dataset::default()
                .name(policy_id.clone())
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(color))
                .data(points)
        })
        .collect();

    let x_axis = Axis::default()
        .title(Span::styled(
            "Iteration",
            Style::default().fg(Color::DarkGray),
        ))
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
            y_label,
            Style::default().fg(Color::DarkGray),
        ))
        .bounds([y_min, y_max]);

    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(x_axis)
        .y_axis(y_axis);

    frame.render_widget(chart, area);
}

fn render_metrics_chart_info(frame: &mut Frame<'_>, area: Rect, app: &App) {
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

    let available_width = area.width.saturating_sub(4) as usize;
    let mut lines = Vec::new();

    // First line: metric info - smart wrap based on width
    if available_width >= 60 {
        // Wide enough for one line
        lines.push(Line::from(vec![
            Span::styled("Metric: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                metric_label,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Value: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                value,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Policy: ", Style::default().fg(Color::DarkGray)),
            Span::styled(policy_label, Style::default().fg(Color::Yellow)),
        ]));
    } else {
        // Narrow - split into two lines
        lines.push(Line::from(vec![
            Span::styled("Metric: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                metric_label,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("Value: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                value,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Policy: ", Style::default().fg(Color::DarkGray)),
            Span::styled(policy_label, Style::default().fg(Color::Yellow)),
        ]));
    }

    // Second line: controls - adaptive based on width
    if available_width >= 90 {
        lines.push(Line::from(vec![Span::styled(
            "↑/↓: navigate history  •  PgUp/PgDn: jump 10  •  Home/End: edges  •  , / . : cycle metric",
            Style::default().fg(Color::DarkGray),
        )]));
    } else if available_width >= 60 {
        lines.push(Line::from(vec![Span::styled(
            "↑/↓: history  •  PgUp/Dn: jump  •  Home/End: edges  •  ,/.: metric",
            Style::default().fg(Color::DarkGray),
        )]));
    } else {
        // Very narrow - show minimal controls
        lines.push(Line::from(vec![Span::styled(
            "↑/↓: history  •  ,/.: metric",
            Style::default().fg(Color::DarkGray),
        )]));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Controls")
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
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

    let highlighted_policy = app
        .current_chart_metric()
        .and_then(|metric| metric.policy_id().map(|p| p.to_string()));

    let available_width = area.width.saturating_sub(4) as usize; // Account for borders
    let available_height = area.height.saturating_sub(2) as usize; // Account for borders
    
    // Collect and sort policies using smart alphanumeric sorting
    let mut policies: Vec<_> = sample.policies().iter().collect();
    policies.sort_by(|a, b| {
        let key_a = alphanumeric_sort_key(a.0);
        let key_b = alphanumeric_sort_key(b.0);
        key_a.cmp(&key_b)
    });
    
    let mut items = Vec::new();

    for (policy_id, metrics) in policies {
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
                Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD),
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

        // Learner stats - show key metrics, wrap intelligently
        // Limit stats shown based on available height
        let max_learner_stats = if available_height < 8 {
            2 // Very limited space, show only 2 stats per policy
        } else if available_height < 12 {
            3 // Limited space, show 3 stats per policy
        } else {
            5 // Normal space, show up to 5 stats per policy
        };
        
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
                        Span::styled(
                            format!("{:.4}", value),
                            Style::default().fg(Color::White),
                        ),
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
    let scroll_offset = app.metrics_policies_scroll().min(total_items.saturating_sub(visible_items.max(1)));
    let items_to_show: Vec<_> = items.into_iter().skip(scroll_offset).collect();

    // Highlight border if this panel is focused
    let is_focused = app.metrics_focus() == MetricsFocus::Policies;
    let border_color = if is_focused { Color::Cyan } else { Color::DarkGray };
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
        let placeholder = Paragraph::new("Policy metrics will appear once available.\n\nPress Enter to return to chart view.")
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
        let placeholder = Paragraph::new("Policy metrics will appear once available.\n\nPress Enter to return to chart view.")
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

    // Collect and sort policies using smart alphanumeric sorting
    let mut policies: Vec<_> = sample.policies().iter().collect();
    policies.sort_by(|a, b| {
        let key_a = alphanumeric_sort_key(a.0);
        let key_b = alphanumeric_sort_key(b.0);
        key_a.cmp(&key_b)
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
    for (idx, ((policy_id, metrics), chunk)) in visible_policies.iter().zip(chunks.iter()).enumerate() {
        render_single_policy_detailed(frame, *chunk, policy_id, metrics, v_scroll, idx == 0);
    }
}

fn render_single_policy_detailed(
    frame: &mut Frame<'_>,
    area: Rect,
    policy_id: &str,
    metrics: &crate::app::PolicyMetrics,
    scroll_offset: usize,
    is_first: bool,
) {
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
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )]));
    lines.push(Line::from(vec![
        Span::raw("  Mean: "),
        Span::styled(
            format_option_f64(metrics.reward_mean()),
            Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD),
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
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
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
    if !metrics.learner_stats().is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "Learner Stats:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]));
        
        for (key, value) in metrics.learner_stats() {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{}: ", key),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:.6}", value),
                    Style::default().fg(Color::White),
                ),
            ]));
        }
        lines.push(Line::from(""));
    }

    // Custom metrics
    if !metrics.custom_metrics().is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "Custom Metrics:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )]));
        
        for (key, value) in metrics.custom_metrics() {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{}: ", key),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:.4}", value),
                    Style::default().fg(Color::White),
                ),
            ]));
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

fn metric_history_line(sample: &MetricSample) -> Line<'static> {
    // the time is in ISO 8601 format, e.g., "2023-10-05T14:48:00Z" extract only the hh:mm:ss part
    let timestamp = sample.timestamp().unwrap_or("—").to_string();
    let timestamp = timestamp.split("T").nth(1).unwrap_or("—").split("Z").nth(0).unwrap_or("—");
    let timestamp = timestamp.split(":").take(3).collect::<Vec<_>>().join(":");
    let timestamp = timestamp.split(".").nth(0).unwrap_or("—");
    let mut spans = Vec::new();
    spans.push(Span::styled(
        format!("{} ", timestamp),
        Style::default().fg(Color::DarkGray),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("iter {}", format_option_u64(sample.training_iteration())),
        Style::default().fg(Color::Cyan),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("steps {}", format_option_u64(sample.timesteps_total())),
        Style::default().fg(Color::LightGreen),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("reward {}", format_option_f64(sample.episode_reward_mean())),
        Style::default().fg(Color::LightMagenta),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("len {}", format_option_f64(sample.episode_len_mean())),
        Style::default().fg(Color::LightBlue),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("Δt {}", format_option_duration(sample.time_this_iter_s())),
        Style::default().fg(Color::DarkGray),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("env {}", format_option_u64(sample.env_steps_this_iter())),
        Style::default().fg(Color::White),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        format!("ckpts {}", format_option_u64(sample.checkpoints())),
        Style::default().fg(Color::Yellow),
    ));
    Line::from(spans)
}

fn summarize_custom_metrics(
    metrics: &BTreeMap<String, f64>,
    limit: usize,
) -> Option<Line<'static>> {
    if metrics.is_empty() {
        return None;
    }

    let mut spans = vec![Span::styled(
        "Custom: ",
        Style::default().fg(Color::DarkGray),
    )];
    for (idx, (key, value)) in metrics.iter().take(limit).enumerate() {
        if idx > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            format!("{key} "),
            Style::default().fg(Color::DarkGray),
        ));
        spans.push(Span::styled(
            format!("{value:.3}"),
            Style::default().fg(Color::White),
        ));
    }
    if metrics.len() > limit {
        spans.push(Span::raw("  "));
        spans.push(Span::styled("…", Style::default().fg(Color::DarkGray)));
    }

    Some(Line::from(spans))
}

fn format_option_rate(value: Option<f64>, suffix: &str) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{:.2}{suffix}", v))
        .unwrap_or_else(|| "—".to_string())
}

fn format_option_u64(value: Option<u64>) -> String {
    value
        .map(|v| format!("{}", v))
        .unwrap_or_else(|| "—".to_string())
}

fn format_option_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "—".to_string())
}

fn format_option_duration(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|seconds| format_duration(seconds.max(0.0)))
        .unwrap_or_else(|| "—".to_string())
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

fn render_training_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let instructions = if app.is_training_running() {
        "Training running… press 'c' to cancel"
    } else {
        match app.input_mode() {
            InputMode::EditingConfig | InputMode::EditingAdvancedConfig => {
                "Editing config: type to update • Enter: save • Esc: cancel"
            }
            InputMode::AdvancedConfig => {
                "Advanced config: ↑/↓ navigate • Enter: edit • a/Esc: close"
            }
            _ => {
                "t: start training • m: toggle mode • d: run demo • g: write RLlib config • p/s/n: edit basics • a: advanced panel • b: browse • ↑/↓: scroll • PgUp/PgDn: fast scroll"
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

    let status_block = Block::default().borders(Borders::ALL).title("Status");
    let status_paragraph = Paragraph::new(lines)
        .block(status_block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    frame.render_widget(status_paragraph, area);
}

fn render_export(frame: &mut Frame<'_>, area: Rect, app: &App) {
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

    let output_text = if app.export_output().is_empty() {
        "No export activity yet. Configure options and press 'x' to export.".to_string()
    } else {
        app.export_output().join("\n")
    };

    let mut paragraph = Paragraph::new(output_text)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::White));

    if !app.export_output().is_empty() {
        let total_lines = app.export_output().len();
        let visible_height = area.height.saturating_sub(2).max(1) as usize;
        let max_offset = total_lines.saturating_sub(visible_height);
        let offset_from_bottom = app.export_output_scroll().min(max_offset);
        let first_line = total_lines.saturating_sub(visible_height + offset_from_bottom) as u16;
        paragraph = paragraph.scroll((first_line, 0));
    }

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

fn render_placeholder(frame: &mut Frame<'_>, area: Rect, tab: TabId, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(app[tab].title);

    let message = match tab {
        TabId::Home => unreachable!(),
        TabId::Train => "Training configuration pending",
        TabId::Metrics => "Metrics view pending",
        TabId::Export => unreachable!(),
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

fn render_help_overlay(frame: &mut Frame<'_>, _app: &App) {
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

    let help_text = vec![
        Line::from(Span::styled(
            "KEYBOARD SHORTCUTS",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Global:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  h / F1     - Show this help"),
        Line::from("  q / Esc    - Quit (with confirmation)"),
        Line::from("  Left/Right - Navigate tabs"),
        Line::from("  1-5        - Jump to tab"),
        Line::from(""),
        Line::from(Span::styled(
            "Home Tab:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  Up/Down / j/k - Navigate projects"),
        Line::from("  Enter         - Activate project"),
        Line::from("  n             - Create new project"),
        Line::from("  r             - Refresh project list"),
        Line::from(""),
        Line::from(Span::styled(
            "Train Tab:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  t - Start training (validates first)"),
        Line::from("  d - Run demo training"),
        Line::from("  m - Toggle training mode (Single/Multi)"),
        Line::from("  g - Generate RLlib config"),
        Line::from("  c - Cancel running training"),
        Line::from("  Up/Down / j/k - Scroll training output"),
        Line::from("  PgUp/PgDn      - Fast scroll training output"),
        Line::from("  p/s/n - Edit basic fields"),
        Line::from("  a - Open advanced settings panel"),
        Line::from("  Esc/q - Close advanced settings panel"),
        Line::from("  b - Browse files for env path"),
        Line::from(""),
        Line::from(Span::styled(
            "Metrics Tab:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  Tab / Shift+Tab - Cycle focus between panels"),
        Line::from("  Enter          - Swap Chart/Policies positions"),
        Line::from("  Up/Down / j/k  - Navigate/scroll focused panel"),
        Line::from("  PgUp/PgDn      - Fast navigate/scroll"),
        Line::from("  Home/End       - Jump to latest/oldest in History"),
        Line::from("  Left/Right     - Scroll policies horizontally (when expanded)"),
        Line::from("  , / <          - Previous chart metric"),
        Line::from("  . / >          - Next chart metric"),
        Line::from(""),
        Line::from(Span::styled(
            "Export Tab:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  x - Start export"),
        Line::from("  c - Cancel export"),
        Line::from("  m - Toggle export mode"),
        Line::from("  Enter - Edit/toggle option"),
        Line::from("  Tab / o - Switch between options and output"),
        Line::from("  Up/Down / j/k - Navigate options or scroll output"),
        Line::from("  PgUp/PgDn - Fast scroll output"),
        Line::from(""),
        Line::from(Span::styled(
            "File Browser:",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from("  Up/Down / j/k  - Navigate"),
        Line::from("  Enter          - Select file / Enter directory"),
        Line::from("  Backspace / h  - Go up one directory"),
        Line::from("  Esc            - Cancel"),
        Line::from(""),
        Line::from(Span::styled(
            "Press any key to close",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )),
    ];

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

fn render_confirm_quit(frame: &mut Frame<'_>, _app: &App) {
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

fn render_file_browser(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();

    let vertical_margin = area.height / 8;
    let horizontal_margin = area.width / 8;
    let browser_area = Rect {
        x: horizontal_margin,
        y: vertical_margin,
        width: area.width.saturating_sub(horizontal_margin * 2),
        height: area.height.saturating_sub(vertical_margin * 2),
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(browser_area);

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
    frame.render_widget(path_para, chunks[0]);

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
    frame.render_stateful_widget(list, chunks[1], &mut state);

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
    let inst_block = Block::default().borders(Borders::ALL);
    let inst_para = Paragraph::new(instructions)
        .block(inst_block)
        .alignment(Alignment::Left);
    frame.render_widget(inst_para, chunks[2]);
}

fn render_advanced_config(frame: &mut Frame<'_>, app: &App) {
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
            Constraint::Length(4),
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
                let spans = vec![
                    Span::styled(
                        field.label(),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(": ", Style::default().fg(Color::DarkGray)),
                    Span::styled(value, Style::default().fg(Color::White)),
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

            (
                vec![
                    Line::from(first_line_spans),
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
