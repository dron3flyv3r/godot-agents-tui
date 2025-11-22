mod app;
mod cli;
mod domain;
mod ui;

use std::io::{self, Stdout};
use std::time::Duration;

use app::{App, ExportFocus, FileBrowserKind, FileBrowserState, InputMode, TabId};
use clap::Parser;
use cli::Cli;
use color_eyre::Result;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

fn setup_terminal() -> Result<(Terminal<CrosstermBackend<Stdout>>, CrosstermGuard)> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok((terminal, CrosstermGuard))
}

struct CrosstermGuard;

impl Drop for CrosstermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
    }
}

fn handle_key_event(app: &mut App, key: KeyCode) -> Result<()> {
    match app.input_mode() {
        InputMode::CreatingProject => handle_project_creation_input(app, key)?,
        InputMode::EditingConfig => handle_config_edit_input(app, key)?,
        InputMode::EditingAdvancedConfig => handle_config_edit_input(app, key)?,
        InputMode::SelectingConfigOption => handle_choice_menu_input(app, key)?,
        InputMode::AdvancedConfig => handle_advanced_config_input(app, key)?,
        InputMode::BrowsingFiles => handle_file_browser_input(app, key)?,
        InputMode::Help => handle_help_input(app, key)?,
        InputMode::ConfirmQuit => handle_confirm_quit_input(app, key)?,
        InputMode::EditingExport => handle_export_edit_input(app, key)?,
        InputMode::Normal => handle_normal_mode_key(app, key)?,
    }
    Ok(())
}

fn handle_project_creation_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Enter => {
            app.confirm_project_creation()?;
        }
        KeyCode::Esc => app.cancel_project_creation(),
        KeyCode::Backspace => app.pop_project_name_char(),
        KeyCode::Char(ch) => app.push_project_name_char(ch),
        _ => {}
    }
    Ok(())
}

fn handle_choice_menu_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => app.move_choice_selection(-1),
        KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => app.move_choice_selection(1),
        KeyCode::Enter => {
            app.confirm_choice_selection()?;
        }
        KeyCode::Esc => app.cancel_choice_selection(),
        _ => {}
    }
    Ok(())
}

fn handle_normal_mode_key(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Char('q') => app.request_quit(),
        KeyCode::Esc => app.request_quit(),
        KeyCode::Char('h') | KeyCode::Char('H') | KeyCode::F(1) => app.show_help(),
        // KeyCode::Left => app.previous_tab(),
        // KeyCode::Right => app.next_tab(),
        KeyCode::Char('1') => app.activate(TabId::Home),
        KeyCode::Char('2') => app.activate(TabId::Train),
        KeyCode::Char('3') => app.activate(TabId::Metrics),
        KeyCode::Char('4') => app.activate(TabId::Simulator),
        KeyCode::Char('5') => app.activate(TabId::Interface),
        KeyCode::Char('6') => app.activate(TabId::Export),
        KeyCode::Char('7') => app.activate(TabId::Settings),
        KeyCode::Home => app.activate(TabId::Home),
        KeyCode::End => app.activate(TabId::Settings),
        _ => {}
    }

    if app.active_tab().id == TabId::Home {
        match key {
            KeyCode::Down | KeyCode::Char('j') => app.select_next_project(),
            KeyCode::Up | KeyCode::Char('k') => app.select_previous_project(),
            KeyCode::Enter => app.set_active_project()?,
            KeyCode::Char('n') => app.start_project_creation(),
            KeyCode::Char('p') | KeyCode::Char('P') => app.refresh_python_environment(),
            KeyCode::Char('r') | KeyCode::Char('R') => app.force_refresh_projects()?,
            _ => {}
        }
    } else if app.active_tab().id == TabId::Train {
        match key {
            KeyCode::Char('t') | KeyCode::Char('T') => app.start_training()?,
            KeyCode::Char('d') | KeyCode::Char('D') => app.start_demo_training()?,
            KeyCode::Char('m') | KeyCode::Char('M') => {
                app.toggle_training_mode();
                app.set_status(
                    format!(
                        "Training mode: {}",
                        match app.training_config().mode {
                            app::TrainingMode::SingleAgent => "Single-Agent (SB3)",
                            app::TrainingMode::MultiAgent => "Multi-Agent (RLlib)",
                        }
                    ),
                    app::StatusKind::Info,
                );
            }
            KeyCode::Char('c') | KeyCode::Char('C') => app.cancel_training(),
            KeyCode::Char('g') | KeyCode::Char('G') => app.generate_rllib_config()?,
            KeyCode::Char('e') | KeyCode::Char('E') => {
                app.set_status(
                    "Training config editing: p (path), s (steps), n (name), a (advanced)",
                    app::StatusKind::Info,
                );
            }
            KeyCode::Char('p') | KeyCode::Char('P') if !app.is_training_running() => {
                app.start_config_edit(app::ConfigField::EnvPath);
            }
            KeyCode::Char('s') | KeyCode::Char('S') if !app.is_training_running() => {
                app.start_config_edit(app::ConfigField::Timesteps);
            }
            KeyCode::Char('n') | KeyCode::Char('N') if !app.is_training_running() => {
                app.start_config_edit(app::ConfigField::ExperimentName);
            }
            KeyCode::Char('b') | KeyCode::Char('B') => {
                app.start_config_file_browser(app::ConfigField::EnvPath);
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                if app.is_training_running() {
                    app.set_status(
                        "Cannot edit advanced settings while training is running",
                        app::StatusKind::Warning,
                    );
                } else {
                    app.open_advanced_config();
                }
            }
            KeyCode::Up => app.scroll_training_output_up(1),
            KeyCode::Down => app.scroll_training_output_down(1),
            KeyCode::PageUp => app.scroll_training_output_up(10),
            KeyCode::PageDown => app.scroll_training_output_down(10),
            KeyCode::Char('k') | KeyCode::Char('K') => app.scroll_training_output_up(1),
            KeyCode::Char('j') | KeyCode::Char('J') => app.scroll_training_output_down(1),
            _ => {}
        }
    } else if app.active_tab().id == TabId::Metrics {
        match key {
            KeyCode::Tab => {
                app.metrics_cycle_focus_next();
            }
            KeyCode::BackTab => {
                app.metrics_cycle_focus_previous();
            }
            KeyCode::Enter => {
                // Toggle chart/policies swap when Policies or Chart is focused
                if app.metrics_focus() == app::MetricsFocus::Policies
                    || app.metrics_focus() == app::MetricsFocus::Chart
                {
                    app.metrics_toggle_policies_expanded();
                }
            }
            KeyCode::Left => {
                // Horizontal scroll in expanded policies view
                if app.metrics_policies_expanded() {
                    app.metrics_scroll_policies_left();
                }
            }
            KeyCode::Right => {
                // Horizontal scroll in expanded policies view
                if app.metrics_policies_expanded() {
                    app.metrics_scroll_policies_right();
                }
            }
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                // If History is focused, move to newer entry (history navigation)
                // Otherwise scroll the focused panel up
                if app.metrics_focus() == app::MetricsFocus::History {
                    app.metrics_history_move_newer();
                } else {
                    app.metrics_scroll_up(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                // If History is focused, move to older entry (history navigation)
                // Otherwise scroll the focused panel down
                if app.metrics_focus() == app::MetricsFocus::History {
                    app.metrics_history_move_older();
                } else {
                    app.metrics_scroll_down(1);
                }
            }
            KeyCode::PageUp => {
                if app.metrics_focus() == app::MetricsFocus::History {
                    app.metrics_history_page_newer(10);
                } else {
                    app.metrics_scroll_up(5);
                }
            }
            KeyCode::PageDown => {
                if app.metrics_focus() == app::MetricsFocus::History {
                    app.metrics_history_page_older(10);
                } else {
                    app.metrics_scroll_down(5);
                }
            }
            KeyCode::Home => app.metrics_history_to_latest(),
            KeyCode::End => app.metrics_history_to_oldest(),
            KeyCode::Char(',') | KeyCode::Char('<') => {
                app.cycle_chart_metric_previous();
            }
            KeyCode::Char('.') | KeyCode::Char('>') => {
                app.cycle_chart_metric_next();
            }
            KeyCode::Char('c') => {
                app.start_run_overlay_browser()?;
            }
            KeyCode::Char('C') => {
                app.clear_run_overlays();
            }
            KeyCode::Char('o') => {
                app.toggle_selected_overlay_view();
            }
            KeyCode::Char('O') => {
                app.cycle_saved_run_overlay(1);
            }
            KeyCode::Char('v') | KeyCode::Char('V') => {
                app.clear_archived_run_view();
            }
            _ => {}
        }
    } else if app.active_tab().id == TabId::Simulator {
        match key {
            KeyCode::Char('s') | KeyCode::Char('S') => app.start_simulator()?,
            KeyCode::Char('c') | KeyCode::Char('C') => app.cancel_simulator(),
            KeyCode::Char('m') | KeyCode::Char('M') => app.toggle_simulator_mode(),
            KeyCode::Char('w') | KeyCode::Char('W') => app.toggle_simulator_show_window(),
            KeyCode::Char('a') | KeyCode::Char('A') => app.toggle_simulator_auto_restart(),
            KeyCode::Char('t') | KeyCode::Char('T') => app.toggle_simulator_tracebacks(),
            KeyCode::Char('p') | KeyCode::Char('P') => app.start_simulator_file_browser(),
            KeyCode::Char('y') | KeyCode::Char('Y') => app.simulator_use_training_env_path(),
            KeyCode::Char('f') | KeyCode::Char('F') | KeyCode::Tab => {
                app.cycle_simulator_focus();
            }
            KeyCode::Char('v') | KeyCode::Char('V') => app.toggle_simulator_compact_view(),
            KeyCode::Char('[') => app.adjust_simulator_step_delay(-0.05),
            KeyCode::Char(']') => app.adjust_simulator_step_delay(0.05),
            KeyCode::Char('-') => app.adjust_simulator_restart_delay(-0.5),
            KeyCode::Char('=') => app.adjust_simulator_restart_delay(0.5),
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                app.simulator_scroll_up(1);
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                app.simulator_scroll_down(1);
            }
            KeyCode::PageUp => app.simulator_scroll_up(10),
            KeyCode::PageDown => app.simulator_scroll_down(10),
            _ => {}
        }
    } else if app.active_tab().id == TabId::Interface {
        match key {
            KeyCode::Char('s') | KeyCode::Char('S') => app.start_interface()?,
            KeyCode::Char('c') | KeyCode::Char('C') => app.cancel_interface(),
            KeyCode::Char('m') | KeyCode::Char('M') => app.toggle_interface_mode(),
            KeyCode::Char('w') | KeyCode::Char('W') => app.toggle_interface_window(),
            KeyCode::Char('a') | KeyCode::Char('A') => app.toggle_interface_auto_restart(),
            KeyCode::Char('t') => app.toggle_interface_agent_type(),
            KeyCode::Char('T') => app.toggle_interface_tracebacks(),
            KeyCode::Char('p') | KeyCode::Char('P') => app.start_interface_agent_browser(),
            KeyCode::Char('y') | KeyCode::Char('Y') => app.interface_use_export_agent_path(),
            KeyCode::Char('f') | KeyCode::Char('F') | KeyCode::Tab => {
                app.cycle_interface_focus();
            }
            KeyCode::Char('v') | KeyCode::Char('V') => app.toggle_interface_compact_view(),
            KeyCode::Char('[') => app.adjust_interface_step_delay(-0.05),
            KeyCode::Char(']') => app.adjust_interface_step_delay(0.05),
            KeyCode::Char('-') => app.adjust_interface_restart_delay(-0.5),
            KeyCode::Char('=') => app.adjust_interface_restart_delay(0.5),
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                app.interface_scroll_up(1);
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                app.interface_scroll_down(1);
            }
            KeyCode::PageUp => app.interface_scroll_up(10),
            KeyCode::PageDown => app.interface_scroll_down(10),
            _ => {}
        }
    } else if app.active_tab().id == TabId::Export {
        match key {
            KeyCode::Char('x') => app.start_export()?,
            KeyCode::Char('c') | KeyCode::Char('C') => app.cancel_export(),
            KeyCode::Char('m') | KeyCode::Char('M') => {
                if app.is_export_running() {
                    app.set_status(
                        "Cannot change export mode while export is running",
                        app::StatusKind::Warning,
                    );
                } else {
                    if let Err(error) = app.toggle_export_mode() {
                        app.set_status(
                            format!("Failed to update export mode: {error}"),
                            app::StatusKind::Error,
                        );
                    } else {
                        app.set_status(
                            format!("Export mode: {}", app.export_mode().label()),
                            app::StatusKind::Info,
                        );
                    }
                }
            }
            KeyCode::Backspace | KeyCode::Delete => {
                if app.export_focus() == ExportFocus::Fields {
                    app.clear_selected_export_field();
                }
            }
            KeyCode::Tab | KeyCode::BackTab | KeyCode::Char('o') | KeyCode::Char('O') => {
                app.toggle_export_focus();
            }
            KeyCode::Enter => {
                if app.is_export_running() {
                    app.set_status(
                        "Cannot edit export options during an active export",
                        app::StatusKind::Warning,
                    );
                } else {
                    app.edit_selected_export_field();
                }
            }
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                if app.export_focus() == ExportFocus::Fields {
                    app.select_previous_export_field();
                } else {
                    app.scroll_export_output_up(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                if app.export_focus() == ExportFocus::Fields {
                    app.select_next_export_field();
                } else {
                    app.scroll_export_output_down(1);
                }
            }
            KeyCode::PageUp => app.scroll_export_output_up(10),
            KeyCode::PageDown => app.scroll_export_output_down(10),
            _ => {}
        }
    } else if app.active_tab().id == TabId::Settings {
        match key {
            KeyCode::Down | KeyCode::Char('j') => app.select_next_setting(),
            KeyCode::Up | KeyCode::Char('k') => app.select_previous_setting(),
            KeyCode::Left | KeyCode::Char('h') => app.adjust_setting(-1),
            KeyCode::Right | KeyCode::Char('l') => app.adjust_setting(1),
            KeyCode::Enter | KeyCode::Char(' ') => app.toggle_current_setting(),
            _ => {}
        }
    }

    Ok(())
}

fn handle_config_edit_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Enter => app.confirm_config_edit(),
        KeyCode::Esc => app.cancel_config_edit(),
        KeyCode::Backspace => app.pop_config_char(),
        KeyCode::Char(ch) => app.push_config_char(ch),
        _ => {}
    }
    Ok(())
}

fn handle_export_edit_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Enter => app.confirm_export_edit(),
        KeyCode::Esc => app.cancel_export_edit(),
        KeyCode::Backspace => app.pop_export_char(),
        KeyCode::Char(ch) => app.push_export_char(ch),
        _ => {}
    }
    Ok(())
}

fn handle_file_browser_input(app: &mut App, key: KeyCode) -> Result<()> {
    match app.file_browser_state() {
        FileBrowserState::Browsing => match key {
            KeyCode::Esc => app.cancel_file_browser(),
            KeyCode::Down | KeyCode::Char('j') => app.file_browser_select_next(),
            KeyCode::Up | KeyCode::Char('k') => app.file_browser_select_previous(),
            KeyCode::Enter => app.file_browser_enter(),
            KeyCode::Backspace | KeyCode::Left | KeyCode::Char('h') => app.file_browser_go_up(),
            KeyCode::Char('f') | KeyCode::Char('F') => app.file_browser_finalize_selection(),
            KeyCode::Char('n') | KeyCode::Char('N') => {
                if matches!(
                    app.file_browser_kind(),
                    FileBrowserKind::Directory {
                        allow_create: true,
                        ..
                    } | FileBrowserKind::OutputFile { .. }
                ) {
                    app.file_browser_begin_new_folder();
                } else {
                    app.set_status(
                        "Folder creation is not allowed here",
                        app::StatusKind::Warning,
                    );
                }
            }
            _ => {}
        },
        FileBrowserState::NamingFolder | FileBrowserState::NamingFile => match key {
            KeyCode::Esc => app.file_browser_cancel_input(),
            KeyCode::Enter => app.file_browser_confirm_input(),
            KeyCode::Backspace => app.file_browser_pop_char(),
            KeyCode::Char(ch) => app.file_browser_push_char(ch),
            _ => {}
        },
    }
    Ok(())
}

fn handle_help_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Esc
        | KeyCode::Char('h')
        | KeyCode::Char('H')
        | KeyCode::Char('q')
        | KeyCode::F(1) => {
            app.hide_help();
        }
        _ => {}
    }
    Ok(())
}

fn handle_confirm_quit_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => app.confirm_quit(),
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => app.cancel_quit(),
        _ => {}
    }
    Ok(())
}

fn handle_advanced_config_input(app: &mut App, key: KeyCode) -> Result<()> {
    match key {
        KeyCode::Esc
        | KeyCode::Char('q')
        | KeyCode::Char('Q')
        | KeyCode::Char('a')
        | KeyCode::Char('A') => app.close_advanced_config(),
        KeyCode::Down | KeyCode::Char('j') => app.select_next_advanced_field(),
        KeyCode::Up | KeyCode::Char('k') => app.select_previous_advanced_field(),
        KeyCode::Backspace | KeyCode::Delete => {
            app.clear_selected_advanced_field();
        }
        KeyCode::Enter => {
            if !app.is_training_running() {
                app.edit_selected_advanced_field();
            } else {
                app.set_status(
                    "Cannot edit settings while training is running",
                    app::StatusKind::Warning,
                );
            }
        }
        _ => {}
    }
    Ok(())
}

fn run() -> Result<()> {
    let (mut terminal, _guard) = setup_terminal()?;
    let mut app = App::new()?;

    while !app.should_quit() {
        app.process_background_tasks();
        app.clamp_all_metrics_scrolls();
        terminal.draw(|frame| ui::render(frame, &app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key_event) = event::read()? {
                handle_key_event(&mut app, key_event.code)?;
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let cli = Cli::parse();
    if cli.command.is_some() {
        cli::handle_cli(cli)?;
    } else {
        run()?;
    }

    Ok(())
}
