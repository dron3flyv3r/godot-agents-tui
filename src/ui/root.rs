use ratatui::Frame;

use crate::app::{App, InputMode, TabId};
use ratatui::widgets::Clear;

use super::components::render_tabs;
use super::layout::split_layout;
use super::screens;

pub fn render(frame: &mut Frame<'_>, app: &App) {
    if app.is_help_visible() {
        screens::render_help_overlay(frame, app);
        return;
    }

    if app.input_mode() == InputMode::ConfirmQuit {
        screens::render_confirm_quit(frame, app);
        return;
    }

    if app.input_mode() == InputMode::ConfirmProjectImport {
        screens::render_project_import_prompt(frame, app);
        return;
    }

    if matches!(
        app.input_mode(),
        InputMode::ChartExportOptions | InputMode::EditingChartExportOption
    ) {
        screens::render_chart_export_options(frame, app);
        return;
    }

    if matches!(
        app.input_mode(),
        InputMode::MetricsSettings | InputMode::EditingMetricsSetting
    ) {
        screens::render_metrics_settings(frame, app);
        return;
    }

    if app.input_mode() == InputMode::SelectingConfigOption {
        screens::render_choice_menu(frame, app);
        return;
    }

    if matches!(
        app.input_mode(),
        InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig
    ) {
        screens::render_advanced_config(frame, app);
        return;
    }

    if app.input_mode() == InputMode::BrowsingFiles {
        screens::render_file_browser(frame, app);
        return;
    }

    let [header_area, content_area] = split_layout(frame.area());
    render_tabs(frame, header_area, app);

    #[allow(unreachable_patterns)]
    match app.active_tab().id {
        TabId::Home => screens::render_home(frame, content_area, app),
        TabId::Train => screens::render_train(frame, content_area, app),
        TabId::Metrics => screens::render_metrics(frame, content_area, app),
        TabId::Simulator => screens::render_simulator(frame, content_area, app),
        TabId::Interface => screens::render_interface(frame, content_area, app),
        TabId::ExportModel => screens::render_export(frame, content_area, app),
        TabId::Projects => screens::render_projects(frame, content_area, app),
        TabId::Settings => screens::render_settings(frame, content_area, app),
        tab => screens::render_placeholder(frame, content_area, tab, app),
    }

    // Loading overlay when long tasks run
    if app.is_export_running() {
        let area = frame.area();
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .title("Working...")
            .style(ratatui::style::Style::default().fg(ratatui::style::Color::Yellow));
        let message = ratatui::widgets::Paragraph::new("Please wait, export in progress...")
            .block(block)
            .alignment(ratatui::layout::Alignment::Center);
        frame.render_widget(Clear, area);
        frame.render_widget(message, area);
    }
}
