use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Tabs;
use ratatui::Frame;

use crate::app::{App, TabId};

pub fn render_tabs(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let titles = app
        .tabs()
        .iter()
        .map(|tab| {
            let mut label = format!(" {} ", tab.title);
            let show_spinner = (tab.id == TabId::Train && app.is_training_running())
                || (tab.id == TabId::ExportModel && app.is_export_running())
                || (tab.id == TabId::Simulator && app.is_simulator_running())
                || (tab.id == TabId::Projects && app.is_project_archive_running());
            if show_spinner {
                if let Some(spinner) = app.spinner_char() {
                    label = format!("{} {}", spinner, label.trim_start());
                }
            }
            Line::from(Span::styled(label, Style::default().fg(Color::Cyan)))
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
