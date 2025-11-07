use ratatui::layout::{Constraint, Direction, Layout, Rect};

pub fn split_layout(area: Rect) -> [Rect; 2] {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
        .split(area);
    [layout[0], layout[1]]
}
