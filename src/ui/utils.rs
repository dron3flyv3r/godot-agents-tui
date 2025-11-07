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

    if !current_alpha.is_empty() {
        parts.push((current_alpha, 0));
    }
    if !current_num.is_empty() {
        let num: u64 = current_num.parse().unwrap_or(0);
        parts.push((String::new(), num));
    }

    parts
}
