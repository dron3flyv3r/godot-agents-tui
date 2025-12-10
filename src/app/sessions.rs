use serde::{Deserialize, Serialize};

/// Increment when the session file format changes.
pub const SESSION_STORE_VERSION: u32 = 2;

fn default_last_used() -> u64 {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRunLink {
    /// Run file path relative to the project root.
    pub run_path: String,
    /// Global iteration where this run should start in the merged timeline.
    pub start_iteration: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    #[serde(default = "default_last_used")]
    pub last_used: u64,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub runs: Vec<SessionRunLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStore {
    #[serde(default = "SessionStore::version")]
    pub version: u32,
    #[serde(default)]
    pub sessions: Vec<SessionRecord>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self {
            version: SESSION_STORE_VERSION,
            sessions: Vec::new(),
        }
    }
}

impl SessionStore {
    fn version() -> u32 {
        SESSION_STORE_VERSION
    }

    /// Migrate an older session store to the latest format.
    pub fn migrate(mut self) -> (Self, bool) {
        let mut changed = false;

        // Upgrade missing fields and bump version when loading older files.
        for session in &mut self.sessions {
            if session.last_used == 0 {
                session.last_used = session.created_at;
                changed = true;
            }
        }

        if self.version != SESSION_STORE_VERSION {
            self.version = SESSION_STORE_VERSION;
            changed = true;
        }

        (self, changed)
    }
}

/// Create a two-word session name from static word lists.
pub fn generate_session_name(seed: u64) -> String {
    const ADJECTIVES: &[&str] = &[
        "amber", "brisk", "crisp", "daring", "eager", "frozen", "gentle", "hollow", "ivory",
        "jade", "kindred", "lively", "mellow", "nimble", "opal", "prism", "quiet", "rustic",
        "solar", "tidy", "urban", "vivid", "willow", "young", "zephyr", "drone",
    ];
    const NOUNS: &[&str] = &[
        "badger", "brook", "cedar", "dawn", "ember", "falcon", "grove", "harbor", "iris", "koi",
        "lagoon", "mesa", "nightjar", "orchid", "prairie", "quill", "ridge", "spruce", "thicket",
        "upland", "valley", "wave", "yew", "zinnia", "flyver",
    ];

    let adj = ADJECTIVES[(seed as usize) % ADJECTIVES.len()];
    let noun = NOUNS[(seed as usize / ADJECTIVES.len()) % NOUNS.len()];
    format!("{adj}-{noun}")
}
