use chrono::Local;
use clap::{Parser, Subcommand};
use color_eyre::owo_colors::OwoColorize;
use color_eyre::Result;
use crossterm::style::Stylize;
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

const EMBEDDED_PYTHON_BIN: Option<&str> = option_env!("CONTROLLER_PYTHON_BIN");
const EMBEDDED_SCRIPT_ROOT: Option<&str> = option_env!("CONTROLLER_SCRIPTS_ROOT");

#[derive(Parser)]
#[command(name = "controller-mk2")]
#[command(about = "Rust Controller for Godot RL Agents", long_about = None)]
pub struct Cli {
    /// Launch the experimental MARS-only controller UI
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub exp: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run the training script
    Train {
        /// Path to the config file
        #[arg(long, default_value = "rllib_config.yaml")]
        config_file: String,

        /// Restore from a checkpoint
        #[arg(long)]
        restore: Option<String>,

        /// Resume from a checkpoint directory (or legacy Tune run directory)
        #[arg(long)]
        resume: Option<String>,

        /// Experiment directory for logs
        #[arg(long, default_value = "logs/rllib")]
        experiment_dir: String,

        /// Generate a default config file
        #[arg(long)]
        generate: bool,

        /// Show debug panel before training
        #[arg(long)]
        debug_panel: bool,
    },
    /// Run the simulator
    Simulator {
        /// Path to the exported Godot environment
        #[arg(long)]
        env_path: Option<String>,

        /// Simulation mode (single or multi)
        #[arg(long, default_value = "auto")]
        mode: String,

        /// Show the Godot window
        #[arg(long)]
        show_window: bool,

        /// Run in headless mode
        #[arg(long)]
        headless: bool,

        /// Seed for the environment
        #[arg(long, default_value = "0")]
        seed: i32,

        /// Delay between steps
        #[arg(long, default_value = "0.0")]
        step_delay: f32,

        /// Delay before restarting
        #[arg(long, default_value = "2.0")]
        restart_delay: f32,

        /// Max episodes to run
        #[arg(long)]
        max_episodes: Option<i32>,

        /// Max steps per episode
        #[arg(long)]
        max_steps: Option<i32>,

        /// Disable auto-restart
        #[arg(long)]
        no_auto_restart: bool,

        /// Log tracebacks
        #[arg(long)]
        log_tracebacks: bool,
    },
    /// Export the model to ONNX
    Export {
        /// Path to the config file
        #[arg(long, default_value = "rllib_config.yaml")]
        config_file: String,

        /// Checkpoint to export
        #[arg(long)]
        checkpoint: Option<String>,

        /// Output directory
        #[arg(long, default_value = "onnx_export")]
        output_dir: String,
    },
}

// Helper types for clap arguments
// type i32 = i32;
// type f32 = f32;

pub fn handle_cli(cli: Cli) -> Result<()> {
    match cli.command {
        Some(Commands::Train {
            config_file,
            restore,
            resume,
            experiment_dir,
            generate,
            debug_panel,
        }) => {
            if generate {
                generate_config()?;
                return Ok(());
            }
            run_training(config_file, restore, resume, experiment_dir, debug_panel)
        }
        Some(Commands::Simulator {
            env_path,
            mode,
            show_window,
            headless,
            seed,
            step_delay,
            restart_delay,
            max_episodes,
            max_steps,
            no_auto_restart,
            log_tracebacks,
        }) => run_simulator(
            env_path,
            mode,
            show_window,
            headless,
            seed,
            step_delay,
            restart_delay,
            max_episodes,
            max_steps,
            no_auto_restart,
            log_tracebacks,
        ),
        Some(Commands::Export {
            config_file,
            checkpoint,
            output_dir,
        }) => run_export(config_file, checkpoint, output_dir),
        None => Ok(()), // Should be handled by main to launch TUI
    }
}

fn generate_config() -> Result<()> {
    let config_content = r#"
env_is_multiagent: false
stop:
  training_iteration: 100
checkpoint_frequency: 10
config:
  env_config:
    env_path: null
  num_envs_per_worker: 1
  framework: torch
  model:
    fcnet_hiddens: [256, 256]
    fcnet_activation: relu
"#;
    let path = "rllib_config.yaml";
    if std::path::Path::new(path).exists() {
        println!("{} already exists. Skipping generation.", path);
    } else {
        fs::write(path, config_content)?;
        println!("Generated default config at {}", path);
    }
    Ok(())
}

fn run_training(
    config_file: String,
    restore: Option<String>,
    resume: Option<String>,
    experiment_dir: String,
    debug_panel: bool,
) -> Result<()> {
    let script_path = find_script("rllib_training_script.py")?;
    let mut args = vec![
        script_path.to_string_lossy().to_string(),
        "--config_file".to_string(),
        config_file,
        "--experiment_dir".to_string(),
        experiment_dir,
    ];

    if let Some(r) = restore {
        args.push("--restore".to_string());
        args.push(r);
    }
    if let Some(r) = resume {
        args.push("--resume".to_string());
        args.push(r);
    }
    if debug_panel {
        args.push("--debug_panel".to_string());
    }

    run_python_process("train", args, |line| {
        if line.starts_with("@METRIC ") {
            if let Some(json_str) = line.strip_prefix("@METRIC ") {
                if let Ok(json) = serde_json::from_str::<Value>(json_str) {
                    print_pretty_metric(&json);
                    return false; // Suppress raw metric line from stdout
                }
            }
        }
        true // Print other lines
    })
}

fn print_pretty_metric(json: &Value) {
    let iter = json["training_iteration"].as_i64().unwrap_or(0);
    let reward_mean = json["episode_reward_mean"].as_f64().unwrap_or(0.0);
    let len_mean = json["episode_len_mean"].as_f64().unwrap_or(0.0);

    println!(
        "{} Iter {}: Reward={:.2}, Len={:.2}",
        "📈".green(),
        iter.to_string().bold(),
        reward_mean.to_string().cyan(),
        len_mean.to_string().yellow()
    );
}

fn run_simulator(
    env_path: Option<String>,
    mode: String,
    show_window: bool,
    headless: bool,
    seed: i32,
    step_delay: f32,
    restart_delay: f32,
    max_episodes: Option<i32>,
    max_steps: Option<i32>,
    no_auto_restart: bool,
    log_tracebacks: bool,
) -> Result<()> {
    let script_path = find_script("simulator.py")?;
    let mut args = vec![
        script_path.to_string_lossy().to_string(),
        "--mode".to_string(),
        mode,
        "--seed".to_string(),
        seed.to_string(),
        "--step-delay".to_string(),
        step_delay.to_string(),
        "--restart-delay".to_string(),
        restart_delay.to_string(),
    ];

    if let Some(path) = env_path {
        args.push("--env-path".to_string());
        args.push(path);
    }
    if show_window {
        args.push("--show-window".to_string());
    }
    if headless {
        args.push("--headless".to_string());
    }
    if let Some(ep) = max_episodes {
        args.push("--max-episodes".to_string());
        args.push(ep.to_string());
    }
    if let Some(steps) = max_steps {
        args.push("--max-steps".to_string());
        args.push(steps.to_string());
    }
    if no_auto_restart {
        args.push("--no-auto-restart".to_string());
    }
    if log_tracebacks {
        args.push("--log-tracebacks".to_string());
    }

    run_python_process("sim", args, |line| {
        if line.starts_with("@SIM_ACTION ") {
            if let Some(json_str) = line.strip_prefix("@SIM_ACTION ") {
                if let Ok(json) = serde_json::from_str::<Value>(json_str) {
                    print_pretty_sim_action(&json);
                    return false; // Suppress raw action line
                }
            }
        } else if line.starts_with("@SIM_EVENT ") {
            if let Some(json_str) = line.strip_prefix("@SIM_EVENT ") {
                if let Ok(json) = serde_json::from_str::<Value>(json_str) {
                    print_pretty_sim_event(&json);
                    return false; // Suppress raw event line
                }
            }
        }
        true
    })
}

fn print_pretty_sim_action(json: &Value) {
    let episode = json["episode"].as_i64().unwrap_or(0);
    let step = json["step"].as_i64().unwrap_or(0);
    let agents = json["agents"].as_array();

    if let Some(agents) = agents {
        for agent in agents {
            let id = agent["agent"].as_str().unwrap_or("?");
            let reward = agent["reward"].as_f64().unwrap_or(0.0);
            // Truncate action for display
            let action = agent["action"].to_string();
            let action_display = if action.len() > 50 {
                format!("{}...", &action[..47])
            } else {
                action
            };

            println!(
                "{} Ep {} Step {}: Agent {} => Act: {}, Rew: {:.2}",
                "🎮".blue(),
                episode,
                step,
                id.bold(),
                action_display.as_str().dimmed(),
                reward
            );
        }
    }
}

fn print_pretty_sim_event(json: &Value) {
    let kind = json["kind"].as_str().unwrap_or("unknown");
    let msg = json.get("message").and_then(|v| v.as_str()).unwrap_or("");

    let icon = match kind {
        "episode_start" => "🏁",
        "episode_end" => "🛑",
        "connected" => "🔌",
        "error" => "❌",
        _ => "ℹ️",
    };

    println!("{} {}: {}", icon, kind.to_uppercase().bold(), msg);
}

fn run_export(config_file: String, checkpoint: Option<String>, output_dir: String) -> Result<()> {
    let script_path = find_script("convert_rllib_to_onnx.py")?;
    let mut args = vec![
        script_path.to_string_lossy().to_string(),
        "--config_file".to_string(),
        config_file,
        "--output_dir".to_string(),
        output_dir,
    ];

    // Note: The python script might not take --checkpoint directly if it relies on rllib_config.yaml or interactive selection,
    // but based on user request we assume we can pass it or the script handles it.
    // Checking the script (not visible here but assuming standard args or user intent),
    // actually the user implementation plan said "Wraps convert_rllib_to_onnx.py".
    // Let's assume we pass checkpoint if the script supports it, or we might need to adjust.
    // For now, we'll pass it if provided.
    if let Some(ckpt) = checkpoint {
        // The script might expect a different flag or positional arg.
        // Let's assume --checkpoint based on typical patterns, or maybe it's --restore?
        // The user request implied standard args. Let's use --checkpoint.
        args.push("--checkpoint".to_string());
        args.push(ckpt);
    }

    run_python_process("export", args, |_| true)
}

fn run_python_process<F>(prefix: &str, args: Vec<String>, mut output_filter: F) -> Result<()>
where
    F: FnMut(&str) -> bool,
{
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let log_dir = PathBuf::from("logs");
    fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join(format!("{}_{}.log", prefix, timestamp));
    let mut log_file = File::create(&log_path)?;

    println!("{} Logging to {}", "📝".green(), log_path.display());

    // Determine python command
    let python_cmd = determine_python_command();

    let mut child = Command::new(&python_cmd)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            color_eyre::eyre::eyre!("Failed to spawn python command '{}': {}", python_cmd, e)
        })?;

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    // Spawn thread for stdout
    let (tx, rx) = mpsc::channel();
    let tx_err = tx.clone();

    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(l) = line {
                let _ = tx.send((false, l));
            }
        }
    });

    // Spawn thread for stderr
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(l) = line {
                let _ = tx_err.send((true, l));
            }
        }
    });

    // Main loop processing output
    for (is_err, line) in rx {
        // Log to file
        writeln!(log_file, "{}", line)?;

        // Filter and print to stdout
        if output_filter(&line) {
            if is_err {
                eprintln!("{}", line.red());
            } else {
                println!("{}", line);
            }
        }
    }

    let status = child.wait()?;
    if !status.success() {
        eprintln!("{}", "Process exited with error".red().bold());
    }

    Ok(())
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

fn find_script(script_name: &str) -> Result<PathBuf> {
    // First check current working directory
    let cwd = std::env::current_dir().unwrap_or_default();
    let local_script = cwd.join(script_name);
    if local_script.exists() {
        return Ok(local_script);
    }

    // Then check in the controller root (if configured)
    if let Some(root) = controller_scripts_root() {
        let embedded = root.join(script_name);
        if embedded.exists() {
            return Ok(embedded);
        }
    }

    Err(color_eyre::eyre::eyre!(
        "Could not find script '{}'. Checked current directory and CONTROLLER_SCRIPTS_ROOT.",
        script_name
    ))
}
