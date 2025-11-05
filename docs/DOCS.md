# Documentation Index

Complete documentation for the Godot RL Training Controller.

## 📚 Documentation Overview

This project includes comprehensive documentation to help you get started, configure, and troubleshoot the training controller.

## 🚀 Getting Started

### [Quick Start Guide](docs/QUICK_START.md)
**Perfect for first-time users**

Get up and running in minutes:
- Installation steps
- Your first training session
- Using models in Godot
- Common first-time issues
- Keyboard shortcuts cheat sheet

**Start here if you're new!**

### [README.md](README.md)
**Project overview and features**

High-level introduction:
- What is this controller?
- Key features
- Installation instructions
- Usage overview for all tabs
- Basic troubleshooting

## 📖 User Guides

### [Training Guide](TRAINING_GUIDE.md)
**Detailed training workflows**

Comprehensive guide to training:
- Single-agent vs multi-agent training
- Training modes (SB3 and RLlib)
- Step-by-step procedures
- Keyboard controls
- Output streaming
- Metrics integration
- Tips and best practices

### [Configuration Guide](docs/CONFIGURATION.md)
**All configuration options explained**

Everything about configuration:
- Configuration file formats
- Training parameters for SB3 and RLlib
- 50+ hyperparameters explained
- Export settings
- Best practices
- Common patterns
- Troubleshooting config issues

## 🔧 Technical Documentation

### [Architecture Documentation](docs/ARCHITECTURE.md)
**Technical design and implementation**

Deep dive into the system:
- System components (Rust + Python)
- Data flow and process management
- Metric system design
- Configuration layers
- UI architecture
- Performance considerations
- Dependencies and testing

### [Python Scripts API Reference](docs/API.md)
**Command-line interfaces and scripts**

Complete API documentation:
- `stable_baselines3_training_script.py`
- `rllib_training_script.py`
- `convert_sb3_to_onnx.py`
- `convert_rllib_to_onnx.py`
- All command-line arguments
- Metric format specification
- Integration with Godot RL Agents

## 🔍 Troubleshooting

### [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
**Solutions to common problems**

Comprehensive problem-solving:
- Installation issues
- Python environment problems
- Training issues
- Metric display problems
- Export issues
- Performance problems
- Configuration issues
- UI/display issues
- Godot integration issues
- Advanced debugging techniques

## 📋 Documentation by Task

### I want to...

#### Install and Setup
→ Start with [Quick Start Guide](docs/QUICK_START.md)  
→ Then read [Installation section in README](README.md#installation)

#### Train My First Agent
→ [Quick Start Guide - First Training](docs/QUICK_START.md#your-first-training-session)  
→ [Training Guide - Quick Start](TRAINING_GUIDE.md#quick-start)

#### Understand All Settings
→ [Configuration Guide](docs/CONFIGURATION.md)  
→ [Training Guide - Training Modes](TRAINING_GUIDE.md#training-modes)

#### Export to ONNX
→ [README - Export Tab](README.md#export-tab)  
→ [Training Guide - Tips](TRAINING_GUIDE.md#tips)  
→ [API Reference - Export Scripts](docs/API.md#export-scripts)

#### Optimize Performance
→ [Configuration Guide - Best Practices](docs/CONFIGURATION.md#configuration-best-practices)  
→ [Troubleshooting - Performance Problems](docs/TROUBLESHOOTING.md#performance-problems)

#### Fix Issues
→ [Troubleshooting Guide](docs/TROUBLESHOOTING.md)  
→ [Quick Start - Common Issues](docs/QUICK_START.md#common-first-time-issues)

#### Understand How It Works
→ [Architecture Documentation](docs/ARCHITECTURE.md)  
→ [API Reference](docs/API.md)

#### Use Multi-Agent Training
→ [Training Guide - Multi-Agent](TRAINING_GUIDE.md#multi-agent-rllib)  
→ [Configuration Guide - RLlib Config](docs/CONFIGURATION.md#rllib-configuration)

#### Integrate with Godot
→ [Quick Start - Using Model in Godot](docs/QUICK_START.md#using-the-model-in-godot)  
→ [Godot RL Agents Documentation](https://github.com/edbeeching/godot_rl_agents)

## 📄 File Reference

```
agents/
├── README.md                          # Main project overview
├── TRAINING_GUIDE.md                  # Training workflows
├── docs/
│   ├── QUICK_START.md                 # Getting started guide
│   ├── CONFIGURATION.md               # Configuration reference
│   ├── ARCHITECTURE.md                # Technical architecture
│   ├── API.md                         # Python scripts API
│   ├── TROUBLESHOOTING.md             # Problem solving
│   └── DOCS.md                        # This file
├── Cargo.toml                         # Rust dependencies
├── requrements.txt                    # Python dependencies
├── src/                               # Rust source code
├── projects/                          # Your training projects
└── logs/                              # Training logs and checkpoints
```

## 🎯 Quick Reference

### Essential Commands

**Build controller**:
```bash
cargo build --release
```

**Run controller**:
```bash
source .venv/bin/activate
./target/release/controller-mk2
```

**Check Python environment**:
```bash
python check_py_env.py
```

### Essential Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-4` | Switch tabs |
| `?` | Show help |
| `q` | Quit |
| `t` | Start training (Train tab) |
| `c` | Cancel training |
| `e` | Export model (Export tab) |
| `s` | Save configuration |
| `p` | Check Python env (Home tab) |

### Important File Locations

- **Projects**: `projects/<project-name>/`
- **SB3 Logs**: `logs/sb3/`
- **RLlib Checkpoints**: `logs/rllib/`
- **ONNX Exports**: `projects/<project-name>/onnx_exports/`
- **Training Config**: `projects/<project-name>/training_config.json`
- **Export Config**: `projects/<project-name>/export_config.json`

## 🔗 External Resources

### Godot RL Agents
- **GitHub**: https://github.com/edbeeching/godot_rl_agents
- **Documentation**: See GitHub repository
- **Purpose**: Core framework this controller integrates with

### Training Frameworks
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/index.html
- **PyTorch**: https://pytorch.org/docs/

### UI Framework
- **Ratatui**: https://github.com/ratatui-org/ratatui
- **Crossterm**: https://github.com/crossterm-rs/crossterm

## 📝 Documentation Standards

All documentation follows these principles:

- **Examples**: Every feature includes usage examples
- **Troubleshooting**: Common issues have solutions
- **Cross-references**: Related topics are linked
- **Up-to-date**: Last updated date on each doc
- **Beginner-friendly**: Clear explanations without jargon

## 🆕 Recent Updates

**November 5, 2025**:
- ✅ Complete documentation suite created
- ✅ Quick start guide added
- ✅ Configuration reference completed
- ✅ Architecture documentation written
- ✅ API reference comprehensive
- ✅ Troubleshooting guide extensive

## 💡 Contributing to Documentation

Found an issue or want to improve the docs?

1. **Typos/Errors**: Submit corrections
2. **Missing Info**: Request additional documentation
3. **Examples**: Share your configurations and tips
4. **Troubleshooting**: Document new solutions

Keep documentation:
- Clear and concise
- Example-driven
- Well-organized
- Cross-referenced

## 🎓 Learning Path

### Beginner
1. Read [README.md](README.md)
2. Follow [Quick Start Guide](docs/QUICK_START.md)
3. Complete first training session
4. Review [Training Guide](TRAINING_GUIDE.md)

### Intermediate
1. Explore [Configuration Guide](docs/CONFIGURATION.md)
2. Experiment with hyperparameters
3. Learn multi-agent training
4. Study [API Reference](docs/API.md)

### Advanced
1. Read [Architecture Documentation](docs/ARCHITECTURE.md)
2. Understand metric system
3. Optimize performance
4. Contribute improvements

## 📬 Feedback

Documentation feedback is valuable:
- What's confusing?
- What's missing?
- What examples would help?
- What worked well?

## 🏆 Best Practices

### When Learning
- Start with Quick Start
- Try examples before experimenting
- Keep notes on what works
- Reference troubleshooting when stuck

### When Training
- Read Training Guide thoroughly
- Start with defaults
- Document your changes
- Monitor metrics carefully

### When Troubleshooting
- Check Troubleshooting Guide first
- Gather error information
- Try simple solutions first
- Document what worked

---

**Documentation Version**: 1.0  
**Last Updated**: November 5, 2025  
**Status**: Complete

For questions about this documentation, refer to specific guide or troubleshooting section.
