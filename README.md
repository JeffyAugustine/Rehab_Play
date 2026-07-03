<div align="center">

# RehabPlay

**AI-Powered Physical Rehabilitation Through Gamified Robotics**

Simulated robotic therapy assistant that turns range-of-motion exercises into an adaptive game, built on NVIDIA Isaac Sim, Isaac Lab, and Isaac GR00T.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3-76B900.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900.svg)](https://developer.nvidia.com/isaac-sim)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange.svg)](#roadmap)

[Overview](#overview) •
[Architecture](#architecture) •
[Getting Started](#getting-started) •
[Project Structure](#project-structure) •
[Roadmap](#roadmap) •
[Contributing](#contributing)

</div>

---

## Overview

RehabPlay is a simulation-first research project exploring how **foundation robot models** and **reinforcement learning** can make physical rehabilitation therapy more engaging and data-driven. A simulated robot arm guides a patient's limb through prescribed range-of-motion (ROM) exercises, while the patient's movements simultaneously control a simple game — turning repetitive, often tedious therapy sessions into an interactive experience.

The system is designed around three pillars:

| Pillar | Description |
|---|---|
| **Safety-Constrained Control** | A reward-shaped RL policy that respects force thresholds and safe motion boundaries at every timestep, with hard safety cutoffs. |
| **Adaptive Assistance** | Sensitivity and resistance parameters adjust to the patient's real-time performance, informed by an LLM-based reasoning layer (Cosmos Reason). |
| **Clinical Visibility** | A therapist-facing dashboard surfaces ROM progress, movement quality, and AI-generated session recommendations. |

This repository documents the project end-to-end — from Isaac Lab environment design through GR00T fine-tuning to the final gamified therapy application — as a public build log and reference implementation.

> **Project status:** Active development. See the [Roadmap](#roadmap) for current progress against the build plan.

---

## Architecture

RehabPlay is composed of five core components:

1. **Rehabilitation Environment** (`rehabplay/envs/`) — A manager-based Isaac Lab RL environment. A Franka Panda arm interacts with a simulated patient limb; reward terms encode ROM progress, force safety, and compensatory-movement penalties.
2. **Synthetic Data Pipeline** (`rehabplay/data/`) — Teleoperated demonstrations expanded via Isaac Lab Mimic and Cosmos Transfer into a diverse training corpus.
3. **Fine-Tuned Policy** (`rehabplay/policy/`) — Isaac GR00T fine-tuned on the synthetic corpus to map visual + language + proprioceptive input to safe, therapy-appropriate motor actions.
4. **Adaptive Game Interface** (`rehabplay/game/`) — Real-time control mapping (e.g., shoulder flexion → jump) with adaptive sensitivity.
5. **Therapist Dashboard** (`rehabplay/dashboard/`) — A Streamlit application surfacing session analytics, ROM trends, and AI-generated recommendations for the treating therapist.

```
Patient Movement → Isaac Sim Simulation → GR00T Policy → Game Engine → Score/Feedback
                                     ↓
                          Therapist Dashboard (Streamlit)
```

### Technology Stack

| Layer | Technology |
|---|---|
| Simulation | NVIDIA Isaac Sim 5.1, Isaac Lab 2.3 |
| Robot Learning | Isaac GR00T (Vision-Language-Action foundation model) |
| Synthetic Data | Isaac Lab Mimic, NVIDIA Cosmos Transfer |
| RL Training | PPO via skrl |
| Reasoning Layer | NVIDIA Cosmos Reason |
| Dashboard | Streamlit |
| Compute | NVIDIA Brev (cloud GPU orchestration) |

---

## Getting Started

### Prerequisites

- Python 3.10 (required by both Isaac Sim and Isaac GR00T — newer versions are not currently supported by either)
- An [NVIDIA Brev](https://brev.dev) account (or local workstation meeting [Isaac Sim's system requirements](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html))
- Git LFS (for pulling demonstration datasets)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/RehabPlay.git
cd RehabPlay

# Install dependencies
pip install -r requirements.txt
```

Detailed environment setup (Isaac Lab installation, Brev configuration) is documented in [`docs/setup.md`](docs/setup.md).

### Running the Environment

```bash
python scripts/train.py --task=RehabPlay-ShoulderFlexion-v0 --headless
```

Full usage instructions, including dashboard launch and GR00T inference, are documented in [`docs/usage.md`](docs/usage.md).

---

## Project Structure

```
RehabPlay/
├── rehabplay/
│   ├── envs/           # Isaac Lab manager-based RL environments
│   ├── data/           # Data collection, Mimic generation, format conversion
│   ├── policy/         # GR00T fine-tuning configs and checkpoints
│   ├── game/           # Gamified control interface
│   └── dashboard/      # Streamlit therapist dashboard
├── scripts/            # Training, evaluation, and utility entry points
├── configs/            # Modality configs, environment configs
├── docs/               # Setup guides, architecture notes, build log
├── tests/              # Unit and integration tests
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Roadmap

Development follows a structured build plan:

- [x] **Phase 1 — Foundations:** Isaac Lab environment validated on cloud GPU infrastructure; core API concepts documented.
- [ ] **Phase 2 — Rehabilitation Environment:** Custom manager-based environment with safety-constrained reward shaping.
- [ ] **Phase 3 — Synthetic Data & Fine-Tuning:** Demonstration collection, Mimic-based data generation, GR00T fine-tuning.
- [ ] **Phase 4+ — Application Layer:** Gamified interface and therapist dashboard integration.

Detailed technical progress notes are maintained in [`docs/build-log.md`](docs/build-log.md).

---

## Contributing

This is currently a solo research build, but issues, suggestions, and discussion are welcome. Please open an [issue](../../issues) before submitting a pull request to discuss proposed changes.

---

## Disclaimer

RehabPlay is a **research and simulation project**. It is not a certified medical device and is not intended for use in actual patient therapy without appropriate clinical validation, regulatory review, and supervision by licensed healthcare professionals.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Built on top of [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T).
