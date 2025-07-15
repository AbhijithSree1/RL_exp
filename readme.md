# 🔤 Spinning Up: Modernized PyTorch Edition

**Author**: MoniGarr  
**Branch**: `monigarr-dev`  
**Status**: Stable, updated, and fully operational with modern deep RL libraries.

---

## 🌟 What It Is
A fully modernized and operational fork of [OpenAI's Spinning Up](https://github.com/openai/spinningup), now compatible with current Python (3.8+), PyTorch, Gymnasium, and MuJoCo ecosystems.

This repository restores the value of one of the most respected educational toolkits for deep reinforcement learning (RL) and makes it accessible again to learners, researchers, and developers working in today's environments.

---

## 💡 Why This Matters
OpenAI's original Spinning Up is a brilliant learning resource, but its dependencies are outdated and unusable with modern toolchains *out of the box*  (7/15/2025).

This project revitalizes Spinning Up:
- Updated for modern library compatibility
- Swapped deprecated packages for maintained successors
- Fixed breaking changes in APIs (Gym ➔ Gymnasium)
- Modular and reproducible with Conda + Pip
- Reinforces accessible RL education for diverse researchers

This version is now usable *out of the box* on most modern systems.

---

## 🚀 Quickstart: Install & Run

### 1. Clone the Repository
```bash
git clone https://github.com/monigarr/spinningup.git
cd spinningup
```

### 2. Create & Activate Conda Environment
```bash
conda create -n spinningup-py38 python=3.8
conda activate spinningup-py38
```

### 3. Install Core Dependencies (Conda First)
```bash
conda install -c conda-forge swig msmpi
```

### 4. Install Python Packages
```bash
pip install "gymnasium[mujoco]"
pip install -e .
```

### 5. Run a PPO Agent
```bash
python -m spinup.run ppo --env LunarLander-v2 --exp_name test-ppo-lander
```
You should see training logs and results in `./data/test-ppo-lander/`

---

## 🏠 Architecture Overview

### Core Algorithms
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin-Delayed DDPG)
- DDPG (Deep Deterministic Policy Gradient)
- VPG (Vanilla Policy Gradient)

### Neural Network Design
- Configurable MLPs for actor and critic
- Clean modular structure
- Easy experimentation with policy/value network changes

### Updated Tech Stack
| Component | Modern Version Used |
|----------|----------------------|
| Python   | 3.8                 |
| RL API   | gymnasium           |
| Deep Learning | PyTorch        |
| Physics Engine | mujoco (pip version) |

---

## 📊 Sample Results

Training PPO on Walker2d-v4 after modernization:
```
|      AverageEpRet |          2431.2 |
|          MaxEpRet |          3979.1 |
|          MinEpRet |           432.9 |
|      AverageVVals |          -8.409 |
|           Entropy |           8.524 |
|           EpLen   |           1000  |
|           Time    |         208.71s |
```
To generate your own performance plots:
```bash
python -m spinup.run plot data/test-ppo-lander --savefig ppo_plot.png
```

---

## 🔎 MoniGarr's Motivation
As an Onkwehonwe technologist and AI developer, I’m committed to building bridges between ancient knowledge systems and modern AI research.

This project is part of my broader mission to:
- Make advanced RL research tools usable again for new generations of learners
- Prepare for deeper AI residency-level research work
- Create more accessible and inclusive ML learning pipelines

By modernizing Spinning Up, I’m contributing to an open-source foundation for applied and ethical intelligence research.

---

## 🗺️ Next Steps
- ✅ Extend PPO with attention or recurrent layers
- ✅ Add performance benchmarking for all core algorithms
- ✅ Prepare companion blog series with lessons learned and code walkthroughs
- ✅ Integrate with small research projects (e.g. Kanien’kéha language RL envs)

Follow my journey: [https://github.com/monigarr](https://github.com/monigarr)

---

## 🚑 Contributing & License
This is a research fork for educational purposes. You’re welcome to fork, contribute, or share ideas.

Original source: [OpenAI Spinning Up](https://github.com/openai/spinningup)  
Fork Author: MoniGarr

License: MIT
