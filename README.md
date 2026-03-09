# AI Agents Mini Project

# Status
Work in Progress

This project is currently under active development. 
Core functionality is implemented, but several features 
and optimizations are still being added.


# Project Goal
This repository provides a lightweight research environment for experimenting with **AI agents based on Small Language Models (SLMs)**.

The project is designed primarily for **students and researchers who only have access to Google Colab or limited compute**, enabling them to prototype agent architectures, run experiments, and fine-tune small models using curated datasets.

The repository includes:

- A **dataset-driven simulated environment**
- A **modular agent implementation**
- Utilities for **working with SLM outputs**
- Example **Colab-compatible experiments**

The goal is to make **AI agent research accessible with minimal infrastructure requirements**.


# Repository Structure
```
research/
│
├── environments/
│ ├── datasets/
│ │ └── GTA_dataset/
│ │   ├── raw/
│ │   └── ...
│ │
│ ├── datahandling/
│ │ └── dataset_handler.py
│ │
│ └── environment_simulation.py
│
├── ai_agents/
│ └── monoagent/
│   ├── linear_agent.py
│   └── prompts/
│
├── experiments/
│ └── dummy_experiment/
│   └── dummy_experiment.ipynb
│
├── utils/
│ └── utils.py
│
└── tests/
  └──...
```


# Core Components

# Environments

The `environments` module provides tools to simulate agent environments based on datasets.


# AI Agents

The `ai_agents` module contains implementations of AI agents.

Currently implemented:

## Linear Memory Agent

Location:
ai_agents/monoagent/linear_agent.py


# Experiments

The `experiments` directory is intended for small experiments on ai agents. For now it contains a dummy notebook file demonstrating how to use the framework in Colab.


# Requirements

- Python **>= 3.11**
- [uv](https://github.com/astral-sh/uv)

`uv` is used for dependency management and reproducible environments.


## Planned Improvements

Planned improvements include:
- Refactoring to improve code robustness (eg. adding more schema validation for dataset_provider)
- additional agent architectures
- multi-agent environments
- improved dataset interfaces
- evaluation benchmarks
- Colab experiments
- Containerizing the project using Docker for easier deployment
