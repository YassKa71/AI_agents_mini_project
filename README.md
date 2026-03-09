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


# Features

- Modular **environment simulation based on datasets**
- Simple **linear-memory AI agent architecture**
- Utilities for interacting with **Small Language Models**
- Example **Supervised Fine-Tuning (SFT)** workflow
- Designed to run easily in **Google Colab**
- Lightweight dependency management using **uv**


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


# Components

## Environments

The `environments` module provides tools to simulate agent environments based on datasets.

### datasets

Contains datasets used to simulate environments for training and evaluating agents.

Example included:

- **GTA_dataset**

This dataset can be used for **fine-tuning agents or simulating interactions**.

Datasets may contain:

- raw dataset files
- structured environment data

---

### datahandling

The `datahandling` module contains utilities for **loading and processing datasets**.

Responsibilities include:

- extracting relevant information from datasets
- preparing data for the environment simulator
- serving data to agent classes when required

---

### environment_simulation.py

Implements a class to **simulate an environment for the agent**.

This file:

- provides observations to agents
- validate agent actions

---

# AI Agents

The `ai_agents` module contains implementations of AI agents.

Currently implemented:

## Linear Memory Agent

Location:
ai_agents/monoagent/linear_agent.py


This agent implements a **simple linear memory architecture**, where:

- interactions are stored sequentially
- past observations and actions can be used as context
- prompts are dynamically generated

The agent interacts with a **Small Language Model (SLM)** to determine its next action.

---

## Prompts

Located in:
ai_agents/monoagent/prompts/


This directory contains prompt templates used by agents to communicate with the language model.

---

# Experiments

The `experiments` directory contains a dummy notebook file demonstrating how to use the framework in Colab.

Example included:
experiments/dummy_experiment/dummy_experiment.ipynb


This dummy notebook demonstrates:

- loading the dataset
- performing **Supervised Fine-Tuning (SFT)** using the GTA dataset

The notebook is designed to run easily in **Google Colab**.

---

# Utilities

The `utils` module contains helper functions used across the project.

File:

utils/utils.py

---

# Requirements

- Python **>= 3.11**
- [uv](https://github.com/astral-sh/uv)

`uv` is used for dependency management and reproducible environments.

---

## Planned Improvements

Planned improvements include:
- Refactoring to improve code robustness (eg. adding more schema validation for dataset_provider)
- additional agent architectures
- multi-agent environments
- improved dataset interfaces
- evaluation benchmarks
- Colab experiments
- Containerizing the project using Docker for easier deployment

---

## Contributing

Contributions, ideas, and experiments are welcome.

You can contribute by:

- adding new agents
- adding new datasets
- improving the environment simulator
- providing new experiment notebooks

---

## License

This project is open source.  
License information will be added soon.