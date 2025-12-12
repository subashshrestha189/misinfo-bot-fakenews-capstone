# Backend Architecture Overview

This backend powers the **Social Media Bot and Fake News Detection System**, providing APIs, configuration utilities, I/O helpers, and a lightweight testing interface. The architecture is modular, ensuring clarity, maintainability, and scalability across the whole project.

---

 # 1. `app.py` — Main FastAPI Application *(Developed in collaboration with Subash)*

`app.py` is the primary API server that integrates **bot detection**, **fake news classification**, and the **ensemble trust-scoring** system.

# Key Responsibilities

* Loads the complete ML stack (DistilBERT, Random Forest, label encoders)
* Provides three main inference endpoints:

  * `/analyze/article` → Fake news classification
  * `/analyze/user` → Bot probability + risk score
  * `/analyze/full` → Unified content + user + heuristic trust score
* Manages input validation, error handling, and standardized JSON output
* Serves as the main backend used by the Streamlit dashboard

# Collaboration Note

The development of this file was a **joint effort between Laxman Neupane and Subash Shrestha**, ensuring backend functionality aligned perfectly with frontend and model requirements.

---

## 📌 2. `config.py` — Central Configuration Module

This module centralizes environment settings and shared constants, keeping configuration clean and maintainable.

### What It Handles

* Defines `API_BASE` with support for environment overrides
* Provides `ensure_dir()` for reliable directory creation
* Eliminates repeated hardcoded paths across files

---

## 📌 3. `utils_io.py` — I/O & File Handling Utilities

A small but essential helper ensuring consistent file and path operations throughout the backend.

### Core Uses

* Safe directory creation
* Simplified path management
* Avoids duplicated file-handling logic in multiple scripts

This keeps backend scripts cleaner and reduces error risks.

---

## 📌 4. `mini_app.py` — Lightweight Local Testing API

A minimal FastAPI instance designed for rapid testing of model components.

### Why It Exists

* Enables quick debugging without running the entire system
* Useful for testing preprocessing, BERT inference, or bot features independently
* Supports fast iteration during development

---

## 🧠 Summary of Backend Structure

| File            | Purpose                                                                      |
| --------------- | ---------------------------------------------------------------------------- |
| **app.py**      | Full inference API (bot + fake news + trust score) — *developed with Subash* |
| **config.py**   | Environment configuration + utility helpers                                  |
| **utils_io.py** | Modular file/directory handling utilities                                    |
| **mini_app.py** | Lightweight inference tester                                                 |

This backend follows industry principles:
✔ modular design
✔ no duplicated logic
✔ scalable structure
✔ clean integration with Streamlit frontend
✔ easy debugging and maintainability

---

If you want, I can also prepare a full README.md including:
📌 setup instructions
📌 environment installation
📌 sample API requests
📌 architecture diagram
📌 contributor credits

Just tell me!
