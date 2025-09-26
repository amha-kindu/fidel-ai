# Fidel AI

This project hosts a **full-stack AI serving & UI** system, combining a FastAPI backend for a custom GPT model with an Open WebUI frontend. It is designed to let you deploy your model and a conversational UI in a containerized, production-style fashion.

---

## 🚀 Project Summary

`fidel-ai` packages together:

- A **FastAPI-based API** exposing an OpenAI-compatible interface (`/v1/models`, `/v1/chat/completions`) with streaming support (SSE).
- A **Open WebUI frontend** container configured to communicate with that backend out of the box (via environment variables).
- Docker and Docker Compose setups (CPU and GPU) for easy deployment.
- Environment-driven customization (CORS, UI banners, prompt suggestions, etc.).
- Health-checking, persistent config, and multi-service orchestration via `docker-compose`.

In essence, `fidel-ai` is the integration glue between your model API and a ready-to-use web UI, all packaged as a deployable stack.

---

## 🏗 Components & Design

### Backend API (FastAPI + PyTorch)

- Implements OpenAI-style endpoints (`/v1/models`, `/v1/chat/completions`) with both **streaming** and **non-streaming** modes.
- Uses an inference lock or similar mechanism to serialize access to GPU during generation.
- Supports prompt rendering, stop‐token logic, and sampling (temperature, top-p, top-k).
- CORS middleware to allow UI origin(s) to access the API.
- Health check endpoint (`/health`) for service monitoring.
- Configured via environment variables (e.g. `MODEL_ID`, `ALLOW_ORIGINS`, `AMP`, etc).

### Frontend (Open WebUI)

- Runs in its own container, pointed to the backend via:
  - `OPENAI_API_BASE_URL` environment variable (e.g. `http://api:8000/v1`)
  - `OPENAI_API_KEY` (dummy or real, depending on your auth)
  - Optional UI customization env vars: banner messages, prompt suggestions, app name, locale defaults, etc.
- Uses persistent configuration by default; initial env config is loaded once and then stored internally.

### Dockerization & Orchestration

- Two Dockerfiles: **CPU** and **CUDA** (for GPU) builds.
- `docker/entrypoint.sh` to bootstrap the backend service.
- `docker/gunicorn_conf.py` containing worker count, keepalive tuning, logging setup.
- `docker-compose.yml` defining two services:
  1. `api` — your model server
  2. `webui` — the Open WebUI frontend, depends on `api`
- Shared `networks` so the WebUI resolves `api` by hostname inside the container network.
- Health checks for the `api` service so WebUI waits until the backend is ready.
- Volume mount for persistent WebUI data (`webui_data`) so UI customizations persist across restarts.

---

## ⚙ Environment Variables & Configuration (UI & Backend)

Some of the key env variables you can set include:

| Variable | Purpose |
|---|---|
| `MODEL_ID` | Identifier used by the backend for the model (sent in response) |
| `ALLOW_ORIGINS` | CORS origin(s) allowed for UI access |
| `AMP` | Whether to use automatic mixed precision (GPU) |
| `OPENAI_API_BASE_URL` | URL the WebUI uses to call your backend |
| `OPENAI_API_KEY` | API key passed by the UI (may be ignored by backend) |
| `WEBUI_NAME` | Custom application name shown in the UI |
| `DEFAULT_LOCALE` | Default UI language / localization code |
| `WEBUI_BANNERS` | JSON array of banner messages for UI (welcome, notices) |
| `DEFAULT_PROMPT_SUGGESTIONS` | JSON array of suggested prompts/cards on New Chat screen |
| `PENDING_USER_OVERLAY_TITLE`, `PENDING_USER_OVERLAY_CONTENT` | Text for overlay if user approval is required |
| `RESPONSE_WATERMARK` | Text appended/copied along with generated responses |
| `ENABLE_PERSISTENT_CONFIG` | Toggle whether UI persists settings (when disabled, env vars override) |

Because Open WebUI uses **PersistentConfig**, env values are usually only applied on first startup (unless you disable persistence).  

---

## 💡 Running & Deployment

1. Copy `.env.example` to `.env` and fill in your values (model ID, CORS origins, etc.).
2. Choose your Dockerfile: use `docker-compose.yml` by default (CPU build) or switch to CUDA/dockerfile.cuda if GPU.
3. Launch:
   ```bash
   docker-compose up --build


4. Access services:

   * API: `http://localhost:8000/health`, `/v1/chat/completions`
   * WebUI: `http://localhost:3000`

---

## 🛠 Extending the Stack

* **Swap in your model**: in the backend container, inject your tokenizer, model loading code, and generation logic.
* **Customize UI texts or experience**: use the env vars above (banners, prompt suggestions, UI name).
* **Add authentication or rate limiting**: decorate endpoints or proxy them as needed.
* **Scale horizontally**: for CPU inference, you can run multiple backend replicas and route through a load balancer; for GPU, usually one worker per GPU.
* **Enable logging / metrics**: add structured logs, Prometheus exporters, or tracing.

---
