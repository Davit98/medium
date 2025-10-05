# dataframe-chatbot

This repository provides a minimal example of how to build a chatbot capable of interacting with pandas DataFrames using LangGraph for agent orchestration and Ollama for local LLM inference.

## Setup

1. Run ```poetry install --with dev``` to install project's dependencies.
2. Start Ollama and serve your LLM:

**Step 1:** Pull a compatible Ollama model, for example:
```bash
ollama pull gpt-oss:20b
```

**Step 2:** Preload the model and keep it in memory:
```bash
curl http://localhost:11434/api/generate -d '{"model": "gpt-oss:20b", "keep_alive": -1, "options": {"num_ctx": 4096}}'
```

This ensures the model remains active in memory for faster inference during development.

## Start the API

Once the dependencies and model are ready, start the FastAPI application:

```bash
uvicorn src.api.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at http://127.0.0.1:8000, ready to serve requests from your chatbot interface.