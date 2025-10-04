# dataframe-chatbot

This repository is a toy example of how to build a chatbot application able to interact with pandas dataframes using `LangGraph` and `Ollama`.

## Setup

1. Run ```poetry install --with dev``` to install project's dependencies.
2. Serve LLMs via Ollama by completing the following steps:

**Step 1:** Pull Ollama model for the chatbot, e.g.
```bash
ollama pull gpt-oss:20b
```

**Step 2:** Preload "gpt-oss:20b" and leave it in memory use:
```bash
curl http://localhost:11434/api/generate -d '{"model": "gpt-oss:20b", "keep_alive": -1, "options": {"num_ctx": 4096}}'
```

## Start the API

```bash
uvicorn src.api.app.main:app --host 0.0.0.0 --port 8000 --reload
```
