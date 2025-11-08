# gpt-oss-vllm-rag

This repository offers a **minimal example** of building a **GPT-OSS-powered RAG chatbot** using:
* [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration
* [Langfuse](https://github.com/langfuse/langfuse) for observability
* [vLLM](https://github.com/vllm-project/vllm) for local LLM inference.

## ⚙️ Setup

1. **Create `.env` file** at the root of the project and add the following  environment variables:

```bash
LANGFUSE_PUBLIC_KEY=<your-public-key>
LANGFUSE_SECRET_KEY=<your-secret-key>
LANGFUSE_BASE_URL=http://localhost:3000
```

2. **Install dependencies** with Poetry (including development tools):

```bash
poetry install --with dev
```

3. **Serve the models** locally using vLLM:

### 🔹 GPT-OSS-20B

```bash
vllm serve openai/gpt-oss-20b --port 3001 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --async-scheduling \
  --no-enable-prefix-caching \
  --max-model-len 8192 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.5 \
  --cuda-graph-sizes 1024 \
  --compilation-config '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

### 🔹 Nomic Embed Text v1.5

```bash
vllm serve nomic-ai/nomic-embed-text-v1.5 --port 3002 \
  --trust-remote-code \
  --gpu-memory-utilization 0.1 \
  --max-num-seqs 64
```

## 💬 Testing the Chatbot

To explore the full workflow, open the Jupyter notebook:
```
notebooks/Demo.ipynb
```
It contains an **end-to-end example** demonstrating how to run and test the RAG chatbot locally.