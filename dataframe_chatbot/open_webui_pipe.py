from typing import Any, Dict, Iterator, Optional

import httpx
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        FASTAPI_BASE_URL: str = Field(
            default="http://127.0.0.1:8000",
            description="Base URL for the FastAPI server.",
        )
        API_PATH: str = Field(
            default="/chat/dataframe-assistant",
            description="Path for the chat endpoint.",
        )

        # httpx timeouts
        CONNECT_TIMEOUT: float = Field(
            default=5.0, description="HTTP connect timeout (s)."
        )
        READ_TIMEOUT: Optional[float] = Field(
            default=None, description="HTTP read timeout (s). None = no limit."
        )
        WRITE_TIMEOUT: float = Field(
            default=10.0, description="HTTP write timeout (s)."
        )
        POOL_TIMEOUT: float = Field(
            default=5.0, description="HTTP connection pool timeout (s)."
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [
            {
                "id": "dataframe_assistant.pipe",
                "name": "Pandas DataFrame Chatbot",
            }
        ]

    # --- helpers -------------------------------------------------------------

    def _extract_last_user_message(self, body: Dict[str, Any]) -> str:
        msgs = body.get("messages") or []
        for m in reversed(msgs):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if (
                            isinstance(p, dict)
                            and p.get("type") == "text"
                            and isinstance(p.get("text"), str)
                        ):
                            parts.append(p["text"])
                    if parts:
                        return "\n".join(parts).strip()
        if isinstance(body.get("prompt"), str):
            return body["prompt"].strip()
        return ""

    def _iter_stream(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: httpx.Timeout,
    ) -> Iterator[str]:
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        # Surface a readable error in-stream
                        try:
                            err = resp.text
                        except Exception:
                            err = ""
                        yield f"API error {resp.status_code} at {url}: {err or 'No error body'}"
                        return
                    # Yield bytes as decoded text chunks
                    for chunk in resp.iter_bytes():
                        if chunk:
                            yield chunk.decode("utf-8", errors="replace")
        except httpx.HTTPError as e:
            yield f"HTTP error calling {url}: {e}"
        except Exception as e:
            yield f"Unhandled error calling {url}: {e}"

    # --- main entrypoint -----------------------------------------------------

    def pipe(self, body: Dict[str, Any], __metadata__: dict):
        base = (self.valves.FASTAPI_BASE_URL or "").rstrip("/")
        path = (self.valves.API_PATH or "").lstrip("/")
        url = f"{base}/{path}"

        user_msg = self._extract_last_user_message(body)
        if not user_msg:
            return "No prompt provided. Please type your question."

        chat_req = {"message": user_msg, "session_id": __metadata__["chat_id"]}

        headers = {"Content-Type": "application/json"}
        timeout = httpx.Timeout(
            connect=self.valves.CONNECT_TIMEOUT,
            read=self.valves.READ_TIMEOUT,
            write=self.valves.WRITE_TIMEOUT,
            pool=self.valves.POOL_TIMEOUT,
        )

        # Stream by default; if the UI sends stream=False, fall back to buffering.
        do_stream = body.get("stream", True)

        if do_stream:
            # Return an iterator/generator -> Open WebUI streams it to the chat
            return self._iter_stream(url, chat_req, headers, timeout)

        # Non-streaming fallback: buffer the whole response, then return
        full = ""
        for piece in self._iter_stream(url, chat_req, headers, timeout):
            full += piece
        return full.strip() or "(empty response)"
