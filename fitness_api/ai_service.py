"""
AI Service with multiple API key fallback support.

Supports Gemini and OpenRouter APIs with automatic fallback mechanism.
Keys are read from .env file and tried sequentially until one succeeds.
"""

import json
import logging
import multiprocessing
import os
import re
import threading
import time
from queue import Empty
from typing import Iterator, List, Optional

import google.genai as genai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
_ai_service_lock = threading.Lock()


def _gemini_process_worker(
    key: str, model: str, prompt: str, response_queue: multiprocessing.Queue
):
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = getattr(response, "text", None) or getattr(response, "content", None)
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        response_queue.put({"text": str(text)})
    except Exception as exc:
        response_queue.put({"error": str(exc)})


class AIService:
    """
    AI Service that supports multiple API keys with automatic fallback.

    Attempts to use OpenRouter keys first, then Gemini keys if OpenRouter fails.
    Returns a static fallback response if all API calls fail.
    """

    # Configuration constants
    GEMINI_KEY_TIMEOUT = 10  # seconds
    OPENROUTER_KEY_TIMEOUT = 10  # seconds
    STATIC_FALLBACK_RESPONSE = (
        "I'm currently experiencing technical difficulties with my AI service. "
        "Please try again later or contact support if the problem persists."
    )

    # HTTP error codes that indicate a failed key
    FAILURE_HTTP_CODES = {401, 429, 500, 503}

    def __init__(self):
        """Initialize AIService and load API keys from .env file."""
        self.gemini_keys = self._load_gemini_keys()
        self.openrouter_keys = self._load_openrouter_keys()
        self.gemini_model = (
            os.getenv("GENAI_MODEL", "models/gemini-2.0-flash").strip()
            or "models/gemini-2.0-flash"
        )
        self.openrouter_model = (
            os.getenv("OPENROUTER_MODEL", "openrouter/auto").strip()
            or "openrouter/auto"
        )
        self.system_prompt = os.getenv("AI_SYSTEM_PROMPT", "").strip()
        self.current_key_index = 0
        self.current_provider = None
        self._state_lock = threading.Lock()

        logger.info(
            f"AIService initialized with {len(self.gemini_keys)} Gemini keys "
            f"and {len(self.openrouter_keys)} OpenRouter keys"
        )

    def _set_current_key_info(self, provider: str, index: int):
        with self._state_lock:
            self.current_provider = provider
            self.current_key_index = index

    def get_last_used_key_info(self) -> dict:
        """
        Get information about the last successfully used API key.

        Returns:
            Dictionary with provider and key index information.
        """
        with self._state_lock:
            return {
                "provider": self.current_provider,
                "key_index": (
                    self.current_key_index + 1 if self.current_provider else None
                ),
            }

    def _load_gemini_keys(self) -> List[str]:
        """
        Load Gemini API keys from environment variables.

        Returns:
            List of non-empty Gemini API keys in order.
        """
        keys = []
        for i in range(1, 20):  # GEMINI_KEY_1 to GEMINI_KEY_19
            key = os.getenv(f"GEMINI_KEY_{i}", "").strip()
            if key:
                keys.append(key)
        return keys

    def _load_openrouter_keys(self) -> List[str]:
        """
        Load OpenRouter API keys from environment variables.

        Returns:
            List of non-empty OpenRouter API keys in order.
        """
        keys = []
        for i in range(1, 20):  # OPENROUTER_KEY_1 to OPENROUTER_KEY_19
            key = os.getenv(f"OPENROUTER_KEY_{i}", "").strip()
            if key:
                keys.append(key)
        return keys

    def sanitize_response(self, text: str) -> str:
        """Sanitize AI response to remove unsafe control characters only.

        Accepts bytes, strings, lists, or dictionaries.
        Decodes raw bytes as UTF-8 and serializes JSON-like objects to text to
        prevent mojibake or unexpected response shapes from provider responses.
        """
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False, indent=2)
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        # Remove dangerous control characters while keeping newlines/tabs
        # (\x09/\x0a/\x0d).
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", cleaned)
        return cleaned.strip()

    def _build_prompt(self, prompt: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{prompt}"
        return prompt

    def _build_openrouter_messages(self, prompt: str) -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def get_response(self, prompt: str) -> str:
        """
        Get AI response for the given prompt using fallback mechanism.

        Tries Gemini keys first, then OpenRouter keys, then returns static fallback.

        Args:
            prompt: The user's prompt/question.

        Returns:
            AI-generated response or static fallback response if all APIs fail.
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to get_response")
            return "Please provide a valid prompt."

        # Try OpenRouter keys first, then Gemini keys.
        response = self._try_openrouter_keys(prompt)
        if response:
            return self.sanitize_response(response)

        # Try Gemini keys
        response = self._try_gemini_keys(prompt)
        if response:
            return self.sanitize_response(response)

        # All keys failed, return static fallback
        logger.error("All API keys failed. Returning static fallback response.")
        return self.STATIC_FALLBACK_RESPONSE

    def _decode_utf8(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if value is None:
            return ""
        return str(value)

    def _try_gemini_keys(self, prompt: str) -> Optional[str]:  # noqa: C901
        """
        Try Gemini API keys sequentially until one succeeds.

        Uses a separate process so the timeout is REAL — if Gemini hangs,
        the worker process is terminated and the next key is tried immediately.

        Args:
            prompt: The user's prompt.

        Returns:
            Response string if successful, None if all keys fail.
        """
        if not self.gemini_keys:
            logger.warning("No Gemini keys configured in environment")
            return None

        for index, key in enumerate(self.gemini_keys):
            try:
                logger.info(
                    f"Attempting Gemini key {index + 1}/{len(self.gemini_keys)}"
                )

                start_time = time.time()
                prompt_text = self._build_prompt(prompt)
                response_queue: multiprocessing.Queue = multiprocessing.Queue()
                proc = multiprocessing.Process(
                    target=_gemini_process_worker,
                    args=(key, self.gemini_model, prompt_text, response_queue),
                    daemon=True,
                )
                proc.start()
                proc.join(timeout=self.GEMINI_KEY_TIMEOUT)

                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    logger.warning(
                        f"Gemini key {index + 1} REAL timeout after "
                        f"{self.GEMINI_KEY_TIMEOUT}s — terminating process and "
                        "skipping to next key"
                    )
                    continue

                try:
                    result = response_queue.get_nowait()
                except Empty:
                    logger.warning(f"Gemini key {index + 1} returned no response")
                    continue

                if "error" in result:
                    logger.warning(f"Gemini key {index + 1} failed: {result['error']}")
                    continue

                response_text = self._decode_utf8(result.get("text", ""))
                elapsed_time = time.time() - start_time

                if not response_text.strip():
                    logger.warning(f"Gemini key {index + 1} returned empty response")
                    continue

                logger.info(
                    f"Gemini key {index + 1} succeeded in {elapsed_time:.2f}s "
                    f"(response: {len(response_text)} chars)"
                )
                self._set_current_key_info("gemini", index)
                return response_text.strip()

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Gemini key {index + 1} failed: {type(e).__name__}: {error_msg}"
                )
                if "401" in error_msg or "authentication" in error_msg.lower():
                    logger.warning(
                        f"Gemini key {index + 1}: Authentication failed (401)"
                    )
                elif "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(f"Gemini key {index + 1}: Quota exceeded (429)")
                elif "500" in error_msg or "503" in error_msg:
                    logger.warning(f"Gemini key {index + 1}: Server error")
                continue

        logger.warning("All Gemini keys exhausted")
        return None

    def _try_openrouter_keys(self, prompt: str) -> Optional[str]:  # noqa: C901
        """
        Try OpenRouter API keys sequentially until one succeeds.

        Args:
            prompt: The user's prompt.

        Returns:
            Response string if successful, None if all keys fail.
        """
        if not self.openrouter_keys:
            logger.warning("No OpenRouter keys configured in environment")
            return None

        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        for index, key in enumerate(self.openrouter_keys):
            try:
                logger.info(
                    f"Attempting OpenRouter key {index + 1}/{len(self.openrouter_keys)}"
                )

                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "HTTP-Referer": "https://gymassistant.local",
                }

                payload = {
                    "model": self.openrouter_model,
                    "messages": self._build_openrouter_messages(prompt),
                    "temperature": 0.7,
                }

                start_time = time.time()
                response = requests.post(
                    openrouter_url,
                    json=payload,
                    headers=headers,
                    timeout=self.OPENROUTER_KEY_TIMEOUT,
                )
                elapsed_time = time.time() - start_time

                # Check HTTP status codes
                if response.status_code in self.FAILURE_HTTP_CODES:
                    logger.warning(
                        "/".join(
                            [
                                f"OpenRouter key {index + 1} failed with HTTP",
                                str(response.status_code),
                            ]
                        )
                    )
                    continue

                # Raise for other HTTP errors
                response.raise_for_status()

                # Parse response with explicit UTF-8 decoding to avoid mojibake.
                try:
                    raw_text = self._decode_utf8(response.content)
                    data = json.loads(raw_text)
                except ValueError as exc:
                    logger.warning(
                        f"OpenRouter key {index + 1} returned invalid JSON: {exc}"
                    )
                    continue

                # Check for error in response
                if "error" in data:
                    logger.warning(
                        f"OpenRouter key {index + 1} returned error: {data['error']}"
                    )
                    continue

                # Extract message content
                if (
                    "choices" in data
                    and len(data["choices"]) > 0
                    and "message" in data["choices"][0]
                    and "content" in data["choices"][0]["message"]
                ):
                    content = data["choices"][0]["message"]["content"].strip()

                    if not content:
                        logger.warning(
                            f"OpenRouter key {index + 1} returned empty content"
                        )
                        continue

                    logger.info(
                        f"OpenRouter key {index + 1} succeeded in {elapsed_time:.2f}s "
                        f"(response: {len(content)} chars)"
                    )
                    self.current_key_index = index
                    self.current_provider = "openrouter"
                    return content
                else:
                    logger.warning(
                        f"OpenRouter key {index + 1} response has unexpected format"
                    )
                    continue

            except requests.Timeout:
                logger.warning(
                    f"OpenRouter key {index + 1} timed out after "
                    f"{self.OPENROUTER_KEY_TIMEOUT}s"
                )
                continue
            except requests.RequestException as e:
                logger.warning(
                    "OpenRouter key %s network error: %s: %s",
                    index + 1,
                    type(e).__name__,
                    str(e),
                )
                continue
            except Exception as e:
                logger.warning(
                    f"OpenRouter key {index + 1} failed: {type(e).__name__}: {str(e)}"
                )
                continue

        logger.warning("All OpenRouter keys exhausted")
        return None

    def _iter_gemini_stream(self, prompt: str) -> Iterator[str]:
        for index, key in enumerate(self.gemini_keys):
            try:
                logger.info(
                    "Attempting Gemini streaming key %s/%s",
                    index + 1,
                    len(self.gemini_keys),
                )
                client = genai.Client(api_key=key)
                stream = client.models.generate_content_stream(
                    model=self.gemini_model,
                    contents=prompt,
                )
                emitted = False
                for chunk in stream:
                    if chunk is None:
                        continue
                    text = getattr(chunk, "text", chunk)
                    text = self._decode_utf8(text)
                    if text:
                        emitted = True
                        yield text
                if emitted:
                    self._set_current_key_info("gemini", index)
                    return
            except Exception as exc:
                logger.warning("Gemini streaming key %s failed: %s", index + 1, exc)

    def _iter_openrouter_stream(self, prompt: str) -> Iterator[str]:  # noqa: C901
        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        for index, key in enumerate(self.openrouter_keys):
            try:
                logger.info(
                    "Attempting OpenRouter streaming key %s/%s",
                    index + 1,
                    len(self.openrouter_keys),
                )
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "HTTP-Referer": "https://gymassistant.local",
                }
                payload = {
                    "model": self.openrouter_model,
                    "messages": self._build_openrouter_messages(prompt),
                    "temperature": 0.7,
                    "stream": True,
                }
                response = requests.post(
                    openrouter_url,
                    json=payload,
                    headers=headers,
                    timeout=self.OPENROUTER_KEY_TIMEOUT,
                    stream=True,
                )
                if response.status_code in self.FAILURE_HTTP_CODES:
                    continue

                response.raise_for_status()
                emitted = False
                for line in response.iter_lines(decode_unicode=False):
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line.startswith("data: "):
                        continue
                    data = self._decode_utf8(line[6:].strip())
                    if data == "[DONE]":
                        break
                    try:
                        payload_data = json.loads(data)
                    except ValueError:
                        continue
                    delta = (
                        payload_data.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        emitted = True
                        yield delta
                if emitted:
                    self._set_current_key_info("openrouter", index)
                    return
            except Exception as exc:
                logger.warning("OpenRouter streaming key %s failed: %s", index + 1, exc)

    def _chunk_text(self, text: str, chunk_size: int = 24) -> Iterator[str]:
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]

    def stream_response(self, prompt: str) -> Iterator[str]:
        """
        Stream AI response chunks with provider fallback.
        Falls back to chunked static response if all providers fail.
        """
        if not prompt or not prompt.strip():
            yield "يرجى إدخال سؤال صالح."
            return

        for stream_provider in (self._iter_openrouter_stream, self._iter_gemini_stream):
            emitted = False
            try:
                for chunk in stream_provider(prompt):
                    emitted = True
                    yield self._decode_utf8(chunk)
            except Exception as exc:
                logger.warning("Streaming provider failed: %s", exc)
                emitted = False
            if emitted:
                return

        logger.error("All streaming providers failed. Using chunked fallback.")
        for chunk in self._chunk_text(self.STATIC_FALLBACK_RESPONSE):
            yield chunk


_ai_service_instance = None


def get_ai_service() -> AIService:
    """
    Get singleton instance of AIService.

    Returns:
        AIService instance.
    """
    global _ai_service_instance
    with _ai_service_lock:
        if _ai_service_instance is None:
            _ai_service_instance = AIService()
    return _ai_service_instance
