
"""
htk_backends.py — Universal backends for Hallucination Toolkit (EDFL/B2T/ISR)

Drop-in adapters for:
- Anthropic (Claude Messages API)
- Hugging Face (Transformers, TGI, or HF Inference API)
- Ollama (local HTTP or python package)
- OpenRouter (OpenAI-compatible aggregator: https://openrouter.ai/)

They are designed to work with the original `hallucination_toolkit.py` **without editing it**:
just pass any of these backends wherever the toolkit expects `OpenAIBackend`.

Each backend implements:
  - chat_create(messages: List[Dict], **kwargs) -> backend-native response (opaque)
  - multi_choice(messages: List[Dict], n: int = 1, **kwargs) -> List[ChoiceLike]
Where a ChoiceLike is a simple object exposing `.message.content: str`

Messages contract (same as the original toolkit):
  messages = [{"role": "system", "content": "..."},
              {"role": "user",   "content": "..."}]

The prompts already instruct the model to return a tiny JSON object:
  {"decision":"answer"}  OR  {"decision":"refuse"}

Notes
-----
- These adapters do **not** depend on the original OpenAI SDK. Optional dependencies are imported at runtime.
- For Hugging Face:
    * mode="transformers" uses local models via `transformers` pipeline
    * mode="tgi" talks to a Text Generation Inference server via HTTP
    * mode="inference_api" calls the hosted HF Inference API (requires HF token)
  For chat-tuned models, if the tokenizer has a chat template, we use it; otherwise we fall back
  to a simple "System:\n.. \nUser:\n..\nAssistant:" prompt.
- For Ollama:
    * If the `ollama` python package is available, we use it. Otherwise we fall back to the HTTP API.
- For Anthropic:
    * Requires `anthropic`>=0.28 installed and `ANTHROPIC_API_KEY` set in the environment.
- For OpenRouter:
    * Uses the OpenAI-compatible Chat Completions endpoint at https://openrouter.ai/api/v1/chat/completions
      with header Authorization: Bearer <OPENROUTER_API_KEY>. Optional headers: HTTP-Referer, X-Title.

MIT License — see original toolkit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helper: a "choice-like" object to match OpenAI-style shape (message.content)
# ---------------------------------------------------------------------------

@dataclass
class _ChoiceLikeMessage:
    content: str

@dataclass
class _ChoiceLike:
    message: _ChoiceLikeMessage


def _as_choices(texts: List[str]) -> List[_ChoiceLike]:
    return [_ChoiceLike(_ChoiceLikeMessage(content=t)) for t in texts]


def _split_system_user(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Extract the (single) system and user strings from toolkit-style messages."""
    system = ""
    user = ""
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system":
            system = str(m.get("content") or "")
        elif role == "user":
            user = str(m.get("content") or "")
    return {"system": system, "user": user}


# ---------------------------------------------------------------------------
# Anthropic (Claude) backend
# ---------------------------------------------------------------------------

class AnthropicBackend:
    """
    Adapter for Anthropic Messages API.
    Usage:
        backend = AnthropicBackend(model="claude-3-5-sonnet-latest")
        choices = backend.multi_choice(messages, n=3, temperature=0.5, max_tokens=16)
    """
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
    ) -> None:
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise ImportError("Install `anthropic>=0.28` to use AnthropicBackend.") from e

        self._anthropic = anthropic
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.request_timeout = float(request_timeout)

    def chat_create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        parts = _split_system_user(messages)
        system = parts["system"]
        user = parts["user"]
        max_tokens = int(kwargs.get("max_tokens", kwargs.get("max_tokens_to_sample", 16)))
        temperature = float(kwargs.get("temperature", 0.5))

        resp = self.client.messages.create(
            model=self.model,
            system=system or None,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp

    def multi_choice(self, messages: List[Dict[str, str]], n: int = 1, **kwargs: Any) -> List[_ChoiceLike]:
        out_texts: List[str] = []
        for _ in range(n):
            resp = self.chat_create(messages, **kwargs)
            # Extract text from content blocks
            text = ""
            try:
                blocks = getattr(resp, "content", []) or []
                for b in blocks:
                    if getattr(b, "type", "") == "text":
                        text += getattr(b, "text", "")
            except Exception:
                pass
            out_texts.append(text)
        return _as_choices(out_texts)


# ---------------------------------------------------------------------------
# Hugging Face backend (transformers / TGI / Inference API)
# ---------------------------------------------------------------------------

class HuggingFaceBackend:
    """
    Flexible adapter for Hugging Face models.

    Modes
    -----
    mode="transformers"   -> local generation via `transformers.pipeline`
    mode="tgi"            -> HTTP to Text Generation Inference server (URL required)
    mode="inference_api"  -> hosted Inference API (HF token required)

    Examples
    --------
    # Local transformers (best for speed/control)
    hf_local = HuggingFaceBackend(
        mode="transformers", model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto"
    )

    # TGI server (e.g., http://localhost:8080)
    hf_tgi = HuggingFaceBackend(mode="tgi", tgi_url="http://localhost:8080", model_id=None)

    # Hosted Inference API
    hf_api = HuggingFaceBackend(mode="inference_api", model_id="mistralai/Mistral-7B-Instruct-v0.3",
                                hf_token=os.environ.get("HF_TOKEN"))
    """
    def __init__(
        self,
        mode: str = "transformers",
        model_id: Optional[str] = None,
        # transformers mode
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
        # tgi mode
        tgi_url: Optional[str] = None,
        # inference_api mode
        hf_token: Optional[str] = None,
        request_timeout: float = 60.0,
    ) -> None:
        mode = mode.lower().strip()
        if mode not in {"transformers", "tgi", "inference_api"}:
            raise ValueError("mode must be one of: 'transformers', 'tgi', 'inference_api'")
        self.mode = mode
        self.model_id = model_id
        self.device_map = device_map
        self.trust_remote_code = bool(trust_remote_code)
        self.model_kwargs = model_kwargs or {}
        self.tgi_url = tgi_url
        self.hf_token = hf_token
        self.request_timeout = float(request_timeout)

        self._init_clients(torch_dtype)

    # -- internal helpers --

    def _init_clients(self, torch_dtype: Optional[str]) -> None:
        self._req = None
        if self.mode == "transformers":
            try:
                from transformers import AutoTokenizer, pipeline  # type: ignore
            except Exception as e:
                raise ImportError("Install `transformers` to use HuggingFaceBackend(mode='transformers').") from e
            if not self.model_id:
                raise ValueError("Provide `model_id` for transformers mode.")
            # Build tokenizer & generator lazily
            self._AutoTokenizer = AutoTokenizer
            self._pipeline = pipeline
            self._generator = None
            self._tokenizer = None
            self._torch_dtype = torch_dtype
        else:
            try:
                import requests  # type: ignore
            except Exception as e:
                raise ImportError("Install `requests` to use HuggingFaceBackend in HTTP modes.") from e
            self._req = requests
            if self.mode == "tgi":
                if not self.tgi_url:
                    raise ValueError("Provide `tgi_url` for TGI mode (e.g., http://localhost:8080).")
            elif self.mode == "inference_api":
                if not self.model_id:
                    raise ValueError("Provide `model_id` for inference_api mode.")
                if not self.hf_token:
                    raise RuntimeError("HF token required for inference_api mode (set `hf_token` or env HF_TOKEN).")

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = _split_system_user(messages)
        system = parts["system"]
        user = parts["user"]

        # For transformers mode: try chat template
        if self.mode == "transformers":
            # Lazy init tokenizer/generator
            if self._tokenizer is None:
                self._tokenizer = self._AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
            tok = self._tokenizer
            try:
                if hasattr(tok, "apply_chat_template") and callable(getattr(tok, "apply_chat_template")) and getattr(tok, "chat_template", None):
                    chat = []
                    if system:
                        chat.append({"role": "system", "content": system})
                    chat.append({"role": "user", "content": user})
                    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass  # fall through to plain text

        # Fallback: plain text chat framing
        prompt = ""
        if system:
            prompt += f"System:\n{system}\n\n"
        prompt += f"User:\n{user}\n\nAssistant:"
        return prompt

    # -- API surface --

    def chat_create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        # Provided for compatibility; not used by the toolkit directly.
        outs = self.multi_choice(messages, n=1, **kwargs)
        return outs[0]

    def multi_choice(self, messages: List[Dict[str, str]], n: int = 1, **kwargs: Any) -> List[_ChoiceLike]:
        temperature = float(kwargs.get("temperature", 0.5))
        max_new_tokens = int(kwargs.get("max_tokens", 16))  # map to generation cap
        prompt = self._format_prompt(messages)

        texts: List[str] = []

        if self.mode == "transformers":
            # Lazy init generator
            if self._generator is None:
                from transformers import AutoModelForCausalLM  # type: ignore  # noqa: F401
                tok = self._tokenizer or self._AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
                self._tokenizer = tok
                pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
                # Give user a robust default; allow overrides via model_kwargs
                gen_kwargs = dict(
                    model=self.model_id,
                    tokenizer=tok,
                    device_map=self.device_map or "auto",
                    model_kwargs=self.model_kwargs,
                    trust_remote_code=self.trust_remote_code,
                )
                self._generator = self._pipeline("text-generation", **gen_kwargs)
                self._pad_token_id = pad_id

            gen = self._generator
            try:
                outputs = gen(
                    prompt,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=n,
                    return_full_text=False,
                    pad_token_id=self._pad_token_id,
                )
                for o in outputs:
                    t = o.get("generated_text", "")
                    texts.append(t)
            except Exception as e:
                raise RuntimeError(f"Transformers generation error: {e}")

        elif self.mode == "tgi":
            assert self._req is not None
            url = self.tgi_url.rstrip("/") + "/generate"
            for _ in range(n):
                r = self._req.post(
                    url,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "do_sample": True,
                            "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                    timeout=self.request_timeout,
                )
                r.raise_for_status()
                data = r.json()
                t = data.get("generated_text") or data.get("output_text") or ""
                texts.append(t)

        elif self.mode == "inference_api":
            assert self._req is not None
            api = f"https://api-inference.huggingface.co/models/{self.model_id}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            for _ in range(n):
                r = self._req.post(
                    api,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "do_sample": True,
                            "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                    timeout=self.request_timeout,
                )
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    t = data[0].get("generated_text", "")
                else:
                    t = ""
                texts.append(t)

        return _as_choices(texts)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend:
    """
    Adapter for Ollama local models.

    Examples
    --------
    backend = OllamaBackend(model="llama3.1:8b-instruct")  # default host http://localhost:11434
    choices = backend.multi_choice(messages, n=3, temperature=0.7, max_tokens=16)
    """
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        request_timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.request_timeout = float(request_timeout)

        # Try python package first
        try:
            import ollama  # type: ignore
            self._ollama = ollama
        except Exception:
            self._ollama = None

        # HTTP client fallback
        try:
            import requests  # type: ignore
            self._req = requests
        except Exception as e:
            if self._ollama is None:
                raise ImportError("Install `ollama` or `requests` to use OllamaBackend.") from e
            self._req = None

    def chat_create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        outs = self.multi_choice(messages, n=1, **kwargs)
        return outs[0]

    def multi_choice(self, messages: List[Dict[str, str]], n: int = 1, **kwargs: Any) -> List[_ChoiceLike]:
        temperature = float(kwargs.get("temperature", 0.5))
        max_tokens = int(kwargs.get("max_tokens", 16))

        # Ollama supports chat messages natively
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        texts: List[str] = []

        if self._ollama is not None:
            for _ in range(n):
                resp = self._ollama.chat(
                    model=self.model,
                    messages=msgs,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    stream=False,
                )
                try:
                    t = resp["message"]["content"]
                except Exception:
                    t = ""
                texts.append(t)
        else:
            assert self._req is not None
            url = f"{self.host}/api/chat"
            for _ in range(n):
                r = self._req.post(
                    url,
                    json={
                        "model": self.model,
                        "messages": msgs,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=self.request_timeout,
                )
                r.raise_for_status()
                data = r.json()
                t = ""
                try:
                    t = data["message"]["content"]
                except Exception:
                    pass
                texts.append(t)

        return _as_choices(texts)


# ---------------------------------------------------------------------------
# OpenRouter backend (OpenAI-compatible aggregator)
# ---------------------------------------------------------------------------

class OpenRouterBackend:
    """
    Adapter for OpenRouter's OpenAI-compatible Chat Completions API.

    Requirements:
        - Set OPENROUTER_API_KEY in the environment, or pass api_key=...
        - (Optional) Provide HTTP-Referer and X-Title for routing transparency.

    Example:
        backend = OpenRouterBackend(model="openrouter/auto")
        choices = backend.multi_choice(messages, n=3, temperature=0.5, max_tokens=16)
    """
    def __init__(
        self,
        model: str = "openrouter/auto",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        request_timeout: float = 60.0,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        providers: Optional[Dict[str, Any]] = None,  # e.g., {"allow": ["anthropic", "openai"]}
    ) -> None:
        try:
            import requests  # type: ignore
        except Exception as e:
            raise ImportError("Install `requests` to use OpenRouterBackend.") from e
        self._req = requests
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        self.http_referer = http_referer
        self.x_title = x_title
        self.extra_headers = extra_headers or {}
        self.providers = providers

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            h["HTTP-Referer"] = self.http_referer
        if self.x_title:
            h["X-Title"] = self.x_title
        h.update(self.extra_headers)
        return h

    def chat_create(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Low-level single call to /chat/completions. Returns parsed JSON dict."""
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(kwargs.get("temperature", 0.5)),
            "max_tokens": int(kwargs.get("max_tokens", 16)),
        }
        # Allow caller to request multiple choices in one call
        if "n" in kwargs:
            payload["n"] = int(kwargs["n"])
        # Optional: route constraints
        if self.providers is not None:
            payload["providers"] = self.providers
        r = self._req.post(url, headers=self._headers(), json=payload, timeout=self.request_timeout)
        r.raise_for_status()
        return r.json()

    def multi_choice(self, messages: List[Dict[str, str]], n: int = 1, **kwargs: Any) -> List[_ChoiceLike]:
        texts: List[str] = []
        # First, try to request n choices in a single API call (if backend supports it)
        try:
            data = self.chat_create(messages, n=n, **kwargs)
            choices = data.get("choices") or []
            for ch in choices:
                # OpenRouter returns OpenAI-like shape: {"message":{"role":"assistant","content":"..."}}
                msg = ch.get("message") or {}
                texts.append(str(msg.get("content", "") or ""))
            if len(texts) == n:
                return _as_choices(texts)
        except Exception:
            texts = []  # fall through to individual calls

        # Fallback: call n times
        for _ in range(n):
            data = self.chat_create(messages, **kwargs)
            ch0 = (data.get("choices") or [{}])[0]
            msg = ch0.get("message") or {}
            texts.append(str(msg.get("content", "") or ""))

        return _as_choices(texts)
