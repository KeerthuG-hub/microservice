"""
LLM Client - unified interface for Groq, Gemini, Ollama.
Switch with LLM_BACKEND env var.
"""

import os
import json
import time
from typing import Optional


class LLMClient:
    """Unified free LLM interface."""

    def __init__(self):
        self.backend = os.getenv('LLM_BACKEND', 'groq').lower()
        self._client = None
        self._model = None
        self._setup()

    def _setup(self):
        if self.backend == 'groq':
            self._setup_groq()
        elif self.backend == 'gemini':
            self._setup_gemini()
        elif self.backend == 'ollama':
            self._setup_ollama()
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {self.backend}. Use groq/gemini/ollama")

    def _setup_groq(self):
        try:
            from groq import Groq
            key = os.getenv('GROQ_API_KEY')
            if not key:
                raise ValueError("GROQ_API_KEY not set")
            self._client = Groq(api_key=key)
            self._model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            print(f"   🤖 LLM: Groq ({self._model})")
        except ImportError:
            raise ImportError("pip install groq")

    def _setup_gemini(self):
        try:
            import google.generativeai as genai
            key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not key:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=key)
            self._model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
            self._client = genai.GenerativeModel(self._model)
            print(f"   🤖 LLM: Gemini ({self._model})")
        except ImportError:
            raise ImportError("pip install google-generativeai")

    def _setup_ollama(self):
        import urllib.request
        try:
            urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3)
        except Exception:
            raise ConnectionError("Ollama not running. Start with: ollama serve")
        self._model = os.getenv('OLLAMA_MODEL', 'mistral:7b')
        print(f"   🤖 LLM: Ollama ({self._model})")

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------

    def complete(self, prompt: str, max_tokens: int = 2048,
                 retries: int = 3) -> Optional[str]:
        """Send prompt, return text response."""
        for attempt in range(retries):
            try:
                return self._complete(prompt, max_tokens)
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"   ⚠️  LLM error (retry {attempt+1}/{retries}): {e}. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   ❌ LLM failed after {retries} attempts: {e}")
                    return None

    def _complete(self, prompt: str, max_tokens: int) -> str:
        if self.backend == 'groq':
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content

        elif self.backend == 'gemini':
            resp = self._client.generate_content(prompt)
            return resp.text

        elif self.backend == 'ollama':
            import urllib.request
            import json as _json
            data = _json.dumps({
                'model': self._model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.1, 'num_predict': max_tokens}
            }).encode()
            req = urllib.request.Request(
                'http://localhost:11434/api/generate',
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = _json.loads(response.read())
                return result.get('response', '')

    def available(self) -> bool:
        """Check if LLM is configured."""
        try:
            if self.backend == 'groq':
                return bool(os.getenv('GROQ_API_KEY'))
            elif self.backend == 'gemini':
                return bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))
            elif self.backend == 'ollama':
                return True
        except Exception:
            return False
