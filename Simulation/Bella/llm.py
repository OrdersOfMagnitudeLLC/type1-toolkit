# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Orders of Magnitude LLC <orders@ofmagnitude.com>
"""Universal LLM client for Bella chat."""
import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / '.bella' / '.env')
    load_dotenv(Path(__file__).resolve().parent / '.env')
except ImportError:
    pass


PROVIDER_DEFAULTS = {
    'anthropic': {
        'base_url': 'https://api.anthropic.com/v1',
        'model': 'claude-sonnet-4-6',
        'chat_path': '/messages',
    },
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o',
        'chat_path': '/chat/completions',
    },
    'openrouter': {
        'base_url': 'https://openrouter.ai/api/v1',
        'model': 'openrouter/auto',
        'chat_path': '/chat/completions',
    },
    'custom': {
        'base_url': None,
        'model': None,
        'chat_path': '/chat/completions',
    },
}


class LLMClient:
    """Route to anthropic, openai, openrouter, or a custom OpenAI-compatible endpoint."""

    def __init__(self):
        self.provider = os.environ.get('BELLA_LLM_PROVIDER', 'anthropic').lower().strip()
        if self.provider not in PROVIDER_DEFAULTS:
            raise RuntimeError(f"Unknown BELLA_LLM_PROVIDER: {self.provider}. Use anthropic, openai, openrouter, or custom.")

        self.key = os.environ.get('BELLA_LLM_API_KEY', '').strip()
        if not self.key:
            raise RuntimeError("no_api_key")

        defaults = PROVIDER_DEFAULTS[self.provider]
        self.base_url = os.environ.get('BELLA_LLM_BASE_URL', defaults['base_url'] or '').rstrip('/')
        if not self.base_url:
            raise RuntimeError("BELLA_LLM_BASE_URL must be set for custom provider.")

        self.model = os.environ.get('BELLA_LLM_MODEL', defaults['model'])
        if self.provider == 'custom' and not self.model:
            raise RuntimeError("BELLA_LLM_MODEL must be set for custom provider.")

        self.chat_path = defaults['chat_path']

    def chat(self, system, messages, max_tokens=256):
        """Return a single text completion using requests (no extra SDK required)."""
        try:
            import requests
            headers = {'content-type': 'application/json'}

            if self.provider == 'anthropic':
                headers['x-api-key'] = self.key
                headers['anthropic-version'] = '2023-06-01'
                payload = {
                    'model': self.model,
                    'max_tokens': max_tokens,
                    'system': system,
                    'messages': messages,
                }
            else:
                headers['Authorization'] = f'Bearer {self.key}'
                payload = {
                    'model': self.model,
                    'messages': [{'role': 'system', 'content': system}] + messages,
                    'max_tokens': max_tokens,
                }

            response = requests.post(
                self.base_url + self.chat_path,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if self.provider == 'anthropic':
                return data['content'][0]['text']
            # OpenAI / OpenRouter / custom
            return data['choices'][0]['message']['content']
        except Exception as e:
            return json.dumps({"tool": "none", "response": f"LLM error: {e}"})

    def stream(self, system, messages, max_tokens=256):
        """Yield text chunks from a streaming completion."""
        try:
            import requests
            headers = {'content-type': 'application/json'}

            if self.provider == 'anthropic':
                headers['x-api-key'] = self.key
                headers['anthropic-version'] = '2023-06-01'
                payload = {
                    'model': self.model,
                    'max_tokens': max_tokens,
                    'system': system,
                    'messages': messages,
                    'stream': True,
                }
            else:
                headers['Authorization'] = f'Bearer {self.key}'
                payload = {
                    'model': self.model,
                    'messages': [{'role': 'system', 'content': system}] + messages,
                    'max_tokens': max_tokens,
                    'stream': True,
                }

            response = requests.post(
                self.base_url + self.chat_path,
                headers=headers,
                json=payload,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    decoded = decoded[6:]
                if decoded.strip() == '[DONE]':
                    break
                if not decoded.strip():
                    continue
                try:
                    chunk = json.loads(decoded)
                    if self.provider == 'anthropic':
                        text = chunk.get('delta', {}).get('text', '')
                    else:
                        text = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            yield f"LLM error: {e}"
