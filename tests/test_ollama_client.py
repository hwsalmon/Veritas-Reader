"""Tests for OllamaClient.

Run with: pytest tests/test_ollama_client.py
Requires a running Ollama instance for integration tests (marked with @pytest.mark.integration).
"""

from unittest.mock import MagicMock, patch

import pytest

from core.ollama_client import OllamaClient, OllamaError


class TestListModels:
    def test_returns_sorted_list(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "codellama:7b"},
                {"name": "mistral:latest"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        client = OllamaClient()
        with patch.object(client._session, "get", return_value=mock_response):
            models = client.list_models()

        assert models == ["codellama:7b", "llama3.2:latest", "mistral:latest"]

    def test_empty_models_list(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()

        client = OllamaClient()
        with patch.object(client._session, "get", return_value=mock_response):
            models = client.list_models()

        assert models == []

    def test_raises_on_connection_error(self):
        import requests
        client = OllamaClient()
        with patch.object(client._session, "get", side_effect=requests.ConnectionError):
            with pytest.raises(OllamaError, match="Cannot connect"):
                client.list_models()


class TestGenerate:
    def test_yields_tokens(self):
        import json
        lines = [
            json.dumps({"response": "Hello", "done": False}).encode(),
            json.dumps({"response": " world", "done": False}).encode(),
            json.dumps({"response": "", "done": True}).encode(),
        ]

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        client = OllamaClient()
        with patch.object(client._session, "post", return_value=mock_response):
            tokens = list(client.generate("llama3.2", "Hi"))

        assert tokens == ["Hello", " world"]
