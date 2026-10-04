from types import SimpleNamespace

from cleancut.llm_utils import chat_for_json, make_ollama_client, strip_to_json


def test_strip_to_json_plain():
    assert strip_to_json('{"key": "val"}') == '{"key": "val"}'


def test_strip_to_json_markdown_fence():
    text = '```json\n{"key": "val"}\n```'
    assert strip_to_json(text) == '{"key": "val"}'


def test_strip_to_json_fence_no_lang():
    text = '```\n{"key": "val"}\n```'
    assert strip_to_json(text) == '{"key": "val"}'


def test_strip_to_json_whitespace():
    assert strip_to_json('  {"key": "val"}  ') == '{"key": "val"}'


def test_make_ollama_client_no_host():
    # Should not raise; we just verify it returns an object
    client = make_ollama_client(None)
    assert client is not None


def test_make_ollama_client_with_host():
    client = make_ollama_client("http://localhost:11434")
    assert client is not None


def test_chat_for_json_disables_thinking():
    class Client:
        def __init__(self):
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"message": {"content": '{"clean": true}', "thinking": "ignored"}}

    client = Client()
    assert chat_for_json(client, model="qwen3.5:9b") == '{"clean": true}'
    assert client.kwargs["think"] is False


def test_chat_for_json_reads_qwen_thinking_fallback():
    class Client:
        def chat(self, **kwargs):
            return SimpleNamespace(
                message=SimpleNamespace(content="", thinking='analysis {"explicit": true}')
            )

    assert "explicit" in chat_for_json(Client(), model="qwen3.5:9b")


def test_chat_for_json_retries_old_client_without_think():
    class OldClient:
        def chat(self, model):
            return {"message": {"content": '{"ok": true}'}}

    assert chat_for_json(OldClient(), model="qwen3.5:9b") == '{"ok": true}'
