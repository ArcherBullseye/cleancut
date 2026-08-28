from cleancut.llm_utils import strip_to_json, make_ollama_client


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
