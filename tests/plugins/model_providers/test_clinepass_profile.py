"""ClinePass discovery, subscription catalogs, and OpenAI wire compatibility."""

import io
import json
import urllib.error

import pytest


@pytest.fixture
def clinepass_profile():
    from providers import get_provider_profile

    profile = get_provider_profile("clinepass")
    assert profile is not None
    return profile


def test_clinepass_key_enables_picker_and_persists_runtime(clinepass_profile, monkeypatch):
    from hermes_cli.auth import resolve_provider
    from hermes_cli.config import OPTIONAL_ENV_VARS, load_config, save_env_value
    from hermes_cli.model_switch import persist_model_selection, switch_model
    from hermes_cli.models import list_available_providers, parse_model_input
    from hermes_cli.provider_catalog import provider_catalog_by_slug
    from hermes_cli.runtime_provider import resolve_runtime_provider

    profile = clinepass_profile
    descriptor = provider_catalog_by_slug()[profile.name]
    assert descriptor.label == "ClinePass"
    assert descriptor.api_key_env_vars == profile.env_vars
    assert OPTIONAL_ENV_VARS[profile.env_vars[0]]["password"] is True
    row = next(p for p in list_available_providers() if p["id"] == profile.name)
    assert not row["authenticated"]

    save_env_value(profile.env_vars[0], "test-cline-key")
    assert resolve_provider("clinepass") == profile.name
    row = next(p for p in list_available_providers() if p["id"] == profile.name)
    assert row["authenticated"]
    chosen = profile.fallback_models[0]
    assert parse_model_input(f"{profile.name}:{chosen}", "openrouter") == (profile.name, chosen)

    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", offline)
    result = switch_model(chosen, "openrouter", "previous-model", explicit_provider="clinepass")
    assert result.success, result.error_message
    assert result.new_model == chosen
    assert result.base_url == profile.base_url
    persist_model_selection(result)
    assert load_config()["model"]["provider"] == profile.name
    runtime = resolve_runtime_provider()
    assert runtime["provider"] == profile.name
    assert runtime["api_key"] == "test-cline-key"
    assert runtime["base_url"] == profile.base_url
    assert runtime["api_mode"] == profile.api_mode


@pytest.mark.parametrize("custom_endpoint", [False, True])
def test_clinepass_catalog_excludes_other_billing_lanes_and_falls_back(
    clinepass_profile, monkeypatch, custom_endpoint
):
    from hermes_cli.models import probe_profile_catalog

    profile = clinepass_profile
    subscription = "cline-pass/future-coding-model"
    rows = [{"id": subscription}, {"id": "paid/model"}, {"id": "cline-free/model"}]
    payload = {"data": rows} if custom_endpoint else {
        "clinePass": rows,
        "recommended": [{"id": "paid/recommended"}],
        "free": [{"id": "cline-free/free"}],
    }
    base_url = "https://proxy.example/v1" if custom_endpoint else profile.base_url + "/"
    requests = []

    def catalog(request, *, timeout):
        requests.append(request)
        assert timeout > 0
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", catalog)
    assert profile.fetch_models(api_key="test-key", base_url=base_url) == [subscription]
    expected_url = base_url + "/models" if custom_endpoint else profile.models_url
    assert requests[0].full_url == expected_url
    if custom_endpoint:
        assert requests[0].get_header("Authorization") == "Bearer test-key"
    else:
        assert requests[0].get_header("Authorization") is None
    models = probe_profile_catalog(profile.name, profile, "test-key", base_url)
    assert subscription in models
    assert all(model.startswith("cline-pass/") for model in models)

    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", offline)
    assert profile.fetch_models(api_key="test-key", base_url=base_url) is None
    assert probe_profile_catalog(profile.name, profile, "test-key", base_url) == list(profile.fallback_models)


@pytest.mark.parametrize("response_kind", ["wrapped", "standard", "streaming"])
@pytest.mark.parametrize("client_route", ["profile", "auxiliary"])
def test_clinepass_client_preserves_streams_and_unwraps_completions(
    clinepass_profile, monkeypatch, response_kind, client_route
):
    import httpx

    profile = clinepass_profile
    model = profile.fallback_models[0]
    streaming = response_kind == "streaming"
    completion = {
        "id": "test-completion", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello", "tool_calls": [
            {"id": "call-test", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}},
        ]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    def respond(request):
        assert str(request.url) == profile.base_url + "/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content)["model"] == model
        if streaming:
            chunk = {**completion, "object": "chat.completion.chunk"}
            chunk["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}]
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        payload = {"success": True, "data": completion} if response_kind == "wrapped" else completion
        return httpx.Response(200, json=payload)

    if client_route == "auxiliary":
        from agent.auxiliary_client import resolve_provider_client

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _transport, request: respond(request))
        client, resolved_model = resolve_provider_client(provider=profile.name, model=model, explicit_api_key="test-key")
        assert resolved_model == model
    else:
        client = profile.create_client(api_key="test-key", base_url=profile.base_url,
                                       http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    with client:
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}],
                                                  stream=streaming)
        if streaming:
            assert "".join(chunk.choices[0].delta.content or "" for chunk in response) == "Hello"
        else:
            assert response.choices[0].message.content == "Hello"
            assert response.choices[0].message.tool_calls[0].function.name == "test_tool"
            assert response.usage.total_tokens == completion["usage"]["total_tokens"]
