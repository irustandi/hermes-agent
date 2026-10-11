"""ClinePass subscription models through Cline's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)


def _subscription_models(models: list[str] | None) -> list[str] | None:
    if models is None:
        return None
    return list(dict.fromkeys(model for model in models if isinstance(model, str) and model.startswith("cline-pass/")))


class ClinePassProfile(ProviderProfile):
    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        caller_base = (base_url or "").strip().rstrip("/")
        if caller_base and caller_base != self.base_url.rstrip("/"):
            return _subscription_models(super().fetch_models(api_key=api_key, base_url=caller_base, timeout=timeout))

        from hermes_cli.chat_catalog import chat_catalog_ids
        from hermes_cli.urllib_security import open_credentialed_url

        # The standard /models catalog omits Pass; this public endpoint separates billing lanes.
        request = urllib.request.Request(self.models_url, headers={
            "Accept": "application/json", "User-Agent": _profile_user_agent(),
        })
        try:
            with open_credentialed_url(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            return _subscription_models(chat_catalog_ids(payload.get("clinePass", [])))
        except (OSError, ValueError, AttributeError):
            logger.debug("ClinePass catalog fetch failed", exc_info=True)
            return None

    def create_client(self, **client_kwargs: Any) -> Any:
        from .client import ClinePassClient

        return ClinePassClient(**client_kwargs)


register_provider(ClinePassProfile(
    name="cline-pass", aliases=("clinepass", "cline_pass"), display_name="ClinePass",
    description="ClinePass (subscription coding models)", signup_url="https://app.cline.bot",
    env_vars=("CLINE_API_KEY", "CLINEPASS_API_KEY"), base_url="https://api.cline.bot/api/v1",
    models_url="https://api.cline.bot/api/v1/ai/cline/recommended-models",
    # Auxiliary tasks use the selected model so they stay on the subscription lane.
    fallback_models=(
        "cline-pass/glm-5.3", "cline-pass/deepseek-v4.1-flash", "cline-pass/mimo-v2.6-flash",
        "cline-pass/mimo-v2.6-pro", "cline-pass/deepseek-v4-pro", "cline-pass/qwen3.8-max",
        "cline-pass/muse-spark-1.3-contributor", "cline-pass/kimi-k3", "cline-pass/glm-5.3-flash",
        "cline-pass/qwen3.7-plus", "cline-pass/qwen3.7-max", "cline-pass/minimax-m3",
        "cline-pass/mimo-v2.5-pro", "cline-pass/mimo-v2.5",
    ),
))
