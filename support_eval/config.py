"""Environment-driven model configuration.

Two roles are configured independently: TARGET (the agent under test) and JUDGE
(the LLM-as-a-judge evaluators). Each can point at Gemini or at any
OpenAI-compatible endpoint -- Fireworks, Modal, Doubleword, Together, vLLM,
Ollama, an internal gateway -- so a judge can be swapped without touching a
rubric, and a target can be swapped without touching the harness.

Keeping the roles separate is the point: judging a model with itself shares its
blind spots, so the ability to put an independent model on the judge side is a
correctness feature, not a convenience.

    {ROLE}_PROVIDER   google | openai        (default: google)
    {ROLE}_MODEL      model name as the provider spells it
    {ROLE}_BASE_URL   OpenAI-compatible endpoint; required when provider=openai
    {ROLE}_API_KEY    falls back to OPENAI_API_KEY / GOOGLE_API_KEY

Optional, for endpoints that do not implement the full OpenAI surface:

    {ROLE}_STRUCTURED_OUTPUT   tool | native | prompted
    {ROLE}_STRICT_TOOLS        true | false  (default: true)
    {ROLE}_TEMPERATURE         float (judge defaults to 0.0; target to the
                               provider default)

`GEMINI_MODEL` is still honoured as the target model name.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

Role = Literal['TARGET', 'JUDGE']

DEFAULTS = {
    'TARGET': 'gemini-3.1-flash-lite',
    'JUDGE': 'gemini-3.5-flash-lite',
}


class ConfigError(RuntimeError):
    """Raised at import time so a misconfigured run fails before it spends money."""


def _env(role: Role, key: str) -> str | None:
    value = os.getenv(f'{role}_{key}')
    return value.strip() if value and value.strip() else None


def _flag(role: Role, key: str, default: bool) -> bool:
    raw = _env(role, key)
    if raw is None:
        return default
    if raw.lower() in {'1', 'true', 'yes', 'on'}:
        return True
    if raw.lower() in {'0', 'false', 'no', 'off'}:
        return False
    raise ConfigError(f'{role}_{key} must be a boolean, got {raw!r}')


def _model_name(role: Role) -> str:
    # GEMINI_MODEL predates the TARGET_/JUDGE_ scheme and still names the target.
    legacy = os.getenv('GEMINI_MODEL') if role == 'TARGET' else None
    return _env(role, 'MODEL') or legacy or DEFAULTS[role]


def _profile(role: Role) -> OpenAIModelProfile | None:
    """Only built when an override is set, so stock OpenAI keeps its own profile.

    Open-weight endpoints vary in what they implement. Two knobs cover most of
    it: many reject `strict: true` on tool definitions, and many do reliable
    JSON only when the schema is prompted rather than sent as a tool.
    """
    overrides: dict[str, Any] = {}

    mode = _env(role, 'STRUCTURED_OUTPUT')
    if mode is not None:
        if mode not in {'tool', 'native', 'prompted'}:
            raise ConfigError(
                f'{role}_STRUCTURED_OUTPUT must be tool, native, or prompted; got {mode!r}'
            )
        overrides['default_structured_output_mode'] = mode

    if _env(role, 'STRICT_TOOLS') is not None:
        overrides['openai_supports_strict_tool_definition'] = _flag(role, 'STRICT_TOOLS', True)

    return OpenAIModelProfile(**overrides) if overrides else None


def build_model(role: Role) -> Model:
    provider_name = (_env(role, 'PROVIDER') or 'google').lower()
    name = _model_name(role)

    if provider_name == 'google':
        api_key = _env(role, 'API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ConfigError(
                f'{role}_PROVIDER=google needs GOOGLE_API_KEY (or {role}_API_KEY).'
            )
        return GoogleModel(name, provider=GoogleProvider(api_key=api_key))

    if provider_name == 'openai':
        base_url = _env(role, 'BASE_URL')
        api_key = _env(role, 'API_KEY') or os.getenv('OPENAI_API_KEY')
        if base_url is None and api_key is None:
            raise ConfigError(
                f'{role}_PROVIDER=openai needs {role}_BASE_URL and {role}_API_KEY '
                '(or OPENAI_API_KEY for api.openai.com).'
            )
        if api_key is None:
            # Local servers (vLLM, Ollama) ignore the key but the client requires one.
            api_key = 'not-required'
        return OpenAIChatModel(
            name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            profile=_profile(role),
        )

    raise ConfigError(
        f'{role}_PROVIDER must be google or openai; got {provider_name!r}'
    )


def _settings(role: Role, default_temperature: float | None) -> dict[str, Any]:
    raw = _env(role, 'TEMPERATURE')
    temperature = float(raw) if raw is not None else default_temperature
    return {} if temperature is None else {'temperature': temperature}


# The target is left at the provider default unless asked otherwise: the point
# of the harness is to grade the agent as it would actually be deployed.
TARGET_MODEL = build_model('TARGET')
TARGET_SETTINGS = _settings('TARGET', None)

# Judges are graded instruments, not creative writers. Pinning temperature means
# a rerun of the same gold set moves only when the rubric moves, which is what
# makes judge-vs-agent regressions attributable to one variable at a time.
JUDGE_MODEL = build_model('JUDGE')
JUDGE_SETTINGS = _settings('JUDGE', 0.0)


def describe() -> str:
    """One line per role. Print this in any run so a report records what produced it."""
    return '\n'.join(
        f'{role:6} {model.system}:{model.model_name}'
        for role, model in (('TARGET', TARGET_MODEL), ('JUDGE', JUDGE_MODEL))
    )
