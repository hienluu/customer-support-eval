# customer-support-eval

Three-layer eval harness for a policy-grounded support agent, following
Airbnb's eval-driven development post. Built with Pydantic AI, Pydantic Evals,
and Logfire, on Gemini.

## Layout

| File | Role |
| --- | --- |
| `config.py` | Model names from `.env`; pinned judge temperature |
| `policies.py` | Policy corpus and a top-k retriever that always returns `k` |
| `agent.py` | The system under test; `SupportResult` carries question + evidence |
| `evaluators.py` | Layer 1 programmatic checks and Layer 2 judges, one per dimension |
| `datasets.py` | Labeled offline cases |
| `run_eval.py` | Offline experiment + CI gate |
| `gold_sets.py` | Layer 3: human-labeled cases, one set per judge |
| `calibrate.py` | Layer 3: judge vs. human labels, agreement + Cohen's kappa |
| `online.py` | Same evaluators on live traffic, judges sampled at 5% |

## Running

```sh
uv run python -m support_eval.calibrate                # grade every judge
uv run python -m support_eval.calibrate over_refusal   # grade one
uv run python -m support_eval.run_eval                 # does the agent pass its gates?
```

Calibrate before trusting a gate. An uncalibrated judge is just another model
output, and both scripts exit non-zero on failure.

`run_eval` exits non-zero when a gate fails, so it works as a CI step.

## Metrics

Layer 1, deterministic:

- `retrieval_recall` — the policies that answer the question were retrieved
- `escalation_correct` — action taken matches the human label (offline only)
- `citations_grounded` — cited IDs exist in retrieved evidence
- `cites_when_answering` — an answer cites something
- `used_policy_lookup` — the retrieval tool ran exactly once
- `within_length_budget` — answer under 80 words

Layer 2, judge-backed, gated below 100% because judges are stochastic:

- `faithful_to_policy` — no claim beyond the cited evidence
- `not_over_refusing` — escalated only when the evidence really lacks the answer
- `concise` — no wasted sentences

Layer 3, `calibrate.py`: replays saved outputs through the judge and compares
against human labels.

## Design notes

**The retriever always returns `k` policies.** Real top-k vector search has no
relevance floor, so "evidence came back" says nothing about whether the question
is answerable. No evaluator may use retrieval non-emptiness as a proxy for
answerability — escalation correctness comes from a human label
(`CaseMeta.expected_action`) instead.

**`SupportResult` carries `question` and `evidence`.** A judge cannot check
faithfulness against retrieval unless retrieval is inside the graded object, and
a self-contained result is what lets `calibrate.py` replay it without rerunning
the agent.

**Calibration replays; it does not regenerate.** The task there is the identity
function, so a disagreement is attributable to the judge rather than to the
agent and the judge having both moved.

**The gate fails on missing verdicts.** If an evaluator errors, averaging the
cases that survived turns a judge outage into a perfect score. `pass_rate`
raises unless every case produced a verdict, and task errors and evaluator
failures are checked separately.

**One evaluator per dimension.** A single boolean spanning several failure modes
cannot be gated at several thresholds, and a regression in it does not say what
regressed.

## Configuration

Two roles are configured independently: **TARGET** (the agent under test) and
**JUDGE** (the LLM-as-a-judge evaluators). Either can be Gemini or any
OpenAI-compatible endpoint.

| Variable | Meaning |
| --- | --- |
| `{ROLE}_PROVIDER` | `google` (default) or `openai` |
| `{ROLE}_MODEL` | model name as that provider spells it |
| `{ROLE}_BASE_URL` | OpenAI-compatible endpoint; required when provider is `openai` |
| `{ROLE}_API_KEY` | falls back to `OPENAI_API_KEY` / `GOOGLE_API_KEY` |
| `{ROLE}_STRUCTURED_OUTPUT` | `tool` \| `native` \| `prompted` — for endpoints with partial OpenAI support |
| `{ROLE}_STRICT_TOOLS` | `true` (default) / `false` — many open-weight endpoints reject `strict: true` |
| `{ROLE}_TEMPERATURE` | float; the judge defaults to `0.0` |

`{ROLE}` is `TARGET` or `JUDGE`. Bad values raise at import, before a run spends
money. `GEMINI_MODEL` is still honoured as the target model name.

Default `.env` (gitignored):

```sh
GOOGLE_API_KEY=...
LOGFIRE_TOKEN=...
GEMINI_MODEL="gemini-3.1-flash-lite"
JUDGE_MODEL="gemini-3.5-flash-lite"
```

### Doubleword (open-weight models)

Both roles on Doubleword, with a deliberately independent judge:

```sh
TARGET_PROVIDER=openai
TARGET_MODEL="openai/gpt-oss-20b"
TARGET_BASE_URL="https://api.doubleword.ai/v1"
TARGET_API_KEY=...
TARGET_STRICT_TOOLS=false

JUDGE_PROVIDER=openai
JUDGE_MODEL="deepseek-ai/deepseek-v4-pro"
JUDGE_BASE_URL="https://api.doubleword.ai/v1"
JUDGE_API_KEY=...
JUDGE_STRICT_TOOLS=false
```

The target needs tool calling (`lookup_policy`) and structured output
(`SupportAnswer`); the judge needs neither tools nor speed, only careful
reading, so spend the capability there. Keep the two from the same family only
if you have measured that it does not matter -- shared lineage means shared
blind spots.

Doubleword's `-dottxt` variants are built around constrained decoding, which is
the exact requirement for a judge that must return `{reason, pass, score}`.
Worth benchmarking against a plain variant with `calibrate.py` if judge verdicts
come back malformed.

Other endpoints follow the same shape — only `BASE_URL` and `MODEL` change:

| Endpoint | `{ROLE}_BASE_URL` |
| --- | --- |
| Doubleword | `https://api.doubleword.ai/v1` |
| Fireworks | `https://api.fireworks.ai/inference/v1` |
| Together | `https://api.together.xyz/v1` |
| Moonshot | `https://api.moonshot.ai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Modal | your deployed app's `/v1` URL |
| vLLM / Ollama | `http://localhost:8000/v1` (key optional) |
| OpenAI | omit — the default |

Since judges and targets are configured separately, a cross-vendor judge is one
env var: an open-weight judge over a Gemini target, or the reverse. That is the
point — a judge sharing its target's lineage shares its blind spots.

**Recalibrate after any judge change.** Agreement and kappa are properties of a
specific judge model against a specific rubric, so a swapped model invalidates
them:

```sh
uv run python -m support_eval.calibrate
```
