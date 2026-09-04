"""The system under test: a support agent that answers from retrieved policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from support_eval.config import TARGET_MODEL
from support_eval.policies import retrieve

logfire.configure(service_name='customer-support-eval', console=False)
logfire.instrument_pydantic_ai()


class SupportAnswer(BaseModel):
    text: str
    cited_policy_ids: list[str]
    action: Literal['answer', 'escalate']


class SupportResult(BaseModel):
    """What every evaluator grades.

    `question` and `evidence` are carried alongside the answer so that a judge
    can check faithfulness and over-refusal from the output alone. That also
    makes a saved result self-contained, which is what lets the calibration
    harness replay it without rerunning the agent.
    """

    question: str
    answer: SupportAnswer
    evidence: dict[str, str]


@dataclass
class SupportDeps:
    evidence: dict[str, str] = field(default_factory=dict)


support_agent = Agent(
    TARGET_MODEL,
    deps_type=SupportDeps,
    output_type=SupportAnswer,
    system_prompt=(
        'You answer customer support questions for an online retailer.\n'
        'Before answering, call `lookup_policy` with the customer question.\n'
        'The lookup always returns policies, and they are often only loosely '
        'related to the question. Use only the ones that actually bear on it.\n'
        'Answer only from those policies and cite every policy you used by ID. '
        'Do not cite a policy you did not use.\n'
        'Escalate when the returned policies do not contain the facts needed to '
        'answer. Escalating because a policy looks unrelated is correct; '
        'escalating when the answer is plainly present is not.\n'
        'Keep answers to at most three sentences. State the rule and what it '
        'means for this customer. Do not restate the question.'
    ),
)


@support_agent.tool
def lookup_policy(ctx: RunContext[SupportDeps], question: str) -> dict[str, str]:
    """Return support policies that may be relevant to the customer's question."""
    evidence = retrieve(question)
    ctx.deps.evidence.update(evidence)
    return evidence


async def answer_support_question(question: str) -> SupportResult:
    deps = SupportDeps()
    result = await support_agent.run(question, deps=deps)
    return SupportResult(question=question, answer=result.output, evidence=deps.evidence)
