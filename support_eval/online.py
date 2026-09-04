"""Production wrapper: the label-free evaluators, applied to live traffic.

Two offline checks cannot come along, both because they read `CaseMeta`, which
production does not have: `escalation_correct` needs `expected_action`, and
`retrieval_recall` needs `answering_policy_ids`. The over-refusal judge is the
unlabelled stand-in for the first; the second has none, so a retrieval
regression surfaces here only indirectly, as faithfulness or over-refusal drift.

The concurrency limits below are placeholders. An online evaluator that hits its
limit drops the evaluation, so a limit set below peak throughput silently thins
the sample. Load-test against real traffic before trusting these numbers.
"""

from __future__ import annotations

import logfire
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.online import OnlineEvaluator, evaluate

from support_eval.agent import answer_support_question
from support_eval.evaluators import (
    CitationDiscipline,
    CitationsGrounded,
    WithinLengthBudget,
    conciseness_judge,
    faithfulness_judge,
    over_refusal_judge,
    used_policy_lookup,
)

JUDGE_SAMPLE_RATE = 0.05


def _dropped(ctx: EvaluatorContext) -> None:
    # Identify the interaction, otherwise you learn that drops happened but not
    # which part of the sample went missing.
    logfire.warn(
        'online evaluation dropped at concurrency limit',
        case_name=ctx.name,
    )


def _cheap(evaluator) -> OnlineEvaluator:
    return OnlineEvaluator(
        evaluator=evaluator,
        sample_rate=1.0,
        max_concurrency=100,
        on_max_concurrency=_dropped,
    )


def _judged(evaluator) -> OnlineEvaluator:
    return OnlineEvaluator(
        evaluator=evaluator,
        sample_rate=JUDGE_SAMPLE_RATE,
        max_concurrency=20,
        on_max_concurrency=_dropped,
    )


answer_support_question_evaluated = evaluate(
    _cheap(CitationsGrounded()),
    _cheap(CitationDiscipline()),
    _cheap(WithinLengthBudget()),
    _cheap(used_policy_lookup),
    _judged(faithfulness_judge),
    _judged(over_refusal_judge),
    _judged(conciseness_judge),
    target='support-policy-agent',
    extract_args=True,
    record_return=True,
)(answer_support_question)
