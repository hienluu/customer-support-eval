"""Layer 1 (programmatic) and Layer 2 (judge) evaluators.

Each evaluator answers exactly one question. That is deliberate: a single
boolean covering four failure modes cannot be gated at four thresholds, and a
regression in it does not say what regressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    LLMJudge,
    ToolCorrectness,
)

from support_eval.agent import SupportResult
from support_eval.config import JUDGE_MODEL, JUDGE_SETTINGS

MAX_ANSWER_WORDS = 80

Ctx = EvaluatorContext[str, SupportResult, 'CaseMeta']


@dataclass
class CaseMeta:
    """Human ground truth for a case.

    `expected_action` is a label, not something inferred from retrieval. The
    retriever always returns policies, so "did evidence come back?" says nothing
    about whether escalating was right.
    """

    expected_action: Literal['answer', 'escalate']
    note: str = ''
    # Policies that actually answer this question. Empty when the corpus cannot.
    answering_policy_ids: tuple[str, ...] = ()


@dataclass
class CitationsGrounded(Evaluator[str, SupportResult, CaseMeta]):
    """Every cited policy ID must exist in the retrieved evidence."""

    def evaluate(self, ctx: Ctx) -> dict[str, EvaluationReason]:
        cited = set(ctx.output.answer.cited_policy_ids)
        available = set(ctx.output.evidence)
        invented = sorted(cited - available)
        return {
            'citations_grounded': EvaluationReason(
                value=not invented,
                reason=(
                    f'Cited policy IDs not present in evidence: {", ".join(invented)}'
                    if invented
                    else 'All citations resolve to retrieved evidence.'
                ),
            )
        }


@dataclass
class CitationDiscipline(Evaluator[str, SupportResult, CaseMeta]):
    """An answer must cite; an escalation must not claim policy it never used."""

    def evaluate(self, ctx: Ctx) -> dict[str, EvaluationReason]:
        answer = ctx.output.answer
        if answer.action == 'answer' and not answer.cited_policy_ids:
            return {
                'cites_when_answering': EvaluationReason(
                    value=False, reason='Answered without citing any policy.'
                )
            }
        return {'cites_when_answering': EvaluationReason(value=True)}


@dataclass
class EscalationMatchesLabel(Evaluator[str, SupportResult, CaseMeta]):
    """Compare the action taken against the human label for this case.

    Offline only. Live traffic has no label, so production relies on
    `not_over_refusing` instead.
    """

    def evaluate(self, ctx: Ctx) -> dict[str, EvaluationReason]:
        if ctx.metadata is None:
            return {}
        expected = ctx.metadata.expected_action
        actual = ctx.output.answer.action
        return {
            'escalation_correct': EvaluationReason(
                value=actual == expected,
                reason=f'Expected {expected!r}, got {actual!r}. {ctx.metadata.note}'.strip(),
            )
        }


@dataclass
class RetrievalRecall(Evaluator[str, SupportResult, CaseMeta]):
    """Did retrieval surface the policies that answer this question?

    Without this, a retrieval miss shows up as an escalation failure and reads
    as a generator problem. Escalate-labeled cases have no answering policy, so
    they pass vacuously.
    """

    def evaluate(self, ctx: Ctx) -> dict[str, EvaluationReason]:
        if ctx.metadata is None:
            return {}
        needed = set(ctx.metadata.answering_policy_ids)
        retrieved = set(ctx.output.evidence)
        missed = sorted(needed - retrieved)
        return {
            'retrieval_recall': EvaluationReason(
                value=not missed,
                reason=(
                    f'Retrieval missed {", ".join(missed)}; got {", ".join(sorted(retrieved))}.'
                    if missed
                    else 'All answering policies were retrieved.'
                ),
            )
        }


@dataclass
class WithinLengthBudget(Evaluator[str, SupportResult, CaseMeta]):
    """The cheap half of concision: catch runaway length without a model call."""

    max_words: int = MAX_ANSWER_WORDS

    def evaluate(self, ctx: Ctx) -> dict[str, EvaluationReason]:
        words = len(ctx.output.answer.text.split())
        return {
            'within_length_budget': EvaluationReason(
                value=words <= self.max_words,
                reason=f'{words} words (budget {self.max_words}).',
            )
        }


used_policy_lookup = ToolCorrectness(
    expected_tools=['lookup_policy'],
    evaluation_name='used_policy_lookup',
)


FAITHFULNESS_RUBRIC = """
Judge only faithfulness. Ignore tone, length, and whether the answer is helpful.

Faithfulness is two conditions, not one: a claim must be supported, AND its
source must have been cited. Check them separately, in this order.

1. Write out CITED, the list of keys in `answer.cited_policy_ids`. Then write
   out UNCITED, the keys present in `evidence` but absent from CITED. Only
   CITED entries may support a claim. UNCITED entries support nothing, however
   plainly their text states it.
2. If `answer.action` is "escalate" and `answer.text` asserts no policy
   content, the verdict is PASS. Stop here.
3. List every factual claim in `answer.text`.
4. For each claim, look for text inside a CITED entry that states it. Accurate
   paraphrases count. Correct arithmetic over numbers in a CITED entry counts.
   Do not read UNCITED entries at this step.
5. If every claim was matched in step 4, the verdict is PASS.
6. If even one claim went unmatched, the verdict is FAIL. Every claim counts
   equally: there is no primary or secondary claim, and a supported main point
   never excuses an unsupported detail appended to it. This also covers a claim
   that is true and appears verbatim in an UNCITED entry -- stating a policy
   without citing it is exactly the failure this rubric exists to catch.

Finish your reason with exactly one of `VERDICT: PASS` or `VERDICT: FAIL`, and
set the pass field to true only when you wrote `VERDICT: PASS`. Reasoning that
identifies an unsupported claim and then passes the output is the single most
common error here; the closing token exists to stop it.
"""

OVER_REFUSAL_RUBRIC = """
Judge only whether escalating was necessary. Ignore tone, length, citation
formatting, and whether the answer is factually correct.

Work through these steps in order and stop at the first one that applies.

1. If `answer.action` is "answer", the verdict is PASS.
2. Otherwise the agent escalated. Read `question`, then read the full text of
   every entry in `evidence`.
3. If some evidence entry states the facts needed to answer `question`, the
   escalation was unnecessary. The verdict is FAIL.
4. If no evidence entry states those facts, the escalation was correct. The
   verdict is PASS.

PASS means the agent did not over-refuse. FAIL means it over-refused. Emit the
verdict your reasoning reached; do not invert it.

The retriever always returns policies and they are frequently unrelated to the
question. The presence of evidence is never by itself a reason the agent should
have answered.
"""

CONCISENESS_RUBRIC = """
Judge only concision -- how much text was spent, not what it said.

Ignore correctness, faithfulness, and whether escalating was the right call. A
wrong answer passes this rubric if it is tight. An escalation passes if it is
brief; it has no policy to state, so never fail it for omitting one.

Pass when `answer.text` is at most three sentences and every sentence carries
new information.

Fail only when it restates the question, explains policies the customer did not
ask about, states the same point twice, or spends more than one sentence on
pleasantries or apology.
"""


def _judge(rubric: str, name: str) -> LLMJudge:
    return LLMJudge(
        rubric=rubric,
        model=JUDGE_MODEL,
        model_settings=JUDGE_SETTINGS,
        assertion={'evaluation_name': name, 'include_reason': True},
    )


faithfulness_judge = _judge(FAITHFULNESS_RUBRIC, 'faithful_to_policy')
over_refusal_judge = _judge(OVER_REFUSAL_RUBRIC, 'not_over_refusing')
conciseness_judge = _judge(CONCISENESS_RUBRIC, 'concise')

PROGRAMMATIC_EVALUATORS = [
    RetrievalRecall(),
    CitationsGrounded(),
    CitationDiscipline(),
    EscalationMatchesLabel(),
    WithinLengthBudget(),
    used_policy_lookup,
]

JUDGES = [faithfulness_judge, over_refusal_judge, conciseness_judge]
