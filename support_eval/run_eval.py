"""Run the offline experiment and gate on it.

The gate treats an evaluator that failed to produce a result as a red build.
Skipping those cases and averaging the rest lets a judge outage read as a
perfect score, which is the one failure a quality gate must never have.
"""

from __future__ import annotations

import asyncio
import sys

import logfire

from support_eval.agent import answer_support_question
from support_eval.config import describe
from support_eval.datasets import dataset

# Deterministic checks have no excuse for failing. Judge-backed metrics are
# stochastic and are gated below 1.0 on purpose.
THRESHOLDS: dict[str, float] = {
    'retrieval_recall': 0.90,
    'citations_grounded': 1.0,
    'cites_when_answering': 1.0,
    'used_policy_lookup': 1.0,
    'within_length_budget': 0.90,
    'escalation_correct': 0.85,
    'faithful_to_policy': 0.90,
    'not_over_refusing': 0.85,
    'concise': 0.80,
}


def pass_rate(report, name: str) -> float:
    """Pass rate over *every* case, or an error if any case lacks a verdict."""
    total = len(report.cases)
    values = [
        case.assertions[name].value for case in report.cases if name in case.assertions
    ]
    if len(values) != total:
        raise RuntimeError(
            f'{name}: only {len(values)} of {total} cases produced a verdict. '
            'Treating this as a pass rate would hide the missing evaluations.'
        )
    return sum(values) / total


def check(report) -> list[str]:
    problems: list[str] = []

    if report.failures:
        names = ', '.join(f.name for f in report.failures)
        problems.append(f'{len(report.failures)} case(s) raised a task error: {names}')

    evaluator_failures = [
        (case.name, failure) for case in report.cases for failure in case.evaluator_failures
    ]
    for case_name, failure in evaluator_failures:
        problems.append(f'evaluator failed on {case_name}: {failure}')

    failing_metrics: list[str] = []
    for name, threshold in THRESHOLDS.items():
        try:
            rate = pass_rate(report, name)
        except RuntimeError as exc:
            problems.append(str(exc))
            continue
        status = 'ok  ' if rate >= threshold else 'FAIL'
        print(f'  {status} {name:24} {rate:6.1%}  (gate {threshold:.0%})')
        if rate < threshold:
            failing_metrics.append(name)
            problems.append(f'{name}: {rate:.1%} is below the {threshold:.0%} gate')

    for name in failing_metrics:
        print(f'\n  Cases failing {name}:')
        for case in report.cases:
            verdict = case.assertions.get(name)
            if verdict is not None and not verdict.value:
                print(f'    {case.name}: {verdict.reason}')

    return problems


async def main() -> int:
    print(describe(), '\n')
    report = await dataset.evaluate(
        answer_support_question,
        name='baseline',
        max_concurrency=4,
    )
    report.print(include_reasons=True, include_output=True)

    print('\nGates:')
    problems = check(report)

    print(f'\nLogfire: {logfire.url_from_eval(report)}')

    if problems:
        print('\nFAILED:')
        for problem in problems:
            print(f'  - {problem}')
        return 1

    print('\nAll gates passed.')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
