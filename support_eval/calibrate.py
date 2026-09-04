"""Grade a judge against human labels.

The task is the identity function, so saved agent outputs are replayed straight
into the judge. Rerunning the agent here would move two variables at once and
make a disagreement unattributable.

Raw agreement is reported but is not the number to trust: on an imbalanced gold
set a judge that always says "pass" scores well. Cohen's kappa corrects for
agreement by chance, and the two error directions are broken out because they
have different costs -- for faithfulness a false pass ships a hallucinated
policy to a customer, and for over-refusal a false pass hides an agent that is
refusing questions it could answer.

Usage:
    uv run python -m support_eval.calibrate                # every judge
    uv run python -m support_eval.calibrate over_refusal   # just one
"""

from __future__ import annotations

import sys

from support_eval.config import describe
from support_eval.gold_sets import GOLD_SETS, GoldSet

TARGET_AGREEMENT = 0.85
TARGET_KAPPA = 0.70


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    observed = sum(h == j for h, j in pairs) / n
    human_pass = sum(h for h, _ in pairs) / n
    judge_pass = sum(j for _, j in pairs) / n
    expected = human_pass * judge_pass + (1 - human_pass) * (1 - judge_pass)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def calibrate(gold: GoldSet) -> bool:
    print(f'\n=== {gold.name} ({gold.assertion_name}) ===')
    report = gold.dataset.evaluate_sync(
        lambda result: result, name=f'{gold.name}-calibration'
    )

    pairs: list[tuple[bool, bool]] = []
    missing: list[str] = []
    disagreements = []
    for case in report.cases:
        verdict = case.assertions.get(gold.assertion_name)
        if verdict is None:
            missing.append(case.name)
            continue
        human = bool(case.metadata[gold.human_key])
        judge = bool(verdict.value)
        pairs.append((human, judge))
        if human != judge:
            disagreements.append((case.name, human, judge, verdict.reason))

    if missing:
        print(f'Judge produced no verdict for: {", ".join(missing)}')
    if not pairs:
        print('No comparable results; cannot calibrate.')
        return False

    n = len(pairs)
    agreement = sum(h == j for h, j in pairs) / n
    false_pass = sum(1 for h, j in pairs if not h and j)
    false_fail = sum(1 for h, j in pairs if h and not j)
    kappa = cohens_kappa(pairs)
    human_pass = sum(h for h, _ in pairs) / n

    print(f'Gold cases compared : {n}  ({human_pass:.0%} labeled pass)')
    print(f'Raw agreement       : {agreement:.1%}  (target {TARGET_AGREEMENT:.0%})')
    print(f"Cohen's kappa       : {kappa:.2f}   (target {TARGET_KAPPA:.2f})")
    print(f'False passes        : {false_pass}  <- real failures the judge let through')
    print(f'False fails         : {false_fail}  <- good outputs the judge rejected')

    print('\nDisagreements:')
    if not disagreements:
        print('  none')
    for name, human, judge, reason in disagreements:
        print(f'  {name}: human={human} judge={judge}\n    {reason}')

    ready = agreement >= TARGET_AGREEMENT and kappa >= TARGET_KAPPA and not missing
    print(
        f'\n{gold.name}: calibrated enough to scale.'
        if ready
        else f'\n{gold.name}: NOT calibrated. Classify each disagreement:\n'
        '  ambiguous rubric -> make the rule observable, add examples\n'
        '  missing context  -> give the judge the source it needs\n'
        '  wrong label      -> fix the gold set and record why\n'
        '  truly subjective -> keep it with humans'
    )
    return ready


def main(argv: list[str]) -> int:
    requested = argv or list(GOLD_SETS)
    unknown = [name for name in requested if name not in GOLD_SETS]
    if unknown:
        print(f'Unknown gold set(s): {", ".join(unknown)}')
        print(f'Available: {", ".join(GOLD_SETS)}')
        return 2

    # Calibration is only meaningful for the judge that produced it -- record which.
    print(describe())
    results = {name: calibrate(GOLD_SETS[name]) for name in requested}

    print('\n--- summary ---')
    for name, ready in results.items():
        print(f'  {"ok  " if ready else "FAIL"} {name}')
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
