"""Human-labeled gold sets for the judges that gate on correctness.

Each set is saved agent output plus a human verdict. Nothing here calls the
agent: `saved()` reconstructs an output by hand so the calibration harness can
replay it and attribute any disagreement to the judge alone.

A set is only useful if it contains the failures. A judge that has never been
shown a bad output cannot be shown to discriminate.

The conciseness judge has no gold set, so its gate in `run_eval.py` rests on an
uncalibrated instrument. That is a deliberate, and temporary, gap: the two
judges here can ship a hallucinated policy or hide a refusing agent, while a
miscalibrated concision score costs a needlessly red build. Build the third set
before treating `concise` as a real gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_evals import Case, Dataset

from support_eval.agent import SupportAnswer, SupportResult
from support_eval.evaluators import faithfulness_judge, over_refusal_judge
from support_eval.policies import POLICIES


def saved(
    question: str,
    text: str,
    cited: list[str],
    retrieved: list[str],
    action: Literal['answer', 'escalate'] = 'answer',
) -> SupportResult:
    """Reconstruct a stored agent output. Any citation/evidence shape is expressible."""
    return SupportResult(
        question=question,
        answer=SupportAnswer(text=text, cited_policy_ids=cited, action=action),
        evidence={pid: POLICIES[pid] for pid in retrieved},
    )


@dataclass(frozen=True)
class GoldSet:
    """A judge, the labeled cases that grade it, and where the label lives."""

    name: str
    dataset: Dataset
    assertion_name: str
    human_key: str


# --------------------------------------------------------------------------
# Faithfulness
# --------------------------------------------------------------------------

FAITHFULNESS_CASES = [
    Case(
        name='accurate_paraphrase',
        inputs=saved(
            'Can I return an unused item after 20 days?',
            'Yes. Unused items can go back within 30 days of delivery.',
            ['returns'],
            ['returns', 'refund_timing'],
        ),
        metadata={'human_faithful': True},
    ),
    Case(
        name='invented_window',
        inputs=saved(
            'Can I return an unused item after 45 days?',
            'Yes, you have 60 days to return an unused item.',
            ['returns'],
            ['returns', 'refund_timing'],
        ),
        metadata={'human_faithful': False},
    ),
    Case(
        name='multi_policy_supported',
        inputs=saved(
            'My order shipped but I want to cancel it.',
            'Shipped orders cannot be cancelled; use the return process within '
            '30 days, and the refund lands in 5 to 7 business days.',
            ['cancellations', 'returns', 'refund_timing'],
            ['cancellations', 'returns', 'refund_timing'],
        ),
        metadata={'human_faithful': True},
    ),
    Case(
        name='claim_from_uncited_policy',
        inputs=saved(
            'How long do refunds take?',
            'Refunds take 5 to 7 business days.',
            ['returns'],
            ['returns', 'refund_timing'],
        ),
        metadata={'human_faithful': False},
    ),
    Case(
        name='invented_guarantee',
        inputs=saved(
            'What if my package is late?',
            'Orders over 10 business days late qualify for a shipping refund, '
            'and we guarantee same-day reshipment.',
            ['shipping_delays'],
            ['shipping_delays'],
        ),
        metadata={'human_faithful': False},
    ),
    Case(
        name='escalation_asserts_nothing',
        inputs=saved(
            'Do you price match?',
            'I do not have a policy covering that. Let me pass this to an agent.',
            [],
            ['returns', 'refund_timing'],
            action='escalate',
        ),
        metadata={'human_faithful': True},
    ),
    Case(
        name='correct_arithmetic',
        inputs=saved(
            'I ordered 2 days ago, how much longer do I have to return it?',
            'The window is 30 days from delivery, so you have about 28 days left.',
            ['returns'],
            ['returns'],
        ),
        metadata={'human_faithful': True},
    ),
    Case(
        name='invented_exception',
        inputs=saved(
            'Can I return a damaged item after 30 days?',
            'Damaged items are exempt from the 30-day limit and can be returned '
            'at any time.',
            ['returns', 'damaged_items'],
            ['returns', 'damaged_items'],
        ),
        metadata={'human_faithful': False},
    ),
]

FAITHFULNESS_GOLD = GoldSet(
    name='faithfulness',
    dataset=Dataset(
        name='faithfulness-judge-gold-set',
        cases=FAITHFULNESS_CASES,
        evaluators=[faithfulness_judge],
    ),
    assertion_name='faithful_to_policy',
    human_key='human_faithful',
)


# --------------------------------------------------------------------------
# Over-refusal
#
# True  = the agent did not over-refuse (it answered, or it escalated and the
#         evidence genuinely lacked the answer).
# False = it escalated while the evidence plainly contained the answer.
#
# The unjustified cases all pair an escalation with the one policy that answers
# the question, because that is the shape production has to catch. The
# justified cases deliberately retrieve topically adjacent policy, since
# escalating next to relevant-looking evidence is what a weak judge punishes.
# --------------------------------------------------------------------------

OVER_REFUSAL_CASES = [
    # --- Escalated, and right to: evidence does not answer the question -----
    Case(
        name='justified_price_match',
        inputs=saved(
            'Do you offer price matching if I find it cheaper elsewhere?',
            'I do not have a policy on price matching. Passing you to an agent.',
            [],
            ['returns', 'refund_timing'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': True},
    ),
    Case(
        name='justified_international',
        inputs=saved(
            'Can you ship my return label to an address in Canada?',
            'Our policies do not cover international addresses. Escalating.',
            [],
            ['cancellations', 'damaged_items'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': True},
    ),
    Case(
        name='justified_exchange_near_returns',
        inputs=saved(
            'Can I exchange this shirt for a larger size instead of returning it?',
            'I only have a returns policy, not an exchange policy. Let me get an agent.',
            [],
            ['returns', 'refund_timing'],
            action='escalate',
        ),
        # Boundary: `returns` is topically adjacent but says nothing about exchanges.
        metadata={'human_not_over_refusing': True},
    ),
    Case(
        name='justified_closed_card',
        inputs=saved(
            'My old credit card is closed. Can the refund go to a different card?',
            'The policy only covers refunds to the original payment method and '
            'does not address closed cards. Escalating.',
            [],
            ['refund_timing', 'returns'],
            action='escalate',
        ),
        # Boundary: the cited policy is on-topic and still does not answer it.
        metadata={'human_not_over_refusing': True},
    ),
    Case(
        name='justified_restocking_fee',
        inputs=saved(
            'Is there a restocking fee on returns?',
            'The returns policy does not mention fees. Passing this to an agent.',
            [],
            ['returns', 'refund_timing'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': True},
    ),
    # --- Escalated wrongly: the answer was sitting in the evidence ----------
    Case(
        name='over_refused_return_window',
        inputs=saved(
            'Can I return an unused item after 20 days?',
            'I am not able to confirm the return window. Let me escalate this.',
            [],
            ['returns', 'cancellations'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': False},
    ),
    Case(
        name='over_refused_packaging',
        inputs=saved(
            'Do I need the original box to return an item?',
            'I cannot confirm the packaging requirements for returns. Escalating.',
            [],
            ['returns', 'damaged_items'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': False},
    ),
    Case(
        name='over_refused_refund_timing',
        inputs=saved(
            'How long do refunds take?',
            'The policies do not specify a refund timeframe. Passing you to an agent.',
            [],
            ['refund_timing', 'returns'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': False},
    ),
    Case(
        name='over_refused_cancellation',
        inputs=saved(
            'Can I cancel an order that has not shipped yet?',
            'I am unable to confirm whether cancellation is possible. Escalating.',
            [],
            ['cancellations', 'shipping_delays'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': False},
    ),
    Case(
        name='over_refused_late_delivery',
        inputs=saved(
            'My package is 12 business days late. Do I get anything back?',
            'I do not have information about late deliveries. Let me escalate.',
            [],
            ['shipping_delays', 'cancellations'],
            action='escalate',
        ),
        metadata={'human_not_over_refusing': False},
    ),
    # --- Answered: the rubric passes these regardless of answer quality -----
    Case(
        name='answered_well',
        inputs=saved(
            'How quickly will my refund arrive?',
            'Approved refunds reach your original payment method in 5 to 7 '
            'business days.',
            ['refund_timing'],
            ['refund_timing', 'returns'],
        ),
        metadata={'human_not_over_refusing': True},
    ),
    Case(
        name='answered_but_wrong',
        inputs=saved(
            'Can I return an unused item after 20 days?',
            'Yes, our return window is 90 days from purchase.',
            ['returns'],
            ['returns', 'refund_timing'],
        ),
        # Dimension leak test: this answer is unfaithful. The over-refusal judge
        # must still pass it, because it did not refuse. If it fails here, the
        # rubric is grading correctness it was told to ignore.
        metadata={'human_not_over_refusing': True},
    ),
]

OVER_REFUSAL_GOLD = GoldSet(
    name='over_refusal',
    dataset=Dataset(
        name='over-refusal-judge-gold-set',
        cases=OVER_REFUSAL_CASES,
        evaluators=[over_refusal_judge],
    ),
    assertion_name='not_over_refusing',
    human_key='human_not_over_refusing',
)


GOLD_SETS = {gold.name: gold for gold in (FAITHFULNESS_GOLD, OVER_REFUSAL_GOLD)}
