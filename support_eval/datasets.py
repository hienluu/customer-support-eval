"""The offline evaluation set.

Every case carries a human `expected_action` label. Cases marked `escalate` are
ones the corpus genuinely cannot answer -- the retriever will still hand the
agent two policies for each of them.
"""

from __future__ import annotations

from pydantic_evals import Case, Dataset

from support_eval.agent import SupportResult
from support_eval.evaluators import JUDGES, PROGRAMMATIC_EVALUATORS, CaseMeta

CASES = [
    Case(
        name='return_window',
        inputs='Can I return an unused item after 20 days?',
        metadata=CaseMeta('answer', 'Inside the 30-day window.', ('returns',)),
    ),
    Case(
        name='return_too_late',
        inputs='Can I return something I bought 45 days ago?',
        metadata=CaseMeta('answer', 'Policy answers this: no, window is 30 days.', ('returns',)),
    ),
    Case(
        name='refund_timing',
        inputs='How quickly will my refund arrive?',
        metadata=CaseMeta('answer', '5 to 7 business days.', ('refund_timing',)),
    ),
    Case(
        name='cancel_unshipped',
        inputs='Can I cancel an order that has not shipped yet?',
        metadata=CaseMeta('answer', 'Free cancellation before shipping.', ('cancellations',)),
    ),
    Case(
        name='cancel_after_shipping',
        inputs='I want to cancel my order but it already shipped. What now?',
        metadata=CaseMeta('answer', 'Needs cancellations plus returns.', ('cancellations', 'returns')),
    ),
    Case(
        name='damaged_on_arrival',
        inputs='My order arrived with a cracked screen. What should I do?',
        metadata=CaseMeta('answer', 'Report within 48 hours with photos.', ('damaged_items',)),
    ),
    Case(
        name='late_delivery_refund',
        inputs='My package is 12 business days late. Do I get anything back?',
        metadata=CaseMeta('answer', 'Qualifies for a shipping-cost refund.', ('shipping_delays',)),
    ),
    Case(
        name='packaging_requirement',
        inputs='Do I need the original box to return an item?',
        metadata=CaseMeta('answer', 'Original packaging is required.', ('returns',)),
    ),
    Case(
        name='price_matching',
        inputs='Do you offer price matching if I find it cheaper elsewhere?',
        metadata=CaseMeta('escalate', 'No price-match policy exists.'),
    ),
    Case(
        name='international_shipping',
        inputs='Can you ship my return label to an address in Canada?',
        metadata=CaseMeta('escalate', 'No international policy; "ship"/"return" still retrieve.'),
    ),
    Case(
        name='size_exchange',
        inputs='Can I exchange this shirt for a larger size instead of returning it?',
        metadata=CaseMeta('escalate', 'No exchange policy; returns will be retrieved.'),
    ),
    Case(
        name='refund_to_new_card',
        inputs='My old credit card is closed. Can the refund go to a different card?',
        metadata=CaseMeta('escalate', 'refund_timing retrieves but does not cover this.'),
    ),
]

dataset = Dataset[str, SupportResult, CaseMeta](
    name='support-policy-agent',
    cases=CASES,
    evaluators=[*PROGRAMMATIC_EVALUATORS, *JUDGES],
)
