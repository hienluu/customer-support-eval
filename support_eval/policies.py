"""The policy corpus and its retriever.

The retriever always returns `k` policies, ranked by term overlap and never
filtered by a relevance floor. That mirrors how top-k vector search actually
behaves, and it is the reason no evaluator in this project may treat "evidence
was retrieved" as a proxy for "the question is answerable".
"""

from __future__ import annotations

import math
import re

POLICIES: dict[str, str] = {
    'returns': (
        'Unused items can be returned within 30 days of delivery. '
        'Items must be in their original packaging.'
    ),
    'refund_timing': (
        'Approved refunds are issued to the original payment method within '
        '5 to 7 business days.'
    ),
    'cancellations': (
        'An order can be cancelled at no cost before it ships. '
        'Once an order has shipped it must go through the return process.'
    ),
    'damaged_items': (
        'Items that arrive damaged must be reported within 48 hours of '
        'delivery, with photographs of the damage.'
    ),
    'shipping_delays': (
        'Standard delivery takes 3 to 5 business days. An order delayed beyond '
        '10 business days qualifies for a refund of the shipping cost.'
    ),
}

_STOPWORDS = frozenset(
    'a an and are as at be by can do does for from how i if in is it my of on or '
    'the to was what when will with you your'.split()
)


def _stem(word: str) -> str:
    """Crude suffix stripping.

    Without this, the question word "return" never matches "returned" or
    "returns" in the returns policy, while the cancellations policy wins on a
    literal "return process". Every returns question then retrieves the wrong
    document.
    """
    for suffix in ('ing', 'ed', 'es', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:  # shipp -> ship, cancell -> cancel
        word = word[:-1]
    return word


def _terms(text: str) -> set[str]:
    return {
        _stem(w) for w in re.findall(r'[a-z]+', text.lower()) if w not in _STOPWORDS
    }


def _document(pid: str) -> str:
    return f'{pid.replace("_", " ")} {POLICIES[pid]}'


_INDEX: dict[str, set[str]] = {pid: _terms(_document(pid)) for pid in POLICIES}

# Rare terms discriminate; terms in every policy do not. "return" appears all
# over the corpus and should not outrank "packaging".
_IDF: dict[str, float] = {
    term: math.log(len(POLICIES) / sum(term in terms for terms in _INDEX.values()))
    for terms in _INDEX.values()
    for term in terms
}


def _score(asked: set[str], pid: str) -> float:
    return sum(_IDF.get(term, 0.0) for term in asked & _INDEX[pid])


def retrieve(question: str, k: int = 2) -> dict[str, str]:
    """Return the `k` best-scoring policies, relevant or not.

    There is deliberately no relevance floor: real top-k search returns `k`
    documents whatever their scores, and the evaluators are built to assume it.
    """
    asked = _terms(question)
    ranked = sorted(POLICIES, key=lambda pid: (-_score(asked, pid), pid))
    return {pid: POLICIES[pid] for pid in ranked[:k]}
