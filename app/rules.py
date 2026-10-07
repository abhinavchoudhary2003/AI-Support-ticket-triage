"""
Escalation rules (no LLM, no third-party imports).

Fix A: legal-threat detection and refund/amount detection.
These rules are deterministic, so they cannot be talked around by the customer
and do not depend on how the LLM happened to label the ticket.
"""

import re
import unicodedata

HUMAN = "Human"
AI = "AI"

CONFIDENCE_THRESHOLD = 0.70
COVERAGE_THRESHOLD = 0.50
REFUND_LIMIT_USD = 200.0
# Another (or an unstated) currency can't be converted reliably, so amounts above
# this face value go to a human instead of being guessed at.
NON_USD_REVIEW_THRESHOLD = 100.0

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalize_text(text: str) -> str:
    """Lower-case, fold unicode tricks, strip zero-width characters, collapse spaces."""
    text = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).casefold().strip()


def _value(x) -> str:
    """Accept an Enum member or a plain string."""
    return str(getattr(x, "value", x)).strip().lower()

# --------------------------------------------------------------------------- prompt injection
# First-layer filter on normalised text (lower-cased, unicode-folded, zero-width
# characters removed, whitespace collapsed), so spacing and case tricks do not
# help an attacker. It is a heuristic, not a guarantee: the prompts must still
# treat ticket text as untrusted data, and the escalation rules never depend on
# anything the customer can talk the model into saying.
_NOT_A_MESSAGE = r"(?!\s+(?:e-?mail|mail|message|ticket|request|reply|response|order|one)\b)"

_INJECTION_PATTERNS = [
    re.compile(p)
    for p in (
        # "ignore / disregard / forget ... (all|the|your) ... instructions|rules|prompt|knowledge base"
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
        r"\b(?:previous|prior|above|earlier|all|your|the|any)\b[^.\n]{0,30}"
        r"\b(?:instructions?|rules?|prompts?|guidelines?|knowledge base)\b",
        # "ignore the above" / "disregard all prior" with no noun (but not "ignore my previous email")
        r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:of\s+)?(?:the\s+)?"
        r"(?:above|previous|prior|preceding|earlier)\b" + _NOT_A_MESSAGE,
        # "do not follow your rules", "stop obeying the guidelines"
        r"\b(?:do not|don't|dont|stop)\s+(?:follow|following|obey|obeying)\b[^.\n]{0,30}"
        r"\b(?:instructions?|rules?|guidelines?|polic(?:y|ies))\b",
        # asking for hidden instructions
        r"\b(?:reveal|show|print|display|repeat|leak|tell me)\b[^.\n]{0,30}"
        r"\b(?:system|hidden|internal|developer)\s+(?:prompt|message|instructions?)\b",
        r"\bsystem prompt\b",
        # role / mode switching
        r"\byou(?:'re| are)\s+(?:now\s+)?(?:in|on)\s+"
        r"(?:developer|dev|admin|god|debug|unrestricted|jailbreak|dan)\s+mode\b",
        r"\b(?:enter|activate|switch to|enable)\s+(?:dan|god|jailbreak|unrestricted)\s+mode\b",
        r"\byou are now\s+(?:an?|the|my|called|free|unrestricted|dan)\b",
        r"\bpretend\s+(?:to be|you are|that you)\b",
        r"\bact as (?:if|though) you (?:have|had) no\b",
        r"\bfrom now on,?\s+you\s+(?:will|must|are|should|shall)\b",
        r"\bdo anything now\b",
        r"\bjailbreak\b",
        r"\bnew instructions?\s*:",
    )
]


def detect_prompt_injection(text: str) -> bool:
    norm = normalize_text(text)
    return any(p.search(norm) for p in _INJECTION_PATTERNS)

# --------------------------------------------------------------------------- legal threats
# Whole-word matching. A plain substring test flags "issue" (contains "sue")
# and "courtesy" (contains "court") as legal threats.
_LEGAL = re.compile(
    r"\b(?:lawyers?|attorneys?|solicitors?|advocates?|lawsuits?|sue|sued|suing|"
    r"court|legal action|legal proceedings|legal notice|legal team|"
    r"small claims|consumer court|consumer forum)\b"
)


def mentions_legal_threat(text: str) -> bool:
    return bool(_LEGAL.search(normalize_text(text)))


# --------------------------------------------------------------------------- amounts
_AMT = r"(?P<amt>\d[\d,]*(?:\.\d+)?)(?!\d)(?:\s?(?P<mult>k|thousand)\b)?"
_CUR = r"(?:usd|dollars?|inr|rs\.?|rupees?|eur|euros?|gbp|pounds?)"
_USD_WORDS = {"usd", "dollar", "dollars"}

_SYMBOL = re.compile(r"(?P<sym>[$\u20b9\u20ac\u00a3])\s*" + _AMT)
_WORD_AFTER = re.compile(_AMT + r"\s*(?P<cur>" + _CUR + r")(?![a-z])")
_WORD_BEFORE = re.compile(r"\b(?P<cur>usd|inr|rs\.?|eur|gbp)\s*" + _AMT)
_BARE = re.compile(
    r"\brefund(?:s|ed)?\s+(?:of|for)\s+" + _AMT
    + r"(?!\s*(?:" + _CUR + r"|days?|weeks?|months?|years?|hours?|"
    r"users?|seats?|licen[cs]es?|times?|items?|orders?))"
)

# Words that mean the customer wants money moved back to them.
# ("credit card" is excluded so "update my credit card" is not a money request.)
_MONEY_MOVEMENT = re.compile(
    r"\b(?:refund\w*|reimburs\w*|money back|pay(?:ment)? back|chargebacks?|"
    r"charged|charges?|overcharg\w*|double[- ]charged|reverse[ds]?|reversal|"
    r"credit(?!\s*cards?)\w*|return(?:ed)?|dispute[ds]?|disputing)\b"
)


def extract_amounts(text: str):
    """Return [(amount, currency)] with currency 'USD', 'OTHER' or 'UNKNOWN'."""
    norm = normalize_text(text)
    found = []

    def add(match, currency):
        amount = float(match.group("amt").replace(",", ""))
        if match.group("mult"):
            amount *= 1000
        found.append((amount, currency))

    for m in _SYMBOL.finditer(norm):
        add(m, "USD" if m.group("sym") == "$" else "OTHER")
    for m in _WORD_AFTER.finditer(norm):
        add(m, "USD" if m.group("cur") in _USD_WORDS else "OTHER")
    for m in _WORD_BEFORE.finditer(norm):
        add(m, "USD" if m.group("cur") in _USD_WORDS else "OTHER")
    for m in _BARE.finditer(norm):
        add(m, "UNKNOWN")
    return found


def refund_review_reason(category, message: str):
    """Reason string if the ticket needs human/Finance review, else None.

    Not gated on the LLM's category: a $1,200 "please reverse this charge" ticket
    labelled Billing must be caught just like one labelled Refund."""
    norm = normalize_text(message)
    is_money_request = _value(category) == "refund" or bool(_MONEY_MOVEMENT.search(norm))
    if not is_money_request:
        return None

    amounts = extract_amounts(message)
    if any(cur == "USD" and amt > REFUND_LIMIT_USD for amt, cur in amounts):
        return "Refund above $200 requires Finance approval."
    if any(cur != "USD" and amt > NON_USD_REVIEW_THRESHOLD for amt, cur in amounts):
        return "Refund amount is in a non-USD or unspecified currency and needs human review."
    return None


# --------------------------------------------------------------------------- decision
def decide_resolution(
    category,
    priority,
    confidence,
    message,
    coverage,
    prompt_injection_detected=False,
):
    """Return ("AI" | "Human", reason). Checked in order; first match wins."""
    if prompt_injection_detected:
        return HUMAN, "Possible prompt injection detected."

    if _value(priority) == "critical":
        return HUMAN, "Critical issue requires human review."

    if confidence < CONFIDENCE_THRESHOLD:
        return HUMAN, "Classification confidence is too low."

    if mentions_legal_threat(message):
        return HUMAN, "Legal threat requires human review."

    if coverage < COVERAGE_THRESHOLD:
        return HUMAN, "Knowledge base does not contain enough information."

    refund_reason = refund_review_reason(category, message)
    if refund_reason:
        return HUMAN, refund_reason

    return AI, "Ticket can be resolved using the available knowledge base."