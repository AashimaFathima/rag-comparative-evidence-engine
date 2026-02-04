import re
from typing import Dict

# -------------------------------------------------
# Regex patterns that indicate RESULT / FINDING
# Domain-agnostic: works for ML, medicine, education,
# economics, public policy, social sciences, etc.
# -------------------------------------------------

RESULT_PATTERNS = [
    # performance / outcome language
    r"\b(performed|performance|outcome|outcomes|effect|effects)\b",
    r"\b(improved|improvement|declined|decrease|increase|reduced|reduction)\b",
    r"\b(higher|lower|better|worse|significant|insignificant)\b",

    # comparison language
    r"\b(compared to|relative to|versus|vs\.?|than)\b",
    r"\b(difference|gap|no difference|similar|equivalent|equally)\b",

    # statistics / evaluation
    r"\b(p\s*[<=>]\s*0?\.\d+)\b",
    r"\b(mean|average|median|coefficient|effect size)\b",
    r"\b(controlled for|after controlling|adjusted for)\b",

    # grades / success metrics
    r"\b(score|scores|grade|grades|test|exam|achievement)\b",
    r"\b(persistence|completion|retention|success rate)\b",
]

# -------------------------------------------------
# Regex patterns that indicate NON-RESULT content
# (used only if NO result pattern matches)
# -------------------------------------------------

NON_RESULT_PATTERNS = [
    # enrollment / adoption / usage
    r"\b(enrollment|enrolled|participation|uptake|adoption|usage rates)\b",

    # attitudes / perception
    r"\b(attitude|perception|acceptance|belief|opinion|satisfaction)\b",

    # logistics / admin / policy 
    r"\b(policy|strategy|planning|implementation|infrastructure)\b",

    # purely descriptive trends
    r"\b(number of students|percent of students|proportion of institutions)\b",

    # methods-only text
    r"\b(method|methodology|experimental design|dataset|sample size)\b",
]


def is_result_claim(text: str) -> bool:
    """
    Returns True if the claim looks like an author-reported
    result, finding, or evaluated outcome.
    """
    t = text.lower()

    for pat in RESULT_PATTERNS:
        if re.search(pat, t):
            return True

    for pat in NON_RESULT_PATTERNS:
        if re.search(pat, t):
            return False

    return True


def filter_result_claims(extracted_claims: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Filters extracted claims paper-wise.
    Keeps only result / finding claims.
    """

    filtered = {}

    for paper_id, data in extracted_claims.items():
        claims = data.get("claims", [])

        kept_claims = []
        for c in claims:
            claim_text = c.get("claim", "")
            if claim_text and is_result_claim(claim_text):
                kept_claims.append(c)

        filtered[paper_id] = {"claims": kept_claims}

    return filtered
