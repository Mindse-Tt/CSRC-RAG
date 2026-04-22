"""Topic guard -- determines whether a query is within the CSRC punishment-case domain.

Two complementary checks:
1. Hard blocklist: keywords clearly outside scope (weather, cooking, games, etc.)
2. Domain presence: at least one domain keyword must appear for non-trivial queries.

Returns a (is_out_of_scope: bool, reason: str) tuple so callers can surface
informative refusal messages.
"""
from __future__ import annotations

# keywords that are definitively off-topic
_BLOCKLIST_TOKENS: frozenset[str] = frozenset([
    # everyday / general
    "\u5929\u6c14", "\u6e29\u5ea6", "\u660e\u5929", "\u98df\u8c31", "\u70f9\u996a",
    "\u505a\u996d", "\u6e38\u620f", "\u5a31\u4e50", "\u7535\u5f71", "\u97f3\u4e50",
    "\u6b4c\u66f2", "\u4f53\u80b2", "\u7403\u8d5b", "\u8db3\u7403", "\u7bee\u7403",
    "\u65c5\u6e38", "\u666f\u70b9", "\u9152\u5e97", "\u673a\u7968",
    "\u8d2d\u7269", "\u5feb\u9012", "\u5916\u5356", "\u51cf\u80a5", "\u5065\u8eab",
    "\u5ba0\u7269", "\u517b\u732b", "\u517b\u72d7",
    # medical / legal outside CSRC
    "\u533b\u9662", "\u770b\u75c5", "\u75c7\u72b6", "\u836f\u7269",
    "\u79bb\u5a5a", "\u7ee7\u627f", "\u623f\u4ea7", "\u4ea4\u901a\u4e8b\u6545",
    # technology unrelated to finance
    "\u7f16\u7a0b\u9898", "leetcode", "\u7b97\u6cd5\u9898", "\u6570\u5b66\u9898", "\u7269\u7406\u9898",
    # sensitive / political
    "\u653f\u6cbb", "\u9009\u4e3e", "\u515a\u6d3e",
])

# domain keywords that indicate CSRC / securities scope
_DOMAIN_TOKENS: frozenset[str] = frozenset([
    # entity types
    "\u8bc1\u76d1\u4f1a", "\u8bc1\u76d1\u5c40", "\u4ea4\u6613\u6240",
    "\u4e0a\u5e02\u516c\u53f8", "\u8bc1\u5238", "\u671f\u8d27", "\u57fa\u91d1",
    "\u80a1\u7968", "\u503a\u5238", "\u884d\u751f\u54c1", "\u6295\u8d44\u8005",
    "\u53d1\u884c\u4eba", "\u5238\u5546",
    # violation types
    "\u5185\u5e55\u4ea4\u6613", "\u4fe1\u606f\u62ab\u9732", "\u64cd\u7eb5\u5e02\u573a",
    "\u865a\u5047\u9648\u8ff0", "\u6b3a\u8bc8", "\u8fdd\u89c4",
    "\u77ed\u7ebf\u4ea4\u6613", "\u4e0d\u5f53\u64cd\u4f5c", "\u8fdd\u53cd",
    # regulatory / legal
    "\u5904\u7f5a", "\u884c\u653f\u5904\u7f5a", "\u8b66\u544a", "\u7f5a\u6b3e",
    "\u6ca1\u6536", "\u5e02\u573a\u7981\u5165", "\u6682\u505c", "\u6492\u9500",
    "\u6cd5\u6761", "\u6cd5\u89c4", "\u8bc1\u5238\u6cd5", "\u57fa\u91d1\u6cd5", "\u5211\u6cd5",
    # analytical
    "\u6848\u4f8b", "\u6848\u4ef6", "\u8d8b\u52bf", "\u5206\u5e03", "\u7edf\u8ba1",
    "\u5386\u53f2", "\u8fd1\u5e74", "\u5e74\u5ea6",
    # company identifiers
    "ST", "\u9000\u5e02", "\u518d\u878d\u8d44", "IPO", "\u91cd\u7ec4",
])

# A query shorter than this character threshold passes without domain check
_SHORT_QUERY_THRESHOLD = 6

_LQUOTE = "\u201c"
_RQUOTE = "\u201d"
_SCOPE_MSG = (
    "\u672c\u7cfb\u7edf\u4ec5\u652f\u6301\u8bc1\u5238\u8fdd\u89c4\u6848\u4f8b\u68c0\u7d22\u3001"
    "\u6cd5\u89c4\u4f9d\u636e\u67e5\u8be2\u3001\u5904\u7f5a\u63a8\u8350\u548c\u8d8b\u52bf\u5206\u6790\u3002"
)


def is_out_of_scope(query: str) -> tuple[bool, str]:
    """Return (True, reason) if the query is out of scope, else (False, '').

    Args:
        query: The user's raw query string.

    Returns:
        A tuple (out_of_scope, reason).  When out_of_scope is True, reason
        contains a human-readable Chinese explanation to show the user.
    """
    stripped = query.strip()
    if not stripped:
        return False, ""

    # 1. Blocklist check -- any blocked token triggers immediate rejection
    for token in _BLOCKLIST_TOKENS:
        if token in stripped:
            reason = (
                "\u8be5\u95ee\u9898\uff08\u542b\u5173\u952e\u8bcd"
                + _LQUOTE + token + _RQUOTE
                + "\uff09\u4e0d\u5728\u8bc1\u76d1\u4f1a\u5904\u7f5a\u6848\u4f8b\u5206\u6790\u8303\u56f4\u5185\u3002"
                + _SCOPE_MSG
            )
            return True, reason

    # 2. Short queries pass directly
    if len(stripped) <= _SHORT_QUERY_THRESHOLD:
        return False, ""

    # 3. Domain presence check for longer queries
    has_domain_token = any(token in stripped for token in _DOMAIN_TOKENS)
    if not has_domain_token:
        reason = (
            "\u8be5\u95ee\u9898\u672a\u68c0\u6d4b\u5230\u4e0e\u8bc1\u76d1\u4f1a\u5904\u7f5a\u6848\u4f8b"
            "\u76f8\u5173\u7684\u5173\u952e\u8bcd\uff0c\u53ef\u80fd\u8d85\u51fa\u7cfb\u7edf\u8303\u56f4\u3002"
            "\u8bf7\u56f4\u7ed5\u8bc1\u5238\u8fdd\u89c4\u3001\u5904\u7f5a\u6848\u4f8b\u3001"
            "\u6cd5\u89c4\u4f9d\u636e\u6216\u8d8b\u52bf\u5206\u6790\u63d0\u95ee\u3002"
        )
        return True, reason

    return False, ""
