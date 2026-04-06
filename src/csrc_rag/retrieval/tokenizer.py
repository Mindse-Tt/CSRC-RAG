from __future__ import annotations

import re


ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
LAW_PATTERN = re.compile(r"《[^》]{1,40}》")


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []

    normalized = text.lower().strip()
    tokens: list[str] = []

    for law_name in LAW_PATTERN.findall(normalized):
        tokens.append(law_name)

    for token in ALNUM_PATTERN.findall(normalized):
        tokens.append(token)

    for chunk in CJK_PATTERN.findall(normalized):
        compact = chunk.strip()
        if not compact:
            continue
        if len(compact) <= 2:
            tokens.append(compact)
            continue
        tokens.append(compact[:6])
        for idx in range(len(compact) - 1):
            tokens.append(compact[idx : idx + 2])

    return tokens

