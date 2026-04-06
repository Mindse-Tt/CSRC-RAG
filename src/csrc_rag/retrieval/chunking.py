from __future__ import annotations

from typing import Any

from csrc_rag.data.builders import _normalize_binary_flag


def sliding_text_chunks(text: str | None, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []

    compact = text.strip()
    if not compact:
        return []
    if len(compact) <= chunk_size:
        return [compact]

    chunks: list[str] = []
    start = 0
    while start < len(compact):
        end = min(len(compact), start + chunk_size)
        chunk = compact[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(compact):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_event_chunks(
    event_documents: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in event_documents:
        event_id = event["event_id"]
        year = event["declare_date"][:4] if event.get("declare_date") else None
        common_meta = {
            "event_id": event_id,
            "title": event.get("title"),
            "declare_date": event.get("declare_date"),
            "supervision_date": event.get("supervision_date"),
            "promulgator": event.get("promulgator"),
            "supervisor": event.get("supervisor"),
            "year": year,
            "is_listed_company": _normalize_binary_flag(event.get("is_listed_company")),
            "violation_types": event.get("violation_types", []),
            "punishment_types": event.get("punishment_types", []),
        }

        summary_parts = [
            f"标题：{event.get('title')}" if event.get("title") else None,
            f"公告日期：{event.get('declare_date')}" if event.get("declare_date") else None,
            f"发布机构：{event.get('promulgator')}" if event.get("promulgator") else None,
            f"处罚机构：{event.get('supervisor')}" if event.get("supervisor") else None,
            f"违规类型：{'；'.join(event.get('violation_types', []))}" if event.get("violation_types") else None,
            f"核心行为：{(event.get('activity') or '')[:180]}",
        ]
        summary_text = "\n".join(part for part in summary_parts if part)
        rows.append(
            {
                "chunk_id": f"{event_id}::summary",
                "chunk_type": "summary",
                "section": "summary",
                "chunk_text": summary_text,
                "retrieval_text": summary_text,
                **common_meta,
            }
        )

        for idx, chunk in enumerate(sliding_text_chunks(event.get("activity"), chunk_size=chunk_size, overlap=overlap)):
            retrieval_text = "\n".join(
                part
                for part in [
                    f"标题：{event.get('title')}" if event.get("title") else None,
                    f"违规行为片段：{chunk}",
                    f"违规类型：{'；'.join(event.get('violation_types', []))}" if event.get("violation_types") else None,
                    f"发布机构：{event.get('promulgator')}" if event.get("promulgator") else None,
                ]
                if part
            )
            rows.append(
                {
                    "chunk_id": f"{event_id}::activity::{idx}",
                    "chunk_type": "activity",
                    "section": "activity",
                    "chunk_text": chunk,
                    "retrieval_text": retrieval_text,
                    **common_meta,
                }
            )

        for idx, chunk in enumerate(sliding_text_chunks(event.get("law"), chunk_size=chunk_size, overlap=overlap)):
            retrieval_text = "\n".join(
                part
                for part in [
                    f"标题：{event.get('title')}" if event.get("title") else None,
                    f"法律依据片段：{chunk}",
                    f"发布机构：{event.get('promulgator')}" if event.get("promulgator") else None,
                ]
                if part
            )
            rows.append(
                {
                    "chunk_id": f"{event_id}::law::{idx}",
                    "chunk_type": "law",
                    "section": "law",
                    "chunk_text": chunk,
                    "retrieval_text": retrieval_text,
                    **common_meta,
                }
            )
    return rows
