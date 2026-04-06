from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csrc_rag.orchestration.intents import IntentSpec
from csrc_rag.settings import CONFIG_DIR
from csrc_rag.utils import read_json


@dataclass(frozen=True)
class ResponseOutput:
    text: str
    backend: str
    model_name: str | None = None


class TemplateResponder:
    def generate(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None = None,
    ) -> ResponseOutput:
        if not ranked_events:
            return ResponseOutput(
                text=f"未检索到与“{query}”高度相关的案例，请缩短描述或换一种问法。",
                backend="template",
            )

        if intent.name == "case_retrieval":
            lines = ["已检索到最相关的历史案例，优先建议查看以下案件："]
            for event in ranked_events[:5]:
                lines.append(f"- {event.title}（{event.declare_date}，{event.promulgator}）")
            return ResponseOutput(text="\n".join(lines), backend="template")

        if intent.name == "law_grounding":
            laws: list[str] = []
            seen: set[str] = set()
            for event in ranked_events:
                for law in event.laws:
                    if law and law not in seen:
                        seen.add(law)
                        laws.append(law[:180])
                    if len(laws) >= 3:
                        break
                if len(laws) >= 3:
                    break
            lines = ["根据当前检索到的相似案例，优先可关注以下法规依据："]
            for law in laws:
                lines.append(f"- {law}")
            lines.append("回答基于相似案例中的法条引用，不构成正式法律意见。")
            return ResponseOutput(text="\n".join(lines), backend="template")

        if intent.name == "sanction_recommendation":
            labels: dict[str, float] = {}
            for event in ranked_events[:5]:
                weight = max(float(event.score), 1.0)
                for label in event.punishment_types:
                    labels[label] = labels.get(label, 0.0) + weight
            lines = ["基于相似案例，当前更可能出现的处罚方式有："]
            for label, _score in sorted(labels.items(), key=lambda item: item[1], reverse=True)[:5]:
                lines.append(f"- {label}")
            lines.append("支撑依据来自已召回的相似处罚案例及其中的法规引用。")
            lines.append("这一步是案例支持下的推荐，不替代正式执法判断。")
            return ResponseOutput(text="\n".join(lines), backend="template")

        top_years = [event.declare_date[:4] for event in ranked_events if event.declare_date]
        lines = ["已返回相关案例，可进一步做趋势统计分析。"]
        if top_years:
            lines.append(f"- 当前相关结果主要集中年份：{'、'.join(top_years[:5])}")
        return ResponseOutput(text="\n".join(lines), backend="template")


class LocalHFResponder:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _build_prompt(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None,
    ) -> str:
        evidence_lines: list[str] = []
        for index, event in enumerate(ranked_events[:4], start=1):
            evidence_lines.append(
                "\n".join(
                    [
                        f"[案例{index}] 标题：{event.title or event.event_id}",
                        f"时间：{event.declare_date or '未知'}",
                        f"机构：{event.promulgator or '未知'}",
                        f"处罚方式：{'、'.join(event.punishment_types) or '未提取'}",
                        f"法规：{'；'.join(filter(None, event.laws[:2])) or '未提取'}",
                        f"证据片段：{' '.join(event.snippets[:2]) or '未提取'}",
                    ]
                )
            )
        history_lines = []
        for turn in (history or [])[-4:]:
            role = "用户" if turn.get("role") == "user" else "助手"
            history_lines.append(f"{role}：{turn.get('content', '').strip()}")

        sections = [
            "你是证监会处罚案例智能分析助手。",
            "你只能根据给定案例证据回答，禁止编造未出现的法条、处罚结果、金额或事实。",
            "如果证据不足，请明确写“证据不足”。",
            f"当前任务：{intent.description}",
        ]
        if history_lines:
            sections.extend(["对话历史：", "\n".join(history_lines)])
        sections.extend(
            [
                f"用户问题：{query}",
                "检索证据：",
                "\n\n".join(evidence_lines),
                "请输出：1. 结论摘要 2. 依据要点 3. 相似案例提示 4. 风险与不足。",
                "输出要求：",
                "- 结论摘要先直接回答问题，不要空话。",
                "- 如果是处罚推荐，明确写出1到3个最可能的处罚方式。",
                "- 依据要点优先引用法规名称或处罚类型，不要逐字复制长段原文。",
                "- 相似案例提示只写案例标题、年份和机构，不要贴整段证据。",
                "- 每一部分尽量简洁，避免超过4条。",
                "- 如果证据不足，直接写“证据不足”。",
                "回答用中文。",
            ]
        )
        return "\n\n".join(sections)

    def generate(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None = None,
    ) -> ResponseOutput:
        if not ranked_events:
            return ResponseOutput(
                text=f"未检索到与“{query}”高度相关的案例，请缩短描述或换一种问法。",
                backend="local_hf_empty",
                model_name=self.model_name,
            )
        self._ensure_model()

        import torch  # type: ignore

        prompt = self._build_prompt(query=query, intent=intent, ranked_events=ranked_events, history=history)
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None or self._device is None:
            raise RuntimeError("Local HF responder failed to initialize.")

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": "你是证监会处罚案例智能分析助手，只能依据给定证据回答。",
                },
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not answer:
            answer = "证据已召回，但本地回复模型未生成有效文本，请先查看下方案例证据。"
        return ResponseOutput(text=answer, backend="local_hf", model_name=self.model_name)


class CompositeResponder:
    def __init__(self, primary: Any, fallback: TemplateResponder) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None = None,
    ) -> ResponseOutput:
        try:
            return self.primary.generate(query=query, intent=intent, ranked_events=ranked_events, history=history)
        except Exception:
            return self.fallback.generate(query=query, intent=intent, ranked_events=ranked_events, history=history)


def build_responder(config_path: str | Path | None = None) -> CompositeResponder:
    config = read_json(config_path or CONFIG_DIR / "models.json")
    response_cfg = config.get("response_generation", {})
    fallback = TemplateResponder()
    backend = response_cfg.get("backend", "template")
    if backend == "local_hf":
        primary = LocalHFResponder(
            model_name=response_cfg["model_name"],
            max_new_tokens=int(response_cfg.get("max_new_tokens", 256)),
            temperature=float(response_cfg.get("temperature", 0.2)),
            top_p=float(response_cfg.get("top_p", 0.9)),
        )
        return CompositeResponder(primary=primary, fallback=fallback)
    return CompositeResponder(primary=fallback, fallback=fallback)
