from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTask:
    name: str
    target_field: str
    allowed_input_fields: list[str]
    blocked_input_fields: list[str]
    metrics: list[str]


PUNISHMENT_TYPE_TASK = TrainingTask(
    name="punishment_type_multilabel",
    target_field="PunishmentType",
    allowed_input_fields=[
        "Activity",
        "Law",
        "Party",
        "Position",
        "Relationship",
        "Promulgator",
        "DeclareDate",
        "IsListedCom",
    ],
    blocked_input_fields=[
        "PunishmentType",
        "PunishmentMeasure",
        "SumPenalty",
    ],
    metrics=["Micro-F1", "Macro-F1", "HammingLoss", "SubsetAccuracy"],
)

