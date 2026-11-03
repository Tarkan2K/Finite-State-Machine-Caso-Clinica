"""Definición de estados del ciclo de vida del agente. (mío)"""

"""Data contracts and schemas for Aura Clinical runtime engine."""

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineState(str, Enum):
    """Execution states for the Finite State Machine."""

    INITIALIZED = "INITIALIZED"
    PLANNING = "PLANNING"
    DISPATCHING_TOOL = "DISPATCHING_TOOL"
    AWAITING_OBSERVATION = "AWAITING_OBSERVATION"
    REFLECTING = "REFLECTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ClinicalSafetyStatus(str, Enum):
    """Terminal safety verdict for a clinical session."""

    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL_BLOCK = "CRITICAL_BLOCK"


class ClinicalNote(BaseModel):
    """Unstructured text written by a physician at bedside."""

    note_id: str = Field(description="Unique UUID for the clinical note")
    patient_ref: str = Field(description="Anonymized patient reference ID")
    raw_text: str = Field(description="Free-text clinical note written by clinician")
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the note was submitted",
    )


class ExtractedMedication(BaseModel):
    """A medication span extracted from unstructured clinical text."""

    name: str = Field(description="Normalized active pharmaceutical ingredient")
    dose_mg: float | None = Field(default=None, description="Extracted dose in milligrams")
    route: str | None = Field(default=None, description="Administration route: oral, IV, IM, etc.")


class DrugInteractionAlert(BaseModel):
    """Structured clinical alert identifying a dangerous drug-drug interaction."""

    drug_a: str = Field(description="First interacting active substance")
    drug_b: str = Field(description="Second interacting active substance")
    severity: Literal["LOW", "MODERATE", "CRITICAL"] = Field(
        description="Clinical severity grade"
    )
    description: str = Field(description="Pathophysiological risk description")
    cie10_codes: list[str] = Field(
        default_factory=list, description="Related ICD-10 diagnostic/adverse effect codes"
    )


class PatientAuditRecord(BaseModel):
    """
    Immutable audit record stored for legal compliance and telemetry.
    Configured with frozen=True to prevent post-creation tampering.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    patient_ref: str
    note_text: str
    medications_extracted: list[str]
    safety_status: ClinicalSafetyStatus
    alerts: list[DrugInteractionAlert]
    fsm_final_state: EngineState
    processing_time_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))