"""Data contracts and schemas for Aura Clinical runtime engine."""

from datetime import timezone, datetime
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


class PatientRecord(BaseModel):
    """EHR patient record with Chilean clinical demographic and pharmacological baseline."""

    model_config = ConfigDict(frozen=True)

    rut: str = Field(description="RUT chileno del paciente (ej. 12.345.678-5)")
    nombre: str = Field(description="Nombre completo del paciente")
    edad: int = Field(description="Edad en años")
    ciudad: str = Field(description="Ciudad de residencia")
    enfermedades_cronicas: list[str] = Field(default_factory=list)
    alergias: list[str] = Field(default_factory=list)
    farmacos_actuales: list[str] = Field(default_factory=list)


class ClinicalNote(BaseModel):
    """Unstructured text written by a physician at bedside."""

    note_id: str = Field(description="Unique ID for the clinical note")
    patient_ref: str = Field(description="RUT del paciente")
    raw_text: str = Field(description="Free-text clinical note written by clinician")
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractedMedication(BaseModel):
    """A medication extracted from clinical text with AST-evaluated daily dosage."""

    name: str = Field(description="Normalized active pharmaceutical ingredient")
    dose_expression: str | None = Field(
        default=None,
        description="Arithmetic expression for daily dose in mg (e.g. '20 * 1000 * (24 / 8)')",
    )
    dose_mg: float | None = Field(
        default=None, description="Evaluated total daily dose in milligrams"
    )
    max_safe_daily_mg: float | None = Field(
        default=None, description="Maximum safe daily dosage ceiling in milligrams"
    )
    route: str | None = Field(default=None, description="Administration route: oral, IV, IM, etc.")


class DrugInteractionAlert(BaseModel):
    """Structured clinical alert identifying a dangerous interaction, allergy, condition, or overdose."""

    severity: Literal["LOW", "MODERATE", "CRITICAL"] = Field(
        description="Clinical severity grade"
    )
    description: str = Field(description="Pathophysiological risk description")
    cie10_codes: list[str] = Field(
        default_factory=list, description="Related ICD-10 diagnostic/adverse effect codes"
    )


class PatientAuditRecord(BaseModel):
    """Immutable audit record stored for legal compliance and telemetry."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    patient_ref: str
    note_text: str
    medications_extracted: list[str]
    safety_status: ClinicalSafetyStatus
    alerts: list[DrugInteractionAlert]
    fsm_final_state: EngineState
    processing_time_ms: float
    created_at: datetime
