"""
Unit and security tests for the AST sandbox and clinical interaction validator.
Validates arithmetic safety, denylist enforcement, drug-drug contraindications, and overdose ceilings.
"""

import pytest

from src.exceptions import ToolSecurityError
from src.sandbox import (
    check_allergy_contraindications,
    check_condition_contraindications,
    check_dosage_safety,
    check_drug_interactions,
    validate_dose_expression,
)
from src.types import ExtractedMedication

# --- AST Expression Sandbox Tests ---


def test_valid_arithmetic_expressions() -> None:
    """Standard dose arithmetic operations must evaluate accurately."""
    assert validate_dose_expression("500 * 2") == 1000.0
    assert validate_dose_expression("(100 + 50) / 2") == 75.0
    assert validate_dose_expression("0.15 * 70") == 10.5
    assert validate_dose_expression("-10 + 20") == 10.0


def test_ast_blocks_arbitrary_function_calls() -> None:
    """Function and method invocations must be intercepted by AST traversal."""
    with pytest.raises(ToolSecurityError) as exc_info:
        validate_dose_expression("__import__('os').system('ls')")
    assert "Function and method calls are strictly forbidden" in str(exc_info.value)

    with pytest.raises(ToolSecurityError) as exc_info:
        validate_dose_expression("eval('2 + 2')")
    assert "Function and method calls are strictly forbidden" in str(exc_info.value)


def test_ast_blocks_attribute_access() -> None:
    """Attribute resolution (e.g. __class__) must be blocked to prevent sandbox escapes."""
    with pytest.raises(ToolSecurityError) as exc_info:
        validate_dose_expression("(1).__class__.__bases__")
    assert "Attribute access is forbidden" in str(exc_info.value)


def test_ast_blocks_blacklisted_identifiers() -> None:
    """Names present in denylist must raise ToolSecurityError before execution."""
    with pytest.raises(ToolSecurityError) as exc_info:
        validate_dose_expression("globals")
    assert "Unauthorized identifier blocked" in str(exc_info.value)


def test_ast_handles_syntax_errors_gracefully() -> None:
    """Malformed arithmetic expressions must be converted into ToolSecurityError."""
    with pytest.raises(ToolSecurityError) as exc_info:
        validate_dose_expression("500 * + / 2")
    assert "Malformed arithmetic expression" in str(exc_info.value)


# --- Clinical Interaction & Overdose Tests ---


def test_critical_interaction_detection_commutative() -> None:
    """Interaction must be flagged regardless of entity ordering in list."""
    alerts_forward = check_drug_interactions(["Metformin", "Radiocontrast"])
    assert len(alerts_forward) == 1
    assert alerts_forward[0].severity == "CRITICAL"
    assert "lactic acidosis" in alerts_forward[0].description.lower()

    alerts_reverse = check_drug_interactions(["  RADIOCONTRAST ", "metformin"])
    assert len(alerts_reverse) == 1
    assert alerts_reverse[0].severity == "CRITICAL"


def test_speedball_hazard_detection() -> None:
    """Cocaine + Fentanyl lethal combination must be intercepted."""
    alerts = check_drug_interactions(["cocaine", "fentanyl"])
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "respiratory depression" in alerts[0].description.lower()


def test_lethal_tramadol_overdose_intercepted() -> None:
    """Tramadol prescribed at 5000mg (5g) must trigger a FATAL OVERDOSE critical alert."""
    meds = [ExtractedMedication(name="tramadol", dose_mg=5000.0)]
    alerts = check_dosage_safety(meds)
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "FATAL OVERDOSE" in alerts[0].description
    assert "400.0mg" in alerts[0].description


def test_safe_dosage_passes() -> None:
    """Normal therapeutic dose within safe ceilings must not generate alerts."""
    meds = [ExtractedMedication(name="tramadol", dose_mg=100.0)]
    alerts = check_dosage_safety(meds)
    assert len(alerts) == 0


def test_multiple_critical_interactions() -> None:
    """All co-occurring dangerous pairs must be identified."""
    meds = ["warfarin", "aspirin", "tramadol", "ssri"]
    alerts = check_drug_interactions(meds)
    assert len(alerts) == 2
    severities = {a.severity for a in alerts}
    assert severities == {"CRITICAL"}


def test_safe_combinations_return_no_alerts() -> None:
    """Non-interacting drugs must yield an empty alert list."""
    alerts = check_drug_interactions(["Paracetamol", "Amoxicillin", "Ibuprofen"])
    assert len(alerts) == 0


def test_lethal_fentanyl_overdose_intercepted() -> None:
    """Fentanyl prescribed at 200 grams (200,000mg) must trigger FATAL OVERDOSE."""
    meds = [ExtractedMedication(name="fentanyl", dose_mg=200000.0)]
    alerts = check_dosage_safety(meds)
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "FATAL OVERDOSE" in alerts[0].description
    assert "0.2mg" in alerts[0].description


def test_fentanyl_missing_dose_blocked() -> None:
    """Unspecified dose for high-alert fentanyl must trigger HIGH_ALERT_MEDICATION hazard."""
    meds = [ExtractedMedication(name="fentanyl", dose_mg=None)]
    alerts = check_dosage_safety(meds)
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "HIGH-ALERT OPIOID HAZARD" in alerts[0].description


def test_renal_failure_condition_contraindications() -> None:
    """Radiocontrast and NSAIDs must be blocked in patients with chronic renal impairment."""
    alerts = check_condition_contraindications(
        ["Insuficiencia Renal Moderada", "Diabetes Tipo 2"],
        ["radiocontrast", "ibuprofen"],
    )
    assert len(alerts) == 2
    assert all(a.severity == "CRITICAL" for a in alerts)
    descriptions = " ".join(a.description for a in alerts)
    assert "CONTRAINDICATED IN RENAL FAILURE" in descriptions


def test_asthma_condition_contraindication() -> None:
    """Aspirin in severe asthma must be blocked due to bronchospasm / AERD."""
    alerts = check_condition_contraindications(["Asma Severa"], ["aspirin"])
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "CONTRAINDICATED IN ASTHMA" in alerts[0].description
