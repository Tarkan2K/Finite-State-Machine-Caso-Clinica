"""
Unit and security tests for the AST sandbox and clinical interaction validator.
Validates arithmetic safety, denylist enforcement, and drug-drug contraindication logic.
"""

import pytest
from src.exceptions import ToolSecurityError
from src.sandbox import check_drug_interactions, validate_dose_expression


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


# --- Clinical Interaction Matrix Tests ---


def test_critical_interaction_detection_commutative() -> None:
    """Interaction must be flagged regardless of entity ordering in list."""
    # Order A
    alerts_forward = check_drug_interactions(["Metformin", "Radiocontrast"])
    assert len(alerts_forward) == 1
    assert alerts_forward[0].severity == "CRITICAL"
    assert "lactic acidosis" in alerts_forward[0].description.lower()

    # Order B (Reverse order + messy casing/whitespace)
    alerts_reverse = check_drug_interactions(["  RADIOCONTRAST ", "metformin"])
    assert len(alerts_reverse) == 1
    assert alerts_reverse[0].severity == "CRITICAL"


def test_multiple_critical_interactions() -> None:
    """All co-occurring dangerous pairs must be identified."""
    meds = ["warfarin", "aspirin", "tramadol", "ssri"]
    alerts = check_drug_interactions(meds)
    assert len(alerts) == 2  # warfarin+aspirin AND tramadol+ssri
    severities = {a.severity for a in alerts}
    assert severities == {"CRITICAL"}


def test_safe_combinations_return_no_alerts() -> None:
    """Non-interacting drugs must yield an empty alert list."""
    alerts = check_drug_interactions(["Paracetamol", "Amoxicillin", "Ibuprofen"])
    assert len(alerts) == 0


def test_single_drug_returns_no_alerts() -> None:
    """Single substance has no contraindication pair."""
    alerts = check_drug_interactions(["Metformin"])
    assert len(alerts) == 0