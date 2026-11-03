"""
Unit and security tests for the AST sandbox, posology ceiling validator, and Pass^k metrics.
"""

import pytest

from src.exceptions import ToolSecurityError
from src.sandbox import (
    check_dosage_safety,
    compute_pass_metrics,
    resolve_medication_doses,
    validate_dose_expression,
)
from src.types import ExtractedMedication


def test_valid_arithmetic_expressions() -> None:
    """Standard daily dose arithmetic operations must evaluate accurately."""
    assert validate_dose_expression("500 * 2") == 1000.0
    assert validate_dose_expression("20 * 1000 * (24 / 8)") == 60000.0
    assert validate_dose_expression("2 * 1000 * (24 / 6)") == 8000.0
    assert validate_dose_expression("(100 + 50) / 2") == 75.0


def test_ast_blocks_arbitrary_function_calls() -> None:
    """Function and method invocations must be intercepted by AST traversal."""
    with pytest.raises(ToolSecurityError) as exc_info:
        validate_dose_expression("__import__('os').system('ls')")
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


def test_resolve_medication_doses_and_ceiling_intercept() -> None:
    """'20 gramos de paracetamol cada 8 horas' (60,000 mg/day) must be calculated by AST and blocked."""
    raw_meds = [
        ExtractedMedication(
            name="paracetamol",
            dose_expression="20 * 1000 * (24 / 8)",
            max_safe_daily_mg=4000.0,
            route="oral",
        )
    ]
    resolved = resolve_medication_doses(raw_meds)
    assert resolved[0].dose_mg == 60000.0

    alerts = check_dosage_safety(resolved)
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "60000.0 mg/día" in alerts[0].description


def test_tau_bench_pass_metrics() -> None:
    """Verify Pass@k vs Pass^k consistency calculation (e.g. p=0.80 at k=8 drops to ~16.78%)."""
    metrics = compute_pass_metrics(successes=8, total_trials=10, k=8)
    assert metrics["p"] == 0.8
    assert metrics["pass_at_k"] == pytest.approx(0.9993, rel=1e-3)
    assert metrics["pass_hat_k"] == pytest.approx(0.1678, rel=1e-2)
