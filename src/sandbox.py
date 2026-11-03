"""
Execution sandbox, deterministic posology validation, and agent reliability metrics.
Provides AST-based arithmetic evaluation for daily dose formulas, mathematical
ceiling checks, and Pass^k / Pass@k consistency metrics.
"""

import ast

from src.exceptions import ToolSecurityError
from src.types import DrugInteractionAlert, ExtractedMedication

BLOCKED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "import",
        "__import__",
        "open",
        "compile",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "subprocess",
        "os",
        "sys",
    }
)


def validate_dose_expression(expression: str) -> float:
    """
    Parse and evaluate a daily dose arithmetic expression using AST pre-flight validation.
    Example: '20 * 1000 * (24 / 8)' -> 60000.0 (mg/day).
    Blocks function calls, attribute access, and unwhitelisted identifiers.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolSecurityError(
            tool_name="dose_calculator",
            message=f"Malformed arithmetic expression: {exc}",
        ) from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            raise ToolSecurityError(
                tool_name="dose_calculator",
                message="Function and method calls are strictly forbidden in dose calculation.",
            )
        if isinstance(node, ast.Attribute):
            raise ToolSecurityError(
                tool_name="dose_calculator",
                message="Attribute access is forbidden in dose calculation.",
            )
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ToolSecurityError(
                tool_name="dose_calculator",
                message=f"Unauthorized identifier blocked by AST policy: {node.id}",
            )

    result = eval(compile(tree, filename="<ast_sandbox>", mode="eval"))
    return float(result)


def resolve_medication_doses(
    medications: list[ExtractedMedication],
) -> list[ExtractedMedication]:
    """
    Evaluate each medication's `dose_expression` inside the AST sandbox
    to populate the exact `dose_mg` (daily dose in milligrams).
    """
    resolved: list[ExtractedMedication] = []
    for med in medications:
        evaluated_dose = med.dose_mg
        if med.dose_expression:
            evaluated_dose = validate_dose_expression(med.dose_expression)
        resolved.append(
            ExtractedMedication(
                name=med.name.lower().strip(),
                dose_expression=med.dose_expression,
                dose_mg=evaluated_dose,
                max_safe_daily_mg=med.max_safe_daily_mg,
                route=med.route,
            )
        )
    return resolved


def check_dosage_safety(
    medications: list[ExtractedMedication],
) -> list[DrugInteractionAlert]:
    """
    Deterministically compare AST-evaluated daily doses (`dose_mg`) against
    each substance's maximum safe daily ceiling (`max_safe_daily_mg`).
    """
    alerts: list[DrugInteractionAlert] = []

    for med in medications:
        name = med.name.upper()
        ceiling = med.max_safe_daily_mg

        if ceiling is not None and ceiling <= 0.0:
            alerts.append(
                DrugInteractionAlert(
                    severity="CRITICAL",
                    description=(
                        f"SUSTANCIA PROHIBIDA / UMBRAL CERO: {name} no posee margen terapéutico "
                        "seguro y su dispensación está estrictamente bloqueada."
                    ),
                    cie10_codes=["T40.5", "T40.9"],
                )
            )
            continue

        if med.dose_mg is not None and ceiling is not None and med.dose_mg > ceiling:
            ratio = (med.dose_mg / ceiling) * 100.0
            alerts.append(
                DrugInteractionAlert(
                    severity="CRITICAL",
                    description=(
                        f"SOBREDOSIS LETAL DETECTADA (AST Sandbox): La dosis diaria calculada de {name} "
                        f"({med.dose_mg:.1f} mg/día) supera el techo máximo seguro de {ceiling:.1f} mg/día "
                        f"({ratio:.0f}% del límite). Riesgo inminente de toxicidad aguda severa o fallo orgánico."
                    ),
                    cie10_codes=["T36-T50", "T40.9"],
                )
            )

    return alerts


def compute_pass_metrics(successes: int, total_trials: int, k: int = 8) -> dict[str, float]:
    """
    Compute Pass@k and Pass^k (tau-bench production consistency metric).
    - Pass@k = 1 - (1 - p)^k (probability that at least 1 of k trials succeeds)
    - Pass^k = p^k (probability that all k consecutive trials succeed consistently)
    """
    if total_trials <= 0 or k <= 0:
        return {"p": 0.0, "pass_at_k": 0.0, "pass_hat_k": 0.0}
    p = max(0.0, min(1.0, successes / total_trials))
    return {
        "p": round(p, 4),
        "pass_at_k": round(1.0 - ((1.0 - p) ** k), 4),
        "pass_hat_k": round(p**k, 4),
    }
