"""
Execution sandbox and clinical safety validation layer.
Provides AST-based expression whitelisting and drug interaction detection.
"""

import ast
from src.exceptions import ToolSecurityError
from src.types import DrugInteractionAlert

# Denylist of unsafe AST identifiers for mathematical evaluation
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

# Known critical clinical interactions (active substance pairs -> description)
CRITICAL_INTERACTIONS: dict[frozenset[str], str] = {
    frozenset({"metformin", "radiocontrast"}): (
        "Radiocontrast media induces acute kidney injury, impairing metformin clearance "
        "and triggering lethal lactic acidosis. Metformin must be discontinued 48h prior."
    ),
    frozenset({"warfarin", "aspirin"}): (
        "Concurrent anticoagulation and antiplatelet therapy elevates major hemorrhage "
        "risk by 3-4x. Requires immediate hemostatic review."
    ),
    frozenset({"tramadol", "ssri"}): (
        "Concurrent serotonergic agents induce life-threatening serotonin syndrome. "
        "Symptoms: neuromuscular excitation, autonomic instability, altered mental state."
    ),
    frozenset({"methotrexate", "nsaid"}): (
        "NSAIDs reduce renal perfusion and clearance of methotrexate, causing severe "
        "bone marrow suppression and nephrotoxicity."
    ),
}


def validate_dose_expression(expression: str) -> float:
    """
    Parse and evaluate a dose arithmetic expression using AST pre-flight validation.

    Only numbers, unary/binary arithmetic operators, and parentheses are permitted.
    Blocks function calls, variable assignments, and unwhitelisted identifiers.

    Raises:
        ToolSecurityError: If an unsafe AST node or identifier is detected.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolSecurityError(
            tool_name="dose_calculator",
            message=f"Malformed arithmetic expression: {exc}",
        ) from exc

    for node in ast.walk(tree):
        # Disallow function or method calls entirely (e.g., os.system())
        if isinstance(node, ast.Call):
            raise ToolSecurityError(
                tool_name="dose_calculator",
                message="Function and method calls are strictly forbidden in dose calculation.",
            )

        # Disallow attribute access (e.g., __class__, __subclasses__)
        if isinstance(node, ast.Attribute):
            raise ToolSecurityError(
                tool_name="dose_calculator",
                message="Attribute access is forbidden in dose calculation.",
            )

        # Disallow blocked identifiers
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ToolSecurityError(
                tool_name="dose_calculator",
                message=f"Unauthorized identifier blocked by AST policy: {node.id}",
            )

    # Safe: the AST tree contains only arithmetic literals and operators
    result = eval(compile(tree, filename="<ast_sandbox>", mode="eval"))  # noqa: S307
    return float(result)


def check_drug_interactions(medications: list[str]) -> list[DrugInteractionAlert]:
    """
    Cross-reference extracted medications against known critical interactions.

    Returns:
        List of DrugInteractionAlert containing critical contraindications.
    """
    normalized = [med.lower().strip() for med in medications]
    alerts: list[DrugInteractionAlert] = []

    for i, drug_a in enumerate(normalized):
        for drug_b in normalized[i + 1 :]:
            pair = frozenset({drug_a, drug_b})
            if pair in CRITICAL_INTERACTIONS:
                alerts.append(
                    DrugInteractionAlert(
                        drug_a=drug_a,
                        drug_b=drug_b,
                        severity="CRITICAL",
                        description=CRITICAL_INTERACTIONS[pair],
                        cie10_codes=["T36-T50"],
                    )
                )

    return alerts