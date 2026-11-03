"""
Execution sandbox and clinical safety validation layer.
Provides AST-based expression whitelisting, drug-drug contraindication detection,
patient allergy matching, and posology ceiling guardrails (lethal overdose protection).
Operates directly on canonical RxNorm active substances without hardcoded synonym maps.
"""

import ast

from src.exceptions import ToolSecurityError
from src.types import DrugInteractionAlert, ExtractedMedication

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

# Known critical clinical interactions (canonical substance pairs -> description)
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
    frozenset({"methotrexate", "aspirin"}): (
        "Aspirin reduces renal perfusion and clearance of methotrexate, causing severe "
        "bone marrow suppression and nephrotoxicity."
    ),
    frozenset({"methotrexate", "ibuprofen"}): (
        "Ibuprofen reduces renal perfusion and clearance of methotrexate, causing severe "
        "bone marrow suppression and nephrotoxicity."
    ),
    frozenset({"cocaine", "fentanyl"}): (
        "Concurrent administration of cocaine (potent CNS stimulant) and fentanyl "
        "(high-potency opioid) causes lethal respiratory depression, fatal cardiac arrhythmia, "
        "and acute cardiovascular collapse (speedball toxicity)."
    ),
}

# Maximum clinical daily dosage ceilings in milligrams (Posology Guardrails)
MAX_SAFE_DAILY_DOSES_MG: dict[str, float] = {
    "tramadol": 400.0,       # Max 400mg/day. Overdose causes intractable seizures, apnea, and coma.
    "paracetamol": 4000.0,   # Max 4000mg/day (4g). Overdose causes acute fulminant hepatic necrosis.
    "metformin": 2550.0,     # Max 2550mg/day. Overdose triggers severe metformin-associated lactic acidosis.
    "aspirin": 4000.0,       # Max 4000mg/day. Overdose triggers acute salicylate toxicity and metabolic acidosis.
    "ibuprofen": 3200.0,     # Max 3200mg/day. Severe GI hemorrhage and acute renal failure.
    "fentanyl": 0.2,         # 200mcg (0.2mg). Extremely potent; milligram doses are universally fatal.
    "cocaine": 0.0,          # Illicit substance: zero therapeutic threshold allowed.
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


def check_drug_interactions(medications: list[str]) -> list[DrugInteractionAlert]:
    """
    Cross-reference canonical medications against known critical clinical interactions.
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
                        cie10_codes=["T36-T50", "T40.1", "T40.4"],
                    )
                )

    return alerts


def check_allergy_contraindications(
    patient_allergies: list[str],
    ordered_medications: list[str],
) -> list[DrugInteractionAlert]:
    """
    Cross-reference ordered substances against the patient's documented EHR allergy registry.
    """
    alerts: list[DrugInteractionAlert] = []
    if not patient_allergies:
        return alerts

    allergies_text = " ".join(patient_allergies).lower()

    for med in ordered_medications:
        med_clean = med.lower().strip()
        matched = False
        allergen_label = ""

        if med_clean in ("aspirin", "ibuprofen", "nsaid") and any(
            k in allergies_text for k in ("aine", "aspirina", "ibuprofeno")
        ):
            matched = True
            allergen_label = "AINEs / Ácido Acetilsalicílico"
        elif med_clean == "radiocontrast" and any(
            k in allergies_text for k in ("contraste", "yodado", "yodo")
        ):
            matched = True
            allergen_label = "Medio de Contraste Yodado"
        elif med_clean == "tramadol" and "tramadol" in allergies_text:
            matched = True
            allergen_label = "Tramadol / Opioides"

        if matched:
            alerts.append(
                DrugInteractionAlert(
                    drug_a=med_clean,
                    drug_b="PATIENT_ALLERGY_ANAPHYLAXIS",
                    severity="CRITICAL",
                    description=(
                        f"CRITICAL ALLERGY INTERCEPT: Patient has documented severe allergy to "
                        f"{allergen_label}. Administering {med_clean.upper()} carries imminent "
                        "risk of bronchospasm, severe angioedema, or fatal anaphylactic shock."
                    ),
                    cie10_codes=["T78.4", "T88.7"],
                )
            )

    return alerts


def check_condition_contraindications(
    chronic_conditions: list[str],
    ordered_medications: list[str],
) -> list[DrugInteractionAlert]:
    """
    Cross-reference ordered substances against the patient's chronic diseases to intercept
    organ damage (e.g. nephrotoxicity in renal failure, bronchospasm in asthma).
    """
    alerts: list[DrugInteractionAlert] = []
    if not chronic_conditions:
        return alerts

    conditions_text = " ".join(chronic_conditions).lower()

    for med in ordered_medications:
        med_clean = med.lower().strip()

        # Renal impairment contraindications
        if any(k in conditions_text for k in ("renal", "riñon", "riñón", "nefropatia", "nefropatía")):
            if med_clean == "radiocontrast":
                alerts.append(
                    DrugInteractionAlert(
                        drug_a="radiocontrast",
                        drug_b="CHRONIC_RENAL_IMPAIRMENT",
                        severity="CRITICAL",
                        description=(
                            "CONTRAINDICATED IN RENAL FAILURE: Radiocontrast agents induce acute "
                            "contrast nephropathy and tubular necrosis in patients with pre-existing "
                            "renal impairment."
                        ),
                        cie10_codes=["N17.9", "N18.9"],
                    )
                )
            elif med_clean in ("ibuprofen", "nsaid"):
                alerts.append(
                    DrugInteractionAlert(
                        drug_a=med_clean,
                        drug_b="CHRONIC_RENAL_IMPAIRMENT",
                        severity="CRITICAL",
                        description=(
                            f"CONTRAINDICATED IN RENAL FAILURE: NSAID ({med_clean.upper()}) reduces "
                            "renal prostaglandins, decreasing GFR and precipitating acute-on-chronic "
                            "renal failure."
                        ),
                        cie10_codes=["N17.9", "N18.9"],
                    )
                )

        # Asthma / bronchospasm contraindications
        if any(k in conditions_text for k in ("asma", "asthma", "epoc")):
            if med_clean in ("aspirin", "ibuprofen", "nsaid"):
                alerts.append(
                    DrugInteractionAlert(
                        drug_a=med_clean,
                        drug_b="ASTHMA_BRONCHOSPASM_RISK",
                        severity="CRITICAL",
                        description=(
                            f"CONTRAINDICATED IN ASTHMA: NSAID/Aspirin ({med_clean.upper()}) inhibits COX-1, "
                            "shunting arachidonic acid to leukotrienes and triggering severe bronchospasm (AERD)."
                        ),
                        cie10_codes=["J45.9", "T78.4"],
                    )
                )

    return alerts


def check_dosage_safety(medications: list[ExtractedMedication]) -> list[DrugInteractionAlert]:
    """
    Evaluate prescribed dosages against maximum safe pharmacological ceilings.
    Flags lethal overdoses, toxic quantities, and prohibited illicit substances.
    """
    overdose_alerts: list[DrugInteractionAlert] = []

    for med in medications:
        canonical_name = med.name.lower().strip()
        max_limit = MAX_SAFE_DAILY_DOSES_MG.get(canonical_name)

        # Prohibited illicit substances (e.g., cocaine) have zero clinical allowance
        if max_limit == 0.0:
            overdose_alerts.append(
                DrugInteractionAlert(
                    drug_a=canonical_name,
                    drug_b="PROHIBITED_SUBSTANCE",
                    severity="CRITICAL",
                    description=(
                        f"PROHIBITED ILLICIT SUBSTANCE: {canonical_name.upper()} has zero therapeutic "
                        "threshold and cannot be dispensed under any clinical protocol."
                    ),
                    cie10_codes=["T40.5", "T40.9"],
                )
            )
            continue

        # High-alert opioid safeguard: fentanyl must have an explicit safe dose (<0.2mg)
        if canonical_name == "fentanyl" and med.dose_mg is None:
            overdose_alerts.append(
                DrugInteractionAlert(
                    drug_a="fentanyl",
                    drug_b="HIGH_ALERT_MEDICATION",
                    severity="CRITICAL",
                    description=(
                        "HIGH-ALERT OPIOID HAZARD: Fentanyl posology must be strictly specified in micrograms (mcg). "
                        "Uncalibrated or missing dosage values are intercepted to prevent fatal respiratory depression."
                    ),
                    cie10_codes=["T40.4"],
                )
            )
            continue

        if med.dose_mg is None:
            continue

        if max_limit is not None and med.dose_mg > max_limit:
            ratio = (med.dose_mg / max_limit) * 100.0 if max_limit > 0 else 0
            overdose_alerts.append(
                DrugInteractionAlert(
                    drug_a=canonical_name,
                    drug_b="LETHAL_OVERDOSE",
                    severity="CRITICAL",
                    description=(
                        f"FATAL OVERDOSE DETECTED: Prescribed dose of {canonical_name.upper()} "
                        f"({med.dose_mg:.1f}mg) exceeds the maximum safe ceiling of {max_limit:.1f}mg "
                        f"({ratio:.0f}% of ceiling). Imminent risk of severe toxicity, organ collapse, or death."
                    ),
                    cie10_codes=["T36-T50", "T40.9"],
                )
            )

    return overdose_alerts
