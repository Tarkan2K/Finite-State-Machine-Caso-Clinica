"""Domain exception hierarchy for fsm-agent-runtime."""


class EngineError(Exception):
    """Base exception for all runtime engine failures."""

    pass


class StateTransitionError(EngineError):
    """Raised when an illegal FSM state transition is attempted."""

    pass


class ToolExecutionError(EngineError):
    """Raised when an external or internal tool execution fails."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolSecurityError(ToolExecutionError):
    """Raised when an AST sandbox violation or untrusted input is intercepted."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(tool_name=tool_name, message=f"[SECURITY BLOCKED] {message}")


class ClinicalSafetyViolationError(EngineError):
    """
    Raised when a CRITICAL contraindication or drug interaction is detected.
    Signals the orchestrator to immediately transition FSM to FAILED state.
    """

    def __init__(
        self,
        medications: list[str],
        interaction_description: str,
        severity: str,
    ) -> None:
        self.medications = medications
        self.interaction_description = interaction_description
        self.severity = severity
        super().__init__(
            f"Safety violation [{severity}]: {interaction_description} "
            f"(interacting medications: {', '.join(medications)})"
        )


class StorageError(EngineError):
    """Raised when an audit record or telemetry write to DuckDB fails."""

    pass