"""
Finite State Machine (FSM) implementation for deterministic agent execution.
Supports strict transition validation, loop prevention, and checkpoint serialization/recovery.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timezone, datetime
from typing import Any, ClassVar

from src.exceptions import StateTransitionError
from src.types import EngineState


@dataclass(frozen=True)
class TransitionRecord:
    """Immutable audit record of a state transition."""

    from_state: EngineState
    to_state: EngineState
    timestamp: datetime
    step_number: int


class AgentFSM:
    """
    Hardened Finite State Machine enforcing strict execution lifecycle,
    checkpoint persistence, and recovery from interrupted runs.
    """

    VALID_TRANSITIONS: ClassVar[dict[EngineState, set[EngineState]]] = {
        EngineState.INITIALIZED: {EngineState.PLANNING, EngineState.FAILED},
        EngineState.PLANNING: {
            EngineState.DISPATCHING_TOOL,
            EngineState.REFLECTING,
            EngineState.FAILED,
        },
        EngineState.DISPATCHING_TOOL: {
            EngineState.AWAITING_OBSERVATION,
            EngineState.FAILED,
        },
        EngineState.AWAITING_OBSERVATION: {
            EngineState.REFLECTING,
            EngineState.FAILED,
        },
        EngineState.REFLECTING: {
            EngineState.PLANNING,
            EngineState.DISPATCHING_TOOL,
            EngineState.COMPLETED,
            EngineState.FAILED,
        },
        EngineState.COMPLETED: set(),
        EngineState.FAILED: set(),
    }

    def __init__(
        self,
        max_steps: int = 10,
        on_transition: Callable[[TransitionRecord], None] | None = None,
    ) -> None:
        self._current_state: EngineState = EngineState.INITIALIZED
        self._step_count: int = 0
        self._max_steps: int = max_steps
        self._history: list[TransitionRecord] = []
        self._on_transition = on_transition

    @property
    def current_state(self) -> EngineState:
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        return self._current_state in (EngineState.COMPLETED, EngineState.FAILED)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def history(self) -> list[TransitionRecord]:
        return list(self._history)

    def transition(self, target_state: EngineState) -> None:
        """Validate and execute a transition to target_state."""
        if self.is_terminal:
            raise StateTransitionError(
                f"Cannot transition from terminal state '{self._current_state.value}' "
                f"to '{target_state.value}'."
            )

        if self._step_count >= self._max_steps:
            raise StateTransitionError(
                f"FSM exceeded maximum execution steps limit ({self._max_steps}). "
                "Halting execution to prevent infinite loop."
            )

        allowed_next = self.VALID_TRANSITIONS.get(self._current_state, set())
        if target_state not in allowed_next:
            raise StateTransitionError(
                f"Illegal state transition from '{self._current_state.value}' "
                f"to '{target_state.value}'. "
                f"Valid transitions are: {[s.value for s in allowed_next]}"
            )

        previous_state = self._current_state
        self._current_state = target_state
        self._step_count += 1

        record = TransitionRecord(
            from_state=previous_state,
            to_state=target_state,
            timestamp=datetime.now(timezone.utc),
            step_number=self._step_count,
        )
        self._history.append(record)

        if self._on_transition:
            self._on_transition(record)

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize current state and transition history for persistence."""
        return {
            "current_state": self._current_state.value,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "history": [
                {
                    "from_state": r.from_state.value,
                    "to_state": r.to_state.value,
                    "timestamp": r.timestamp.isoformat(),
                    "step_number": r.step_number,
                }
                for r in self._history
            ],
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        on_transition: Callable[[TransitionRecord], None] | None = None,
    ) -> "AgentFSM":
        """Rehydrate an AgentFSM instance from a persisted checkpoint dictionary."""
        fsm = cls(
            max_steps=int(checkpoint.get("max_steps", 10)),
            on_transition=on_transition,
        )
        fsm._current_state = EngineState(checkpoint["current_state"])
        fsm._step_count = int(checkpoint["step_count"])
        fsm._history = [
            TransitionRecord(
                from_state=EngineState(item["from_state"]),
                to_state=EngineState(item["to_state"]),
                timestamp=datetime.fromisoformat(item["timestamp"]),
                step_number=int(item["step_number"]),
            )
            for item in checkpoint.get("history", [])
        ]
        return fsm
