"""
Finite State Machine (FSM) implementation for deterministic agent execution.
Guarantees legal transitions, audit history tracking, and loop prevention.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
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
    Hardened Finite State Machine enforcing strict execution lifecycle.
    Guarantees that agents cannot skip safety checks or enter infinite loops.
    """

    # Explicit transition matrix: current_state -> set of valid next states
    VALID_TRANSITIONS: dict[EngineState, set[EngineState]] = {
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
            EngineState.PLANNING,          # Loop for multi-step reasoning
            EngineState.DISPATCHING_TOOL,  # Direct tool retry
            EngineState.COMPLETED,         # Clean finish
            EngineState.FAILED,            # Safety block or fatal error
        },
        EngineState.COMPLETED: set(),      # Terminal absorbing state
        EngineState.FAILED: set(),         # Terminal absorbing state
    }

    def __init__(
        self,
        max_steps: int = 10,
        on_transition: Callable[[TransitionRecord], None] | None = None,
    ) -> None:
        """
        Initialize the FSM in the INITIALIZED state.

        Args:
            max_steps: Maximum allowed state transitions before loop abort.
            on_transition: Optional callback triggered on every legal transition.
        """
        self._current_state: EngineState = EngineState.INITIALIZED
        self._step_count: int = 0
        self._max_steps: int = max_steps
        self._history: list[TransitionRecord] = []
        self._on_transition = on_transition

    @property
    def current_state(self) -> EngineState:
        """Read-only access to current FSM state."""
        return self._current_state

    @property
    def step_count(self) -> int:
        """Total number of transitions executed."""
        return self._step_count

    @property
    def history(self) -> list[TransitionRecord]:
        """Read-only snapshot of all historical transitions."""
        return list(self._history)

    def transition(self, target_state: EngineState) -> None:
        """
        Attempt a state transition to target_state.

        Raises:
            StateTransitionError: If transition is illegal, machine is terminated,
                                  or max_steps limit is exceeded.
        """
        # Guardrail 1: Absorbing terminal states cannot transition anywhere
        if self._current_state in (EngineState.COMPLETED, EngineState.FAILED):
            raise StateTransitionError(
                f"Cannot transition from terminal state '{self._current_state.value}' "
                f"to '{target_state.value}'."
            )

        # Guardrail 2: Loop prevention threshold
        if self._step_count >= self._max_steps:
            raise StateTransitionError(
                f"FSM exceeded maximum execution steps limit ({self._max_steps}). "
                f"Halting execution to prevent infinite loop."
            )

        # Guardrail 3: Strict transition matrix validation
        allowed_next = self.VALID_TRANSITIONS.get(self._current_state, set())
        if target_state not in allowed_next:
            raise StateTransitionError(
                f"Illegal state transition from '{self._current_state.value}' "
                f"to '{target_state.value}'. "
                f"Valid transitions are: {[s.value for s in allowed_next]}"
            )

        # Apply transition
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

    def to_checkpoint(self) -> dict[str, object]:
        """Serialize current state and history for persistence/recovery."""
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