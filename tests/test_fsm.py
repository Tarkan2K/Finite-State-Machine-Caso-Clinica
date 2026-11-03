"""
Unit tests for the Agent Finite State Machine (AgentFSM).
Validates legal transitions, illegal state rejections, loop guardrails, and audit trails.
"""

import pytest
from src.exceptions import StateTransitionError
from src.fsm import AgentFSM
from src.types import EngineState


def test_initial_state(fsm: AgentFSM) -> None:
    """FSM must initialize in INITIALIZED state with 0 steps executed."""
    assert fsm.current_state == EngineState.INITIALIZED
    assert fsm.step_count == 0
    assert len(fsm.history) == 0


def test_legal_happy_path(fsm: AgentFSM) -> None:
    """Verify standard happy-path lifecycle transitions."""
    fsm.transition(EngineState.PLANNING)
    assert fsm.current_state == EngineState.PLANNING

    fsm.transition(EngineState.DISPATCHING_TOOL)
    assert fsm.current_state == EngineState.DISPATCHING_TOOL

    fsm.transition(EngineState.AWAITING_OBSERVATION)
    assert fsm.current_state == EngineState.AWAITING_OBSERVATION

    fsm.transition(EngineState.REFLECTING)
    assert fsm.current_state == EngineState.REFLECTING

    fsm.transition(EngineState.COMPLETED)
    assert fsm.current_state == EngineState.COMPLETED
    assert fsm.step_count == 5
    assert len(fsm.history) == 5


def test_illegal_transition_raises(fsm: AgentFSM) -> None:
    """Attempting an invalid jump must immediately raise StateTransitionError."""
    with pytest.raises(StateTransitionError) as exc_info:
        # Cannot jump from INITIALIZED directly to COMPLETED
        fsm.transition(EngineState.COMPLETED)
    assert "Illegal state transition" in str(exc_info.value)
    assert fsm.current_state == EngineState.INITIALIZED


def test_terminal_states_are_absorbing(fsm: AgentFSM) -> None:
    """Once in COMPLETED or FAILED, no further transitions are allowed."""
    fsm.transition(EngineState.PLANNING)
    fsm.transition(EngineState.FAILED)
    assert fsm.current_state == EngineState.FAILED

    # Attempt transition from terminal state FAILED
    with pytest.raises(StateTransitionError) as exc_info:
        fsm.transition(EngineState.PLANNING)
    assert "Cannot transition from terminal state 'FAILED'" in str(exc_info.value)


def test_max_steps_guardrail_prevents_infinite_loops() -> None:
    """FSM must abort execution if transition count hits max_steps threshold."""
    loop_fsm = AgentFSM(max_steps=4)

    loop_fsm.transition(EngineState.PLANNING)  # step 1
    loop_fsm.transition(EngineState.REFLECTING)  # step 2
    loop_fsm.transition(EngineState.PLANNING)  # step 3
    loop_fsm.transition(EngineState.REFLECTING)  # step 4

    # 5th transition must be blocked by loop guardrail
    with pytest.raises(StateTransitionError) as exc_info:
        loop_fsm.transition(EngineState.COMPLETED)
    assert "exceeded maximum execution steps limit (4)" in str(exc_info.value)


def test_to_checkpoint_serialization(fsm: AgentFSM) -> None:
    """to_checkpoint() must produce serializable snapshot with full audit history."""
    fsm.transition(EngineState.PLANNING)
    fsm.transition(EngineState.DISPATCHING_TOOL)

    checkpoint = fsm.to_checkpoint()
    assert checkpoint["current_state"] == "DISPATCHING_TOOL"
    assert checkpoint["step_count"] == 2
    assert checkpoint["max_steps"] == 10
    assert len(checkpoint["history"]) == 2  # type: ignore
    assert checkpoint["history"][0]["from_state"] == "INITIALIZED"  # type: ignore
    assert checkpoint["history"][0]["to_state"] == "PLANNING"  # type: ignore