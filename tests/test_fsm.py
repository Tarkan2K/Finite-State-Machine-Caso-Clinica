"""
Unit tests for the Agent Finite State Machine (AgentFSM).
Validates legal transitions, illegal state rejections, loop guardrails, and checkpoint recovery.
"""

import pytest

from src.exceptions import StateTransitionError
from src.fsm import AgentFSM
from src.types import EngineState


def test_initial_state(fsm: AgentFSM) -> None:
    """FSM must initialize in INITIALIZED state with 0 steps executed."""
    assert fsm.current_state == EngineState.INITIALIZED
    assert fsm.step_count == 0
    assert not fsm.is_terminal
    assert len(fsm.history) == 0


def test_legal_happy_path(fsm: AgentFSM) -> None:
    """Verify standard happy-path lifecycle transitions."""
    fsm.transition(EngineState.PLANNING)
    fsm.transition(EngineState.DISPATCHING_TOOL)
    fsm.transition(EngineState.AWAITING_OBSERVATION)
    fsm.transition(EngineState.REFLECTING)
    fsm.transition(EngineState.COMPLETED)

    assert fsm.current_state == EngineState.COMPLETED
    assert fsm.is_terminal
    assert fsm.step_count == 5
    assert len(fsm.history) == 5


def test_illegal_transition_raises(fsm: AgentFSM) -> None:
    """Attempting an invalid jump must immediately raise StateTransitionError."""
    with pytest.raises(StateTransitionError) as exc_info:
        fsm.transition(EngineState.COMPLETED)
    assert "Illegal state transition" in str(exc_info.value)
    assert fsm.current_state == EngineState.INITIALIZED


def test_terminal_states_are_absorbing(fsm: AgentFSM) -> None:
    """Once in COMPLETED or FAILED, no further transitions are allowed."""
    fsm.transition(EngineState.PLANNING)
    fsm.transition(EngineState.FAILED)
    assert fsm.current_state == EngineState.FAILED
    assert fsm.is_terminal

    with pytest.raises(StateTransitionError) as exc_info:
        fsm.transition(EngineState.PLANNING)
    assert "Cannot transition from terminal state 'FAILED'" in str(exc_info.value)


def test_max_steps_guardrail_prevents_infinite_loops() -> None:
    """FSM must abort execution if transition count hits max_steps threshold."""
    loop_fsm = AgentFSM(max_steps=4)
    loop_fsm.transition(EngineState.PLANNING)
    loop_fsm.transition(EngineState.REFLECTING)
    loop_fsm.transition(EngineState.PLANNING)
    loop_fsm.transition(EngineState.REFLECTING)

    with pytest.raises(StateTransitionError) as exc_info:
        loop_fsm.transition(EngineState.COMPLETED)
    assert "exceeded maximum execution steps limit (4)" in str(exc_info.value)


def test_checkpoint_serialization_and_rehydration(fsm: AgentFSM) -> None:
    """to_checkpoint() and from_checkpoint() must preserve exact state and history for resumption."""
    fsm.transition(EngineState.PLANNING)
    fsm.transition(EngineState.DISPATCHING_TOOL)

    checkpoint = fsm.to_checkpoint()
    resumed_fsm = AgentFSM.from_checkpoint(checkpoint)

    assert resumed_fsm.current_state == EngineState.DISPATCHING_TOOL
    assert resumed_fsm.step_count == 2
    assert len(resumed_fsm.history) == 2

    # Continue seamlessly from rehydrated state
    resumed_fsm.transition(EngineState.AWAITING_OBSERVATION)
    resumed_fsm.transition(EngineState.REFLECTING)
    resumed_fsm.transition(EngineState.COMPLETED)
    assert resumed_fsm.current_state == EngineState.COMPLETED
    assert resumed_fsm.step_count == 5