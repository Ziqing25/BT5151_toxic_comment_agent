from __future__ import annotations

from typing import Literal, Union

from .build import compile_build_graph
from .graph import compile_runtime_graph
from .state import BuildState, RuntimeState


def run(
    mode: Literal["build", "moderate"],
    state: Union[BuildState, RuntimeState],
) -> Union[BuildState, RuntimeState]:
    """Deterministic router: call the build layer or the runtime layer."""
    if mode == "build":
        return compile_build_graph().invoke(state)
    if mode == "moderate":
        return compile_runtime_graph().invoke(state)
    raise ValueError(
        f"Unknown mode {mode!r}. Expected 'build' or 'moderate'."
    )
