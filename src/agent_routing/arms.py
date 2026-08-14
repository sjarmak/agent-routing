"""Routing arms under comparison.

An arm is a total function from step kind to model tier. Every arm must cover
every step kind, so a comparison cannot silently differ by an unrouted step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StepKind(str, Enum):
    """The step kinds an agent workload is decomposed into.

    The decomposition is the experiment's unit of routing. Adding a kind
    invalidates existing arms until every arm assigns it a tier.
    """

    CLASSIFY = "classify"
    PLAN = "plan"
    HARD_REASONING = "hard_reasoning"
    TOOL_FORMAT = "tool_format"
    SUMMARIZE = "summarize"
    VERIFY = "verify"


class Tier(str, Enum):
    SMALL = "small"
    LARGE = "large"


@dataclass(frozen=True)
class Arm:
    """One routing configuration.

    `expectation` records, before any run, what this arm is predicted to do.
    An arm predicted to lose is not decoration: if the losing arm scores the
    same as the winning arm, the task set cannot detect routing quality and no
    routing claim is supportable from it.
    """

    name: str
    routing: dict[StepKind, Tier]
    expectation: str
    is_distinguishing_control: bool = False

    def __post_init__(self) -> None:
        missing = [kind.value for kind in StepKind if kind not in self.routing]
        if missing:
            raise ValueError(f"arm {self.name!r} does not route: {', '.join(sorted(missing))}")

    def tier_for(self, kind: StepKind) -> Tier:
        return self.routing[kind]


ALL_LARGE = Arm(
    name="all-large",
    routing={kind: Tier.LARGE for kind in StepKind},
    expectation=(
        "Highest task success and highest cost. The quality ceiling every other "
        "arm is measured against."
    ),
)

ALL_SMALL = Arm(
    name="all-small",
    routing={kind: Tier.SMALL for kind in StepKind},
    expectation=(
        "Lowest cost per call and lowest task success. The cost floor, and the "
        "check that the task set is hard enough for model choice to matter."
    ),
)

ROUTED = Arm(
    name="routed",
    routing={
        StepKind.CLASSIFY: Tier.SMALL,
        StepKind.PLAN: Tier.LARGE,
        StepKind.HARD_REASONING: Tier.LARGE,
        StepKind.TOOL_FORMAT: Tier.SMALL,
        StepKind.SUMMARIZE: Tier.SMALL,
        StepKind.VERIFY: Tier.SMALL,
    },
    expectation=(
        "Task success statistically indistinguishable from all-large at lower "
        "cost per successful task. This is the hypothesis."
    ),
)

INVERTED = Arm(
    name="inverted",
    routing={
        StepKind.CLASSIFY: Tier.LARGE,
        StepKind.PLAN: Tier.SMALL,
        StepKind.HARD_REASONING: Tier.SMALL,
        StepKind.TOOL_FORMAT: Tier.LARGE,
        StepKind.SUMMARIZE: Tier.LARGE,
        StepKind.VERIFY: Tier.LARGE,
    },
    expectation=(
        "Worse task success than routed at comparable or higher cost. Spends the "
        "large model where capability does not change the outcome and starves the "
        "steps where it does."
    ),
    is_distinguishing_control=True,
)

ARMS: dict[str, Arm] = {arm.name: arm for arm in (ALL_LARGE, ALL_SMALL, ROUTED, INVERTED)}


def arm(name: str) -> Arm:
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; known arms: {', '.join(sorted(ARMS))}") from None
