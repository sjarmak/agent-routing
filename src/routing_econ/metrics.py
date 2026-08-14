"""Metrics computed from raw trial records.

The headline is cost per successful task: every attempt's cost in the
numerator, only successes in the denominator. A router that saves money per
call and buys extra failed attempts does not look cheaper here, which is the
whole point.

No third-party dependencies. Percentiles use the nearest-rank method so a
result can be checked by hand against a sorted list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Prices = dict[str, "ModelPrice"]


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens, as published by the endpoint operator.

    Cached input is priced separately because prefix reuse is only an economic
    argument if the provider discounts it.
    """

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int | None) -> float:
        cached = cached_input_tokens or 0
        if cached and self.cached_input_per_mtok is None:
            raise ValueError(
                "provider reported cached input tokens but no cached price is configured; "
                "set cached_input_per_mtok or the saving cannot be claimed"
            )
        fresh = input_tokens - cached
        total = fresh * self.input_per_mtok + output_tokens * self.output_per_mtok
        if cached:
            assert self.cached_input_per_mtok is not None
            total += cached * self.cached_input_per_mtok
        return total / 1_000_000


class MissingPriceError(KeyError):
    """Raised when a record names a model with no configured price.

    Fails closed. An unpriced model must not be silently costed at zero.
    """


def call_cost(call: dict, prices: Prices) -> float:
    model = call["model"]
    try:
        price = prices[model]
    except KeyError:
        raise MissingPriceError(
            f"no price configured for model {model!r}; "
            f"priced models: {', '.join(sorted(prices)) or '(none)'}"
        ) from None
    return price.cost(call["input_tokens"], call["output_tokens"], call.get("cached_input_tokens"))


def trial_cost(trial: dict, prices: Prices) -> float:
    """Cost of every attempt in the trial, including the ones that failed."""
    return sum(
        call_cost(call, prices) for attempt in trial["attempts"] for call in attempt["calls"]
    )


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    trials: int
    successes: int
    total_cost_usd: float
    total_attempts: int

    @property
    def pass_rate(self) -> float:
        return self.successes / self.trials

    @property
    def cost_per_successful_task(self) -> float | None:
        """None when the arm succeeded at nothing. Zero successes is not free."""
        if self.successes == 0:
            return None
        return self.total_cost_usd / self.successes

    @property
    def cost_per_task(self) -> float:
        return self.total_cost_usd / self.trials

    @property
    def attempts_per_success(self) -> float | None:
        if self.successes == 0:
            return None
        return self.total_attempts / self.successes


def summarize_arm(trials: list[dict], prices: Prices) -> ArmSummary:
    if not trials:
        raise ValueError("cannot summarize an arm with no trials")
    names = {trial["arm"] for trial in trials}
    if len(names) != 1:
        raise ValueError(f"trials span several arms: {', '.join(sorted(names))}")
    successes = sum(1 for trial in trials if trial["attempts"][-1]["succeeded"])
    return ArmSummary(
        arm=names.pop(),
        trials=len(trials),
        successes=successes,
        total_cost_usd=sum(trial_cost(trial, prices) for trial in trials),
        total_attempts=sum(len(trial["attempts"]) for trial in trials),
    )


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. `fraction` is in (0, 1]."""
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must fall in (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[rank - 1]


def task_latencies_ms(trials: list[dict]) -> list[float]:
    """End-to-end wall clock per task, summed across attempts.

    A task that needed three attempts took the time of all three. Reporting
    only the successful attempt's latency understates what a user waited for.
    """
    return [sum(attempt["wall_clock_ms"] for attempt in trial["attempts"]) for trial in trials]


def ttft_ms(trials: list[dict]) -> list[float]:
    return [
        call["ttft_ms"]
        for trial in trials
        for attempt in trial["attempts"]
        for call in attempt["calls"]
    ]


def inference_calls_per_task(trials: list[dict]) -> list[int]:
    return [sum(len(attempt["calls"]) for attempt in trial["attempts"]) for trial in trials]


def model_time_share(trials: list[dict]) -> float | None:
    """Fraction of wall clock spent inside model calls rather than tools.

    A faster endpoint can only move end-to-end latency by this much. Below
    roughly a half, endpoint speed is not the lever the workload needs.
    """
    total = sum(sum(a["wall_clock_ms"] for a in trial["attempts"]) for trial in trials)
    if total <= 0:
        return None
    model_ms = sum(
        call["latency_ms"] for trial in trials for a in trial["attempts"] for call in a["calls"]
    )
    return model_ms / total


def cache_reuse_rate(trials: list[dict]) -> float | None:
    """Cached share of input tokens, or None when the endpoint never reported it.

    None means unmeasured. It does not mean no reuse happened.
    """
    reported = [
        call
        for trial in trials
        for attempt in trial["attempts"]
        for call in attempt["calls"]
        if call.get("cached_input_tokens") is not None
    ]
    if not reported:
        return None
    input_tokens = sum(call["input_tokens"] for call in reported)
    if input_tokens == 0:
        return None
    return sum(call["cached_input_tokens"] for call in reported) / input_tokens


def token_totals(trials: list[dict]) -> dict[str, int]:
    calls = [call for trial in trials for attempt in trial["attempts"] for call in attempt["calls"]]
    return {
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "cached_input_tokens": sum(call.get("cached_input_tokens") or 0 for call in calls),
    }


def unmatched_pairs(left: list[dict], right: list[dict]) -> list[str]:
    """Pair keys present in one arm but not the other.

    Two arms are only comparable over the tasks both actually ran. A non-empty
    result means the comparison is between different workloads.
    """
    left_keys = {trial["pair_key"] for trial in left}
    right_keys = {trial["pair_key"] for trial in right}
    return sorted(left_keys ^ right_keys)
