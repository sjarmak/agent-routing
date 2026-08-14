import pytest

from routing_econ import (
    MissingPriceError,
    ModelPrice,
    cache_reuse_rate,
    inference_calls_per_task,
    model_time_share,
    percentile,
    summarize_arm,
    task_latencies_ms,
    token_totals,
    trial_cost,
    unmatched_pairs,
)

PRICES = {
    "large-1": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.3),
    "small-1": ModelPrice(input_per_mtok=0.3, output_per_mtok=1.2, cached_input_per_mtok=0.03),
}


def call(model, input_tokens=1000, output_tokens=200, cached=None, ttft=100.0, latency=900.0):
    return {
        "step_kind": "hard_reasoning",
        "tier": "large" if model.startswith("large") else "small",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        "ttft_ms": ttft,
        "latency_ms": latency,
    }


def trial(arm, pair_key, attempts):
    return {
        "run_id": "r1",
        "arm": arm,
        "task_id": pair_key.split("|")[0],
        "seed": 7,
        "pair_key": pair_key,
        "endpoint": "shared",
        "attempts": attempts,
    }


def attempt(index, succeeded, calls, wall_clock_ms=1000.0):
    return {
        "attempt_index": index,
        "succeeded": succeeded,
        "wall_clock_ms": wall_clock_ms,
        "tool_ms": 0.0,
        "calls": calls,
        "failure_reason": None if succeeded else "verifier_failed",
    }


def test_cost_counts_failed_attempts_against_successes():
    """Two cheap attempts to reach one success is not cheaper than one that worked."""
    cheap_but_retried = [
        trial(
            "routed",
            "t1|seed=7|env=e",
            [
                attempt(1, False, [call("small-1")]),
                attempt(2, True, [call("small-1")]),
            ],
        )
    ]
    summary = summarize_arm(cheap_but_retried, PRICES)
    per_attempt_cost = PRICES["small-1"].cost(1000, 200, None)
    assert summary.cost_per_successful_task == pytest.approx(2 * per_attempt_cost)
    assert summary.attempts_per_success == pytest.approx(2.0)
    assert summary.cost_per_task == pytest.approx(2 * per_attempt_cost)


def test_zero_successes_is_not_free():
    failing = [trial("all-small", "t1|seed=7|env=e", [attempt(1, False, [call("small-1")])])]
    summary = summarize_arm(failing, PRICES)
    assert summary.successes == 0
    assert summary.cost_per_successful_task is None
    assert summary.attempts_per_success is None
    assert summary.total_cost_usd > 0


def test_unpriced_model_fails_closed():
    unpriced = [trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("mystery-1")])])]
    with pytest.raises(MissingPriceError, match="mystery-1"):
        summarize_arm(unpriced, PRICES)


def test_cached_tokens_are_priced_at_the_cached_rate():
    record = trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("large-1", cached=800)])])
    expected = (200 * 3.0 + 800 * 0.3 + 200 * 15.0) / 1_000_000
    assert trial_cost(record, PRICES) == pytest.approx(expected)


def test_cached_tokens_without_a_cached_price_is_an_error():
    prices = {"large-1": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)}
    record = trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("large-1", cached=500)])])
    with pytest.raises(ValueError, match="cached price"):
        trial_cost(record, prices)


def test_unreported_cache_is_none_not_zero():
    records = [trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("large-1")])])]
    assert cache_reuse_rate(records) is None
    with_cache = [
        trial("routed", "t2|seed=7|env=e", [attempt(1, True, [call("large-1", cached=250)])])
    ]
    assert cache_reuse_rate(with_cache) == pytest.approx(0.25)


def test_latency_sums_every_attempt_the_user_waited_through():
    record = trial(
        "routed",
        "t1|seed=7|env=e",
        [
            attempt(1, False, [call("small-1")], wall_clock_ms=1200.0),
            attempt(2, True, [call("large-1")], wall_clock_ms=2000.0),
        ],
    )
    assert task_latencies_ms([record]) == [3200.0]
    assert inference_calls_per_task([record]) == [2]


def test_model_time_share_bounds_what_a_faster_endpoint_can_buy():
    record = trial(
        "routed",
        "t1|seed=7|env=e",
        [attempt(1, True, [call("large-1", latency=400.0)], wall_clock_ms=2000.0)],
    )
    assert model_time_share([record]) == pytest.approx(0.2)


def test_percentile_uses_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.5) == 20.0
    assert percentile(values, 0.95) == 40.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_unmatched_pairs_names_the_tasks_only_one_arm_ran():
    left = [trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("small-1")])])]
    right = [trial("all-large", "t2|seed=7|env=e", [attempt(1, True, [call("large-1")])])]
    assert unmatched_pairs(left, right) == ["t1|seed=7|env=e", "t2|seed=7|env=e"]


def test_summarize_refuses_mixed_arms():
    mixed = [
        trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("small-1")])]),
        trial("all-large", "t1|seed=7|env=e", [attempt(1, True, [call("large-1")])]),
    ]
    with pytest.raises(ValueError, match="several arms"):
        summarize_arm(mixed, PRICES)


def test_token_totals_reports_cached_separately():
    records = [
        trial("routed", "t1|seed=7|env=e", [attempt(1, True, [call("large-1", cached=400)])])
    ]
    assert token_totals(records) == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cached_input_tokens": 400,
    }
