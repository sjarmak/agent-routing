import json

import pytest

from routing_econ import (
    ARMS,
    INVERTED,
    ROUTED,
    Arm,
    Attempt,
    InferenceCall,
    ManifestTaskSource,
    StepKind,
    Task,
    Tier,
    TrialRecord,
    TrialWriter,
    arm,
    pair_key,
    read_trials,
)


def a_call(**overrides):
    defaults = dict(
        step_kind=StepKind.HARD_REASONING,
        tier=Tier.LARGE,
        model="large-1",
        input_tokens=1000,
        output_tokens=200,
        ttft_ms=100.0,
        latency_ms=900.0,
    )
    defaults.update(overrides)
    return InferenceCall(**defaults)


def a_record(**overrides):
    defaults = dict(
        run_id="r1",
        arm="routed",
        task_id="t1",
        seed=7,
        pair_key="t1|seed=7|env=e",
        endpoint="shared",
        attempts=[Attempt(attempt_index=1, succeeded=True, wall_clock_ms=1000.0, calls=[a_call()])],
    )
    defaults.update(overrides)
    return TrialRecord(**defaults)


def test_every_arm_routes_every_step_kind():
    for name, configured in ARMS.items():
        assert set(configured.routing) == set(StepKind), name


def test_an_arm_missing_a_step_kind_is_rejected():
    with pytest.raises(ValueError, match="does not route"):
        Arm(name="partial", routing={StepKind.CLASSIFY: Tier.SMALL}, expectation="incomplete")


def test_the_losing_arm_is_marked_as_the_distinguishing_control():
    assert INVERTED.is_distinguishing_control
    assert not ROUTED.is_distinguishing_control
    assert INVERTED.tier_for(StepKind.HARD_REASONING) is Tier.SMALL
    assert ROUTED.tier_for(StepKind.HARD_REASONING) is Tier.LARGE


def test_unknown_arm_names_the_known_ones():
    with pytest.raises(KeyError, match="all-large"):
        arm("nope")


def test_time_to_first_token_cannot_exceed_call_latency():
    with pytest.raises(ValueError, match="first token"):
        a_call(ttft_ms=1000.0, latency_ms=500.0)


def test_cached_tokens_cannot_exceed_input_tokens():
    with pytest.raises(ValueError, match="cached input tokens"):
        a_call(input_tokens=100, cached_input_tokens=101)


def test_a_trial_cannot_continue_after_success():
    attempts = [
        Attempt(attempt_index=1, succeeded=True, wall_clock_ms=10.0, calls=[a_call()]),
        Attempt(attempt_index=2, succeeded=True, wall_clock_ms=10.0, calls=[a_call()]),
    ]
    with pytest.raises(ValueError, match="after a successful attempt"):
        a_record(attempts=attempts)


def test_attempt_indices_must_be_ordered_from_one():
    attempts = [Attempt(attempt_index=2, succeeded=True, wall_clock_ms=10.0, calls=[a_call()])]
    with pytest.raises(ValueError, match="1..n"):
        a_record(attempts=attempts)


def test_an_attempt_needs_at_least_one_call():
    with pytest.raises(ValueError, match="no inference calls"):
        Attempt(attempt_index=1, succeeded=True, wall_clock_ms=10.0, calls=[])


def test_writer_appends_and_round_trips(tmp_path):
    writer = TrialWriter(tmp_path / "run", "r1")
    writer.append(a_record(task_id="t1", pair_key="t1|seed=7|env=e"))
    writer.append(a_record(task_id="t2", pair_key="t2|seed=7|env=e"))
    trials = read_trials(writer.path)
    assert [trial["task_id"] for trial in trials] == ["t1", "t2"]
    assert trials[0]["attempts"][0]["calls"][0]["tier"] == "large"
    assert trials[0]["schema_version"] == "routing-econ-trial-v1"


def test_writer_refuses_to_reuse_another_runs_root(tmp_path):
    TrialWriter(tmp_path / "run", "r1")
    with pytest.raises(FileExistsError, match="already holds run"):
        TrialWriter(tmp_path / "run", "r2")


def test_writer_rejects_a_record_from_a_different_run(tmp_path):
    writer = TrialWriter(tmp_path / "run", "r1")
    with pytest.raises(ValueError, match="belongs to run"):
        writer.append(a_record(run_id="r2"))


def test_manifest_source_rejects_duplicate_task_ids(tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "t1", "prompt": "p", "verifier": "v"},
                    {"task_id": "t1", "prompt": "p", "verifier": "v"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate task_id"):
        list(ManifestTaskSource(manifest))


def test_manifest_source_reads_optional_fields(tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {"tasks": [{"task_id": "t1", "prompt": "p", "verifier": "v", "work_type": "debug"}]}
        )
    )
    tasks = list(ManifestTaskSource(manifest))
    assert tasks == [Task(task_id="t1", prompt="p", verifier="v", work_type="debug")]


def test_a_task_without_a_verifier_is_rejected():
    with pytest.raises(ValueError, match="verifier"):
        Task(task_id="t1", prompt="p", verifier="")


def test_pair_key_binds_task_seed_and_environment():
    task = Task(task_id="t1", prompt="p", verifier="v")
    assert pair_key(task, 7, "e") == "t1|seed=7|env=e"
    assert pair_key(task, 8, "e") != pair_key(task, 7, "e")
