"""Keep the published schema and the writer in sync.

No jsonschema dependency: these checks compare the schema's declared fields
against what the writer actually emits, which is the drift that matters.
"""

import json
from pathlib import Path

from agent_routing import Attempt, InferenceCall, StepKind, Tier, TrialRecord

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "trial_record.schema.json"


def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def written_record():
    record = TrialRecord(
        run_id="r1",
        arm="routed",
        task_id="t1",
        seed=7,
        pair_key="t1|seed=7|env=e",
        endpoint="shared",
        attempts=[
            Attempt(
                attempt_index=1,
                succeeded=True,
                wall_clock_ms=1000.0,
                calls=[
                    InferenceCall(
                        step_kind=StepKind.HARD_REASONING,
                        tier=Tier.LARGE,
                        model="large-1",
                        input_tokens=1000,
                        output_tokens=200,
                        ttft_ms=100.0,
                        latency_ms=900.0,
                        cached_input_tokens=400,
                    )
                ],
            )
        ],
    )
    return json.loads(record.to_json())


def test_record_fields_match_the_schema_exactly():
    emitted = set(written_record())
    declared = set(schema()["properties"])
    assert emitted == declared, f"missing {declared - emitted}, undeclared {emitted - declared}"


def test_attempt_fields_match_the_schema_exactly():
    emitted = set(written_record()["attempts"][0])
    declared = set(schema()["definitions"]["attempt"]["properties"])
    assert emitted == declared, f"missing {declared - emitted}, undeclared {emitted - declared}"


def test_call_fields_match_the_schema_exactly():
    emitted = set(written_record()["attempts"][0]["calls"][0])
    declared = set(schema()["definitions"]["call"]["properties"])
    assert emitted == declared, f"missing {declared - emitted}, undeclared {emitted - declared}"


def test_schema_arm_enum_covers_every_defined_arm():
    from agent_routing import ARMS

    assert set(schema()["properties"]["arm"]["enum"]) == set(ARMS)


def test_schema_step_kind_enum_covers_every_step_kind():
    declared = set(schema()["definitions"]["call"]["properties"]["step_kind"]["enum"])
    assert declared == {kind.value for kind in StepKind}


def test_schema_version_constant_matches_the_writer():
    assert schema()["properties"]["schema_version"]["const"] == written_record()["schema_version"]
