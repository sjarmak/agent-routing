# Preregistration v1

Frozen before any run. Changing anything below starts a v2 and does not
retroactively apply to runs already recorded under v1.

**Status:** drafted, not yet frozen. The fields marked `TBD` must be filled and
this document committed before the first live trial. A run whose configuration
disagrees with the frozen document is not admissible.

## Question

Does routing agent steps across model tiers lower cost per successful task
without lowering task success?

## Arms

Four, defined in `src/routing_econ/arms.py`: `all-large`, `all-small`,
`routed`, `inverted`. Each is a total function from step kind to tier, so no
arm can differ from another by an unrouted step.

The step decomposition is fixed at six kinds: classify, plan, hard_reasoning,
tool_format, summarize, verify. Adding a kind invalidates every arm until all
of them assign it a tier.

## Models

- Large tier: TBD
- Small tier: TBD
- Endpoint: TBD (shared or dedicated; record which, and the region and GPU type
  when dedicated)

Model identity is recorded per call, not per arm, so a mid-run substitution is
visible in the raw records.

## Task set

- Source manifest: TBD
- Task count: TBD
- Selection rule: fixed before running; no task is added or dropped after the
  first trial.
- Verifier: each task carries one, and it decides success independently of the
  agent's own report.

## Seeds and pairing

- Seeds: TBD, fixed list.
- Every arm runs every (task, seed) combination in the same tool environment.
- `pair_key` is `task_id|seed|env`. Arms are compared only over pair keys both
  completed.

## Retry policy

- Maximum attempts per task: TBD, identical across arms.
- An attempt ends when the verifier decides, not when the agent claims done.
- Escalation on failure, if enabled, is part of the arm definition and must be
  declared here rather than applied ad hoc.

## Primary outcome

Cost per successful task, per arm, over the matched pair set.

## Secondary outcomes

Pass rate, attempts per success, p50 and p95 end-to-end task latency, time to
first token, model time share, input/output/cached tokens, cache reuse rate,
inference calls per task.

## Decision rules

Declared before seeing results:

1. The hypothesis is supported if `routed` shows lower cost per successful task
   than `all-large` while its pass rate is not lower by more than TBD
   percentage points.
2. The comparison is void if `inverted` does not score worse than `routed` on
   pass rate. That outcome means the task set cannot detect routing quality,
   and it gets published as that, not reframed.
3. The comparison is void if `all-small` matches `all-large` on pass rate.
   Model choice does not matter on these tasks.
4. Latency results are reported as bounded by endpoint contention unless a
   dedicated-endpoint arm ran alongside the shared one.

## What invalidates a run

- A model substituted mid-run without a new run root.
- A task added, dropped, or edited after the first trial.
- An arm run against a different tool environment than its pair.
- A price applied to a model that had no configured price at run time.
- Missing raw records for any trial included in an aggregate.

An invalidated run is preserved and labeled, never deleted or rerun over.

## What this design cannot answer

Whether these results transfer to a workload with a different step mix, a
different task difficulty distribution, or a different tool latency profile.
Whether the endpoint's latency behavior holds under load other than the load
present during the run.
