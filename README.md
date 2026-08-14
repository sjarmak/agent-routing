# Agent routing economics

**Status: apparatus only. No results yet.** The harness, arms, and metrics are
built and tested. Nothing has been run against a live endpoint, so this
repository currently supports no claim about any model, router, or provider.

## The question

Agents call a model many times per task: to classify, to plan, to reason, to
format a tool call, to summarize, to verify. Sending every one of those steps
to the most capable model is the default, and it is expensive. Does routing
cheap steps to a small model pay for itself?

The answer is not a per-token price comparison. A small model that fails and
escalates costs the small call, plus the large call, plus the latency of
discovering the failure. So the metric is **cost per successful task**: every
attempt in the numerator, only successes in the denominator.

## Hypothesis

An agent does not need its most capable model for every step. Routing
classification, tool formatting, summarization, and verification to a small
model holds task success while lowering cost per successful task.

## Arms

Four routing configurations over the same tasks, the same seeds, and the same
tool environment. Only the routing differs.

| Arm | Routing | Predicted before running |
| --- | --- | --- |
| `all-large` | every step to the large model | quality ceiling, highest cost |
| `all-small` | every step to the small model | cost floor, lowest success |
| `routed` | plan and hard reasoning large; the rest small | success near `all-large`, lower cost per success |
| `inverted` | hard reasoning small; formatting and summarizing large | worse success at comparable cost |

`inverted` is the distinguishing control and it is the reason this is an
experiment rather than a demo. It spends the large model where capability does
not change the outcome and starves the steps where it does. **If `inverted`
scores the same as `routed`, the task set cannot detect routing quality, and no
routing claim is supportable from it.** That result would be worth publishing
too.

`all-small` is the second control: if it matches `all-large`, model choice does
not matter on these tasks and the whole comparison is measuring noise.

## What gets measured

Per task, per arm:

- **cost per successful task** — the headline. Failed attempts included.
- **attempts per success** — where a cheap router hides its cost.
- **pass rate** against a verifier that decides success independently.
- **end-to-end task latency**, p50 and p95, summed across every attempt the
  user waited through, not just the attempt that worked.
- **time to first token** per call, and **model time share**: the fraction of
  wall clock spent inside model calls. Model time share bounds what a faster
  endpoint can buy. Below roughly half, endpoint speed is not the lever this
  workload needs.
- **tokens** in, out, and cached, with **cache reuse rate** reported as
  unmeasured rather than zero when the endpoint does not report it.
- **inference calls per task**.

Prices are supplied by the operator in a price file. A model with no configured
price raises `MissingPriceError` rather than costing zero.

## What this will not claim

- No provider ranking. Latency and cache behavior measured against a shared or
  serverless endpoint are partly a measurement of that endpoint's load while the
  run happened. A dedicated-endpoint arm is the only way to separate those, and
  until one runs, latency results are bounded by that.
- No transfer to workloads with a different step mix. A workload whose wall
  clock is dominated by tools cannot be improved much by faster inference,
  whatever the routing.
- No claim that benchmark quality scores predict task success on your workload.
  That assumption is what the arms exist to test.

## Method

The comparison is between matched pairs: the same task, the same seed, the same
tool environment, run under each arm. `unmatched_pairs` names any task only one
arm completed, because a comparison across different workloads is not a
comparison.

Arms, task set, metrics, and decision rules are frozen in
[`docs/preregistration.md`](docs/preregistration.md) before any run. Trial
records are append-only JSONL; the writer refuses a run root that already
belongs to a different run.

## Task sets

No benchmark task content is vendored here. Tasks come from a manifest the
operator supplies, so the harness can point at a public suite, a suite you are
licensed to redistribute, or tasks you authored. See
[`tasks/example-manifest.json`](tasks/example-manifest.json) for the shape,
which matches the fields a CodeScaleBench suite carries so an existing suite can
be adapted with a projection.

## Run the tests

```bash
python3 -m pytest -q
```

The apparatus has no third-party runtime dependencies. Percentiles use the
nearest-rank method so any number here can be checked by hand against a sorted
list.

## Prior art in this line of work

The evidence discipline — preregistered arms, a control designed to fail,
matched pairs, append-only raw records, admission before analysis — comes from
[agent-durability-lab](https://github.com/sjarmak/agent-durability-lab). The
per-task cost and trace layout follows
[CodeScaleBench](https://github.com/sourcegraph/CodeScaleBench) (Apache-2.0);
its methodology is adapted here, its task content is not redistributed.

## License

MIT. See [LICENSE](LICENSE).
