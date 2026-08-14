# Why these metrics

Notes on the inference behavior each metric is trying to catch, and what it
cannot tell you.

## Time to first token versus generation

TTFT is dominated by prefill: the model processing the prompt before it emits
anything. Output throughput governs the rest of the generation. Agents feel
TTFT more than a chat product does, because a single task issues many model
invocations and each one contributes another pause before anything happens.

The trap: end-to-end agent latency also contains tool execution, retrieval, and
orchestration. A model endpoint that is twice as fast does not make the agent
twice as fast. `model_time_share` reports the fraction of wall clock spent
inside model calls, which is the ceiling on what any endpoint improvement can
buy for that workload. Report it beside any latency claim.

## KV cache and prefix reuse

Transformer decoding stores the keys and values of already-processed tokens so
they are not recomputed for every new token. When a provider also reuses that
state across requests sharing a prefix, the repeated part of the prompt gets
cheaper and faster to prefill.

Agents are unusually well suited to this: system prompt, tool schemas, and
conversation history are large and stable across the calls within a task.
Routing interacts with it directly — sending consecutive steps to different
models splits the workload across two caches, and a router that thrashes
between tiers can lose more in cache reuse than it saves in per-token price.

Two measurement rules follow. Cache reuse is reported as `None` when the
endpoint does not report it, because unmeasured is not zero. And cached input
is priced at the cached rate or not counted as a saving at all: `ModelPrice`
raises rather than assume a discount that was never configured.

## Routing

Route on where marginal capability changes the task outcome, not on where the
prompt looks hard. Escalation triggers worth distinguishing: task complexity
known up front, model confidence, failure and retry behavior, and workflow
stage.

Evaluate at the task outcome level. Benchmark quality scores do not
automatically predict success on your workload, which is the assumption the
`inverted` and `all-small` arms exist to test rather than assume.

## Speculative decoding

A small draft model proposes several tokens ahead and the target model verifies
them in one pass. Higher acceptance means fewer sequential decoding steps on
the expensive model. The metric that matters is acceptance rate, and it is
workload-specific: a draft model trained on your traffic accepts more than a
generic one.

It changes latency and cost without changing which model answers, so it is
orthogonal to routing and should be a separate arm rather than folded into the
routing comparison. This harness does not measure it yet.

## Shared versus dedicated endpoints

Shared and serverless endpoints are economically attractive for variable
workloads and require no capacity decision. Dedicated endpoints matter when
hardware choice, scaling bounds, region or data residency, predictable tail
latency, or production isolation matter.

For measurement, the distinction is decisive: latency and cache behavior on a
shared endpoint are partly a measurement of other tenants' load while the run
happened. p95 in particular is not reproducible across time on shared capacity.
The `endpoint` field on every trial record exists so this cannot be quietly
forgotten during analysis, and the preregistration requires latency results to
be reported as contention-bounded until a dedicated arm runs alongside.
