# Task interaction contract

`TaskInteractionSpec` is the semantic boundary shared by `infra-bench` and
`infra-aware-mas`. It contains the objective, initial artifacts, allowed semantic
operators, observable operation results, runtime verifier, and hidden external
evaluator. Infrastructure remains a separate `InfraState`/world object.

Operator IDs are stable semantics, not runtime tool names. Each MAS runtime maps
its concrete tools to these IDs through an `OperatorRegistry`. A task is rejected
as `not_realizable` before execution when any allowed operator has no runtime
binding. A benchmark workflow is rejected from the formal workflow bank when an
operator is unregistered, unbound, disallowed for the task, or consumes an
artifact that cannot be derived from the task inputs and prior nodes.

Formal admission is the conjunction:

```text
MAS-realizable AND benchmark-correct AND infra-sensitive
```

Runtime verification has three levels: `none`, `partial`, and `terminal`.
SWE-bench local test feedback is `partial`; the official SWE-bench evaluator is
an `external_only` evaluator and is never exposed to the Planner.
