# Project continuity rules

## Context compaction recovery

When the conversation context has been compacted, summarized, or otherwise replaced while work is in progress, do not rely on the compacted summary alone.

Before planning, editing, or running project code after compaction:

1. Run `rg --files` to refresh the current repository file inventory.
2. Read this `AGENTS.md` and `docs/CURRENT_STATE.md` completely.
3. Inspect `git status --short` and the diff relevant to the active task so uncommitted user and agent work is not mistaken for baseline state.
4. Read only the project documents and source files identified by `docs/CURRENT_STATE.md` or directly relevant to the active task. Do not reread every Markdown document unless the user explicitly requests a full repository-document audit.
5. Reconcile the compacted summary with the current-state record, relevant documents, and implementation. Explicit user approvals and rejections recorded in `docs/CURRENT_STATE.md` take precedence over roadmap wording or the fact that code was already implemented.
6. Identify the latest design that the user explicitly approved, and distinguish it from later proposals, assumptions, implemented experiments, and unresolved objections. Do not infer approval from implementation state or roadmap wording.
7. Only then resume the pending phase or make a new implementation decision.

If `docs/CURRENT_STATE.md` is missing, inaccessible, stale, or conflicts with relevant evidence, report that explicitly and reconcile the conflict before continuing. Do not automatically substitute a full Markdown reread or the compacted summary.

## Current-state maintenance

Keep `docs/CURRENT_STATE.md` short and operational. Update it when the user explicitly approves or rejects a material design, stops a workflow, changes the next action, or when a phase gate is actually evaluated. It must distinguish:

- explicitly approved behavior;
- implemented but unapproved or rejected work;
- proposed but unapproved changes;
- the current blocker or decision required before implementation;
- the small set of documents and source files needed to resume.

Do not copy long histories or duplicate the full roadmap into the current-state file. Link to detailed evidence instead. Do not rewrite an approval record merely to match code that was implemented without approval.

## Design approval checkpoints

Before implementing a change that materially affects product behavior, user workflow or workload, evaluation meaning, training-data meaning, candidate generation, the model or algorithm, or roadmap order:

1. Present a concise design checkpoint to the user before editing code or launching a costly experiment.
2. State the objective, exact proposed behavior or candidate variants, what the user would need to do, what will be saved and how it will be used, the acceptance and stopping criteria, and which existing behavior would change or be removed.
3. Wait for explicit user approval of that checkpoint. A previous `go` approves only the concrete design presented immediately before it; it does not authorize a materially different interpretation or a later redesign.
4. If an ambiguity or mismatch is discovered after work starts, stop the affected implementation or collection, explain the mismatch plainly, and obtain approval before replacing the design, changing the evaluation target, or asking the user to perform a different task.

Small implementation details that do not alter an approved design, and straightforward fixes that restore already-approved behavior, may proceed without a new checkpoint.

Do not create an allegedly lower-burden detour, substitute proxy task, or new data-collection workflow merely because it is easier to implement. User effort must be justified by its exact role in the approved evaluation or training plan.
