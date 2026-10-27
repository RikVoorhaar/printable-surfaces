# Agent instructions

## When to use a subagent

Use a subagent when the task is **clearly bounded in scope** but expensive in the main
session — specifically when it needs:

- many tool calls (exploration across many files, repeated grep/read cycles), or
- long-running shell commands (lattice generation, `phase16` regeneration, batch
  verification, slicing, log archaeology) that would otherwise flood the main context with
  waiting and noisy output.

Typical good fits: "verify these invariants on all five solids", "regenerate all artifacts",
"find every place that uses X", "run the full test suite and report failures".

Do **not** use a subagent for: small edits you can make in one or two tool calls, quick
greps, decisions that need the user's input, or work that will need several rounds of
back-and-forth with the user.

## Model routing

- **Implementation / investigation:** the default agent model (`deepseek-v4.1-flash-1017`).
- **Long batch work, regeneration, verification:** `openrouter/z-ai/glm-5.3-flash`
  (`model: "openrouter/z-ai/glm-5.3-flash"`).

## Subagent hygiene (learned the hard way)

- **Never let a subagent background a long job and exit.** State explicitly at the top of
  the prompt: no `background=true`, no trailing `&`, no `nohup`/`disown`, big timeouts
  (e.g. 1 200 000 ms), and **do not end the turn until the command has finished and its
  output has been reported**. Subagents otherwise return "still running" and leave orphan
  processes.
- **Give it the exact commands and pass criteria.** Include the file paths, the script to
  run, the numeric checks that must hold, and the exact report you want back.
- **One writer at a time.** Two agents must never edit the same file in parallel; run them
  sequentially. Read-only investigations may run in parallel.
- **Ask for the numbers, not prose.** Require the per-item results (table or per-solid
  lines) so the main session can judge them without re-running anything.
- **Regeneration is far too slow to babysit in the main session.** Delegate it.
- **Commit policy:** implementation subagents leave changes uncommitted; a regeneration
  agent may `git add -A && git commit` only after its verification passes.
- If a subagent exits early or reports an unverified result, re-dispatch it with stricter
  instructions rather than redoing the long work in the main session.

`output/` is gitignored: regenerated STL/PLY files live on disk and are served by the
dashboard from there, so regeneration does not produce a git diff.

## Context

- Printability rule: a rod is printable if its elevation is ≥ 48° (`SIN48`), or it is a
  horizontal bridge ≤ 10 mm (`SIN5`, `BRIDGE_MAX`); rods lying on the build plate are
  exempt.
- **The angle rule is never negotiable.** Forced post-optimization rods may break the
  length window (`L_MIN`/`L_MAX`) or the degree cap, but never the 48° angle rule.
