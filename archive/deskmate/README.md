# DeskMate archive pointer

DeskMate is no longer an active product dependency of this branch. Its latest
unfinished state is preserved locally at:

- branch: `codex/archive-deskmate-20260721`
- commit: `668528f2f8d016d76545b7a24859914184e36300`
- snapshot validation: project `.venv`, `213 passed in 31.97s`

The commit intentionally records work in progress; target-camera evidence,
stereo hardware gates and controller integration were not complete. Inspect it
without copying it into Poker Dealer:

```powershell
git show 668528f:docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md
git diff main..codex/archive-deskmate-20260721
```

The branch has not been pushed. Do not delete ignored local model/data assets
based on this pointer; they require a separate read-only inventory and explicit
human approval.
