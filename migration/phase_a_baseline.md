# Phase A baseline

Recorded: 2026-10-03 (Asia/Shanghai)

## Frozen revision

- Branch used for the refactor: `codex/nwp-unified-refactor`
- Freeze tag: `pre-nwp-refactor-20261002`
- Frozen commit: `09230bce481bca561aef94b64f0d9a02479b41ae`
- Frozen commit subject: `clean figure exports to svg and png only`
- Remote at freeze audit: `origin = https://github.com/chennn1228/meteo-prediction.git`

The tag, the refactor branch starting point, local `main`, and `origin/main` all
resolved to the same commit when this record was produced.

## Baseline test execution

The frozen revision was checked out in a detached temporary Git worktree.  The
project's existing virtual environment was used without installing or updating
packages.  Because `data/` and `NWP_website_handoff/` are local project inputs
that are not present in the frozen Git tree, read-only directory junctions from
the temporary worktree to the corresponding directories in the working project
were added before the final baseline run.  Pytest temporary files were directed
to a directory inside the temporary worktree.

Final command (shown with paths shortened only for readability):

```text
.venv/Scripts/python.exe -m pytest -q --disable-warnings --basetemp .pytest-baseline-final
```

Final result:

```text
........................................................................ [ 91%]
.......                                                                  [100%]
79 passed in 17.68s
BASELINE_EXIT_CODE=0
```

## Failed setup attempts retained as evidence

1. Without the local ignored directories and without an explicit pytest temp
   directory: `76 passed, 2 failed, 1 error`.  The error was a Windows
   permission failure under `%TEMP%`; both failures were readiness checks that
   could not see local project inputs.
2. With the local `data/` directory and explicit pytest temp directory, but
   without `NWP_website_handoff/`: `77 passed, 2 failed`.  The sole failing
   structural prerequisite was the missing frozen website handoff directory,
   and the orchestrator test failed because it consumes that readiness result.

No test, source, configuration, data, figure, or report file in the frozen
revision was changed to obtain the passing result.
