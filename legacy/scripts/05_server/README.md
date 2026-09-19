# Server execution status

The old GPU queues are intentionally blocked at process start. They encoded the
retired month-balanced protocol, trained cloud as a co-equal target, used stale
internal model numbering, and could delete result directories before validation.
They remain below the guard only as auditable legacy implementation evidence.

Do not remove the guards until all of the following are true:

1. the 18-variable GFS forecast archive and all featured tables have been rebuilt;
2. the nested purged rolling-origin selection artifacts exist for the GHI task;
3. every compared deep model has the same six-trial selection budget;
4. the final fit → early-stop → calibration → frozen-test chronology passes tests;
5. PINN constraint ablations are defined in physical units;
6. the province-wide hourly Himawari truth-availability gate has run;
7. a new non-destructive queue reads `project_manifest.yaml` and model_registry.py;
8. a smoke run passes before any full GPU run or automatic shutdown is enabled.

`run_server_full_v2.sh` does not exist and must not be inferred or auto-created.
The active local protocol source of truth is `project_manifest.yaml`.
