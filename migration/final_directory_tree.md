# Final directory tree

Generated for the unified refactor closeout. Local environments, caches,
credentials, data partitions, and individual migration evidence rows are
collapsed intentionally.

```text
.
├── project_manifest.yaml
├── pyproject.toml
├── README.md
├── requirements.in
├── requirements.txt
├── requirements-lock.txt
├── config/
│   ├── protocol.yaml
│   ├── data.yaml
│   ├── features.yaml
│   ├── sites.yaml
│   ├── models.yaml
│   ├── experiments.yaml
│   └── local.example.yaml
├── src/nwp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core/
│   │   ├── config.py
│   │   ├── context.py
│   │   ├── hashing.py
│   │   ├── paths.py
│   │   ├── provenance.py
│   │   ├── schema.py
│   │   └── validation.py
│   ├── data/
│   │   ├── audit.py
│   │   ├── clean.py
│   │   ├── client.py
│   │   ├── contracts.py
│   │   └── fetch.py
│   ├── features/
│   │   ├── build.py
│   │   ├── engineering.py
│   │   ├── physics.py
│   │   └── preprocessing.py
│   ├── splits/
│   │   ├── diagnostics.py
│   │   └── rolling.py
│   ├── models/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── baselines.py
│   │   ├── statistical.py
│   │   ├── trees.py
│   │   └── deep/
│   │       ├── architectures.py
│   │       └── training.py
│   ├── experiment/
│   │   ├── calibration.py
│   │   ├── fitting.py
│   │   ├── prediction.py
│   │   └── tuning.py
│   ├── evaluation/
│   │   ├── grouped.py
│   │   ├── interpretation.py
│   │   ├── metrics.py
│   │   └── spatial.py
│   ├── visualization/
│   │   ├── figures.py
│   │   └── style.py
│   └── workflow/
│       └── pipeline.py
├── data/                         # ignored local research material
│   ├── raw/<source>/<site>/<month>.*
│   ├── clean/<config-hash>/<site>/<month>.*
│   ├── features/<config-hash>/<site>/<month>.*
│   ├── registry/
│   ├── catalog.json
│   └── data_inventory.csv
├── outputs/                      # ignored; one isolated directory per run
│   ├── development/
│   └── official/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── 01_protocol.md
│   ├── 02_status.md
│   ├── 03_data_contract.md
│   ├── 04_repository_guide.md
│   ├── 05_literature.md
│   ├── 06_issues.md
│   └── 07_history.md
└── migration/
    ├── repository_inventory.csv
    ├── repository_inventory_summary.json
    ├── migration_map.csv
    ├── figure_inventory.csv
    ├── data_*_plan.csv
    ├── development_run*_inventory.csv
    ├── phase_a_baseline.md … phase_k_cleanup.md
    ├── final_residual_search.md
    ├── final_acceptance.md
    └── recovered_artifacts/      # isolated inherited evidence, never active output
```

Ignored local support directories (`.venv/`, `.cache/`, `secrets/`,
`literature/`, `deploy/`, and `NWP_website_handoff/`) are intentionally not
part of the active research package or Git upload. The inaccessible local
`.pytest_cache/` is likewise ignored and has no repository effect.
