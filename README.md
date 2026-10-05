# Jiangsu GFS–GHI probabilistic post-processing

This repository has four layers: `config/` declares what an experiment uses,
`src/nwp/` implements how it is done, `data/` stores receipt-backed research
material, and `outputs/` isolates every run. Scientific settings belong in
configuration, not in site-, model-, or date-specific scripts.

Install the base package (configuration, status, inventory, and data stages):

```powershell
python -m pip install -e .
```

Install CPU experiment support, deep-development support, or the complete
development-test environment explicitly. Deep architectures and bounded
training utilities currently require the CPU stack; they do not constitute a
complete unified deep workflow:

```powershell
python -m pip install -e ".[cpu]"
python -m pip install -e ".[cpu,deep]"
python -m pip install -e ".[cpu,deep,dev]"
```

The sole command-line interface is `python -m nwp`:

```powershell
python -m nwp validate --profile nanjing_cpu_diagnostic
python -m nwp data inventory
python -m nwp run --profile nanjing_cpu_diagnostic --to-stage splits
python -m nwp run --profile cpu_20site --models raw_gfs,xgboost --n-sites 10
python -m nwp evaluate --run-id <development-run-id>
python -m nwp figures --run-id <development-run-id>
python -m nwp status
```

Model stages require the explicit `--execute-model-stages` switch. Official
runs additionally require the fixed official profile, configuration hash,
readiness pass, validated models, and registered official result set. This
refactor did not run formal/full training or create official results.

Start with [the protocol](docs/01_protocol.md),
[current status](docs/02_status.md),
[data contract](docs/03_data_contract.md), and
[repository guide](docs/04_repository_guide.md).
