# Research protocol

## Research question and evidence chain

The project tests the value of location-specific GFS GHI forecasts at the
registered leads and the incremental value of increasingly complex
post-processing in Jiangsu. GHI is the formal target. The scientific claim,
scope, target, development interval, final-test interval, and evidence boundary
are authoritative only in `config/protocol.yaml`.

The evidence chain is:

```text
receipt-backed data → issue-time-safe features → purged rolling validation
→ model selection and fitting → prediction → later causal calibration
→ evaluation and mechanism analysis → figures and report
```

## Data and feature mechanisms

Forecast and truth products are defined by `config/data.yaml`; feature groups,
derived mechanisms, and forbidden fields are defined by
`config/features.yaml`. Requested coordinates are provenance. Returned service
coordinates are the physical/spatial study object. Identity fields, future
truth, and any feature unavailable at forecast issue time are forbidden model
inputs. Preprocessing is fitted within the applicable fit block.

## Validation and final-test isolation

The outer windows, inner-fold construction, purge derivation, early-stop block,
scoring block, and gap sensitivity are referenced from
`config/protocol.yaml::validation`. Inner fit, early-stop, and score blocks are
disjoint. Outer scoring never selects final-test behavior. Final fitting, early
stopping, later calibration, and final testing follow the declared chronology;
final-test truth cannot affect tuning, fitting, stopping, or calibration.

## Probability, evaluation, and interpretation

Quantile levels, the primary selection metric, and calibration method are
defined by `config/protocol.yaml::probability`. Probability metrics are the
primary comparison; deterministic metrics are auxiliary and use the registered
reference model on matched samples. Calibration is causal by forecast issue
time. Feature-group evidence and interpretation settings come from
`config/features.yaml`, and interpretation remains development evidence unless
the protocol explicitly promotes it.

## Spatial scope and formal evidence

Site membership and density selection are defined by `config/sites.yaml`.
Spatial conclusions require the configured truth gate and a converged returned-
service-point registry. Until those gates pass, spatial outputs are preflight
only. Development runs, imported historical artifacts, inherited incomplete data,
and figures outside a traceable official run are not formal evidence. The
official gate is defined by `config/protocol.yaml::execution_gate` and fails
closed.
