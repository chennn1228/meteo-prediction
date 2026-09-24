# GFS / Open-Meteo data-quality audit

This audit uses raw API values from 20 requested sites over 2024-02–2026-08; it does not train a model. Solar geometry and Ineichen clear-sky GHI are evaluated at API-returned GFS service coordinates. Ratios are not clipped. Requested coordinates are provenance only.

## Main evidence

- Rows: 1,357,920; returned GFS service points: 20.
- Raw GFS GHI negative rate: 0.0000%. These values would become zeros in the legacy clean stage.
- Raw GFS GHI zero rate: 46.7483% overall and 0.0011% at solar elevation >15°.
- Severe raw zeros with solar elevation >15° and Himawari GHI >300 W/m²: 1 rows.
- Raw GHI=0 with DHI>0 or DNI>0: 41 rows; these require lead/month/site inspection in the CSV before attributing cause.
- Raw total-cloud endpoints: P(0)=20.76%, P(100)=54.96%, interior=24.18%.
- Unclipped kt: p99=2.359, max=3.932, P(kt>1.5)=4.963%.
- Duplicate requested-site weighting at identical service-point × time × lead keys: 0.00%. The before/after effect is in `service_grid_duplication.csv`.

## Answers required by the protocol

### A — artifacts created by source processing

The exact pile at kt=1.5 was created by the former hard clip. The same mechanism affected kni at 1.5 and diffuse fraction at 1.0. Those clips are removed; raw and model columns are now separate. Legacy clean also converts negative radiation to zero and bounds cloud cover to [0,100], so cleaned zero counts cannot by themselves diagnose the provider. Old clipped feature-distribution figures are provisional/legacy evidence.

### B — real product distributions still usable for modelling

Raw total cloud cover genuinely has endpoint mass at 0 and 100, and raw radiation/ratios are discretized because the API product is hourly and numerically quantized. These are product characteristics, not automatically errors. Their magnitude and sensitivity to deduplicating returned service cells are machine-readable in the supplied CSVs.

### C — issues that can threaten source usability

Daytime raw GHI zeros co-occurring with substantial Himawari GHI, and raw GHI=0 while DHI or DNI is positive, are the serious subset. Their counts are retained by lead, month, and site rather than hidden by clipping. Attribution to time alignment or provider semantics requires inspecting those strata; this audit does not infer that all GFS data are unusable from visual banding alone.

## Figure contract

Core conclusion: separate processing-created boundaries from raw product structure before judging data-source suitability. Evidence chain: raw-zero stratification → cloud endpoints → unclipped kt tail → solar-angle-stratified cloud–kt density → lead-specific error coupling. Archetype: quantitative grid. Exports: editable SVG/PDF plus PNG previews, with CSV source data.
