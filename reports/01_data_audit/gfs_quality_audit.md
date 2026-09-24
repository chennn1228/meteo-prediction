# GFS / Open-Meteo data-quality audit

This audit uses raw API values from 20 requested sites over 2024-02–2026-08; it does not train a model. Solar elevation uses the preceding-hour midpoint and Ineichen clear-sky GHI uses twelve five-minute midpoint samples over that interval, both at API-returned GFS service coordinates. Ratios are not clipped. Requested coordinates are provenance only.

## Main evidence

- Rows: 1,357,920; returned GFS service points: 20.
- Raw GFS GHI negative rate: 0.0000%. These values would become zeros in the legacy clean stage.
- Raw GFS GHI zero rate: 46.7483% overall and 0.0005% at preceding-hour midpoint solar elevation >15° (3 rows).
- Severe raw zeros with solar elevation >15° and Himawari GHI >300 W/m²: 1 rows.
- Raw GHI=0 with DHI>0 or DNI>0: 41 rows. All are low-light intervals with clear-sky GHI ≤50 W/m², total cloud=100%, DNI=0 and DHI ≤6 W/m². They are consistent with preceding-hour interval semantics plus independent integer quantization, not a field shift.
- Raw total-cloud endpoints: P(0)=20.76%, P(100)=54.96%, interior=24.18%.
- Unclipped kt: p99=1.246, max=1.677, P(kt>1.5)=0.038%.
- Duplicate requested-site weighting at identical service-point × time × lead keys: 0.00%. The before/after effect is in `service_grid_duplication.csv`.

## Answers required by the protocol

### A — artifacts created by source processing

The exact pile at kt=1.5 was created by the former hard clip. The same mechanism affected kni at 1.5 and diffuse fraction at 1.0. Those clips are removed; raw and model columns are now separate. Legacy clean also converts negative radiation to zero and bounds cloud cover to [0,100], so cleaned zero counts cannot by themselves diagnose the provider. Old clipped feature-distribution figures are provisional/legacy evidence.

### B — real product distributions still usable for modelling

Raw total cloud cover genuinely has endpoint mass at 0 and 100, and raw radiation/ratios are discretized because the API product is hourly and numerically quantized. These are product characteristics, not automatically errors. Their magnitude and sensitivity to deduplicating returned service cells are machine-readable in the supplied CSVs.

### C — issues that can threaten source usability

After aligning solar geometry to the preceding-hour radiation interval, only 3 rows remain above 15° (two unique events). The one row with Himawari GHI >300 W/m² is a coherent all-radiation-zero, 100%-cloud D+1 forecast miss surrounded by valid hours and leads: retain it as a genuine extreme forecast error, not missing/corrupt data. The 41 component-difference rows are twilight quantization cases below the registered ratio denominator threshold and are not a source-usability blocker.

## Figure contract

Core conclusion: separate processing-created boundaries from raw product structure before judging data-source suitability. Evidence chain: raw-zero stratification → cloud endpoints → unclipped kt tail → solar-angle-stratified cloud–kt density → lead-specific error coupling. Archetype: quantitative grid. Exports: editable SVG plus PNG previews, with CSV source data.
