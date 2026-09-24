# Source-to-plot processing trace

| Variable | Original API field | Clean stage | Feature stage | Final model value | Artificially changes distribution? |
|---|---|---|---|---|---|
| GFS GHI | `shortwave_radiation_previous_day{1,2,3}` | negatives set to 0 in legacy clean | none | `ghi_fcst` | Yes for negative raw values; audit uses raw API |
| DHI | `diffuse_radiation_previous_day{1,2,3}` | negatives set to 0 | none | `dhi_fcst` | Yes for negative raw values; audit uses raw API |
| DNI | `direct_normal_irradiance_previous_day{1,2,3}` | negatives set to 0 | none | `dni_fcst` | Yes for negative raw values; audit uses raw API |
| GTI | `global_tilted_irradiance_previous_day{1,2,3}` | negatives set to 0 | none | `gti_fcst` | Yes for negative raw values; audit uses raw API |
| total cloud cover | `cloud_cover_previous_day{1,2,3}` | bounded to [0,100] in legacy clean | lag/change/rolling derived later | `cloud_cover_fcst` plus dynamics | Only if API is outside physical range; audit uses raw API |
| clear-sky GHI | none; pvlib Ineichen at returned GFS coordinate | not applicable | mean of 12 five-minute midpoints over preceding hour | `ghi_clear_sky` | Derived, not clipped |
| kt | none | not applicable | `kt_raw=GHI_GFS/GHI_clear` only when clear GHI >50 | `kt_model` currently identity | Legacy [0,1.5] clip did; removed |
| kni | none | not applicable | `kni_raw=DNI_GFS/DNI_clear` only when clear DNI >50 | `kni_model` currently identity | Legacy [0,1.5] clip did; removed |
| diffuse fraction | none | not applicable | `diffuse_fraction_raw=DHI/GHI` only when GHI >50 | `diffuse_fraction_model` currently identity | Legacy [0,1] clip did; removed |

Raw diagnostic columns and model columns are separate by contract. Fold-local missing-value imputation and standardization occur only at model preprocessing and are forbidden in this audit. Lag/rolling features sort by returned-service identity, lead, and issue time; they are not used in the raw distribution figures.
