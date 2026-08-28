# Data dictionary for analysis-used fields

This file documents only the workbook fields used directly by the reproducibility code. It does not attempt to reconstruct metadata that are not present in the supplied analytical workbook.

## Workbook

- File expected by the analysis: `Dataset.xlsx`
- Sheet: `Data_Tong_hop`
- Time zone used in the study: Vietnam Standard Time (UTC+7)
- Nominal time step: 5 minutes

## Core fields

| Workbook column | Role in analysis | Unit / interpretation |
|---|---|---|
| `Thời gian ghi nhận` | Synchronized observation timestamp | Local timestamp, UTC+7 |
| `Flow out 1 - Giá trị (m3/h)` | Final-effluent flow `Q` | m3/h |
| `COD - Giá trị (mg/L)` | COD concentration `C` | mg/L |
| `COD - Trạng thái` | COD station status | Operational metadata |
| `TSS - Giá trị (mg/L)` | TSS concentration `C` | mg/L |
| `TSS - Trạng thái` | TSS station status | Operational metadata |
| `NH4+ - Giá trị (mg/L)` | Archived study ammonium channel | Interpreted in the manuscript as NH4-N, mg N/L |
| `NH4+ - Trạng thái` | Archived ammonium-channel status | Operational metadata |
| `pH - Giá trị` | pH value used in status/QC summary | pH units |
| `pH - Trạng thái` | pH station status | Operational metadata |
| `Temp - Giá trị (oC)` | Temperature value used in status/QC summary | degrees C |
| `Temp - Trạng thái` | Temperature station status | Operational metadata |

## Status handling

The primary chemistry analyses retain observations with either of the following status labels:

- `Hoạt động tổt`
- `Vượt qui chuẩn`

The latter is retained as station-configured exceedance metadata; the code does not interpret it as a reconstructed legal-compliance determination.

The status audit also reports time labelled:

- `Hiệu chuẩn` (calibration)
- `Lỗi thiết bị` (device error)

Blank/unaccepted statuses and non-positive chemistry values are excluded from the primary pollutant-specific analyses.

## Derived analysis fields

For an eligible pollutant observation:

```text
load_rate = C * Q / 1000
interval_mass = load_rate * (5 / 60)
```

For COD and TSS this gives kg/h and kg per five-minute interval. For NH4-N it gives kg N/h and kg N per five-minute interval.

Exact zero-flow observations are retained as observed hydraulic states in the source record, but primary mass-ranking analyses require positive flow. Missing timestamps are not treated as zero flow.

## Important naming note for NH4-N

The workbook field name contains `NH4+`, while the study documentation and manuscript treat the archived channel as NH4-N (as N). The reproducibility code preserves the workbook column name for data access but reports the channel as `NH4-N`. No chemical-form conversion is applied.
