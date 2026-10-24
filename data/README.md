# Data

This folder holds the dataset used by the project.

## Automatic download (recommended)

From the repository root:

```bash
python -m src.data
```

This downloads the **AI4I 2020 Predictive Maintenance Dataset** archive from
the UCI Machine Learning Repository, extracts `ai4i2020.csv` into
`data/raw/`, and verifies it against a known SHA-256 checksum.

The CSV ships with this repository, so the command is a no-op when the file
is already present and correct.

## Manual fallback

If automatic downloading fails (no internet, proxy, etc.):

1. Open <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>.
2. Download the dataset archive and extract `ai4i2020.csv`.
3. Place the file at `data/raw/ai4i2020.csv` (create the `raw` folder if needed).

Expected checksum:

```
sha256: dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e
```

## Contents

| Path | Committed? | Purpose |
|---|---|---|
| `raw/ai4i2020.csv` | yes (~500 KB) | Raw source dataset (10,000 rows) |
| `raw/` | - | Put manual downloads here |

Generated outputs (EDA figures, reports) go to `reports/` and can be
regenerated at any time with `python -m src.eda` and `python -m src.train`.

## Dataset summary

- **Rows:** 10,000 machines, **339 failures (3.39%)**
- **Sensors:** air temperature [K], process temperature [K], rotational
  speed [rpm], torque [Nm], tool wear [min]
- **Product type:** L / M / H
- **Target:** `Machine failure`
- **Excluded from modelling:** `UDI`, `Product ID` (identifiers) and the
  failure-mode columns `TWF, HDF, PWF, OSF, RNF` (they *compose* the target
  - using them as features would be target leakage)
