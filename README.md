# Jet-Impingement Data-Reduction Pipeline

A measurement data-reduction pipeline for an **impinging slot-jet heat-transfer
experiment**: a heated slot jet impinging on an instrumented plate, with an
optional cylindrical vortex generator placed at the nozzle exit. The code turns
raw wind-tunnel acquisitions (four-hole Cobra probe velocity time-histories,
thermocouple temperatures, electrical-power and pumping-pressure logs) into
spatially-resolved thermal and aerodynamic fields, with **first-order
uncertainty propagated end-to-end**.

> **Method & code only — no data, no results.**
> The study these scripts were written for is currently **under peer review at
> the *International Journal of Heat and Mass Transfer* (IJHMT)**. To respect the
> review embargo, this repository contains **only the processing code and the
> method**. It ships with **no measured data, no calibration data, and no
> results** (no reduced tables, no figures, no fitted values). The scripts are
> the machinery that *produces* results from data you supply yourself.

The pipeline was built to make one thing explicit: **how a measurement is
reduced, and what each reduced number is worth.** Every derived quantity
(Reynolds number, nozzle velocity, Nusselt number, enhancement ratio, Strouhal
number, thermal-hydraulic performance) carries a propagated 1σ uncertainty from
the instrument specifications through the calibration to the final estimate.

---

## What it computes

| Stage | Module | Method |
|-------|--------|--------|
| **Flow / Reynolds** | `processors/Re_Ve_processor.py` | Nozzle-exit velocity and Reynolds numbers from the tunnel differential pressure (Bernoulli + tunnel-to-nozzle area ratio), expressed in a pitot reference via an affine channel→pitot calibration. |
| **Cobra decoding** | `processors/read_th_file_processor.py` | Binary `.thA` time-history decoding, a three-stage coordinate rotation into the flow-aligned frame, and a per-sample affine Cobra→pitot velocity calibration. |
| **Mean fields** | `processors/mean_processor.py` | Time-averaged velocity maps, turbulence intensities, and plane-jet decay metrics (potential-core length, decay rate, virtual origin) with Monte-Carlo error propagation. |
| **Spectra** | `processors/frequency_processor.py` | Welch PSDs, dominant shedding frequency, Strouhal number, shedding coherence, and an inertial-range (Kolmogorov −5/3) cascade-slope diagnostic. |
| **Thermal / Nusselt** | `processors/thermal_processor.py` | Local and global Nusselt numbers from thermocouple temperatures and electrical heat flux, a constant additive heat-loss correction, and enhancement ratios referenced to literature slot-jet correlations. |
| **Thermal-hydraulic PEC** | `processors/pec_processor.py` | Webb performance-evaluation criterion `PEC = (Nu/Nu₀)/(f/f₀)^(1/3)` at constant pumping power, with matched-Reynolds loss interpolation. |
| **Cross-checks** | `processors/Re_comparison_processor.py` | Reynolds cross-validation: theoretical (pressure-based) vs directly measured (Cobra) nozzle-exit velocities. |
| **Uncertainty** | `utils/uncertainty.py` | Shared RSS (first-order Gaussian) propagation for every derived quantity, including calibration-aware terms. |
| **Air properties** | `utils/air_properties.py` | Density, viscosity, conductivity and Prandtl number via the ideal-gas law and Sutherland's law. |
| **Visualisation** | `visualizers/*.py` | Publication-style 1D profiles, 2D fields, spectra, thermal maps, and cross-case superpositions. |
| **Diagnostics** | `diagnostics/motor_drift_investigation.py` | Detects and corrects a low-frequency wind-tunnel "breathing" drift that would otherwise bias temporal means. |
| **Calibration tools** | `calibration/` | Standalone helpers to reduce velocity-calibration and nozzle-profile check-up runs. |

Literature baselines used (and cited in the code where applied): Hofmann,
Kind & Martin (2007), *Int. J. Heat Mass Transfer* 50, 3957–3965; Martin (1977),
*Adv. Heat Transfer* 13.

## Layout

```
usage.py                     # Tkinter hub: runs the pipeline and browses figures
scripts/
  pipeline/                  # orchestration + new-case intake
  processors/                # the reduction stages (see table above)
  utils/                     # uncertainty, air properties, case labelling
  visualizers/               # figure generation
  diagnostics/               # motor-drift investigation
calibration/                 # velocity-calibration + nozzle-profile helpers
```

## Requirements

Python 3.10+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`,
`Pillow` (and `tkinter`, which ships with the standard CPython installer):

```bash
pip install -r requirements.txt
```

## Running

The scripts are the reduction engine, not a dataset. To use them on your own
experiment, point them at a workspace folder holding your raw acquisitions in
the expected layout. The interactive hub is the entry point:

```bash
python usage.py
```

Each processor also runs standalone (e.g. `python scripts/processors/Re_Ve_processor.py`)
and most expose a headless entry point for scripted/batch use.

## Calibration & configuration (bring your own)

The processors read rig geometry, fluid properties, acquisition settings, and
**velocity/thermocouple calibration constants** from `config/*.xlsx` workbooks.
Those workbooks are **not included** — they hold rig- and calibration-specific
values that are out of scope for a code-only release.

When no configuration is present, the code falls back to **sensible, non-secret
defaults and, crucially, to *no extra calibration***:

- the Cobra→pitot and channel→pitot affine maps default to the **identity**
  (slope `a = 1`, intercept `b = 0`, i.e. raw = calibrated);
- the calibration **fit standard errors default to `0`** (no calibration
  residual added);
- thermocouple channels are used **uncorrected** unless a calibration table is
  supplied.

To apply your own calibration, run your velocity- and thermocouple-calibration
campaigns, then provide:

- `config/config_calibration.xlsx` — affine slopes/intercepts and the Cobra
  validity limit;
- `config/config_uncertainty.xlsx` — the 1σ instrument uncertainties and the
  calibration fit standard errors (`cobra_fit_SE`, `channel_fit_SE`);
- `thermal/raw_inputs/calibration_TC.csv` — per-thermocouple calibration points.

The parameter names the loaders expect are documented inline in
`utils/uncertainty.py` and at the top of each processor.

## Scope & data policy

- **In scope:** the reduction methods, the uncertainty framework, the file
  parsers, and the visualisation code.
- **Out of scope (never committed):** any measured or reduced data
  (`.thA`/`.asA`/`.csv`/`.mat`/`.xlsx`), calibration data, result figures, and
  the manuscript. A strict [`.gitignore`](.gitignore) enforces this.

## License

Released under the [MIT License](LICENSE). © 2026 Octave Aldebert.
