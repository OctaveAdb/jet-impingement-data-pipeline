#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_drift_investigation.py
================================================================================
Standalone diagnostic for a low-frequency wind-tunnel motor drift ("breathing")
that biases the TEMPORAL-MEAN velocity.

Physical problem
----------------
The blower/motor introduces a slow, quasi-sinusoidal oscillation in the supply
flow.  A finite 32 s record almost never spans an integer number of drift
periods, so the simple time average  mean(v(t))  retains a fraction of the last
(partial) oscillation -> a *truncation bias* on the mean velocity.

What this script does (per case)
--------------------------------
TASK 1 - Diagnose & visualise
    * load the raw velocity time series v(t) at a representative core point,
    * plot v(t) over the full 32 s (is the breathing visible?),
    * compute a full-resolution amplitude spectrum (CLASSIC FFT, not Welch):
        mean removed -> Hann window (full length) -> optional zero-pad -> rFFT,
      zoomed on [0, 10] Hz, and auto-detect/print the oscillation peak f_osc.

TASK 2 - Correct via time-domain fitting
    * low-pass v(t) (default 5 Hz) to strip turbulence/probe noise and isolate
      the drift,
    * fit   v_model(t) = V_true + A*sin(2*pi*f*t + phi)   with curve_fit, using
      f_osc as the initial guess,
    * overlay v_model on the time trace, and report A, f, raw mean, corrected
      mean V_true and the relative bias.

The script is self-contained: edit the CONFIG block and run.  No pipeline import.

Author: diagnostic tool for the jet-impingement project.
================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit

# =============================================================================
# CONFIG  -- edit here
# =============================================================================
FS          = 2000.0          # sampling frequency [Hz]
SIGNAL_COL  = "Velocity_Magnitude (m/s)"   # column to analyse (mean-speed drift)
#   alternatives: "u (m/s)" (streamwise only), "v (m/s)", ...

# Peak-search band for the motor oscillation (avoid the DC skirt at the low end)
F_MIN       = 0.10            # [Hz] lower bound of the f_osc search
F_MAX       = 10.0            # [Hz] upper bound (also the FFT zoom limit)

ZERO_PAD_FACTOR = 4           # FFT zero-padding (>=1). Interpolates the spectrum
                              # for a cleaner peak; does NOT add true resolution.
LOWPASS_CUTOFF  = 5.0         # [Hz] low-pass cutoff to isolate the drift
LOWPASS_ORDER   = 4           # Butterworth order

# The drift is < LOWPASS_CUTOFF, so the curve_fit can run on a decimated copy of
# the (already low-passed) signal -> ~20x faster with identical parameters.
FIT_DECIMATE    = 20          # keep every 20th sample for the fit (2000 -> 100 Hz)

# Reliability filter for the "worst case": ignore near-zero-velocity edge points,
# where a relative bias is meaningless and the sinusoid fit is unreliable.
CORE_FRACTION   = 0.5         # a point is "reliable" if mean_raw >= this * case-max

# --- Broadband-drift handling (the motor tone may be multi-component) ---------
K_TONES         = 3           # (A) number of spectral peaks fitted (sum of sinusoids)
CONC_HALFWIDTH  = 0.25        # [Hz] +/- window around the main peak for "concentration"
CONC_TONAL      = 0.50        # concentration >= this -> "tonal", else "broadband"
BLOCK_S         = 4.0         # [s] block length for the (C) batch-means mean uncertainty

PLOT_DECIMATE   = 10          # decimate the raw trace when plotting (speed only)
SAVE_FIGS       = True
SHOW_FIGS       = True

# Resolve the experiments directory relative to this script
# (script lives in  Data/Data/scripts/diagnostics/ ).  Override if needed.
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT       = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
EXPERIMENTS_DIR = os.path.join(DATA_ROOT, "experiments")
FIG_DIR         = os.path.join(_SCRIPT_DIR, "figures")

# If True, process EVERY case folder found under experiments/ (recommended).
# If False, process only the explicit CASES dict below.
AUTO_DISCOVER_ALL = True

# Explicit case list (used only when AUTO_DISCOVER_ALL = False).
#   label    -> {folder, [optional] point}
#   'folder' : case sub-folder under experiments/
#   'point'  : (optional) exact Raw_Data CSV file name to use as the probe;
#              if omitted, the core point (max mean speed) is auto-selected.
CASES = {
    "Free 5Pa (free plane)":     {"folder": "Free5PaFree"},
    "Free 5Pa (plate)":          {"folder": "Free5PaPla"},
    "Free 10Pa (free plane)":    {"folder": "Free10PaFree"},
    "Free 10Pa (plate)":         {"folder": "Free10PaPla"},
    "Cyl 12mm 5Pa (free plane)": {"folder": "Cyl125PaFree"},
    "Cyl 12mm 5Pa (plate)":      {"folder": "Cyl125PaPla"},
    "Cyl 12mm 10Pa (free plane)":{"folder": "Cyl1210PaFree"},
    "Cyl 12mm 10Pa (plate)":     {"folder": "Cyl1210PaPla"},
}


# =============================================================================
# DATA LOADING
# =============================================================================
def _raw_data_dir(case_folder):
    """Return the Raw_Data directory for a case (handles the usual layout)."""
    d = os.path.join(EXPERIMENTS_DIR, case_folder, "Processed_CSVs", "Raw_Data")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Raw_Data folder not found: {d}")
    return d


def discover_all_cases():
    """Build the case dict from every experiments/<case> folder that has data.
    Label = folder name, so it is unambiguous and future-proof."""
    cases = {}
    if not os.path.isdir(EXPERIMENTS_DIR):
        raise FileNotFoundError(f"experiments/ not found: {EXPERIMENTS_DIR}")
    for d in sorted(os.listdir(EXPERIMENTS_DIR)):
        rd = os.path.join(EXPERIMENTS_DIR, d, "Processed_CSVs", "Raw_Data")
        if os.path.isdir(rd) and glob.glob(os.path.join(rd, "*.csv")):
            cases[d] = {"folder": d}
    return cases


def select_core_point(case_folder, signal_col=SIGNAL_COL):
    """Scan every grid-point file and return the one with the highest mean speed
    (the jet core), where the drift is cleanest and the mean velocity matters."""
    files = glob.glob(os.path.join(_raw_data_dir(case_folder), "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {case_folder}/.../Raw_Data")
    best_file, best_mean = None, -np.inf
    for f in files:
        try:                               # read only the one column (fast)
            m = pd.read_csv(f, usecols=[signal_col])[signal_col].mean()
        except Exception:
            continue
        if np.isfinite(m) and m > best_mean:
            best_mean, best_file = m, f
    if best_file is None:
        raise RuntimeError(f"Could not read column '{signal_col}' in {case_folder}")
    return best_file


def load_case_series(case_def, signal_col=SIGNAL_COL):
    """Load (t, v, point_name) for one case definition."""
    folder = case_def["folder"]
    if case_def.get("point"):
        path = os.path.join(_raw_data_dir(folder), case_def["point"])
    else:
        path = select_core_point(folder, signal_col)
    df = pd.read_csv(path)
    v = df[signal_col].to_numpy(dtype=float)
    # Use the recorded time vector if present and sane, else reconstruct from FS.
    if "Time (s)" in df.columns:
        t = df["Time (s)"].to_numpy(dtype=float)
    else:
        t = np.arange(v.size) / FS
    # Drop any NaNs (rare QC gaps) by linear interpolation to keep uniform dt.
    if np.isnan(v).any():
        good = ~np.isnan(v)
        v = np.interp(t, t[good], v[good])
    return t, v, os.path.basename(path)


# =============================================================================
# TASK 1 -- FULL-RESOLUTION FFT + PEAK DETECTION
# =============================================================================
def amplitude_spectrum(v, fs=FS, zero_pad_factor=ZERO_PAD_FACTOR):
    """Single-sided amplitude spectrum of a mean-removed, Hann-windowed signal.

    Returns (freqs, amp).  `amp` is normalised by the window's coherent gain so
    that a pure sinusoid of amplitude A0 produces a peak of height ~A0.
    """
    x = v - np.mean(v)                       # centre the signal
    n = x.size
    w = np.hanning(n)                        # Hann window over the FULL record
    xw = x * w
    nfft = int(n * max(1, zero_pad_factor))  # optional zero-padding
    X = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    # coherent gain of the window = sum(w); single-sided -> factor 2
    amp = 2.0 * np.abs(X) / np.sum(w)
    return freqs, amp


def detect_fosc(freqs, amp, f_min=F_MIN, f_max=F_MAX):
    """Auto-detect the dominant oscillation peak within [f_min, f_max]."""
    band = (freqs >= f_min) & (freqs <= f_max)
    fb, ab = freqs[band], amp[band]
    if fb.size == 0:
        return np.nan, np.nan
    # Prefer a true local maximum (prominence-based); fall back to the band max.
    peaks, props = find_peaks(ab, prominence=np.nanmax(ab) * 0.05)
    if peaks.size > 0:
        k = peaks[np.argmax(ab[peaks])]
    else:
        k = int(np.argmax(ab))
    return fb[k], ab[k]


def top_peaks_and_concentration(freqs, amp, k=K_TONES,
                                f_min=F_MIN, f_max=F_MAX, halfwidth=CONC_HALFWIDTH):
    """Return (peak_freqs[<=k], f_main, concentration) for the drift band.

    `concentration` = fraction of band energy within +/- halfwidth of the main
    peak.  High (-> ~1) means the drift is essentially ONE tone (use the fit);
    low means the energy is spread = BROADBAND (trust the model-free SE instead).
    """
    band = (freqs >= f_min) & (freqs <= f_max)
    fb, ab = freqs[band], amp[band]
    if fb.size == 0:
        return np.array([]), np.nan, np.nan
    peaks, props = find_peaks(ab, prominence=np.nanmax(ab) * 0.05)
    if peaks.size == 0:
        order = np.array([int(np.argmax(ab))])
    else:
        order = peaks[np.argsort(props["prominences"])[::-1]]   # most prominent first
    f_main = fb[order[0]]
    peak_freqs = np.sort(fb[order[:k]])
    e_band = float(np.sum(ab ** 2))
    near = np.abs(fb - f_main) <= halfwidth
    conc = float(np.sum(ab[near] ** 2) / e_band) if e_band > 0 else np.nan
    return peak_freqs, f_main, conc


# =============================================================================
# TASK 2 -- LOW-PASS + SINUSOIDAL DRIFT FIT
# =============================================================================
def lowpass(v, fs=FS, cutoff=LOWPASS_CUTOFF, order=LOWPASS_ORDER):
    """Zero-phase Butterworth low-pass to isolate the slow drift."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, v)


def multi_drift_model(t, V_true, *params):
    """(A) Steady mean + a SUM of K low-frequency sinusoids.
    `params` is a flat list A_1,f_1,phi_1, A_2,f_2,phi_2, ...  (K = len/3)."""
    out = np.full_like(t, V_true, dtype=float)
    for k in range(len(params) // 3):
        A, f, phi = params[3 * k:3 * k + 3]
        out = out + A * np.sin(2.0 * np.pi * f * t + phi)
    return out


def fit_multi_drift(t, v_filt, seed_freqs):
    """(A) Fit V_true + sum_k A_k sin(2*pi*f_k*t + phi_k), seeded by the FFT peaks.
    Reduces to the single-sine model when one seed frequency is given.  V_true is
    the multi-component bias-corrected mean.  Returns dict (V_true, dominant tone,
    full popt, n_tones)."""
    seeds = [f for f in seed_freqs if np.isfinite(f) and f > 0] or [1.0]
    A0 = 0.5 * (np.max(v_filt) - np.min(v_filt))
    p0, lo, hi = [float(np.mean(v_filt))], [-np.inf], [np.inf]
    for f in seeds:
        p0 += [A0, f, 0.0]
        lo += [0.0, max(F_MIN, 0.5 * f), -np.inf]      # each f_k stays near its seed
        hi += [np.inf, min(F_MAX, 1.5 * f),  np.inf]
    popt, _ = curve_fit(multi_drift_model, t, v_filt,
                        p0=p0, bounds=(lo, hi), maxfev=40000)
    comps = sorted(((abs(popt[1 + 3 * k]), popt[2 + 3 * k])
                    for k in range(len(seeds))), reverse=True)   # (A, f) desc
    return {"V_true": popt[0], "A_main": comps[0][0], "f_main": comps[0][1],
            "popt": popt, "n_tones": len(seeds)}


def block_mean_uncertainty(v, fs=FS, block_s=BLOCK_S):
    """(C) Model-FREE uncertainty of the mean from low-frequency content: split the
    record into blocks, take each block mean, SE = std(block means)/sqrt(M).  This
    captures single-tone, multi-tone AND broadband drift with NO model assumption
    (it is the batch-means standard error, robust to the drift's spectral shape)."""
    n = int(block_s * fs)
    if n < 1 or v.size < 2 * n:
        return float(np.mean(v)), np.nan
    m = v.size // n
    bm = v[:m * n].reshape(m, n).mean(axis=1)
    return float(bm.mean()), float(bm.std(ddof=1) / np.sqrt(m))


# =============================================================================
# PLOTTING
# =============================================================================
def plot_case(label, t, v, r, file_tag=None):
    """Two-panel figure: (a) v(t)+multi-tone model, (b) zoomed spectrum w/ tones.
    `r` is a compute_point() result dict; `label` titles the figure."""
    mean_raw, fit = r["mean_raw"], r["fit"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Motor-drift diagnosis - {label}", fontsize=14, fontweight="bold")

    # --- (a) time series --------------------------------------------------
    d = max(1, PLOT_DECIMATE)
    ax1.plot(t[::d], v[::d], color="0.75", lw=0.6, label="raw $v(t)$")
    ax1.plot(t, r["v_filt"], color="tab:blue", lw=1.0, alpha=0.8,
             label=f"low-pass (<{LOWPASS_CUTOFF:.0f} Hz)")
    if fit is not None:
        ax1.plot(t, multi_drift_model(t, *fit["popt"]),
                 color="tab:red", lw=2.0, ls="--",
                 label=(f"{fit['n_tones']}-tone fit "
                        f"($A_{{band}}$={r['A_band']:.3f} m/s)"))
        ax1.axhline(fit["V_true"], color="tab:red", lw=1.0, ls=":",
                    label=f"$V_{{true}}$={fit['V_true']:.3f} m/s")
    se_txt = f"  (block SE {r['se_pct']:.3f}%)" if np.isfinite(r["se_pct"]) else ""
    ax1.axhline(mean_raw, color="k", lw=1.0, ls="-.",
                label=f"raw mean={mean_raw:.3f}{se_txt}")
    ax1.set_xlabel("time [s]"); ax1.set_ylabel(f"{SIGNAL_COL}")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_title(f"(a) Time series and fitted drift  [regime: {r['regime']}]",
                  loc="left", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # --- (b) spectrum zoom (mark every fitted tone) -----------------------
    freqs, amp = r["freqs"], r["amp"]
    ax2.plot(freqs, amp, color="tab:purple", lw=1.0)
    for pf in r["peak_freqs"]:
        kk = int(np.argmin(np.abs(freqs - pf)))
        ax2.plot(pf, amp[kk], "rv", ms=8)
    if np.isfinite(r["f_peak"]):
        kk = int(np.argmin(np.abs(freqs - r["f_peak"])))
        ax2.annotate(f"$f_{{peak}}$={r['f_peak']:.3f} Hz, conc={r['conc']:.2f}",
                     xy=(r["f_peak"], amp[kk]),
                     xytext=(r["f_peak"] + 0.6, amp[kk]),
                     fontsize=10, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax2.set_xlim(0, F_MAX)
    ax2.set_xlabel("frequency [Hz]"); ax2.set_ylabel("amplitude [m/s]")
    ax2.set_title("(b) Classic FFT (Hann, mean-removed, zero-padded) - "
                  "red v = fitted tones", loc="left", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        tag = file_tag if file_tag else label
        safe = "".join(c if c.isalnum() else "_" for c in tag)
        fig.savefig(os.path.join(FIG_DIR, f"motor_drift_{safe}.png"), dpi=200)
        fig.savefig(os.path.join(FIG_DIR, f"motor_drift_{safe}.pdf"))
    return fig


# =============================================================================
# MAIN
# =============================================================================
# =============================================================================
# PER-POINT ANALYSIS (every grid point of a case)
# =============================================================================
def _load_series_from_file(path):
    """Load (t, v) from one Raw_Data CSV (reads only the needed columns)."""
    try:
        df = pd.read_csv(path, usecols=[SIGNAL_COL, "Time (s)"])
    except Exception:
        df = pd.read_csv(path)
    v = df[SIGNAL_COL].to_numpy(dtype=float)
    t = (df["Time (s)"].to_numpy(dtype=float)
         if "Time (s)" in df.columns else np.arange(v.size) / FS)
    if np.isnan(v).any():                        # patch rare QC gaps
        good = ~np.isnan(v)
        v = np.interp(t, t[good], v[good])
    return t, v


def compute_point(t, v):
    """Full Task 1 + Task 2 for one time series, with broadband handling.
    Returns arrays + scalars (peak, A_main, A_band, regime, V_true, bias, SE)."""
    mean_raw = float(np.mean(v))
    freqs, amp = amplitude_spectrum(v)                          # Task 1 (FFT)
    peak_freqs, f_peak, conc = top_peaks_and_concentration(freqs, amp)
    regime = "tonal" if (np.isfinite(conc) and conc >= CONC_TONAL) else "broadband"

    v_filt = lowpass(v)                                         # Task 2
    A_band = float(np.std(v_filt))             # (B) RMS of all <cutoff fluctuation
    try:                                       # (A) K-sinusoid deterministic fit
        dec = max(1, FIT_DECIMATE)
        fit = fit_multi_drift(t[::dec], v_filt[::dec], peak_freqs)
        V_true, A_main, f_fit = fit["V_true"], fit["A_main"], fit["f_main"]
        bias = (mean_raw - V_true) / mean_raw * 100.0
    except Exception:
        fit, V_true, A_main, f_fit, bias = None, np.nan, np.nan, np.nan, np.nan

    _, se = block_mean_uncertainty(v)          # (C) model-free mean uncertainty
    se_pct = se / mean_raw * 100.0 if np.isfinite(se) else np.nan

    return dict(mean_raw=mean_raw, f_peak=f_peak, peak_freqs=peak_freqs, conc=conc,
                regime=regime, A_band=A_band, A_main=A_main, f_fit=f_fit,
                V_true=V_true, bias=bias, se_pct=se_pct,
                freqs=freqs, amp=amp, v_filt=v_filt, fit=fit)


def _worst_reliable(pdf):
    """Worst |bias| among reliable (core) points; fall back to overall worst."""
    if pdf.empty or pdf["bias_pct"].notna().sum() == 0:
        return None
    vmax = pdf["mean_raw"].max()
    reliable = pdf[(pdf["mean_raw"] >= CORE_FRACTION * vmax) & pdf["bias_pct"].notna()]
    pool = reliable if not reliable.empty else pdf[pdf["bias_pct"].notna()]
    return pool.loc[pool["bias_pct"].abs().idxmax()]


def analyze_case_all_points(label, case_def):
    """Run every grid point of a case. Returns (per_point_df, worst_row)."""
    files = sorted(glob.glob(os.path.join(_raw_data_dir(case_def["folder"]), "*.csv")))
    rows = []
    for path in files:
        try:
            t, v = _load_series_from_file(path)
            r = compute_point(t, v)
        except Exception as e:
            print(f"   [!] {os.path.basename(path)}: {e}")
            continue
        rows.append(dict(
            case=label, point=os.path.basename(path),
            f_peak=r["f_peak"], f_fit=r["f_fit"],
            A_main=r["A_main"], A_band=r["A_band"],
            conc=r["conc"], regime=r["regime"],
            mean_raw=r["mean_raw"], V_true=r["V_true"],
            bias_pct=r["bias"], se_pct=r["se_pct"]))
    pdf = pd.DataFrame(rows)
    return pdf, _worst_reliable(pdf)


# =============================================================================
# DISTRIBUTION HISTOGRAMS (per-point, overlaid by case)
# =============================================================================
def plot_distributions(df):
    """Histograms of f_osc, A, raw mean, V_true and bias across all points."""
    quantities = [("f_peak", r"$f_{peak}$ [Hz]"), ("A_main", "main tone $A$ [m/s]"),
                  ("A_band", r"drift RMS $A_{band}$ [m/s]"), ("mean_raw", "raw mean [m/s]"),
                  ("V_true", r"$V_{true}$ [m/s]"), ("bias_pct", "fit bias [%]"),
                  ("se_pct", "block-mean SE [%]")]
    cases = list(dict.fromkeys(df["case"]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(cases))))
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    axes = axes.ravel()
    for ax, (col, xlabel) in zip(axes, quantities):
        d_all = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if d_all.empty:
            ax.set_visible(False); continue
        lo, hi = np.nanpercentile(d_all, [1, 99])          # robust range
        if lo == hi:
            lo, hi = float(d_all.min()), float(d_all.max()) + 1e-9
        bins = np.linspace(lo, hi, 30)
        for c, color in zip(cases, colors):
            d = df.loc[df["case"] == c, col].replace([np.inf, -np.inf], np.nan).dropna()
            if not d.empty:
                ax.hist(d, bins=bins, histtype="step", lw=1.6, color=color, label=c)
        ax.set_xlabel(xlabel); ax.set_ylabel("point count"); ax.grid(alpha=0.3)
        ax.set_title(xlabel, fontsize=11, loc="left")
    axes[7].axis("off")                                    # 8th panel = legend
    h, l = axes[0].get_legend_handles_labels()
    axes[7].legend(h, l, loc="center", fontsize=9, title="case", frameon=True)
    fig.suptitle(f"Per-point distributions across all cases (N = {len(df)} points)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "motor_drift_distributions.png"), dpi=200)
        fig.savefig(os.path.join(FIG_DIR, "motor_drift_distributions.pdf"))
    return fig


# =============================================================================
# MAIN
# =============================================================================
def investigate_case(label, case_def):
    """All points of one case: tabulate, report the worst, plot the worst point."""
    pdf, worst = analyze_case_all_points(label, case_def)
    if pdf.empty:
        print(f"\n[!] No usable points for '{label}'")
        return pdf, None

    abs_bias = pdf["bias_pct"].abs()
    n_broad = int((pdf["regime"] == "broadband").sum())
    print(f"\n=== {label}   ({len(pdf)} points; {n_broad} broadband) ===")
    print(f"  median |bias| / mean |bias| : "
          f"{np.nanmedian(abs_bias):.4f} % / {np.nanmean(abs_bias):.4f} %")
    print(f"  max block-mean SE           : {np.nanmax(pdf['se_pct']):.4f} %")
    if worst is not None:
        print(f"  WORST (core) point          : {worst['point']}  "
              f"(mean {worst['mean_raw']:.2f} m/s, {worst['regime']})")
        print(f"    f_peak / A_main / A_band  : {worst['f_peak']:.3f} Hz / "
              f"{worst['A_main']:.4f} / {worst['A_band']:.4f} m/s")
        print(f"    raw mean -> V_true        : "
              f"{worst['mean_raw']:.4f} -> {worst['V_true']:.4f} m/s")
        print(f"    WORST bias | block SE     : {worst['bias_pct']:+.4f} % | "
              f"{worst['se_pct']:.4f} %")
        # re-load and plot the worst point in detail
        path = os.path.join(_raw_data_dir(case_def["folder"]), worst["point"])
        t, v = _load_series_from_file(path)
        r = compute_point(t, v)
        plot_case(f"{label} - worst pt {worst['point']} "
                  f"(bias {worst['bias_pct']:+.3f}%)",
                  t, v, r, file_tag=f"worst_{case_def['folder']}")
    return pdf, worst


def main():
    print("=" * 70)
    print(" WIND-TUNNEL MOTOR-DRIFT INVESTIGATION  (every point, every case)")
    print(f" Fs = {FS:.0f} Hz | signal = '{SIGNAL_COL}' | low-pass = {LOWPASS_CUTOFF:.0f} Hz")
    print("=" * 70)

    cases = discover_all_cases() if AUTO_DISCOVER_ALL else CASES
    print(f" Processing {len(cases)} case(s), all grid points each.")

    all_points, case_summ = [], []
    for label, case_def in cases.items():
        try:
            pdf, worst = investigate_case(label, case_def)
        except Exception as e:
            print(f"\n[!] Skipped '{label}': {e}")
            continue
        if pdf is None or pdf.empty:
            continue
        all_points.append(pdf)
        ab = pdf["bias_pct"].abs()
        case_summ.append(dict(
            case=label, n_points=len(pdf),
            n_broadband=int((pdf["regime"] == "broadband").sum()),
            worst_point=(worst["point"] if worst is not None else ""),
            worst_bias_pct=(worst["bias_pct"] if worst is not None else np.nan),
            worst_mean=(worst["mean_raw"] if worst is not None else np.nan),
            worst_f_peak=(worst["f_peak"] if worst is not None else np.nan),
            worst_A_band=(worst["A_band"] if worst is not None else np.nan),
            max_se_pct=float(np.nanmax(pdf["se_pct"])),
            median_abs_bias_pct=float(np.nanmedian(ab)),
            mean_abs_bias_pct=float(np.nanmean(ab))))

    if not all_points:
        print("\nNo data processed.")
        return
    big = pd.concat(all_points, ignore_index=True)

    # ---- worst-case summary table ---------------------------------------
    print("\n" + "=" * 92)
    print(" WORST-CASE BIAS & MEAN-UNCERTAINTY PER CASE  (core points)")
    print(" brd = #points flagged broadband | A_band = drift RMS | maxSE = block-mean SE")
    print("=" * 92)
    hdr = (f"{'case':14s}{'N':>4s}{'brd':>4s}{'worst pt':>14s}{'mean':>7s}"
           f"{'f_pk':>7s}{'A_band':>8s}{'WORST%':>9s}{'med|%|':>8s}{'maxSE%':>8s}")
    print(hdr); print("-" * len(hdr))
    for s in sorted(case_summ, key=lambda x: -abs(x["worst_bias_pct"])):
        print(f"{s['case'][:14]:14s}{s['n_points']:4d}{s['n_broadband']:4d}"
              f"{s['worst_point'][:14]:>14s}{s['worst_mean']:7.2f}"
              f"{s['worst_f_peak']:7.3f}{s['worst_A_band']:8.4f}"
              f"{s['worst_bias_pct']:+9.4f}{s['median_abs_bias_pct']:8.4f}"
              f"{s['max_se_pct']:8.4f}")

    # ---- save CSVs + distribution histograms ----------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    big.to_csv(os.path.join(FIG_DIR, "motor_drift_per_point.csv"), index=False)
    pd.DataFrame(case_summ).to_csv(
        os.path.join(FIG_DIR, "motor_drift_case_summary.csv"), index=False)
    plot_distributions(big)
    print(f"\nFigures + CSVs written to: {FIG_DIR}")

    if SHOW_FIGS:
        plt.show()


if __name__ == "__main__":
    main()
