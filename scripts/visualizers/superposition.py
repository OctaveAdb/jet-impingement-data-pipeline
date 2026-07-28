# =============================================================================
# SUPERPOSITION
# =============================================================================
# Purpose:
#   Performs cross-case comparison and superposition of aerodynamic results
#   across all experiment subfolders (all 8 or more conditions). The script
#   scans each case subfolder for processed frequency and velocity data,
#   extracts the centerline normalized velocity profile U/U_max, the dominant
#   shedding frequency profile along X/B, and global summary metrics (L_c, K,
#   Strouhal number, virtual jet origin X0/B), then plots all cases on shared
#   axes for direct visual comparison. Four outputs are produced: (1) a
#   superposed centerline velocity decay plot U/U_max vs X/B, annotated with
#   potential core lengths L_c per case; (2) the plane-jet linear decay
#   evaluation (U_max/U_c)^2 vs X/B for free-jet cases with fitted decay
#   rates K and intercepts C; (3) a superposed dominant frequency profile
#   along the centerline for all cases; and (4) a grid of individual
#   centerline PSD panels (one panel per case, with peaks marked and global
#   mean frequency annotated), exported as a single multi-panel figure. A
#   literature comparison table consolidating all key non-dimensional
#   parameters is also exported in CSV/XLSX format.
#
# Inputs:
#   - Raw streamwise velocity maps per case (normalised at runtime):
#       experiments/<case>/1D_Profiles_Results/CSV/Map_u_m_s.csv   (centreline-only)
#       experiments/<case>/2D_Profiles_Results/CSV/Map_u_m_s.csv   (transverse-map)
#   - Velocity summary metrics per case:
#       experiments/<case>/1D_Profiles_Results/CSV/Global_Velocity_Summary.csv
#   - Dominant frequency maps per case:
#       experiments/<case>/Frequency_Results/CSV/Map_Dominant_Freq_1st.csv
#       or 1D_Profiles_Results/CSV/Map_Dominant_Freq_1st.csv
#   - Per-point PSD data per case (for the mega-grid):
#       experiments/<case>/Processed_CSVs/FFT/<grid>_FFT.csv
#   - Global frequency summaries per case:
#       experiments/<case>/Frequency_Results/CSV/Global_Frequency_Summary.csv
#
# Outputs:
#   - Literature comparison table:
#       summary/CSV/Literature_Comparison_Table.csv / .xlsx
#   - Superposed velocity decay figure:
#       summary/PNG/Superposed_U_Normalized.png / .pdf
#   - Plane-jet decay rate figure:
#       summary/PNG/Superposed_Plane_Jet_Decay.png / .pdf
#   - Superposed dominant frequency figure:
#       summary/PNG/Superposed_1st_Dominant_Freq.png / .pdf
#   - Multi-panel centerline PSD grid:
#       summary/PNG/All_Centerline_PSDs_Grid.png / .pdf
#   - Merged superposition data tables:
#       summary/CSV/Superposed_U_Normalized.csv
#       summary/CSV/Superposed_1st_Dominant_Freq.csv
#
# Dependencies:
#   - mean_processor and frequency_processor (produce all input CSVs)
#   - Called by usage.py via the "Generate Parent Summary" button
#
# Usage:
#   - Standalone: python superposition.py  (opens folder dialog)
#   - Via hub:    called by usage.py as generate_superposition(parent_dir)
# =============================================================================

import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# --- case_labels import (optional; graceful fallback if not found) ---
try:
    import sys as _sys, os as _os
    _lp = _os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
    if _lp not in _sys.path:
        _sys.path.insert(0, _lp)
    from case_labels import get_label as _cl_get_label, set_main_folder as _cl_set_folder
    _CL_OK = True
except Exception:
    _CL_OK = False
    def _cl_get_label(x, **_): return x
    def _cl_set_folder(_): pass

# --- Probe POSITION 1-sigma (mm) for the independent-axis (X/B) error bars -----
# Velocity profiles get an explicit position error bar on the position axis; the
# value-axis uncertainty already folds position in via the local gradient, so the
# bar is mildly conservative (never understates) — see UNCERTAINTY_SPEC.md.
try:
    from uncertainty import U_POS_X_MM as _U_POS_X_MM, U_POS_Y_MM as _U_POS_Y_MM
except Exception:
    _U_POS_X_MM, _U_POS_Y_MM = 1.0, 0.5

# --- CONFIG LOADER ---
def _load_cfg(filename):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    _df = pd.read_excel(_p, index_col='parameter')
    return _df['value'] if 'value' in _df.columns else _df.iloc[:, 0]

try:
    _cfg_lit   = _load_cfg('config_literature_nu.csv')
    _cfg_cases = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cases.xlsx'))
    _cfg_style = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_plot_styles.xlsx'))
except Exception:
    _cfg_lit   = None
    _cfg_cases = None
    _cfg_style = None

# --- Professional Plotting Configuration ---
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", 
    "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 300,
})

def get_plot_style(name):
    """Returns (color_hex, marker, linestyle) for a case folder name from config_plot_styles.csv."""
    if _cfg_style is not None:
        for _, row in _cfg_style.iterrows():
            prefix = str(row.get('case_prefix', ''))
            suffix = str(row.get('pressure_suffix', ''))
            # Skip rows that are not case-prefix-based style entries
            if not prefix or prefix in ('default', 'suffix_Pla', 'suffix_Free', 'config_Free'):
                continue
            if name.startswith(prefix):
                if suffix == 'other' or not suffix or suffix in name:
                    ls_raw = str(row.get('linestyle', 'solid'))
                    ls = '--' if ls_raw == 'dashed' else '-'
                    return str(row['color_hex']), str(row['marker']), ls
    # Fallback: original hardcoded logic
    color = '#888888'
    ls = '-'
    marker = 'o'
    if name.startswith('Cyl'):
        color = '#005b96' if '10Pa' in name else '#6497b1'
    elif name.startswith('Free'):
        color = '#b30000' if '10Pa' in name else '#e34a33'
    if name.endswith('Pla'):
        ls = '--'; marker = 's'
    elif name.endswith('Free'):
        ls = '-'; marker = 'o'
    return color, marker, ls


def _norm_ref(df):
    """U0 normalisation reference for velocity profiles. Prefers U0_ref_ms
    (= V_noz for free AND cylinder cases); falls back to U_max_ms for legacy
    summaries that predate the U0 column."""
    for _c in ('U0_ref_ms', 'U_max_ms'):
        if _c in df.columns:
            try:
                _v = float(df[_c].iloc[0])
                if _v == _v and _v > 0:
                    return _v
            except Exception:
                pass
    return float('nan')


def _u0_unc(folder):
    """1-sigma uncertainty of the U0 normalisation reference (u_U0_ref_ms) for a
    case, from Global_Velocity_Summary. NaN if unavailable."""
    gsp = os.path.join(folder, "1D_Profiles_Results", "CSV", "Global_Velocity_Summary.csv")
    if not os.path.exists(gsp):
        return float('nan')
    try:
        d = pd.read_csv(gsp)
        if 'u_U0_ref_ms' in d.columns:
            return float(d['u_U0_ref_ms'].iloc[0])
    except Exception:
        pass
    return float('nan')


def _read_B_mm(folder):
    """Slot height B (mm) for normalising the position error bars into X/B units.
    From Global_Velocity_Summary if present, else the established 30 mm default."""
    gsp = os.path.join(folder, "1D_Profiles_Results", "CSV", "Global_Velocity_Summary.csv")
    try:
        if os.path.exists(gsp):
            d = pd.read_csv(gsp)
            if 'B_mm' in d.columns:
                _b = float(d['B_mm'].iloc[0])
                if _b > 0:
                    return _b
    except Exception:
        pass
    return 30.0


def _pos_unc_xb(folder):
    """Streamwise probe-position 1-sigma expressed in X/B units (u_pos_x / B)."""
    _b = _read_B_mm(folder)
    return (_U_POS_X_MM / _b) if _b else None


def _pos_unc_yb(folder):
    """Transverse probe-position 1-sigma expressed in Y/B units (u_pos_y / B)."""
    _b = _read_B_mm(folder)
    return (_U_POS_Y_MM / _b) if _b else None


def _decay_y_err(folder, vnorm_series):
    """1-sigma of the decay ordinate y = (U_max/U_c)^2 = v^-2, where v is the
    normalised centerline velocity. Propagates the per-point velocity 1-sigma from
    the companion map (Map_u_u_m_s), incl. the U0-reference term, through y=v^-2:
    u_y = 2 v^-3 u_v. Returns an array aligned to `vnorm_series`, or None if the
    companion uncertainty map is unavailable."""
    try:
        _up = _resolve_centerline_csv(folder, "Map_u_u_m_s")
        if not _up:
            return None
        _du = pd.read_csv(_up, index_col=0)
        _du.index = pd.to_numeric(_du.index, errors='coerce')
        _du = _du[_du.index.notnull()]
        _du.columns = pd.to_numeric(_du.columns, errors='coerce')
        _ucy = min(_du.index, key=lambda y: abs(y - 0.0))
        _uabs = _du.loc[_ucy].reindex(vnorm_series.index).values.astype(float)
        _dsu = pd.read_csv(os.path.join(folder, "1D_Profiles_Results", "CSV",
                                        "Global_Velocity_Summary.csv"))
        _v = vnorm_series.values.astype(float)
        _uv = _normalized_band(_v, _uabs, _norm_ref(_dsu), _u0_unc(folder))
        with np.errstate(divide='ignore', invalid='ignore'):
            _uy = 2.0 * np.abs(_v) ** (-3.0) * _uv
        _uy[~np.isfinite(_uy)] = np.nan
        return _uy
    except Exception:
        return None


def _normalized_band(n_vals, u_abs, U0, u_U0):
    """1-sigma band for a normalised velocity n = U/U0, combining the per-point
    velocity uncertainty `u_abs` (ABSOLUTE m/s) and the REFERENCE uncertainty
    `u_U0`:  u_n = sqrt[(u_abs/U0)^2 + (n·u_U0/U0)^2].  Including the U0 term is
    what can bring U/U0 ≈ 1 inside the band even when the central value exceeds 1."""
    n_vals = np.asarray(n_vals, float)
    u_abs = np.asarray(u_abs, float)
    if not (U0 and U0 == U0 and U0 > 0):
        return np.zeros_like(n_vals)
    t1 = u_abs / U0
    t2 = n_vals * ((u_U0 / U0) if (u_U0 == u_U0) else 0.0)
    return np.sqrt(t1 ** 2 + t2 ** 2)


def get_label(name, context_cases=None):
    """Return display label for a case folder name, using case_labels if available."""
    return _cl_get_label(name, context_cases=context_cases) if _CL_OK else name

# =============================================================================
# HELPER: Generic superposed centerline plot
# =============================================================================
def _plot_superposed_centerline(ax, cases, subfolders, csv_stem, ylabel,
                                normalise_by_umax=False):
    """
    Iterate over cases and plot each case's centerline profile on *ax*.

    Parameters
    ----------
    ax               : matplotlib Axes to draw on.
    cases            : list of case folder-name strings (already sorted).
    subfolders       : dict {case_name: folder_path}.
    csv_stem         : CSV filename stem, e.g. "Map_I_u_%_Centerline".
    ylabel           : Y-axis label string (LaTeX okay).
    normalise_by_umax: If True, divide values by U_max_ms from
                       Global_Velocity_Summary.csv.
    """
    import sys as _sys
    _utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'utils')
    if _utils_dir not in _sys.path:
        _sys.path.insert(0, _utils_dir)
    try:
        from case_labels import get_label, set_main_folder as _smf
        _smf(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        _labels_ok = True
    except ImportError:
        _labels_ok = False

    plotted = False
    for name in cases:
        folder = subfolders.get(name)
        if folder is None:
            continue
        csv_path = _resolve_centerline_csv(folder, csv_stem)
        if csv_path is None:
            print(f"  -> Skipping {name}: {csv_stem}.csv not found")
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_numeric(df.index, errors='coerce')
            df = df[df.index.notnull()]
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            center_y = min(df.index, key=lambda y: abs(y - 0.0))
            series = df.loc[center_y].dropna()

            if normalise_by_umax:
                # Derive U_max from the U_Normalized_Centerline max (which is 1.0
                # by construction) — to get V/U_max we need to divide raw v [m/s]
                # by the maximum streamwise velocity U_max [m/s].
                # U_max is stored in Global_Velocity_Summary.csv.
                sum_path = os.path.join(
                    folder, "1D_Profiles_Results", "CSV",
                    "Global_Velocity_Summary.csv")
                u_max = np.nan
                if os.path.exists(sum_path):
                    try:
                        df_su = pd.read_csv(sum_path)
                        if 'U_max_ms' in df_su.columns:
                            u_max = _norm_ref(df_su)
                    except Exception:
                        pass
                if np.isnan(u_max) or u_max == 0:
                    print(f"  -> Skipping {name}: U_max not found for normalisation")
                    continue
                series = series / u_max

            c, m, ls = get_plot_style(name)
            label = get_label(name) if _labels_ok else name

            # Per-point error bars (replaces the shaded band): x = streamwise
            # position 1-sigma (u_pos_x/B); y = value 1-sigma from the companion
            # uncertainty map (Map_u_<var>.csv). For normalised velocities the y
            # error also includes the U0-reference term.
            _uv = None
            try:
                _base = csv_stem[:-len("_Centerline")] if csv_stem.endswith("_Centerline") else csv_stem
                _ustem = _base.replace("Map_", "Map_u_", 1)
                _upath = _resolve_centerline_csv(folder, _ustem)
                if _upath is not None:
                    _du = pd.read_csv(_upath, index_col=0)
                    _du.index = pd.to_numeric(_du.index, errors='coerce')
                    _du = _du[_du.index.notnull()]
                    _du.columns = pd.to_numeric(_du.columns, errors='coerce')
                    _ucy = min(_du.index, key=lambda y: abs(y - 0.0))
                    _uabs = _du.loc[_ucy].reindex(series.index).values.astype(float)
                    if normalise_by_umax and u_max and u_max > 0:
                        _uv = _normalized_band(series.values, _uabs, u_max, _u0_unc(folder))
                    else:
                        _uv = _uabs   # intensities: no U0 normalisation
            except Exception:
                _uv = None
            ax.errorbar(series.index, series.values, xerr=_pos_unc_xb(folder), yerr=_uv,
                        color=c, marker=m, linestyle=ls, label=label, lw=2, ms=5,
                        capsize=2, elinewidth=0.7, ecolor=c)
            plotted = True
        except Exception as e:
            print(f"  -> Error processing {csv_stem} for {name}: {e}")

    return plotted


# =============================================================================
# HELPER: Near-plate (maximum X/B column) superposed profile — Y/B on x-axis
# =============================================================================
def _pick_station_col(df, station, min_rows=3):
    """Pick the X/B station column from a (Y/B index × X/B columns) map.

    Only columns with MORE THAN `min_rows` valid (non-NaN) Y/B rows qualify — the
    transverse Y-traverse is only done at a few X stations (the rest are
    centerline-only, 1 row), and picking one of those would collapse the profile
    to a single point. Among the qualifying columns, in increasing X order:
    'nozzle' = first, 'midfield' = SECOND, 'plate' = last. Falls back to all
    columns if none clear the threshold."""
    cols = sorted(df.columns)
    good = [c for c in cols if int(df[c].notna().sum()) > min_rows]
    if not good:
        good = cols
    if station == 'nozzle':
        return good[0]
    if station == 'midfield':
        return good[1] if len(good) >= 2 else good[0]
    return good[-1]


def _plot_superposed_nearplate(ax, cases, subfolders, map_subdir, csv_stem,
                               ylabel, normalise_by_umax=False, station='plate'):
    """
    For each case, read <case>/<map_subdir>/CSV/<csv_stem>.csv (Y/B index × X/B columns),
    pick a boundary X/B column and plot the resulting Y/B profile.
    station='plate'  -> MAXIMUM X/B column (closest to the impingement plate);
    station='nozzle' -> MINIMUM X/B column (first slice, closest to the nozzle).

    Parameters
    ----------
    ax               : matplotlib Axes to draw on.
    cases            : list of case folder-name strings (Pla cases only).
    subfolders       : dict {case_name: folder_path}.
    map_subdir       : sub-directory inside the case folder, e.g. "2D_Profiles_Results".
    csv_stem         : CSV file stem, e.g. "Map_u_m_s".
    ylabel           : Y-axis label string (LaTeX okay).
    normalise_by_umax: If True, divide values by U_max_ms from Global_Velocity_Summary.csv.

    Axis convention: x-axis = Y/B (cross-jet position), y-axis = the variable.
    """
    import sys as _sys
    _utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'utils')
    if _utils_dir not in _sys.path:
        _sys.path.insert(0, _utils_dir)
    try:
        from case_labels import get_label as _gl, set_main_folder as _smf
        _smf(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        _lbl_ok = True
    except ImportError:
        _lbl_ok = False

    plotted = False
    for name in cases:
        folder = subfolders.get(name)
        if folder is None:
            continue
        csv_path = os.path.join(folder, map_subdir, "CSV", csv_stem + ".csv")
        if not os.path.exists(csv_path):
            print(f"  -> Skipping {name}: {csv_stem}.csv not found in {map_subdir}/CSV/")
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_numeric(df.index, errors='coerce')
            df = df[df.index.notnull()]
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            df = df[df.columns[df.columns.notnull()]]
            if df.shape[0] < 2:
                print(f"  -> Skipping {name}: fewer than 2 Y/B rows in {csv_stem}")
                continue

            # Pick the X/B station column: min (nozzle), middle (mid-field), or
            # max (near plate).
            xb_col = _pick_station_col(df, station)
            series = df[xb_col].dropna()
            if series.empty:
                print(f"  -> Skipping {name}: column X/B={xb_col:.3f} is empty")
                continue

            if normalise_by_umax:
                sum_path = os.path.join(
                    folder, "1D_Profiles_Results", "CSV",
                    "Global_Velocity_Summary.csv")
                u_max = np.nan
                if os.path.exists(sum_path):
                    try:
                        df_su = pd.read_csv(sum_path)
                        if 'U_max_ms' in df_su.columns:
                            u_max = _norm_ref(df_su)
                    except Exception:
                        pass
                if np.isnan(u_max) or u_max == 0:
                    print(f"  -> Skipping {name}: U_max not found for normalisation")
                    continue
                series = series / u_max

            c, m, ls = get_plot_style(name)
            lbl = _gl(name) if _lbl_ok else name
            # Per-point error bars (replaces the shaded band): x = transverse
            # position 1-sigma (u_pos_y/B, this profile runs along Y/B); y = value
            # 1-sigma from the companion map (incl. U0-reference term if normalised).
            _bd = None
            try:
                _ustem = csv_stem.replace("Map_", "Map_u_", 1)
                _up = os.path.join(folder, map_subdir, "CSV", _ustem + ".csv")
                if os.path.exists(_up):
                    _du = pd.read_csv(_up, index_col=0)
                    _du.index = pd.to_numeric(_du.index, errors='coerce')
                    _du = _du[_du.index.notnull()]
                    _du.columns = pd.to_numeric(_du.columns, errors='coerce')
                    _uabs = _du[xb_col].reindex(series.index).values.astype(float)
                    if normalise_by_umax:
                        _bd = _normalized_band(series.values, _uabs, u_max, _u0_unc(folder))
                    else:
                        _bd = _uabs
            except Exception:
                _bd = None
            ax.errorbar(series.index, series.values, xerr=_pos_unc_yb(folder), yerr=_bd,
                        color=c, marker=m, linestyle=ls, label=lbl, lw=2, ms=5,
                        capsize=2, elinewidth=0.7, ecolor=c)
            plotted = True
        except Exception as e:
            print(f"  -> Error processing {csv_stem} for {name}: {e}")

    return plotted


# =============================================================================
# HELPER: Detect 2D dominant-frequency map filename for a case folder
# =============================================================================
_FREQ_MAP_CANDIDATES = [
    "Map_Dominant_Freq_1st",
    "Map_Mean_Resonance_Hz",
    "Map_Dominant_Freq",
]

def _detect_freq_map_stem(folder, map_subdir="Frequency_Results"):
    """Return the CSV stem for the dominant-frequency 2D map, or None if not found."""
    for stem in _FREQ_MAP_CANDIDATES:
        p = os.path.join(folder, map_subdir, "CSV", stem + ".csv")
        if os.path.exists(p):
            return stem
    return None


def _resolve_centerline_csv(folder, csv_stem):
    """Resolve the path to a case's Map CSV for centerline extraction.

    mean_processor now writes exactly one CSV per variable:
    - Centreline-only cases (Free traverse) → 1D_Profiles_Results/CSV/<stem>.csv
    - Transverse-map cases (Pla traverse)   → 2D_Profiles_Results/CSV/<stem>.csv

    The '_Centerline' suffix in the old csv_stem is stripped before lookup so
    that callers using legacy stem names (e.g. 'Map_I_u_%_Centerline') still
    resolve correctly.  The caller is responsible for extracting the Y/B≈0 row
    from the returned (potentially multi-row) map.

    Search order:
      1. 1D_Profiles_Results/CSV/<base_stem>.csv   (centreline-only case)
      2. 2D_Profiles_Results/CSV/<base_stem>.csv   (transverse-map case)
    Returns None if neither file exists.
    """
    # Strip legacy _Centerline suffix so both old and new stem names work
    base_stem = csv_stem[: -len("_Centerline")] if csv_stem.endswith("_Centerline") else csv_stem

    p1 = os.path.join(folder, "1D_Profiles_Results", "CSV", base_stem + ".csv")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(folder, "2D_Profiles_Results", "CSV", base_stem + ".csv")
    if os.path.exists(p2):
        return p2
    return None


def generate_superposition(parent_dir):
    _cl_set_folder(parent_dir)
    print(f"Scanning directory: {parent_dir}")

    # Support new structure where experiments live in experiments/ subfolder
    experiments_sub = os.path.join(parent_dir, "experiments")
    scan_dir = experiments_sub if os.path.isdir(experiments_sub) else parent_dir

    try:
        _exclude = {'summary', 'scripts', 'thermal', '__pycache__'}
        subfolders = [f.path for f in os.scandir(scan_dir) if f.is_dir() and os.path.basename(f.path).lower() not in _exclude]
    except Exception as e:
        raise ValueError(f"Could not read directory structure: {e}")
        
    if not subfolders:
        raise ValueError("No subfolders found. Please select a valid Parent folder.")
    
    freq_data = {}
    u_norm_data = {}
    comparison_table_rows = []

    # Build case-name → folder-path mapping (used by _plot_superposed_centerline)
    _subfolder_map = {os.path.basename(f): f for f in subfolders}

    for folder in subfolders:
        name = os.path.basename(folder)
        row_data = {
            'Case': name, 'Freq_Hz': np.nan, 'St': np.nan,
            'L_c_XB': np.nan, 'K_decay': np.nan, 'C_int': np.nan,
            'X0_B': np.nan, 'U_max_ms': np.nan, 'C_profile': np.nan
        }
        
        # 1. Frequency Superposition Data Setup
        map_paths = [
            os.path.join(folder, "Frequency_Results", "CSV", "Map_Dominant_Freq_1st.csv"),
            os.path.join(folder, "1D_Profiles_Results", "CSV", "Map_Dominant_Freq_1st.csv")
        ]
        
        target_csv = None
        for path in map_paths:
            if os.path.exists(path):
                target_csv = path
                break
                
        if target_csv:
            try:
                df = pd.read_csv(target_csv, index_col=0)
                df.index = pd.to_numeric(df.index, errors='coerce')
                df = df[df.index.notnull()]
                df.columns = pd.to_numeric(df.columns, errors='coerce')
                
                center_y = min(df.index, key=lambda y: abs(y - 0.0))
                centerline_profile = df.loc[center_y]
                freq_data[name] = centerline_profile
                
                summary_path = os.path.join(folder, "Frequency_Results", "CSV", "Global_Frequency_Summary.csv")
                if os.path.exists(summary_path):
                    df_sum = pd.read_csv(summary_path)
                    row_data['Freq_Hz'] = df_sum['Global_Mean_Freq_Hz'].iloc[0]
                    row_data['St'] = df_sum['Global_Mean_St'].iloc[0]
            except Exception as e:
                print(f"  -> Error processing Frequency for {name}: {e}")

        # 2. Normalized U Velocity Data Setup
        # mean_processor no longer writes Map_U_Normalized_Centerline.csv.
        # Read the raw Map_u_m_s.csv (from whichever result folder it lives in),
        # extract the Y/B≈0 centerline row, and normalise by U_max_ms from the
        # velocity summary so the series is dimensionless (U/U_max).
        u_raw_path = _resolve_centerline_csv(folder, "Map_u_m_s")
        if u_raw_path and os.path.exists(u_raw_path):
            try:
                df_u = pd.read_csv(u_raw_path, index_col=0)
                df_u.columns = pd.to_numeric(df_u.columns, errors='coerce')
                df_u.index   = pd.to_numeric(df_u.index,   errors='coerce')
                df_u = df_u[df_u.index.notnull()]
                center_y = min(df_u.index, key=lambda y: abs(y - 0.0))
                u_series = df_u.loc[center_y].dropna()
                # Normalise: prefer U_max_ms from summary; fall back to series max
                _u_max_norm = np.nan
                sum_path_u = os.path.join(folder, "1D_Profiles_Results", "CSV",
                                          "Global_Velocity_Summary.csv")
                if os.path.exists(sum_path_u):
                    try:
                        _dsu = pd.read_csv(sum_path_u)
                        if 'U_max_ms' in _dsu.columns:
                            _u_max_norm = _norm_ref(_dsu)
                    except Exception:
                        pass
                if np.isnan(_u_max_norm) or _u_max_norm == 0:
                    _u_max_norm = float(u_series.max()) if not u_series.empty else np.nan
                if not np.isnan(_u_max_norm) and _u_max_norm > 0:
                    u_norm_data[name] = u_series / _u_max_norm
            except Exception as e:
                print(f"  -> Error processing U profile for {name}: {e}")
                
        # 3. Velocity Summary Meta-Data Setup
        sum_u_path = os.path.join(folder, "1D_Profiles_Results", "CSV", "Global_Velocity_Summary.csv")
        if os.path.exists(sum_u_path):
            try:
                df_su = pd.read_csv(sum_u_path)
                l_c = df_su['L_c_XB'].iloc[0] if 'L_c_XB' in df_su.columns else np.nan
                k_val = df_su['K_decay'].iloc[0] if 'K_decay' in df_su.columns else np.nan
                c_val = df_su['C_intercept'].iloc[0] if 'C_intercept' in df_su.columns else np.nan
                
                row_data['L_c_XB'] = l_c
                row_data['u_L_c_XB'] = df_su['u_L_c_XB'].iloc[0] if 'u_L_c_XB' in df_su.columns else np.nan
                row_data['K_decay'] = k_val
                row_data['C_int'] = c_val
                row_data['u_K_decay'] = df_su['u_K_decay'].iloc[0] if 'u_K_decay' in df_su.columns else np.nan
                row_data['u_C_int']   = df_su['u_C_intercept'].iloc[0] if 'u_C_intercept' in df_su.columns else np.nan
                row_data['U_max_ms'] = df_su['U_max_ms'].iloc[0] if 'U_max_ms' in df_su.columns else np.nan
                row_data['C_profile'] = df_su['C_profile'].iloc[0] if 'C_profile' in df_su.columns else np.nan
                
                # Calculate Virtual Origin X0/B
                if not pd.isna(k_val) and not pd.isna(c_val) and k_val != 0:
                    row_data['X0_B'] = -c_val / k_val

            except Exception as e:
                print(f"  -> Error processing Vel Summary for {name}: {e}")

        comparison_table_rows.append(row_data)

    if not freq_data and not u_norm_data:
        raise ValueError("No valid map data found in subdirectories.")

    summary_dir = os.path.join(parent_dir, "summary")
    for d in ["CSV", "XLSX", "PNG", "PDF"]: os.makedirs(os.path.join(summary_dir, d), exist_ok=True)

    # --- EXPORT LITERATURE COMPARISON TABLE ---
    if comparison_table_rows:
        df_table = pd.DataFrame(comparison_table_rows).sort_values("Case")
        # Ensure clean column ordering
        cols = ['Case', 'U_max_ms', 'C_profile', 'L_c_XB', 'u_L_c_XB', 'K_decay', 'u_K_decay',
                'C_int', 'u_C_int', 'X0_B', 'Freq_Hz', 'St']
        df_table = df_table[[c for c in cols if c in df_table.columns]]
        
        df_table.to_csv(os.path.join(summary_dir, "CSV", "Literature_Comparison_Table.csv"), index=False)
        df_table.to_excel(os.path.join(summary_dir, "XLSX", "Literature_Comparison_Table.xlsx"), index=False)
        print(" -> Literature Comparison Table generated.")

    # ====================================================
    # PLOT 1: Normalized Velocity (Showing L_c)
    # ====================================================
    if u_norm_data:
        print(" -> Generating U/Umax Superposition...")
        df_u_merged = pd.DataFrame(u_norm_data).sort_index()
        df_u_merged.index.name = "X/B"
        df_u_merged.to_csv(os.path.join(summary_dir, "CSV", "Superposed_U_Normalized.csv"))
        df_u_merged.to_excel(os.path.join(summary_dir, "XLSX", "Superposed_U_Normalized.xlsx"))

        fig_u, ax_u = plt.subplots(figsize=(12, 6))
        _all_u_cases = list(sorted(df_u_merged.columns))

        for col in _all_u_cases:
            valid_data = df_u_merged[col].dropna()
            c, m, ls = get_plot_style(col)

            _lbl = _cl_get_label(col, context_cases=_all_u_cases)
            l_c_val = next((r['L_c_XB'] for r in comparison_table_rows if r['Case'] == col), np.nan)

            if not pd.isna(l_c_val):
                label = f"{_lbl}, $L_c = {l_c_val:.2f}B$"
                # Vertical L_c annotation line removed (Task 5B) — value kept in legend
            else:
                label = _lbl

            # Per-point error bars: x = position (u_pos_x/B); y = centerline
            # velocity 1-sigma from the companion map (incl. U0-reference term).
            _fld_u = _subfolder_map.get(col)
            _bd_u = None
            try:
                _up = _resolve_centerline_csv(_fld_u, "Map_u_u_m_s") if _fld_u else None
                if _up:
                    _du = pd.read_csv(_up, index_col=0)
                    _du.index = pd.to_numeric(_du.index, errors='coerce')
                    _du = _du[_du.index.notnull()]
                    _du.columns = pd.to_numeric(_du.columns, errors='coerce')
                    _ucy = min(_du.index, key=lambda y: abs(y - 0.0))
                    _uabs = _du.loc[_ucy].reindex(valid_data.index).values.astype(float)
                    _dsu = pd.read_csv(os.path.join(_fld_u, "1D_Profiles_Results", "CSV",
                                                    "Global_Velocity_Summary.csv"))
                    _bd_u = _normalized_band(valid_data.values, _uabs, _norm_ref(_dsu), _u0_unc(_fld_u))
            except Exception:
                _bd_u = None
            ax_u.errorbar(valid_data.index, valid_data.values,
                          xerr=(_pos_unc_xb(_fld_u) if _fld_u else None), yerr=_bd_u,
                          color=c, marker=m, linestyle=ls, label=label, lw=2, ms=5,
                          capsize=2, elinewidth=0.7, ecolor=c)

        ax_u.set_title(r'Centerline $U/U_{0}$ ($Y/B \approx 0$)')
        ax_u.set_xlabel(r'$X/B$ [-]')
        ax_u.set_ylabel(r'$U/U_{0}$ [-]')
        ax_u.grid(True, ls='--', alpha=0.4)
        ax_u.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

        fig_u.savefig(os.path.join(summary_dir, "PNG", "Superposed_U_Normalized.png"), dpi=300, bbox_inches='tight')
        ax_u.set_title('')
        fig_u.savefig(os.path.join(summary_dir, "PDF", "Superposed_U_Normalized.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig_u)

    # ====================================================
    # PLOT 2: Plane Jet Decay Linear Form (U_max/U_c)^2
    # ====================================================
    if u_norm_data:
        print(" -> Generating Plane Jet Decay Rate Plot...")
        fig_d, ax_d = plt.subplots(figsize=(12, 6))
        _all_decay_cases = [col for col in sorted(df_u_merged.columns) if col.endswith('Free')]

        for col in _all_decay_cases:
            # STRICTLY filter to only plot cases that END WITH 'Free'
            if col.endswith('Free'):
                c, m, ls = get_plot_style(col)
                valid_data = df_u_merged[col].dropna()

                # Restrict plotting to strictly AFTER the maximum velocity peak
                x_peak = valid_data.idxmax()
                valid_data = valid_data[valid_data.index >= x_peak]

                if len(valid_data) > 0:
                    decay_y = (1.0 / valid_data.values)**2

                    k_val = next((r['K_decay'] for r in comparison_table_rows if r['Case'] == col), np.nan)
                    c_val = next((r['C_int'] for r in comparison_table_rows if r['Case'] == col), np.nan)
                    uk = next((r.get('u_K_decay', np.nan) for r in comparison_table_rows if r['Case'] == col), np.nan)
                    uc = next((r.get('u_C_int', np.nan) for r in comparison_table_rows if r['Case'] == col), np.nan)

                    _lbl = _cl_get_label(col, context_cases=_all_decay_cases)
                    if not pd.isna(k_val) and not pd.isna(c_val):
                        _kstr = f"$K={k_val:.3f}" + (rf"\pm{uk:.3f}$" if not pd.isna(uk) else "$")
                        label = f"{_lbl}, {_kstr}, $C={c_val:.3f}$"
                    else:
                        label = _lbl

                    # Per-point error bars (replaces the regression confidence
                    # band): y propagates the centerline-velocity 1-sigma through
                    # y=v^-2 (u_y = 2 v^-3 u_v); x error = position (u_pos_x/B).
                    _fld_d = _subfolder_map.get(col)
                    _ydec = _decay_y_err(_fld_d, valid_data) if _fld_d else None
                    ax_d.errorbar(valid_data.index, decay_y,
                                  xerr=(_pos_unc_xb(_fld_d) if _fld_d else None), yerr=_ydec,
                                  color=c, marker=m, linestyle=ls, label=label, lw=2, ms=5,
                                  capsize=2, elinewidth=0.7, ecolor=c)
            
        ax_d.set_title(r'Plane Jet Decay: $(U_{max}/U_c)^2$ vs $X/B$')
        ax_d.set_xlabel(r'$X/B$ [-]')
        ax_d.set_ylabel(r'$(U_{max}/U_c)^2$ [-]')
        ax_d.grid(True, ls='--', alpha=0.4)
        ax_d.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        
        # Prevent extreme Y limits if deceleration hits near zero
        ax_d.set_ylim(bottom=0.5)

        fig_d.savefig(os.path.join(summary_dir, "PNG", "Superposed_Plane_Jet_Decay.png"), dpi=300, bbox_inches='tight')
        ax_d.set_title('')
        fig_d.savefig(os.path.join(summary_dir, "PDF", "Superposed_Plane_Jet_Decay.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig_d)

    # ====================================================
    # PLOTS 2b–2g: Superposed Centerline profiles (Iu, Iv, Iw, Iuvw, V, W)
    # ====================================================
    _sorted_cases = sorted(_subfolder_map.keys())

    _centerline_specs = [
        # (csv_stem,                        ylabel,                              out_stem,                       normalise)
        ("Map_I_u_%_Centerline",        r'$I_u$ [%]',                        "Superposed_I_u_Centerline",       False),
        ("Map_I_v_%_Centerline",        r'$I_v$ [%]',                        "Superposed_I_v_Centerline",       False),
        ("Map_I_w_%_Centerline",        r'$I_w$ [%]',                        "Superposed_I_w_Centerline",       False),
        ("Map_I_uvw_%_Centerline",      r'$I_{uvw}$ [%]',                    "Superposed_I_uvw_Centerline",     False),
        ("Map_v_m_s_Centerline",        r'$V/U_{0}$ [-]',                  "Superposed_V_Centerline",         True),
        ("Map_w_m_s_Centerline",        r'$W/U_{0}$ [-]',                  "Superposed_W_Centerline",         True),
        ("Map_Vel_Mag_m_s_Centerline",  r'$V_{mag}/U_{0}$ [-]',            "Superposed_Vmag_Centerline",      True),
    ]

    _title_map = {
        "Superposed_I_u_Centerline":    r'Centerline $I_u$ ($Y/B \approx 0$)',
        "Superposed_I_v_Centerline":    r'Centerline $I_v$ ($Y/B \approx 0$)',
        "Superposed_I_w_Centerline":    r'Centerline $I_w$ ($Y/B \approx 0$)',
        "Superposed_I_uvw_Centerline":  r'Centerline $I_{uvw}$ ($Y/B \approx 0$)',
        "Superposed_V_Centerline":      r'Centerline $V/U_{0}$ ($Y/B \approx 0$)',
        "Superposed_W_Centerline":      r'Centerline $W/U_{0}$ ($Y/B \approx 0$)',
        "Superposed_Vmag_Centerline":   r'Centerline $V_{mag}/U_{0}$ ($Y/B \approx 0$)',
    }

    for _csv_stem, _ylabel, _out_stem, _norm in _centerline_specs:
        print(f" -> Generating {_out_stem}...")
        _fig_cl, _ax_cl = plt.subplots(figsize=(12, 6))
        _plotted = _plot_superposed_centerline(
            _ax_cl, _sorted_cases, _subfolder_map,
            _csv_stem, _ylabel, normalise_by_umax=_norm
        )
        if not _plotted:
            plt.close(_fig_cl)
            print(f"    (no data found — skipping {_out_stem})")
            continue

        _ax_cl.set_title(_title_map.get(_out_stem, _out_stem))
        _ax_cl.set_xlabel(r'$X/B$ [-]')
        _ax_cl.set_ylabel(_ylabel)
        _ax_cl.grid(True, ls='--', alpha=0.4)
        _ax_cl.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

        _fig_cl.savefig(
            os.path.join(summary_dir, "PNG", _out_stem + ".png"),
            dpi=300, bbox_inches='tight')
        _ax_cl.set_title('')
        _fig_cl.savefig(
            os.path.join(summary_dir, "PDF", _out_stem + ".pdf"),
            format='pdf', bbox_inches='tight')
        plt.close(_fig_cl)
        print(f"    -> {_out_stem} saved.")

        # --- Export merged CSV (cases as columns, X/B as index) ---
        _cl_series = {}
        for _cn in _sorted_cases:
            _cf = _subfolder_map.get(_cn)
            if _cf is None:
                continue
            _cp = _resolve_centerline_csv(_cf, _csv_stem)
            if _cp is None:
                continue
            try:
                _df_tmp = pd.read_csv(_cp, index_col=0)
                _df_tmp.index   = pd.to_numeric(_df_tmp.index,   errors='coerce')
                _df_tmp.columns = pd.to_numeric(_df_tmp.columns, errors='coerce')
                _df_tmp = _df_tmp[_df_tmp.index.notnull()]
                _cy = min(_df_tmp.index, key=lambda _y: abs(_y - 0.0))
                _series = _df_tmp.loc[_cy].dropna()
                if _norm:
                    _gsp = os.path.join(_cf, "1D_Profiles_Results", "CSV",
                                        "Global_Velocity_Summary.csv")
                    _umax = np.nan
                    if os.path.exists(_gsp):
                        try:
                            _dsu = pd.read_csv(_gsp)
                            if 'U_max_ms' in _dsu.columns:
                                _umax = _norm_ref(_dsu)
                        except Exception:
                            pass
                    if np.isnan(_umax) or _umax == 0:
                        continue
                    _series = _series / _umax
                _cl_series[_cn] = _series
            except Exception:
                pass
        if _cl_series:
            _df_cl_merged = pd.DataFrame(_cl_series).sort_index()
            _df_cl_merged.index.name = "X/B"
            _df_cl_merged.to_csv(
                os.path.join(summary_dir, "CSV", _out_stem + ".csv"))
            print(f"    -> {_out_stem}.csv saved.")

    # ====================================================
    # PLOT 3: Frequency
    # ====================================================
    if freq_data:
        print(" -> Generating Frequency Superposition...")
        df_merged = pd.DataFrame(freq_data).sort_index()
        df_merged.index.name = "X/B"
        df_merged.to_csv(os.path.join(summary_dir, "CSV", "Superposed_1st_Dominant_Freq.csv"))
        df_merged.to_excel(os.path.join(summary_dir, "XLSX", "Superposed_1st_Dominant_Freq.xlsx"))

        fig, ax = plt.subplots(figsize=(12, 6))
        _all_freq_cases = list(sorted(df_merged.columns))

        for col in _all_freq_cases:
            valid_data = df_merged[col].dropna()
            c, m, ls = get_plot_style(col)

            _lbl = _cl_get_label(col, context_cases=_all_freq_cases)
            f_val = next((r['Freq_Hz'] for r in comparison_table_rows if r['Case'] == col), np.nan)
            st_val = next((r['St'] for r in comparison_table_rows if r['Case'] == col), np.nan)
            if not pd.isna(f_val):
                legend_label = f"{_lbl} ({f_val:.0f} Hz, St: {st_val:.2f})"
            else:
                legend_label = _lbl

            ax.plot(valid_data.index, valid_data.values, color=c, marker=m, linestyle=ls, label=legend_label, lw=2, ms=5)
            
            if not pd.isna(f_val) and f_val > 0:
                ax.hlines(f_val, xmin=valid_data.index.min(), xmax=valid_data.index.max(), 
                          color=c, linestyle=':', lw=1.5, alpha=0.6)

        ax.set_title(r'Centerline $f_{dom}$ ($Y/B \approx 0$)')
        ax.set_xlabel(r'$X/B$ [-]')
        ax.set_ylabel(r'$f_{dom}$ [Hz]')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

        fig.savefig(os.path.join(summary_dir, "PNG", "Superposed_1st_Dominant_Freq.png"), dpi=300, bbox_inches='tight')
        ax.set_title('')
        fig.savefig(os.path.join(summary_dir, "PDF", "Superposed_1st_Dominant_Freq.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)
        
    # ====================================================
    # 4. MEGA-GRID OF CENTERLINE PSDs (improved)
    # ====================================================
    # Also collect representative centerline PSDs for Plot 5 (superposed)
    _superposed_psd_data = {}   # case_name -> (freq_array, psd_array)

    if freq_data:
        print(" -> Generating Grid of Centerline PSDs...")
        valid_folders = [f for f in subfolders if os.path.basename(f) in freq_data]
        num_folders = len(valid_folders)
        _psd_xlim = float(_cfg_lit['psd_xlim_hz']) if _cfg_lit is not None else 1000

        # Dynamic grid: ceil(sqrt(n)) columns, enough rows to fit all panels
        cols = math.ceil(math.sqrt(num_folders))
        rows = math.ceil(num_folders / cols)
        fig_psd, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
        axes_flat = axes.flatten()

        for idx, folder in enumerate(valid_folders):
            folder_name = os.path.basename(folder)
            ax_psd = axes_flat[idx]
            row_idx = idx // cols
            col_idx = idx % cols

            # Peak threshold from config_cases.csv; fallback to 75 if not found
            peak_threshold = 75
            if _cfg_cases is not None:
                for _, _row in _cfg_cases.iterrows():
                    _pat = str(_row.get('case_pattern', ''))
                    if _pat and _pat in folder_name:
                        peak_threshold = float(_row.get('peak_threshold_hz', 75))
                        break
            fft_dir = os.path.join(folder, "Processed_CSVs", "FFT")
            raw_dir = os.path.join(folder, "Processed_CSVs", "Raw_Data")
            if not os.path.exists(raw_dir): raw_dir = os.path.join(folder, "Processed_CSVs")

            raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
            if not raw_files or not os.path.exists(fft_dir):
                ax_psd.set_title(f"{_cl_get_label(folder_name, context_cases=_all_freq_cases)}\n(No FFT Data)")
                ax_psd.set_visible(False)
                continue

            closest_y = None
            for f in raw_files:
                try:
                    y = pd.read_csv(f, nrows=1)['Y/B'].iloc[0]
                    if closest_y is None or abs(y) < abs(closest_y): closest_y = y
                except: pass

            center_files = []
            for f in raw_files:
                try:
                    df_t = pd.read_csv(f, nrows=1)
                    if abs(df_t['Y/B'].iloc[0] - closest_y) < 1e-4:
                        center_files.append((df_t['X/B'].iloc[0], os.path.splitext(os.path.basename(f))[0]))
                except: pass

            center_files.sort()
            colors_grid = plt.cm.viridis(np.linspace(0, 1, len(center_files)))

            # Accumulate PSD arrays for the averaged representative curve
            _psd_accumulator = []
            _freq_ref = None

            for c_idx, (x, b_name) in enumerate(center_files):
                f_p = os.path.join(fft_dir, f"{b_name}_FFT.csv")
                if os.path.exists(f_p):
                    df_f = pd.read_csv(f_p)
                    freq_arr = df_f['Frequency (Hz)'].values
                    psd_arr  = df_f['PSD (m^2/s^2/Hz)'].values
                    ax_psd.semilogy(freq_arr, psd_arr, color=colors_grid[c_idx], lw=1.2, alpha=0.8)
                    v_idx = np.where(freq_arr > peak_threshold)[0]
                    if len(v_idx) > 0:
                        p_i = v_idx[np.argmax(psd_arr[v_idx])]
                        ax_psd.plot(freq_arr[p_i], psd_arr[p_i], 'o',
                                    color=colors_grid[c_idx], ms=4, markeredgecolor='black')
                    # Collect for average
                    if _freq_ref is None:
                        _freq_ref = freq_arr
                        _psd_accumulator.append(psd_arr)
                    else:
                        _psd_accumulator.append(np.interp(_freq_ref, freq_arr, psd_arr))

            # Store averaged representative PSD for Plot 5
            if _freq_ref is not None and _psd_accumulator:
                _superposed_psd_data[folder_name] = (
                    _freq_ref,
                    np.mean(np.vstack(_psd_accumulator), axis=0)
                )

            f_val = next((r['Freq_Hz'] for r in comparison_table_rows if r['Case'] == folder_name), np.nan)
            st_val = next((r['St'] for r in comparison_table_rows if r['Case'] == folder_name), np.nan)

            if not pd.isna(f_val) and f_val > 0:
                ax_psd.axvline(f_val, color='red', linestyle='--', lw=1.2, alpha=0.7)
                # Top-right corner text annotation
                _st_txt = f"St = {st_val:.3f}" if not pd.isna(st_val) else ""
                _ann_txt = f"$f_d$ = {f_val:.1f} Hz\n{_st_txt}".strip()
                ax_psd.text(0.97, 0.97, _ann_txt,
                            transform=ax_psd.transAxes,
                            ha='right', va='top', fontsize=9,
                            color='red',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            yh_subtitle = fr'$Y/B \approx {closest_y:.2f}$'
            ax_psd.set_title(fr'{_cl_get_label(folder_name, context_cases=_all_freq_cases)} ({yh_subtitle})')

            # X-label only on bottom row panels
            if row_idx == rows - 1 or idx + cols >= num_folders:
                ax_psd.set_xlabel(r'$f$ [Hz]')

            # Y-label only on leftmost column panels
            if col_idx == 0:
                ax_psd.set_ylabel(r'PSD [m$^2$/s$^2$/Hz]')

            ax_psd.set_xlim(0, _psd_xlim)
            ax_psd.grid(True, alpha=0.3)

        # Hide unused panels
        for idx in range(num_folders, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig_psd.suptitle(r"Centerline PSD — All Cases ($Y/B \approx 0$)", fontsize=14, y=1.01)
        plt.tight_layout()

        fig_psd.savefig(os.path.join(summary_dir, "PNG", "All_Centerline_PSDs_Grid.png"), dpi=300, bbox_inches='tight')

        # PDF: remove suptitle only; per-panel titles (case names) stay
        fig_psd.suptitle("")
        fig_psd.savefig(os.path.join(summary_dir, "PDF", "All_Centerline_PSDs_Grid.pdf"), format='pdf', bbox_inches='tight')
        fig_psd.suptitle(r"Centerline PSD — All Cases ($Y/B \approx 0$)", fontsize=14, y=1.01)

        plt.close(fig_psd)

    # ====================================================
    # (Removed) Superposed Centerline PSD single-axes plot
    # The "PSD (FFT) - Superposed_Centerline" figure/data was removed per
    # request; the per-case PSD mega-grid below is retained.
    # ====================================================

    # ====================================================
    # 6. ALL-CASES CENTERLINE PSD GRID (standalone, no freq map needed)
    # ====================================================
    png_path = generate_all_psds_grid(parent_dir)
    if png_path:
        print(f" -> All Centerline PSDs Grid saved: {png_path}")

    # ====================================================
    # 7. NEAR-PLATE (last X/B slice) SUPERPOSED PROFILES — Pla cases only
    # ====================================================
    # The general summary also emits the near-plate profiles (velocity, TI and
    # frequency/Strouhal at the maximum X/B column) for the plate-on cases.
    # The category generators below produce BOTH the all-8-case centerline and
    # the Pla-only near-plate figures/data, so the summary is a strict superset.
    generate_velocity_plots(parent_dir)
    generate_turbulence_plots(parent_dir)

    # ====================================================
    # 8. FREQUENCY SUBCATEGORY: centerline + near-plate f_dom and St
    # ====================================================
    generate_frequency_plots(parent_dir)

    # ====================================================
    # 9. Nu <-> UNSTEADINESS COUPLING TABLE  (Task 4)
    # ====================================================
    generate_coupling_table(parent_dir)

    print(f"--- SUCCESS: Summary generated in {summary_dir} ---")


# =============================================================================
# STANDALONE: ALL-CASES CENTERLINE PSD GRID
# =============================================================================
def generate_all_psds_grid(main_folder: str) -> str:
    """
    Build a single grid figure with one subplot per experiment case.
    Each subplot superimposes all centerline (Y/B≈0) PSD curves for that case,
    stacked with small vertical offsets for readability.

    Works independently of frequency-map CSVs: uses Raw_Data CSVs to identify
    the centerline Y-row and FFT CSVs for the spectral data.

    Parameters
    ----------
    main_folder : str
        Project root directory (contains experiments/ subfolder).

    Returns
    -------
    str
        Path to the saved PNG, or empty string if no data was found.
    """
    import sys as _sys

    # --- Import case_labels utility ---
    _utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
    if _utils_dir not in _sys.path:
        _sys.path.insert(0, _utils_dir)
    try:
        from case_labels import get_label, set_main_folder as _set_mf
        _set_mf(main_folder)
        _labels_available = True
    except ImportError:
        _labels_available = False

    # --- Discover experiment case folders ---
    experiments_sub = os.path.join(main_folder, "experiments")
    scan_dir = experiments_sub if os.path.isdir(experiments_sub) else main_folder

    _exclude = {'summary', 'scripts', 'thermal', '__pycache__'}
    try:
        subfolders = sorted(
            [f.path for f in os.scandir(scan_dir)
             if f.is_dir() and os.path.basename(f.path).lower() not in _exclude]
        )
    except Exception as e:
        print(f"  generate_all_psds_grid: could not scan {scan_dir}: {e}")
        return ""

    # --- Collect per-case centerline PSD data ---
    case_data = []   # list of dicts: {name, label, center_files [(xb, base_name)], peak_threshold}

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        raw_dir = os.path.join(folder, "Processed_CSVs", "Raw_Data")
        fft_dir = os.path.join(folder, "Processed_CSVs", "FFT")

        if not os.path.isdir(raw_dir) or not os.path.isdir(fft_dir):
            continue

        raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
        if not raw_files:
            continue

        # Find Y/B closest to 0 across all raw files
        closest_y = None
        for f in raw_files:
            try:
                df_t = pd.read_csv(f, nrows=1)
                y = float(df_t['Y/B'].iloc[0])
                if closest_y is None or abs(y) < abs(closest_y):
                    closest_y = y
            except Exception:
                pass

        if closest_y is None:
            continue

        # Collect (X/B, base_name) pairs for centerline row that have matching FFT files
        center_files = []
        for f in raw_files:
            try:
                df_t = pd.read_csv(f, nrows=1)
                y = float(df_t['Y/B'].iloc[0])
                if abs(y - closest_y) < 1e-4:
                    base = os.path.splitext(os.path.basename(f))[0]
                    fft_p = os.path.join(fft_dir, base + "_FFT.csv")
                    if os.path.exists(fft_p):
                        center_files.append((float(df_t['X/B'].iloc[0]), base))
            except Exception:
                pass

        if not center_files:
            continue

        center_files.sort()

        # Peak threshold from config_cases.csv
        peak_threshold = 75.0
        if _cfg_cases is not None:
            for _, _row in _cfg_cases.iterrows():
                _pat = str(_row.get('case_pattern', ''))
                if _pat and _pat in folder_name:
                    peak_threshold = float(_row.get('peak_threshold_hz', 75.0))
                    break

        # Case display label
        if _labels_available:
            try:
                lbl = get_label(folder_name)
            except Exception:
                lbl = folder_name
        else:
            lbl = folder_name

        case_data.append({
            'name': folder_name,
            'label': lbl,
            'folder': folder,
            'fft_dir': fft_dir,
            'center_files': center_files,
            'closest_y': closest_y,
            'peak_threshold': peak_threshold,
        })

    if not case_data:
        print("  generate_all_psds_grid: no valid case data found — skipping.")
        return ""

    # --- Grid layout ---
    n_cases = len(case_data)
    n_cols = math.ceil(math.sqrt(n_cases))
    n_rows = math.ceil(n_cases / n_cols)

    fig_w = max(4.5 * n_cols, 9)
    fig_h = max(3.5 * n_rows, 5)
    fig_grid, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(fig_w, fig_h),
                                  squeeze=False)
    axes_flat = axes.flatten()

    # PSD x-axis limit from config
    psd_xlim = float(_cfg_lit['psd_xlim_hz']) if _cfg_lit is not None else 1000

    for idx, cd in enumerate(case_data):
        ax = axes_flat[idx]
        row_idx = idx // n_cols
        col_idx = idx % n_cols

        center_files = cd['center_files']
        n_stations = len(center_files)
        colors = plt.cm.viridis(np.linspace(0, 0.85, max(n_stations, 1)))
        fft_dir = cd['fft_dir']
        peak_threshold = cd['peak_threshold']

        offset_step = 0.0
        cumulative_offset = 0.0

        psd_arrays = []  # collect all for offset calculation

        # First pass: load all PSDs and compute offset
        loaded = []
        for c_idx, (xb, base) in enumerate(center_files):
            fft_p = os.path.join(fft_dir, base + "_FFT.csv")
            try:
                df_f = pd.read_csv(fft_p)
                freq_col = next((c for c in df_f.columns
                                 if 'freq' in c.lower() or 'hz' in c.lower()), None)
                psd_col  = next((c for c in df_f.columns
                                 if 'psd' in c.lower() or 'm^2' in c.lower()
                                    or 'm2' in c.lower()), None)
                if freq_col is None or psd_col is None:
                    continue
                freq_arr = df_f[freq_col].values.astype(float)
                psd_arr  = df_f[psd_col].values.astype(float)
                loaded.append((xb, base, freq_arr, psd_arr))
                psd_arrays.append(psd_arr)
            except Exception:
                pass

        if not loaded:
            ax.set_title(f"{cd['label']}\n(no FFT data)")
            ax.set_visible(False)
            continue

        # Offset step = 10 % of the maximum PSD across all stations
        global_max = max(np.nanmax(p) for p in psd_arrays) if psd_arrays else 1.0
        offset_step = global_max * 0.10

        for c_idx, (xb, base, freq_arr, psd_arr) in enumerate(loaded):
            psd_shifted = psd_arr + cumulative_offset
            ax.semilogy(freq_arr, psd_shifted,
                        color=colors[c_idx], lw=1.2, alpha=0.85)

            # Annotate X/B value at the right end of the line
            if len(freq_arr) > 0:
                mask = freq_arr <= psd_xlim
                if mask.any():
                    last_f = freq_arr[mask][-1]
                    last_p = psd_shifted[mask][-1]
                    ax.annotate(f"{xb:.1f}",
                                xy=(last_f, last_p),
                                xytext=(3, 0), textcoords='offset points',
                                fontsize=6, color=colors[c_idx], va='center')

            # Mark dominant peak above threshold
            v_idx = np.where(freq_arr > peak_threshold)[0]
            if len(v_idx) > 0:
                p_i = v_idx[np.argmax(psd_arr[v_idx])]
                ax.plot(freq_arr[p_i], psd_shifted[p_i], 'o',
                        color=colors[c_idx], ms=3.5,
                        markeredgecolor='black', markeredgewidth=0.5)

            cumulative_offset += offset_step

        # Frequency summary annotation (if available)
        freq_summary_path = os.path.join(
            cd['folder'], "Frequency_Results", "CSV", "Global_Frequency_Summary.csv")
        if os.path.exists(freq_summary_path):
            try:
                df_sum = pd.read_csv(freq_summary_path)
                f_val  = df_sum['Global_Mean_Freq_Hz'].iloc[0]
                st_val = df_sum['Global_Mean_St'].iloc[0] if 'Global_Mean_St' in df_sum.columns else np.nan
                if not pd.isna(f_val) and f_val > 0:
                    ax.axvline(f_val, color='red', linestyle='--', lw=1.0, alpha=0.6)
                    _st_txt = f"St = {st_val:.3f}" if not pd.isna(st_val) else ""
                    _ann = f"$f_d$ = {f_val:.1f} Hz\n{_st_txt}".strip()
                    ax.text(0.97, 0.97, _ann,
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=7, color='red',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            except Exception:
                pass

        # Axes labels (sparse: bottom row / left column only)
        ax.set_title(cd['label'], fontsize=9)
        if row_idx == n_rows - 1 or idx + n_cols >= n_cases:
            ax.set_xlabel(r'$f$ [Hz]', fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(r'PSD [m$^2$/s$^2$/Hz] (staggered)', fontsize=8)

        ax.set_xlim(0, psd_xlim)
        ax.tick_params(labelsize=7)
        ax.grid(True, ls='--', alpha=0.3)

    # Hide unused panels
    for idx in range(n_cases, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig_grid.suptitle(r"Centerline PSD — All Cases ($Y/B \approx 0$)",
                      fontsize=13, y=1.01)
    plt.tight_layout()

    # --- Save ---
    summary_dir = os.path.join(main_folder, "summary")
    for d in ["PNG", "PDF"]:
        os.makedirs(os.path.join(summary_dir, d), exist_ok=True)

    out_png = os.path.join(summary_dir, "PNG", "All_Centerline_PSDs_Grid.png")
    out_pdf = os.path.join(summary_dir, "PDF", "All_Centerline_PSDs_Grid.pdf")

    fig_grid.savefig(out_png, dpi=300, bbox_inches='tight')
    fig_grid.suptitle("")
    fig_grid.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig_grid)

    return out_png


# =============================================================================
# CATEGORY-SPECIFIC ENTRY POINTS  (called from usage.py category buttons)
# =============================================================================

def _get_subfolder_map(parent_dir):
    """Return (subfolders_list, subfolder_map_dict, summary_dir) for parent_dir."""
    experiments_sub = os.path.join(parent_dir, "experiments")
    scan_dir = experiments_sub if os.path.isdir(experiments_sub) else parent_dir
    _exclude = {'summary', 'scripts', 'thermal', '__pycache__'}
    subfolders = sorted(
        [f.path for f in os.scandir(scan_dir)
         if f.is_dir() and os.path.basename(f.path).lower() not in _exclude]
    )
    sfmap = {os.path.basename(f): f for f in subfolders}
    summary_dir = os.path.join(parent_dir, "summary")
    for d in ["CSV", "XLSX", "PNG", "PDF"]:
        os.makedirs(os.path.join(summary_dir, d), exist_ok=True)
    return subfolders, sfmap, summary_dir


def _emit_nearstation_profiles(specs, station, sfmap, pla_cases, summary_dir):
    """Generate near-station superposed Y/B profiles (PNG+PDF+merged CSV) for a
    list of 5-tuple specs (csv_stem, map_subdir, ylabel, out_stem, normalise).

    station='plate'  -> MAX X/B (near plate); 'nozzle' -> MIN X/B (near nozzle);
    'midfield' -> middle X/B (mid-field). Mirrors the inline near-plate blocks but
    with a selectable station column, so the extra stations are produced without
    altering the near-plate ones."""
    _words = {'nozzle': ('nozzle', 'min'), 'midfield': ('mid-field', 'mid'),
              'plate': ('plate', 'max')}
    word, note = _words.get(station, ('plate', 'max'))
    for csv_stem, subdir, ylabel, out_stem, norm in specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = _plot_superposed_nearplate(
            ax, pla_cases, sfmap, subdir, csv_stem, ylabel,
            normalise_by_umax=norm, station=station)
        if not plotted:
            plt.close(fig)
            print(f"    (no near-{word} data — skipping {out_stem})")
            continue
        _title_pre = "Mid-field" if station == 'midfield' else f"Near-{word}"
        ax.set_title(rf"{_title_pre} profile of {ylabel} ($X/B$ = {note})")
        ax.set_xlabel(r'$Y/B$ [-]')
        ax.set_ylabel(ylabel)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        fig.savefig(os.path.join(summary_dir, "PNG", out_stem + ".png"),
                    dpi=300, bbox_inches='tight')
        ax.set_title('')
        fig.savefig(os.path.join(summary_dir, "PDF", out_stem + ".pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)
        # Merged CSV (Y/B index × case-label columns)
        series = {}
        for name in pla_cases:
            folder = sfmap.get(name)
            if folder is None:
                continue
            csvp = os.path.join(folder, subdir, "CSV", csv_stem + ".csv")
            if not os.path.exists(csvp):
                continue
            try:
                df = pd.read_csv(csvp, index_col=0)
                df.index = pd.to_numeric(df.index, errors='coerce')
                df.columns = pd.to_numeric(df.columns, errors='coerce')
                df = df[df.index.notnull()]
                if df.shape[0] < 2:
                    continue
                col = _pick_station_col(df, station)
                s = df[col].dropna()
                if norm:
                    gsp = os.path.join(folder, "1D_Profiles_Results", "CSV",
                                       "Global_Velocity_Summary.csv")
                    umax = np.nan
                    if os.path.exists(gsp):
                        try:
                            dsu = pd.read_csv(gsp)
                            if 'U_max_ms' in dsu.columns:
                                umax = _norm_ref(dsu)
                        except Exception:
                            pass
                    if np.isnan(umax) or umax == 0:
                        continue
                    s = s / umax
                series[get_label(name)] = s
            except Exception:
                pass
        if series:
            merged = pd.DataFrame(series).sort_index()
            merged.index.name = "Y/B"
            merged.to_csv(os.path.join(summary_dir, "CSV", out_stem + ".csv"))
        print(f" -> {out_stem} saved.")


def generate_velocity_plots(parent_dir):
    """Regenerate U/Umax, V/Umax, W/Umax centerline superposed plots."""
    subfolders, sfmap, summary_dir = _get_subfolder_map(parent_dir)
    sorted_cases = sorted(sfmap.keys())

    # ── U/Umax (re-uses existing logic inline) ──────────────────────────────
    u_norm_data = {}
    comparison_table_rows = []
    for folder in subfolders:
        name = os.path.basename(folder)
        row_data = {'Case': name, 'L_c_XB': np.nan, 'K_decay': np.nan, 'C_int': np.nan}
        # Read raw Map_u_m_s.csv (1D or 2D folder) and normalise at runtime.
        u_raw_path = _resolve_centerline_csv(folder, "Map_u_m_s")
        if u_raw_path and os.path.exists(u_raw_path):
            try:
                df_u = pd.read_csv(u_raw_path, index_col=0)
                df_u.columns = pd.to_numeric(df_u.columns, errors='coerce')
                df_u.index   = pd.to_numeric(df_u.index,   errors='coerce')
                df_u = df_u[df_u.index.notnull()]
                cy = min(df_u.index, key=lambda y: abs(y - 0.0))
                u_series = df_u.loc[cy].dropna()
                _umax_n = float(u_series.max()) if not u_series.empty else np.nan
                # Prefer U_max_ms from velocity summary when available
                _gvs = os.path.join(folder, "1D_Profiles_Results", "CSV",
                                    "Global_Velocity_Summary.csv")
                if os.path.exists(_gvs):
                    try:
                        _dsu = pd.read_csv(_gvs)
                        if 'U_max_ms' in _dsu.columns:
                            _umax_n = _norm_ref(_dsu)
                    except Exception:
                        pass
                if not np.isnan(_umax_n) and _umax_n > 0:
                    u_norm_data[name] = u_series / _umax_n
            except Exception as e:
                print(f"  -> Error U profile {name}: {e}")
        sum_u_path = os.path.join(folder, "1D_Profiles_Results", "CSV",
                                  "Global_Velocity_Summary.csv")
        if os.path.exists(sum_u_path):
            try:
                df_su = pd.read_csv(sum_u_path)
                row_data['L_c_XB']  = df_su['L_c_XB'].iloc[0]  if 'L_c_XB'      in df_su.columns else np.nan
                row_data['K_decay'] = df_su['K_decay'].iloc[0]  if 'K_decay'     in df_su.columns else np.nan
                row_data['C_int']   = df_su['C_intercept'].iloc[0] if 'C_intercept' in df_su.columns else np.nan
            except Exception:
                pass
        comparison_table_rows.append(row_data)

    if u_norm_data:
        _cl_set_folder(parent_dir)
        df_u_merged = pd.DataFrame(u_norm_data).sort_index()
        fig_u, ax_u = plt.subplots(figsize=(12, 6))
        _all_vel_cases = list(sorted(df_u_merged.columns))
        for col in _all_vel_cases:
            valid_data = df_u_merged[col].dropna()
            c, m, ls = get_plot_style(col)
            _lbl = _cl_get_label(col, context_cases=_all_vel_cases)
            l_c_val = next((r['L_c_XB'] for r in comparison_table_rows
                            if r['Case'] == col), np.nan)
            if not pd.isna(l_c_val):
                label = f"{_lbl}, $L_c = {l_c_val:.2f}B$"
            else:
                label = _lbl
            # Per-point error bars (replaces the shaded band): x = streamwise
            # position 1-sigma (u_pos_x/B); y = value 1-sigma incl. the U0-reference
            # term (so U/U0 ~ 1 may fall inside even where the central value > 1).
            _fld = sfmap.get(col)
            _bd = None
            try:
                _up = _resolve_centerline_csv(_fld, "Map_u_u_m_s") if _fld else None
                if _up:
                    _du = pd.read_csv(_up, index_col=0)
                    _du.index = pd.to_numeric(_du.index, errors='coerce')
                    _du = _du[_du.index.notnull()]
                    _du.columns = pd.to_numeric(_du.columns, errors='coerce')
                    _ucy = min(_du.index, key=lambda y: abs(y - 0.0))
                    _uabs = _du.loc[_ucy].reindex(valid_data.index).values.astype(float)
                    _dsu = pd.read_csv(os.path.join(_fld, "1D_Profiles_Results", "CSV",
                                                    "Global_Velocity_Summary.csv"))
                    _bd = _normalized_band(valid_data.values, _uabs, _norm_ref(_dsu), _u0_unc(_fld))
            except Exception:
                _bd = None
            ax_u.errorbar(valid_data.index, valid_data.values,
                          xerr=(_pos_unc_xb(_fld) if _fld else None), yerr=_bd,
                          color=c, marker=m, linestyle=ls, label=label, lw=2, ms=5,
                          capsize=2, elinewidth=0.7, ecolor=c)
        ax_u.set_title(r'Centerline $U/U_{0}$ ($Y/B \approx 0$)')
        ax_u.set_xlabel(r'$X/B$ [-]')
        ax_u.set_ylabel(r'$U/U_{0}$ [-]')
        ax_u.grid(True, ls='--', alpha=0.4)
        ax_u.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        fig_u.savefig(os.path.join(summary_dir, "PNG", "Superposed_U_Normalized.png"),
                      dpi=300, bbox_inches='tight')
        ax_u.set_title('')
        fig_u.savefig(os.path.join(summary_dir, "PDF", "Superposed_U_Normalized.pdf"),
                      format='pdf', bbox_inches='tight')
        plt.close(fig_u)
        print(" -> Superposed_U_Normalized saved.")

    # ── V/Umax, W/Umax, and Vmag/Umax centerline ────────────────────────────
    for _csv_stem, _ylabel, _out_stem, _title in [
        ("Map_v_m_s_Centerline",       r'$V/U_{0}$ [-]',       "Superposed_V_Centerline",
         r'Centerline $V/U_{0}$ ($Y/B \approx 0$)'),
        ("Map_w_m_s_Centerline",       r'$W/U_{0}$ [-]',       "Superposed_W_Centerline",
         r'Centerline $W/U_{0}$ ($Y/B \approx 0$)'),
        ("Map_Vel_Mag_m_s_Centerline", r'$V_{mag}/U_{0}$ [-]', "Superposed_Vmag_Centerline",
         r'Centerline $V_{mag}/U_{0}$ ($Y/B \approx 0$)'),
    ]:
        fig_cl, ax_cl = plt.subplots(figsize=(12, 6))
        plotted = _plot_superposed_centerline(
            ax_cl, sorted_cases, sfmap, _csv_stem, _ylabel, normalise_by_umax=True)
        if not plotted:
            plt.close(fig_cl)
            continue
        ax_cl.set_title(_title)
        ax_cl.set_xlabel(r'$X/B$ [-]')
        ax_cl.set_ylabel(_ylabel)
        ax_cl.grid(True, ls='--', alpha=0.4)
        ax_cl.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        fig_cl.savefig(os.path.join(summary_dir, "PNG", _out_stem + ".png"),
                       dpi=300, bbox_inches='tight')
        ax_cl.set_title('')
        fig_cl.savefig(os.path.join(summary_dir, "PDF", _out_stem + ".pdf"),
                       format='pdf', bbox_inches='tight')
        plt.close(fig_cl)
        print(f" -> {_out_stem} saved.")

    # ── Near-plate velocity profiles (Pla cases, max X/B column) ────────────
    _pla_cases = [n for n in sorted_cases if n.endswith('Pla')]
    _nearplate_vel_specs = [
        # (csv_stem,          map_subdir,             ylabel,                    out_stem,                         normalise)
        ("Map_u_m_s",       "2D_Profiles_Results",  r'$U/U_{0}$ [-]',        "Superposed_NearPlate_U_over_Umax",    True),
        ("Map_v_m_s",       "2D_Profiles_Results",  r'$V/U_{0}$ [-]',        "Superposed_NearPlate_V_over_Umax",    True),
        ("Map_w_m_s",       "2D_Profiles_Results",  r'$W/U_{0}$ [-]',        "Superposed_NearPlate_W_over_Umax",    True),
        ("Map_Vel_Mag_m_s", "2D_Profiles_Results",  r'$V_{mag}/U_{0}$ [-]',  "Superposed_NearPlate_Vmag_over_Umax", True),
    ]
    for _np_stem, _np_subdir, _np_ylabel, _np_out, _np_norm in _nearplate_vel_specs:
        _np_title = rf"Near-plate profile of {_np_ylabel} ($X/B$ = max)"
        _np_fig, _np_ax = plt.subplots(figsize=(10, 6))
        _np_plotted = _plot_superposed_nearplate(
            _np_ax, _pla_cases, sfmap, _np_subdir, _np_stem,
            _np_ylabel, normalise_by_umax=_np_norm)
        if not _np_plotted:
            plt.close(_np_fig)
            print(f"    (no near-plate data — skipping {_np_out})")
            continue
        _np_ax.set_title(_np_title)
        _np_ax.set_xlabel(r'$Y/B$ [-]')
        _np_ax.set_ylabel(_np_ylabel)
        _np_ax.grid(True, ls='--', alpha=0.4)
        _np_ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        _np_fig.savefig(os.path.join(summary_dir, "PNG", _np_out + ".png"),
                        dpi=300, bbox_inches='tight')
        _np_ax.set_title('')
        _np_fig.savefig(os.path.join(summary_dir, "PDF", _np_out + ".pdf"),
                        format='pdf', bbox_inches='tight')
        plt.close(_np_fig)
        # Export merged CSV (Y/B index × case-label columns)
        _np_series = {}
        for _np_name in _pla_cases:
            _np_folder = sfmap.get(_np_name)
            if _np_folder is None:
                continue
            _np_csv = os.path.join(_np_folder, _np_subdir, "CSV", _np_stem + ".csv")
            if not os.path.exists(_np_csv):
                continue
            try:
                _np_df = pd.read_csv(_np_csv, index_col=0)
                _np_df.index   = pd.to_numeric(_np_df.index,   errors='coerce')
                _np_df.columns = pd.to_numeric(_np_df.columns, errors='coerce')
                _np_df = _np_df[_np_df.index.notnull()]
                if _np_df.shape[0] < 2:
                    continue
                _np_maxcol = _np_df.columns.max()
                _np_s = _np_df[_np_maxcol].dropna()
                if _np_norm:
                    _np_gsp = os.path.join(_np_folder, "1D_Profiles_Results", "CSV",
                                           "Global_Velocity_Summary.csv")
                    _np_umax = np.nan
                    if os.path.exists(_np_gsp):
                        try:
                            _np_dsu = pd.read_csv(_np_gsp)
                            if 'U_max_ms' in _np_dsu.columns:
                                _np_umax = _norm_ref(_np_dsu)
                        except Exception:
                            pass
                    if np.isnan(_np_umax) or _np_umax == 0:
                        continue
                    _np_s = _np_s / _np_umax
                _np_series[get_label(_np_name)] = _np_s
            except Exception:
                pass
        if _np_series:
            _np_df_merged = pd.DataFrame(_np_series).sort_index()
            _np_df_merged.index.name = "Y/B"
            _np_df_merged.to_csv(os.path.join(summary_dir, "CSV", _np_out + ".csv"))
        print(f" -> {_np_out} saved.")

    # ── Near-nozzle velocity profiles (Pla cases, FIRST/min X/B column) ──────
    _nearnozzle_vel_specs = [
        (_s, _sub, _yl, _out.replace("NearPlate", "NearNozzle"), _nm)
        for (_s, _sub, _yl, _out, _nm) in _nearplate_vel_specs
    ]
    _emit_nearstation_profiles(_nearnozzle_vel_specs, 'nozzle', sfmap, _pla_cases, summary_dir)

    # ── Mid-field velocity profiles (Pla cases, MIDDLE X/B column) ──────────
    _midfield_vel_specs = [
        (_s, _sub, _yl, _out.replace("NearPlate", "MidField"), _nm)
        for (_s, _sub, _yl, _out, _nm) in _nearplate_vel_specs
    ]
    _emit_nearstation_profiles(_midfield_vel_specs, 'midfield', sfmap, _pla_cases, summary_dir)


def generate_decay_plot(parent_dir):
    """Regenerate only the Plane Jet Decay figure."""
    subfolders, sfmap, summary_dir = _get_subfolder_map(parent_dir)
    u_norm_data = {}
    comparison_table_rows = []
    for folder in subfolders:
        name = os.path.basename(folder)
        row_data = {'Case': name, 'K_decay': np.nan, 'C_int': np.nan}
        u_raw_path = _resolve_centerline_csv(folder, "Map_u_m_s")
        if u_raw_path and os.path.exists(u_raw_path):
            try:
                df_u = pd.read_csv(u_raw_path, index_col=0)
                df_u.columns = pd.to_numeric(df_u.columns, errors='coerce')
                df_u.index   = pd.to_numeric(df_u.index,   errors='coerce')
                df_u = df_u[df_u.index.notnull()]
                cy = min(df_u.index, key=lambda y: abs(y - 0.0))
                u_series = df_u.loc[cy].dropna()
                _umax_d = float(u_series.max()) if not u_series.empty else np.nan
                # Prefer U_max_ms from velocity summary
                _gvs_d = os.path.join(folder, "1D_Profiles_Results", "CSV",
                                      "Global_Velocity_Summary.csv")
                if os.path.exists(_gvs_d):
                    try:
                        _dsu_d = pd.read_csv(_gvs_d)
                        if 'U_max_ms' in _dsu_d.columns:
                            _umax_d = _norm_ref(_dsu_d)
                    except Exception:
                        pass
                if not np.isnan(_umax_d) and _umax_d > 0:
                    u_norm_data[name] = u_series / _umax_d
            except Exception as e:
                print(f"  -> Error U profile {name}: {e}")
        sum_path = os.path.join(folder, "1D_Profiles_Results", "CSV",
                                "Global_Velocity_Summary.csv")
        if os.path.exists(sum_path):
            try:
                df_su = pd.read_csv(sum_path)
                row_data['K_decay'] = df_su['K_decay'].iloc[0]     if 'K_decay'      in df_su.columns else np.nan
                row_data['C_int']   = df_su['C_intercept'].iloc[0] if 'C_intercept'  in df_su.columns else np.nan
                row_data['u_K_decay'] = df_su['u_K_decay'].iloc[0]    if 'u_K_decay'    in df_su.columns else np.nan
                row_data['u_C_int']   = df_su['u_C_intercept'].iloc[0] if 'u_C_intercept' in df_su.columns else np.nan
            except Exception:
                pass
        comparison_table_rows.append(row_data)

    if not u_norm_data:
        print(" -> No U data found for decay plot.")
        return

    _cl_set_folder(parent_dir)
    df_u_merged = pd.DataFrame(u_norm_data).sort_index()
    _all_decay_free = [col for col in sorted(df_u_merged.columns) if col.endswith('Free')]
    fig_d, ax_d = plt.subplots(figsize=(12, 6))
    for col in _all_decay_free:
        c, m, ls = get_plot_style(col)
        valid_data = df_u_merged[col].dropna()
        x_peak = valid_data.idxmax()
        valid_data = valid_data[valid_data.index >= x_peak]
        if len(valid_data) == 0:
            continue
        decay_y = (1.0 / valid_data.values) ** 2
        k_val = next((r['K_decay'] for r in comparison_table_rows if r['Case'] == col), np.nan)
        c_val = next((r['C_int']   for r in comparison_table_rows if r['Case'] == col), np.nan)
        uk = next((r.get('u_K_decay', np.nan) for r in comparison_table_rows if r['Case'] == col), np.nan)
        uc = next((r.get('u_C_int', np.nan)   for r in comparison_table_rows if r['Case'] == col), np.nan)
        _lbl = _cl_get_label(col, context_cases=_all_decay_free)
        if not pd.isna(k_val) and not pd.isna(c_val):
            _kstr = f"$K={k_val:.3f}" + (rf"\pm{uk:.3f}$" if not pd.isna(uk) else "$")
            label = f"{_lbl}, {_kstr}, $C={c_val:.3f}$"
        else:
            label = _lbl
        # Per-point error bars (replaces the regression confidence band): the data
        # are (U_max/U_c)^2; y error propagates the centerline-velocity 1-sigma
        # through y=v^-2 (u_y = 2 v^-3 u_v); x error = position (u_pos_x/B).
        _fld_d = sfmap.get(col)
        _ydec = _decay_y_err(_fld_d, valid_data) if _fld_d else None
        ax_d.errorbar(valid_data.index, decay_y,
                      xerr=(_pos_unc_xb(_fld_d) if _fld_d else None), yerr=_ydec,
                      color=c, marker=m, linestyle=ls, label=label, lw=2, ms=5,
                      capsize=2, elinewidth=0.7, ecolor=c)
    ax_d.set_title(r'Plane Jet Decay: $(U_{max}/U_c)^2$ vs $X/B$')
    ax_d.set_xlabel(r'$X/B$ [-]')
    ax_d.set_ylabel(r'$(U_{max}/U_c)^2$ [-]')
    ax_d.grid(True, ls='--', alpha=0.4)
    ax_d.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax_d.set_ylim(bottom=0.5)
    fig_d.savefig(os.path.join(summary_dir, "PNG", "Superposed_Plane_Jet_Decay.png"),
                  dpi=300, bbox_inches='tight')
    ax_d.set_title('')
    fig_d.savefig(os.path.join(summary_dir, "PDF", "Superposed_Plane_Jet_Decay.pdf"),
                  format='pdf', bbox_inches='tight')
    plt.close(fig_d)
    print(" -> Superposed_Plane_Jet_Decay saved.")


def generate_psd_plots(parent_dir):
    """Regenerate only the PSD grid figure(s)."""
    png_path = generate_all_psds_grid(parent_dir)
    if png_path:
        print(f" -> PSD grid saved: {png_path}")
    else:
        print(" -> No PSD data found.")


def generate_turbulence_plots(parent_dir):
    """Regenerate Iu, Iv, Iw, Iuvw centerline superposed plots."""
    _, sfmap, summary_dir = _get_subfolder_map(parent_dir)
    sorted_cases = sorted(sfmap.keys())

    _specs = [
        ("Map_I_u_%_Centerline",   r'$I_u$ [%]',       "Superposed_I_u_Centerline",
         r'Centerline $I_u$ ($Y/B \approx 0$)'),
        ("Map_I_v_%_Centerline",   r'$I_v$ [%]',       "Superposed_I_v_Centerline",
         r'Centerline $I_v$ ($Y/B \approx 0$)'),
        ("Map_I_w_%_Centerline",   r'$I_w$ [%]',       "Superposed_I_w_Centerline",
         r'Centerline $I_w$ ($Y/B \approx 0$)'),
        ("Map_I_uvw_%_Centerline", r'$I_{uvw}$ [%]',   "Superposed_I_uvw_Centerline",
         r'Centerline $I_{uvw}$ ($Y/B \approx 0$)'),
    ]
    for _csv_stem, _ylabel, _out_stem, _title in _specs:
        fig_cl, ax_cl = plt.subplots(figsize=(12, 6))
        plotted = _plot_superposed_centerline(
            ax_cl, sorted_cases, sfmap, _csv_stem, _ylabel, normalise_by_umax=False)
        if not plotted:
            plt.close(fig_cl)
            print(f"    (no data — skipping {_out_stem})")
            continue
        ax_cl.set_title(_title)
        ax_cl.set_xlabel(r'$X/B$ [-]')
        ax_cl.set_ylabel(_ylabel)
        ax_cl.grid(True, ls='--', alpha=0.4)
        ax_cl.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        fig_cl.savefig(os.path.join(summary_dir, "PNG", _out_stem + ".png"),
                       dpi=300, bbox_inches='tight')
        ax_cl.set_title('')
        fig_cl.savefig(os.path.join(summary_dir, "PDF", _out_stem + ".pdf"),
                       format='pdf', bbox_inches='tight')
        plt.close(fig_cl)
        print(f" -> {_out_stem} saved.")

    # ── Near-plate TI profiles (Pla cases, max X/B column) ──────────────────
    _pla_cases_ti = [n for n in sorted_cases if n.endswith('Pla')]
    _nearplate_ti_specs = [
        # (csv_stem,      map_subdir,             ylabel,           out_stem)
        ("Map_I_u_%",   "2D_Profiles_Results",  r'$I_u$ [%]',    "Superposed_NearPlate_I_u"),
        ("Map_I_v_%",   "2D_Profiles_Results",  r'$I_v$ [%]',    "Superposed_NearPlate_I_v"),
        ("Map_I_w_%",   "2D_Profiles_Results",  r'$I_w$ [%]',    "Superposed_NearPlate_I_w"),
        ("Map_I_uvw_%", "2D_Profiles_Results",  r'$I_{uvw}$ [%]',"Superposed_NearPlate_I_uvw"),
    ]
    for _ti_stem, _ti_subdir, _ti_ylabel, _ti_out in _nearplate_ti_specs:
        _ti_title = rf"Near-plate profile of {_ti_ylabel} ($X/B$ = max)"
        _ti_fig, _ti_ax = plt.subplots(figsize=(10, 6))
        _ti_plotted = _plot_superposed_nearplate(
            _ti_ax, _pla_cases_ti, sfmap, _ti_subdir, _ti_stem,
            _ti_ylabel, normalise_by_umax=False)
        if not _ti_plotted:
            plt.close(_ti_fig)
            print(f"    (no near-plate data — skipping {_ti_out})")
            continue
        _ti_ax.set_title(_ti_title)
        _ti_ax.set_xlabel(r'$Y/B$ [-]')
        _ti_ax.set_ylabel(_ti_ylabel)
        _ti_ax.grid(True, ls='--', alpha=0.4)
        _ti_ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        _ti_fig.savefig(os.path.join(summary_dir, "PNG", _ti_out + ".png"),
                        dpi=300, bbox_inches='tight')
        _ti_ax.set_title('')
        _ti_fig.savefig(os.path.join(summary_dir, "PDF", _ti_out + ".pdf"),
                        format='pdf', bbox_inches='tight')
        plt.close(_ti_fig)
        # Export merged CSV (Y/B index × case-label columns)
        _ti_series = {}
        for _ti_name in _pla_cases_ti:
            _ti_folder = sfmap.get(_ti_name)
            if _ti_folder is None:
                continue
            _ti_csv = os.path.join(_ti_folder, _ti_subdir, "CSV", _ti_stem + ".csv")
            if not os.path.exists(_ti_csv):
                continue
            try:
                _ti_df = pd.read_csv(_ti_csv, index_col=0)
                _ti_df.index   = pd.to_numeric(_ti_df.index,   errors='coerce')
                _ti_df.columns = pd.to_numeric(_ti_df.columns, errors='coerce')
                _ti_df = _ti_df[_ti_df.index.notnull()]
                if _ti_df.shape[0] < 2:
                    continue
                _ti_maxcol = _ti_df.columns.max()
                _ti_s = _ti_df[_ti_maxcol].dropna()
                _ti_series[get_label(_ti_name)] = _ti_s
            except Exception:
                pass
        if _ti_series:
            _ti_merged = pd.DataFrame(_ti_series).sort_index()
            _ti_merged.index.name = "Y/B"
            _ti_merged.to_csv(os.path.join(summary_dir, "CSV", _ti_out + ".csv"))
        print(f" -> {_ti_out} saved.")

    # ── Near-nozzle TI profiles (Pla cases, FIRST/min X/B column) ───────────
    _nearnozzle_ti_specs = [
        (_s, _sub, _yl, _out.replace("NearPlate", "NearNozzle"), False)
        for (_s, _sub, _yl, _out) in _nearplate_ti_specs
    ]
    _emit_nearstation_profiles(_nearnozzle_ti_specs, 'nozzle', sfmap, _pla_cases_ti, summary_dir)

    # ── Mid-field TI profiles (Pla cases, MIDDLE X/B column) ────────────────
    _midfield_ti_specs = [
        (_s, _sub, _yl, _out.replace("NearPlate", "MidField"), False)
        for (_s, _sub, _yl, _out) in _nearplate_ti_specs
    ]
    _emit_nearstation_profiles(_midfield_ti_specs, 'midfield', sfmap, _pla_cases_ti, summary_dir)


def generate_frequency_plots(parent_dir):
    """
    Produce two cross-case Frequency-subcategory superposed plots:
      1. Superposed_Centerline_Freq  — Mean_Resonance_Hz vs X/B
      2. Superposed_Centerline_St    — Strouhal_Number   vs X/B

    Data source per case:
        experiments/<case>/1D_Profiles_Results/CSV/Centerline_Freq_Profile.csv
    Columns used: X/B, Mean_Resonance_Hz, Strouhal_Number
    NaN rows are dropped so the line connects only valid measurement points.
    """
    _cl_set_folder(parent_dir)
    _, sfmap, summary_dir = _get_subfolder_map(parent_dir)
    sorted_cases = sorted(sfmap.keys())

    # NOTE: 'Superposed_Centerline_Freq' was removed — it duplicated
    # 'Superposed_1st_Dominant_Freq' (produced in generate_superposition). Only the
    # centerline Strouhal profile is emitted here.
    _freq_specs = [
        (
            'Strouhal_Number',
            r'$St$ [-]',
            'Superposed_Centerline_St',
            r'Centerline Strouhal Number $St$ ($Y/B \approx 0$)',
        ),
    ]

    for y_col, ylabel, out_stem, title in _freq_specs:
        print(f" -> Generating {out_stem}...")
        fig_f, ax_f = plt.subplots(figsize=(12, 6))
        any_plotted = False
        merged_series = {}

        for name in sorted_cases:
            folder = sfmap.get(name)
            if folder is None:
                continue
            csv_path = os.path.join(
                folder, "1D_Profiles_Results", "CSV", "Centerline_Freq_Profile.csv")
            if not os.path.exists(csv_path):
                print(f"    -> Skipping {name}: Centerline_Freq_Profile.csv not found")
                continue
            try:
                df_fp = pd.read_csv(csv_path)
                # Keep only rows where the target column is not NaN
                df_fp = df_fp.dropna(subset=['X/B', y_col])
                if df_fp.empty:
                    print(f"    -> Skipping {name}: no valid rows for {y_col}")
                    continue
                xb_vals = df_fp['X/B'].values
                y_vals  = df_fp[y_col].values

                c_hex, mkr, ls = get_plot_style(name)
                lbl = _cl_get_label(name, context_cases=sorted_cases)
                ax_f.plot(xb_vals, y_vals,
                          color=c_hex, marker=mkr, linestyle=ls,
                          label=lbl, lw=2, ms=5)
                any_plotted = True

                # Collect for merged CSV (indexed by X/B)
                merged_series[name] = pd.Series(y_vals, index=xb_vals)

            except Exception as exc:
                print(f"    -> Error processing {out_stem} for {name}: {exc}")

        if not any_plotted:
            plt.close(fig_f)
            print(f"    (no data found — skipping {out_stem})")
            continue

        ax_f.set_title(title)
        ax_f.set_xlabel(r'$X/B$ [-]')
        ax_f.set_ylabel(ylabel)
        ax_f.grid(True, ls='--', alpha=0.4)
        ax_f.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)

        fig_f.savefig(os.path.join(summary_dir, "PNG", out_stem + ".png"),
                      dpi=300, bbox_inches='tight')
        ax_f.set_title('')
        fig_f.savefig(os.path.join(summary_dir, "PDF", out_stem + ".pdf"),
                      format='pdf', bbox_inches='tight')
        plt.close(fig_f)
        print(f"    -> {out_stem} PNG/PDF saved.")

        # Export merged CSV (X/B as index, cases as columns)
        if merged_series:
            df_merged_f = pd.DataFrame(merged_series).sort_index()
            df_merged_f.index.name = "X/B"
            df_merged_f.to_csv(os.path.join(summary_dir, "CSV", out_stem + ".csv"))
            print(f"    -> {out_stem}.csv saved.")

    # ── Near-plate freq + Strouhal profiles (Pla cases, max X/B column) ─────
    _pla_cases_fr = [n for n in sorted_cases if n.endswith('Pla')]

    # Detect dominant-freq map filename from the first available Pla case
    _detected_freq_stem = None
    for _fr_name in _pla_cases_fr:
        _fr_folder = sfmap.get(_fr_name)
        if _fr_folder is None:
            continue
        _stem = _detect_freq_map_stem(_fr_folder, "Frequency_Results")
        if _stem is not None:
            _detected_freq_stem = _stem
            print(f"  -> Detected freq map stem: {_stem!r} (from {_fr_name})")
            break
    if _detected_freq_stem is None:
        print("  -> No 2D dominant-freq map found for near-plate freq/St plots — skipping.")
        _nearplate_freq_specs = []
    else:
        _nearplate_freq_specs = [
            (_detected_freq_stem,       "Frequency_Results", r'$f_{dom}$ [Hz]', "Superposed_NearPlate_Dominant_Freq", False),
            ("Map_Strouhal_Number",     "Frequency_Results", r'$St$ [-]',       "Superposed_NearPlate_Strouhal",      False),
        ]
    # Energy-cascade (Kolmogorov) slope near-plate profile — added unconditionally
    # since Map_Energy_Cascade_Slope is written independently of the dominant-freq map.
    _nearplate_freq_specs.append(
        ("Map_Energy_Cascade_Slope", "Frequency_Results",
         r'Energy-cascade slope $\beta$ [-]', "Superposed_NearPlate_Cascade_Slope", False)
    )

    for _fr_stem, _fr_subdir, _fr_ylabel, _fr_out, _fr_norm in _nearplate_freq_specs:
        _fr_title = rf"Near-plate profile of {_fr_ylabel} ($X/B$ = max)"
        _fr_fig, _fr_ax = plt.subplots(figsize=(10, 6))
        _fr_plotted = _plot_superposed_nearplate(
            _fr_ax, _pla_cases_fr, sfmap, _fr_subdir, _fr_stem,
            _fr_ylabel, normalise_by_umax=_fr_norm)
        if not _fr_plotted:
            plt.close(_fr_fig)
            print(f"    (no near-plate data — skipping {_fr_out})")
            continue
        _fr_ax.set_title(_fr_title)
        _fr_ax.set_xlabel(r'$Y/B$ [-]')
        _fr_ax.set_ylabel(_fr_ylabel)
        _fr_ax.grid(True, ls='--', alpha=0.4)
        _fr_ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        _fr_fig.savefig(os.path.join(summary_dir, "PNG", _fr_out + ".png"),
                        dpi=300, bbox_inches='tight')
        _fr_ax.set_title('')
        _fr_fig.savefig(os.path.join(summary_dir, "PDF", _fr_out + ".pdf"),
                        format='pdf', bbox_inches='tight')
        plt.close(_fr_fig)
        # Export merged CSV (Y/B index × case-label columns)
        _fr_series = {}
        for _fr_name in _pla_cases_fr:
            _fr_folder = sfmap.get(_fr_name)
            if _fr_folder is None:
                continue
            _fr_csv = os.path.join(_fr_folder, _fr_subdir, "CSV", _fr_stem + ".csv")
            if not os.path.exists(_fr_csv):
                continue
            try:
                _fr_df = pd.read_csv(_fr_csv, index_col=0)
                _fr_df.index   = pd.to_numeric(_fr_df.index,   errors='coerce')
                _fr_df.columns = pd.to_numeric(_fr_df.columns, errors='coerce')
                _fr_df = _fr_df[_fr_df.index.notnull()]
                if _fr_df.shape[0] < 2:
                    continue
                _fr_maxcol = _fr_df.columns.max()
                _fr_s = _fr_df[_fr_maxcol].dropna()
                _fr_series[get_label(_fr_name)] = _fr_s
            except Exception:
                pass
        if _fr_series:
            _fr_merged = pd.DataFrame(_fr_series).sort_index()
            _fr_merged.index.name = "Y/B"
            _fr_merged.to_csv(os.path.join(summary_dir, "CSV", _fr_out + ".csv"))
        print(f" -> {_fr_out} saved.")

    # ── Near-nozzle freq/St/cascade profiles (Pla cases, FIRST/min X/B) ─────
    _nearnozzle_freq_specs = [
        (_s, _sub, _yl, _out.replace("NearPlate", "NearNozzle"), _nm)
        for (_s, _sub, _yl, _out, _nm) in _nearplate_freq_specs
    ]
    _emit_nearstation_profiles(_nearnozzle_freq_specs, 'nozzle', sfmap, _pla_cases_fr, summary_dir)

    # ── Mid-field freq/St/cascade profiles (Pla cases, MIDDLE X/B column) ───
    _midfield_freq_specs = [
        (_s, _sub, _yl, _out.replace("NearPlate", "MidField"), _nm)
        for (_s, _sub, _yl, _out, _nm) in _nearplate_freq_specs
    ]
    _emit_nearstation_profiles(_midfield_freq_specs, 'midfield', sfmap, _pla_cases_fr, summary_dir)


def generate_coupling_table(parent_dir):
    """
    Task 4 — Nu <-> unsteadiness coupling table, produced from the pipeline itself.

    Joins the thermal Global Nusselt summary with the aerodynamic frequency /
    turbulence outputs, case-by-case, for the heated forced cases (`...PaPla`
    with a cylinder) that have BOTH thermal and aerodynamic data. For each such
    case it reports the measured Reynolds number and global Nusselt number, the
    baseline efficiency, the near-plate coherent shedding frequency and the
    core turbulence intensity, plus a derived coupling ratio.

    Baseline efficiency convention (decided after comparing the two candidates):
        eta_nearestRe = Global_Nu_Exp / Nu_baseline_nearestRe
    is used as the PRIMARY efficiency rather than Nu/Nu_theo, because
      * it isolates the forcing enhancement (forced vs unforced jet measured
        identically, so the heat-loss / calibration biases largely cancel),
      * Hofmann's Nu_theo only describes the UNFORCED slot jet, so Nu/Nu_theo
        for a forced case conflates real enhancement with correlation error.
    The same-pressure baseline (eta_samePa) and the literature efficiency
    (eta_theo) are retained as secondary reference columns.

    Output: summary/CSV/Nu_Unsteadiness_Coupling.csv (+ XLSX), via the existing
    summary directory convention.
    """
    thermal_summary = os.path.join(parent_dir, "thermal", "results",
                                   "Thermal_Global_Summary.csv")
    if not os.path.exists(thermal_summary):
        print(f"  -> Coupling table skipped: {thermal_summary} not found.")
        return None

    try:
        df_th = pd.read_csv(thermal_summary)
    except Exception as e:
        print(f"  -> Coupling table skipped: could not read thermal summary: {e}")
        return None

    _, sfmap, summary_dir = _get_subfolder_map(parent_dir)

    def _is_cyl(row):
        return str(row.get('Has_Cylinder', '')).strip().lower() in ('yes', 'true', '1')

    # Free-jet baselines (Re, Nu) for nearest-Re matching
    baselines = []
    for _, r in df_th.iterrows():
        if not _is_cyl(r) and pd.notna(r.get('Re_used')) and pd.notna(r.get('Global_Nu_Exp')):
            baselines.append((float(r['Re_used']), float(r['Global_Nu_Exp']), str(r['Case_ID'])))

    rows = []
    for _, r in df_th.iterrows():
        case_id = str(r['Case_ID'])
        if not _is_cyl(r):
            continue
        # Only forced cases that have a matching aerodynamic experiment folder
        folder = sfmap.get(case_id)
        if folder is None:
            continue

        re_used = float(r['Re_used']) if pd.notna(r.get('Re_used')) else np.nan
        nu_exp  = float(r['Global_Nu_Exp']) if pd.notna(r.get('Global_Nu_Exp')) else np.nan
        eta_samepa = float(r['eta_baseline']) if pd.notna(r.get('eta_baseline')) else np.nan
        eta_theo   = float(r['eta_theo']) if pd.notna(r.get('eta_theo')) else np.nan

        # Nearest-Re free-jet baseline
        nu_base_near, base_case_near, eta_near = np.nan, "", np.nan
        if baselines and not np.isnan(re_used):
            re_b, nu_b, cid_b = min(baselines, key=lambda b: abs(b[0] - re_used))
            nu_base_near, base_case_near = nu_b, cid_b
            if nu_b > 0 and not np.isnan(nu_exp):
                eta_near = nu_exp / nu_b

        # Near-plate coherent shedding frequency (Task 3b output)
        f_dom_np, st_np = np.nan, np.nan
        freq_sum = os.path.join(folder, "Frequency_Results", "CSV",
                                "Global_Frequency_Summary.csv")
        if os.path.exists(freq_sum):
            try:
                df_fs = pd.read_csv(freq_sum)
                if 'NearPlate_Coherent_Freq_Hz' in df_fs.columns and pd.notna(df_fs['NearPlate_Coherent_Freq_Hz'].iloc[0]):
                    f_dom_np = float(df_fs['NearPlate_Coherent_Freq_Hz'].iloc[0])
                elif 'Global_Mean_Freq_Hz' in df_fs.columns:
                    f_dom_np = float(df_fs['Global_Mean_Freq_Hz'].iloc[0])
                if 'Global_Mean_St' in df_fs.columns:
                    st_np = float(df_fs['Global_Mean_St'].iloc[0])
            except Exception:
                pass

        # Core (centerline) turbulence intensity I_uvw
        i_uvw_core = np.nan
        iuvw_csv = _resolve_centerline_csv(folder, "Map_I_uvw_%_Centerline")
        if iuvw_csv is not None:
            try:
                df_iu = pd.read_csv(iuvw_csv, index_col=0)
                df_iu.index = pd.to_numeric(df_iu.index, errors='coerce')
                df_iu.columns = pd.to_numeric(df_iu.columns, errors='coerce')
                df_iu = df_iu[df_iu.index.notnull()]
                cy = min(df_iu.index, key=lambda y: abs(y - 0.0))
                i_uvw_core = float(df_iu.loc[cy].dropna().mean())
            except Exception:
                pass

        # Derived coupling ratio: fractional Nu enhancement per unit core TI.
        coupling_ratio = np.nan
        if not np.isnan(eta_near) and not np.isnan(i_uvw_core) and i_uvw_core > 0:
            coupling_ratio = (eta_near - 1.0) / (i_uvw_core / 100.0)

        rows.append({
            'Case': get_label(case_id),
            'Re_used': round(re_used, 0) if not np.isnan(re_used) else np.nan,
            'Global_Nu_Exp': round(nu_exp, 2) if not np.isnan(nu_exp) else np.nan,
            'Nu_baseline_nearestRe': round(nu_base_near, 2) if not np.isnan(nu_base_near) else np.nan,
            'Baseline_case': get_label(base_case_near) if base_case_near else "",
            'eta_nearestRe': round(eta_near, 4) if not np.isnan(eta_near) else np.nan,
            'eta_samePa': round(eta_samepa, 4) if not np.isnan(eta_samepa) else np.nan,
            'eta_theo': round(eta_theo, 4) if not np.isnan(eta_theo) else np.nan,
            'f_dom_nearplate_Hz': round(f_dom_np, 1) if not np.isnan(f_dom_np) else np.nan,
            'St_global': round(st_np, 4) if not np.isnan(st_np) else np.nan,
            'I_uvw_core_pct': round(i_uvw_core, 2) if not np.isnan(i_uvw_core) else np.nan,
            'Nu_enhancement_pct': round((eta_near - 1.0) * 100.0, 1) if not np.isnan(eta_near) else np.nan,
            'Coupling_ratio': round(coupling_ratio, 3) if not np.isnan(coupling_ratio) else np.nan,
        })

    if not rows:
        print("  -> Coupling table: no forced heated case had both thermal and aerodynamic data.")
        return None

    df_out = pd.DataFrame(rows).sort_values('Re_used').reset_index(drop=True)
    csv_path = os.path.join(summary_dir, "CSV", "Nu_Unsteadiness_Coupling.csv")
    df_out.to_csv(csv_path, index=False)
    try:
        df_out.to_excel(os.path.join(summary_dir, "XLSX", "Nu_Unsteadiness_Coupling.xlsx"), index=False)
    except Exception:
        pass
    print(f"  -> Nu<->unsteadiness coupling table saved: {csv_path}")
    return df_out


if __name__ == "__main__":
    import tkinter
    from tkinter import filedialog, messagebox
    root = tkinter.Tk()
    root.withdraw()
    target_dir = filedialog.askdirectory(title="Select Parent Directory")
    if target_dir:
        try:
            generate_superposition(target_dir)
            messagebox.showinfo("Success", "Superposition Summary Generated.")
        except Exception as e:
            messagebox.showerror("Error", str(e))