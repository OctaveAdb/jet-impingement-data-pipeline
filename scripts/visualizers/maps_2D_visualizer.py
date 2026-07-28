# =============================================================================
# MAPS_2D_VISUALIZER  —  Clean Shell
# =============================================================================
# Purpose:
#   GUI shell for 2D spatial visualisations of the mean flow field.
#   The folder-selection bar, canvas area, and PNG/PDF save infrastructure
#   are in place. Specific plot functions will be added one by one.
#
# Usage:
#   - Standalone: python maps_2D_visualizer.py  (opens Tkinter GUI)
#   - Via hub:    launched from usage.py as "2D Flow Visualizer"
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Professional Plotting & LaTeX Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


class DataVisualizerApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("2D Flow Field Visualizations")
        self.root.geometry("1100x850")

        self.folder_path = tk.StringVar(value=default_folder or "")
        self.current_fig = None

        self.setup_ui()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Case Folder")
        if folder:
            self.folder_path.set(folder)

    def log(self, message):
        print(message)

    def save_figure_files(self, fig, filename_base):
        main_folder = self.folder_path.get()
        png_dir = os.path.join(main_folder, "2D_Profiles_Results", "PNG")
        pdf_dir = os.path.join(main_folder, "2D_Profiles_Results", "PDF")

        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        # 1. Save PNG (Includes Title)
        fig.savefig(os.path.join(png_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')

        # 2. Temporarily hide title for PDF
        original_titles = [ax.get_title() for ax in fig.axes]
        for ax in fig.axes:
            ax.set_title("")

        # 3. Save PDF (No Title)
        fig.savefig(os.path.join(pdf_dir, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight')

        # 4. Restore the title for UI Canvas
        for ax, original_title in zip(fig.axes, original_titles):
            ax.set_title(original_title)

    def setup_ui(self):
        # --- Top bar: folder selector ---
        control_frame = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        tk.Label(
            control_frame,
            text="Data Source (Case Folder):",
            bg="#e0e0e0",
            font=("Arial", 10, "bold"),
        ).pack(side="left", padx=10, pady=10)
        tk.Entry(
            control_frame,
            textvariable=self.folder_path,
            state="readonly",
            width=60,
        ).pack(side="left", padx=5)
        tk.Button(
            control_frame,
            text="Browse...",
            command=self.select_folder,
        ).pack(side="left", padx=5)

        # --- Left sidebar: placeholder ---
        sidebar = tk.Frame(self.root, width=220, bg="#f4f4f4")
        sidebar.pack(side="left", fill="y", padx=10, pady=(0, 10))

        tk.Frame(sidebar, height=2, bg="#ccc").pack(fill="x", pady=8)
        tk.Label(
            sidebar,
            text="Plot functions coming soon",
            font=("Arial", 10, "italic"),
            bg="#f4f4f4",
            fg="#888",
            wraplength=200,
            justify="center",
        ).pack(pady=20, padx=10)

        # --- Main canvas area ---
        self.canvas_frame = tk.Frame(self.root, bg="white", relief="sunken", bd=2)
        self.canvas_frame.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=(0, 10))

        tk.Label(
            self.canvas_frame,
            text="Select a folder and use the sidebar buttons to generate plots.",
            bg="white",
            fg="#888",
            font=("Arial", 14),
            wraplength=600,
            justify="center",
        ).pack(expand=True)


# =============================================================================
# TRANSVERSE SPATIAL PROFILE PLOTS  (Pla cases)
# =============================================================================
# Standalone function — no Tkinter required.  Called from pipeline.py for
# every case whose folder name ends with "Pla".
#
# Visual recipe
# -------------
# For each X/B station a vertical profile is drawn as a horizontal offset from
# the station baseline, creating the characteristic transverse profile appearance:
#
#   x_profile = x_station + profile_values * scale
#   fill_betweenx(y_vals, x_station, x_profile)
#   plot(x_profile, y_vals)
#   axvline(x_station, ...)
#
# Output locations
# ----------------
#   <case_folder>/2D_Profiles_Results/PNG/Transverse_<var>.png   (300 dpi)
#   <case_folder>/2D_Profiles_Results/PDF/Transverse_<var>.pdf
# =============================================================================

def _pick_value_map(candidates, stem_fragment):
    """From loosely-matched 'Map_*' candidates (stem_fragment found anywhere in the
    name), return the VALUE map — never an uncertainty companion Map_u_<var> added
    in the position-uncertainty step, which also contains the stem and would shadow
    the value map. Prefer the exact name 'Map_<stem>.csv' (or the legacy
    '_Centerline' variant); else the shortest candidate (the value map is shorter
    than its 'Map_u_<stem>' companion)."""
    if not candidates:
        return None
    for exact in (f"Map_{stem_fragment}.csv", f"Map_{stem_fragment}_Centerline.csv"):
        if exact in candidates:
            return exact
    return min(candidates, key=len)


def save_transverse_profiles(case_folder: str) -> None:
    """Generate and save 2D transverse spatial profile figures for all available
    variables in a Pla experiment case.

    Parameters
    ----------
    case_folder : str
        Absolute path to the experiment case folder (must end with 'Pla').
    """
    import os
    import gc
    import pandas as pd
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # ------------------------------------------------------------------
    # 0.  Guard: only run for Pla cases
    # ------------------------------------------------------------------
    case_name = os.path.basename(os.path.normpath(case_folder))
    if not case_name.endswith("Pla"):
        return

    # ------------------------------------------------------------------
    # 0b.  Resolve human-readable display label via case_labels dictionary.
    #      Falls back to raw case_name if the module is unavailable.
    # ------------------------------------------------------------------
    try:
        import sys as _sys, os as _os
        _lp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils')
        if _lp not in _sys.path: _sys.path.insert(0, _lp)
        from case_labels import get_label as _get_label, set_main_folder as _smf
        # main folder = two levels up from the case folder (experiments/<case> -> root)
        _smf(_os.path.dirname(_os.path.dirname(_os.path.normpath(case_folder))))
        _disp = _get_label(case_name)
    except Exception:
        _disp = case_name

    # ------------------------------------------------------------------
    # 1.  Locate CSV folder (try 2D first, fall back to 1D)
    # ------------------------------------------------------------------
    csv_dir_2d = os.path.join(case_folder, "2D_Profiles_Results", "CSV")
    csv_dir_1d = os.path.join(case_folder, "1D_Profiles_Results", "CSV")

    if os.path.isdir(csv_dir_2d):
        csv_dir = csv_dir_2d
    elif os.path.isdir(csv_dir_1d):
        csv_dir = csv_dir_1d
    else:
        print(f"[Transverse] No CSV folder found in {case_name} — skipping.")
        return

    # Frequency pivot tables live in a separate folder regardless of which
    # velocity/TI folder was selected above.
    freq_csv_dir = os.path.join(case_folder, "Frequency_Results", "CSV")

    # ------------------------------------------------------------------
    # 2.  Variable definitions:
    #       (csv_stem_fragment, latex_label, filename_tag, normalise, skip_nan)
    #     normalise=True  → divide by U_max from u_m_s file
    #     normalise=False → use raw values (turbulence intensity already in %)
    #     skip_nan=True   → plot only finite points per column (no zero-fill)
    #                        used for frequency variables where many cells are NaN
    # ------------------------------------------------------------------
    VARIABLES = [
        # (filename stem contains this fragment, LaTeX label, output tag, normalise, skip_nan)
        ("u_m_s",              r"$U/U_{\mathrm{max}}$",                "U_over_Umax",    True,  False),
        ("v_m_s",              r"$V/U_{\mathrm{max}}$",                "V_over_Umax",    True,  False),
        ("w_m_s",              r"$W/U_{\mathrm{max}}$",                "W_over_Umax",    True,  False),
        ("Vel_Mag_m_s",        r"$V_{\mathrm{mag}}/U_{\mathrm{max}}$", "Vmag_over_Umax", True,  False),
        ("I_u_%",              r"$I_u$ [%]",                           "I_u_Pct",        False, False),
        ("I_v_%",              r"$I_v$ [%]",                           "I_v_Pct",        False, False),
        ("I_w_%",              r"$I_w$ [%]",                           "I_w_Pct",        False, False),
        ("I_uvw_%",            r"$I_{uvw}$ [%]",                       "I_uvw_Pct",      False, False),
        ("Dominant_Freq_1st",  r"$f_{dom}$ [Hz]",                      "Dominant_Freq",  False, True),
        ("Strouhal_Number",    r"$St$ [-]",                             "Strouhal",       False, True),
    ]

    # ------------------------------------------------------------------
    # 3.  Determine U_max for normalisation (from u_m_s file)
    # ------------------------------------------------------------------
    u_max = None
    u_files = [
        f for f in os.listdir(csv_dir)
        if f.startswith("Map_") and "u_m_s" in f and f.endswith(".csv")
        and "rms" not in f and "Normalized" not in f
    ]
    _u_val = _pick_value_map(u_files, "u_m_s")
    if _u_val:
        try:
            _df_u = pd.read_csv(os.path.join(csv_dir, _u_val), index_col=0)
            u_max = float(np.nanmax(_df_u.values))
        except Exception:
            u_max = None

    # ------------------------------------------------------------------
    # 4.  Output directories
    # ------------------------------------------------------------------
    png_dir = os.path.join(case_folder, "2D_Profiles_Results", "PNG")
    pdf_dir = os.path.join(case_folder, "2D_Profiles_Results", "PDF")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Styling
    FILL_COLOR = "#4472C4"   # steel blue

    # ------------------------------------------------------------------
    # 5.  Loop over variables
    # ------------------------------------------------------------------
    for stem_fragment, latex_label, out_tag, do_normalise, do_skip_nan in VARIABLES:

        # Frequency variables are stored in a different folder
        search_dir = freq_csv_dir if do_skip_nan else csv_dir

        # Skip silently if the source folder does not exist
        if not os.path.isdir(search_dir):
            continue

        # Find matching CSV (allow _Centerline suffix or plain)
        candidates = [
            f for f in os.listdir(search_dir)
            if f.startswith("Map_") and stem_fragment in f and f.endswith(".csv")
            and "rms" not in f and "Normalized" not in f
        ]
        _chosen = _pick_value_map(candidates, stem_fragment)
        if not _chosen:
            continue  # variable not available — skip silently

        csv_path = os.path.join(search_dir, _chosen)

        # ---- Load pivot table ----
        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index   = pd.to_numeric(df.index,   errors='coerce')
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df = df.sort_index()
            df = df.reindex(sorted(df.columns), axis=1)
        except Exception as e:
            print(f"[Butterfly] Could not load {_chosen}: {e}")
            continue

        # Guard: need at least 2 Y/B rows for a meaningful spatial profile
        if len(df.index) < 2:
            print(
                f"[Transverse] {case_name}/{_chosen} has only {len(df.index)} "
                "Y/B row — skipping (need ≥2 rows for transverse plot)."
            )
            continue

        # ---- Normalise if required ----
        if do_normalise:
            if u_max is None or u_max == 0:
                print(f"[Transverse] U_max unavailable for {case_name} — skipping {stem_fragment}.")
                continue
            values_df = df / u_max
        else:
            values_df = df.copy()

        # ---- Filter out columns with ≤1 finite data points ----
        cols_all = values_df.columns.values.astype(float)
        valid_cols = [c for c in cols_all if np.sum(np.isfinite(values_df[c].values)) > 1]
        if len(valid_cols) < 2:
            continue  # skip this variable — not enough spatial data
        cols = np.array(valid_cols)

        # ---- Compute scale factor based on actual profile range ----
        profile_max = max(
            np.nanmax(np.abs(values_df[c].values)) for c in valid_cols
            if np.any(np.isfinite(values_df[c].values))
        )
        if profile_max == 0:
            continue
        station_spacing = (cols[-1] - cols[0]) / (len(cols) - 1) if len(cols) > 1 else 1.0
        scale = (station_spacing * 0.7) / profile_max  # widest profile fills 70% of gap

        # ---- Build figure (non-interactive — no Tkinter backend) ----
        fig = Figure(figsize=(12, 7), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        y_vals = values_df.index.values.astype(float)

        for x_station in cols:
            prof_vals = values_df[x_station].values.astype(float)

            if do_skip_nan:
                # For frequency variables: plot only finite points so that
                # missing-peak NaNs do not produce spurious zero-spikes.
                finite_mask = np.isfinite(prof_vals)
                if np.sum(finite_mask) < 2:
                    # Not enough finite points to draw a meaningful profile;
                    # still draw the baseline and label.
                    ax.axvline(
                        x_station,
                        color='gray', linestyle='--', alpha=0.4, linewidth=0.8
                    )
                    ax.text(x_station, y_vals.max() * 1.02, f"{x_station:.2f}",
                            ha='center', va='bottom', fontsize=7, color='gray')
                    continue

                y_finite   = y_vals[finite_mask]
                pv_finite  = prof_vals[finite_mask]
                x_profile  = x_station + pv_finite * scale

                ax.fill_betweenx(
                    y_finite, x_station, x_profile,
                    alpha=0.30, color=FILL_COLOR
                )
                ax.plot(
                    x_profile, y_finite,
                    color=FILL_COLOR, linewidth=1.5
                )
            else:
                # Standard recipe: replace NaN with zero so fill works cleanly
                prof_vals = np.where(np.isfinite(prof_vals), prof_vals, 0.0)
                x_profile = x_station + prof_vals * scale

                ax.fill_betweenx(
                    y_vals, x_station, x_profile,
                    alpha=0.30, color=FILL_COLOR
                )
                ax.plot(
                    x_profile, y_vals,
                    color=FILL_COLOR, linewidth=1.5
                )

            ax.axvline(
                x_station,
                color='gray', linestyle='--', alpha=0.4, linewidth=0.8
            )
            # Station label at top of plot
            ax.text(x_station, y_vals.max() * 1.02, f"{x_station:.2f}",
                    ha='center', va='bottom', fontsize=7, color='gray')

        # Centreline reference
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

        # Nozzle-mouth edges project to Y/B = +/-0.5 (the slot height B). Thin dotted
        # reference lines show the jet width relative to the nozzle exit.
        for _ye in (0.5, -0.5):
            ax.axhline(_ye, color='0.35', linestyle=':', linewidth=0.9, alpha=0.8)
        ax.text(cols[0], 0.5, ' nozzle edge ($Y/B=\\pm0.5$)', fontsize=6.5,
                color='0.35', va='bottom', ha='left')

        # ---- Scale bar (overlaid; no added blank space) --------------------
        # Profiles are drawn as lobes whose horizontal half-width equals value*scale.
        # For U0-normalised velocity the bar represents the FULL scale (value=1) and
        # is labelled with the actual normalising velocity U_max in m/s, so the reader
        # converts a lobe directly to a physical speed; for other variables it shows a
        # round reference in the variable's own units. Placed inside the axes
        # (lower-left) with AnchoredSizeBar.
        import math as _m
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as _fm
        def _nice(v):
            if not np.isfinite(v) or v <= 0:
                return v
            _e = _m.floor(_m.log10(v)); _b = v / 10 ** _e
            _n = 1 if _b < 1.5 else 2 if _b < 3.5 else 5 if _b < 7.5 else 10
            return _n * 10 ** _e
        if do_normalise and u_max and np.isfinite(u_max) and u_max > 0:
            _ref = 1.0                                  # full normalised scale
            _label = rf"$U_\mathrm{{max}}={u_max:.0f}$ m/s (full scale)"
        else:
            _ref = _nice(profile_max)
            if _ref > profile_max:
                _ref = _nice(profile_max * 0.5)
            _label = rf"{_ref:g}  {latex_label}"
        _bar = _ref * scale              # length in X/B units
        try:
            _sb = AnchoredSizeBar(ax.transData, _bar, _label,
                                  loc='lower left', pad=0.3, borderpad=0.4, sep=3,
                                  frameon=True, color='black',
                                  fontproperties=_fm.FontProperties(size=8.5))
            _sb.patch.set_alpha(0.75); _sb.patch.set_edgecolor('0.6')
            ax.add_artist(_sb)
        except Exception:
            pass

        # X/B tick labels at each station
        ax.set_xticks(cols)
        ax.set_xticklabels([f"{c:.2f}" for c in cols], fontsize=7, rotation=45, ha='right')

        ax.set_xlabel(r"$X/B$ [-]", fontsize=12)
        ax.set_ylabel(r"$Y/B$ [-]", fontsize=12)
        ax.set_title(rf"Transverse Profiles of {latex_label}  — {_disp}", fontsize=14)
        ax.grid(color='gray', linestyle='--', alpha=0.3)

        # ---- Save PNG (with title) ----
        png_path = os.path.join(png_dir, f"Transverse_{out_tag}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')

        # ---- Save PDF (without title) ----
        original_title = ax.get_title()
        ax.set_title("")
        pdf_path = os.path.join(pdf_dir, f"Transverse_{out_tag}.pdf")
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        ax.set_title(original_title)  # restore (not strictly needed)

        fig.clear()
        del fig
        gc.collect()
        print(f"[Transverse] Saved: {os.path.basename(png_path)}")

    print(f"[Transverse] Done — {case_name}")


# =============================================================================
# VERTICAL SLICE PROFILE PLOTS  (transverse-map cases)
# =============================================================================
# Standalone function — no Tkinter required.  Called from pipeline.py step 5
# for every case that has multiple Y/B rows (plate-present traverse).
#
# For each available variable (velocity components, TI, frequency metrics),
# three figures are produced: one at the minimum X/B station, one at the
# first X/B station after the minimum that has strictly more than 2 finite
# data points (ensures a "good" middle cross-section even when frequency maps
# have sparse coverage near the nozzle), and one at the maximum X/B station.
# Each figure shows the variable on the x-axis vs Y/B on the y-axis.
#
# Output locations
# ----------------
#   <case_folder>/2D_Profiles_Results/PNG/VSlice_<var>_XB<val:.2f>.png
#   <case_folder>/2D_Profiles_Results/PDF/VSlice_<var>_XB<val:.2f>.pdf
# =============================================================================

def save_vertical_slice_profiles(case_folder: str) -> None:
    """Generate and save vertical-slice figures (variable vs Y/B at 3 X/B stations)
    for all available variables in a transverse-map experiment case.

    Parameters
    ----------
    case_folder : str
        Absolute path to the experiment case folder.
    """
    import os
    import gc
    import re as _re
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    case_name = os.path.basename(os.path.normpath(case_folder))

    # ------------------------------------------------------------------
    # 0.  Human-readable display label
    # ------------------------------------------------------------------
    try:
        import sys as _sys, os as _os
        _lp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils')
        if _lp not in _sys.path: _sys.path.insert(0, _lp)
        from case_labels import get_label as _get_label, set_main_folder as _smf
        _smf(_os.path.dirname(_os.path.dirname(_os.path.normpath(case_folder))))
        _disp = _get_label(case_name)
    except Exception:
        _disp = case_name

    # ------------------------------------------------------------------
    # 1.  Locate CSV folders
    # ------------------------------------------------------------------
    csv_dir_2d   = os.path.join(case_folder, "2D_Profiles_Results", "CSV")
    freq_csv_dir = os.path.join(case_folder, "Frequency_Results",   "CSV")

    # ------------------------------------------------------------------
    # 2.  Variable definitions  (same registry as save_transverse_profiles)
    #     (csv_stem_fragment, latex_label, filename_tag, normalise, from_freq)
    #     normalise=True  → divide by U_max (from u_m_s map)
    #     from_freq=True  → CSV lives in Frequency_Results/CSV
    # ------------------------------------------------------------------
    VARIABLES = [
        ("u_m_s",             r"$U/U_{\mathrm{max}}$",                "U_over_Umax",    True,  False),
        ("v_m_s",             r"$V/U_{\mathrm{max}}$",                "V_over_Umax",    True,  False),
        ("w_m_s",             r"$W/U_{\mathrm{max}}$",                "W_over_Umax",    True,  False),
        ("Vel_Mag_m_s",       r"$V_{\mathrm{mag}}/U_{\mathrm{max}}$", "Vmag_over_Umax", True,  False),
        ("I_u_%",             r"$I_u$ [%]",                           "I_u_Pct",        False, False),
        ("I_v_%",             r"$I_v$ [%]",                           "I_v_Pct",        False, False),
        ("I_w_%",             r"$I_w$ [%]",                           "I_w_Pct",        False, False),
        ("I_uvw_%",           r"$I_{uvw}$ [%]",                       "I_uvw_Pct",      False, False),
        ("Dominant_Freq_1st", r"$f_{dom}$ [Hz]",                      "Dominant_Freq",  False, True),
        ("Strouhal_Number",   r"$St$ [-]",                             "Strouhal",       False, True),
    ]

    # ------------------------------------------------------------------
    # 3.  Determine U_max for normalisation
    # ------------------------------------------------------------------
    u_max = None
    if os.path.isdir(csv_dir_2d):
        u_files = [
            f for f in os.listdir(csv_dir_2d)
            if f.startswith("Map_") and "u_m_s" in f and f.endswith(".csv")
            and "rms" not in f
        ]
        _u_val = _pick_value_map(u_files, "u_m_s")
        if _u_val:
            try:
                _df_u = pd.read_csv(os.path.join(csv_dir_2d, _u_val), index_col=0)
                u_max = float(np.nanmax(_df_u.values))
            except Exception:
                u_max = None

    # ------------------------------------------------------------------
    # 4.  Output directories
    # ------------------------------------------------------------------
    png_dir = os.path.join(case_folder, "2D_Profiles_Results", "PNG")
    pdf_dir = os.path.join(case_folder, "2D_Profiles_Results", "PDF")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    LINE_COLOR = "#C0392B"   # deep red — distinct from butterfly fill blue

    # ------------------------------------------------------------------
    # 5.  Loop over variables
    # ------------------------------------------------------------------
    for stem_fragment, latex_label, out_tag, do_normalise, from_freq in VARIABLES:

        search_dir = freq_csv_dir if from_freq else csv_dir_2d
        if not os.path.isdir(search_dir):
            continue

        candidates = [
            f for f in os.listdir(search_dir)
            if f.startswith("Map_") and stem_fragment in f and f.endswith(".csv")
            and "rms" not in f
        ]
        _chosen = _pick_value_map(candidates, stem_fragment)
        if not _chosen:
            continue

        csv_path = os.path.join(search_dir, _chosen)

        try:
            df = pd.read_csv(csv_path, index_col=0)
            df.index   = pd.to_numeric(df.index,   errors='coerce')
            df.columns = pd.to_numeric(df.columns, errors='coerce')
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df = df.sort_index()
            df = df.reindex(sorted(df.columns), axis=1)
        except Exception as e:
            print(f"[VSlice] Could not load {_chosen}: {e}")
            continue

        # Need ≥2 Y/B rows for a transverse profile to be meaningful
        if df.index.nunique() < 2:
            continue

        # Normalise if required
        if do_normalise:
            if u_max is None or u_max == 0:
                print(f"[VSlice] U_max unavailable for {case_name} — skipping {stem_fragment}.")
                continue
            df = df / u_max

        # ------------------------------------------------------------------
        # 6.  Select the 3 X/B stations
        #     min  — first column
        #     mid  — first column *after* min with strictly >2 finite values
        #     max  — last column
        # ------------------------------------------------------------------
        sorted_cols = sorted(df.columns.tolist())
        if len(sorted_cols) < 1:
            continue

        xb_min = sorted_cols[0]
        xb_max = sorted_cols[-1]

        # Find middle: first col after the minimum that has >2 finite values
        xb_mid = None
        for col in sorted_cols[1:]:
            n_finite = int(np.sum(np.isfinite(df[col].values.astype(float))))
            if n_finite > 2:
                xb_mid = col
                break
        # Fallback: if no column qualifies use the second column
        if xb_mid is None and len(sorted_cols) >= 2:
            xb_mid = sorted_cols[1]

        selected = list(dict.fromkeys([xb_min, xb_mid, xb_max]))  # deduplicate

        y_vals = df.index.values.astype(float)

        for xb_val in selected:
            if xb_val is None:
                continue
            col = min(df.columns, key=lambda c: abs(c - xb_val))
            profile = df[col].values.astype(float)

            # For sparse frequency maps: only plot finite points
            finite_mask = np.isfinite(profile)
            if np.sum(finite_mask) < 2:
                continue
            y_plot = y_vals[finite_mask]
            p_plot = profile[finite_mask]

            fig = Figure(figsize=(6, 7), layout='constrained')
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            ax.plot(p_plot, y_plot, '-o', color=LINE_COLOR, lw=2, ms=5)
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax.set_xlabel(latex_label, fontsize=12)
            ax.set_ylabel(r"$Y/B$ [-]", fontsize=12)
            ax.set_title(rf"{latex_label},  $X/B = {col:.2f}$  — {_disp}", fontsize=13)
            ax.grid(color='gray', linestyle='--', alpha=0.3)

            fname_base = f"VSlice_{out_tag}_XB{col:.2f}"

            png_path = os.path.join(png_dir, f"{fname_base}.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')

            original_title = ax.get_title()
            ax.set_title("")
            pdf_path = os.path.join(pdf_dir, f"{fname_base}.pdf")
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            ax.set_title(original_title)

            fig.clear()
            del fig
            gc.collect()
            print(f"[VSlice] Saved: {fname_base}")

    print(f"[VSlice] Done — {case_name}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataVisualizerApp(root)
    root.mainloop()
