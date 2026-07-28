# =============================================================================
# GRAPHS_1D_VISUALIZER
# =============================================================================
# Purpose:
#   Generates interactive 1D cross-sectional profiles from the spatial map
#   CSV files produced by mean_processor and frequency_processor. The user
#   selects any available variable (velocity components, turbulence intensity,
#   dominant frequency, Strouhal number, shedding
#   coherence, or static pressure), specifies whether to slice along an X/B
#   or Y/B station, and plots the resulting profile with proper scientific axis
#   labels. For velocity profiles, the plot annotates the jet potential core
#   length L_c and the plane-jet decay rate K from the Global_Velocity_Summary.
#   For frequency variables, the global mean dominant frequency and mean
#   Strouhal number from Global_Frequency_Summary are overlaid as reference
#   lines. Each generated figure is saved to PNG (300 dpi) and PDF formats.
#
# Inputs:
#   - Spatial map CSV files (automatically discovered):
#       experiments/<case>/1D_Profiles_Results/CSV/Map_*.csv
#       experiments/<case>/2D_Profiles_Results/CSV/Map_*.csv
#       experiments/<case>/Frequency_Results/CSV/Map_*.csv
#   - Jet characteristic metrics:
#       experiments/<case>/1D_Profiles_Results/CSV/Global_Velocity_Summary.csv
#   - Frequency summary:
#       experiments/<case>/Frequency_Results/CSV/Global_Frequency_Summary.csv
#
# Outputs:
#   - 1D profile figures (PNG and PDF) named by variable and slice value:
#       experiments/<case>/1D_Profiles_Results/PNG/Profile_<var>_<type>_<val>.png
#       experiments/<case>/1D_Profiles_Results/PDF/Profile_<var>_<type>_<val>.pdf
#
# Dependencies:
#   - mean_processor and frequency_processor (produce the Map CSV inputs)
#   - Imported by pipeline.py (Step 7) and usage.py
#
# Usage:
#   - Standalone: python graphs_1D_visualizer.py  (opens Tkinter GUI)
#   - Via pipeline: called programmatically by pipeline.py (Step 7)
#   - Via hub:      launched from usage.py as "1D Profiles Viewer"
# =============================================================================

import sys
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- case_labels import (Task 12) ---
try:
    import sys as _sys, os as _os
    _lp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils')
    if _lp not in _sys.path: _sys.path.insert(0, _lp)
    from case_labels import get_label as _cl_get_label, set_main_folder as _cl_set_folder
    _CL_OK = True
except Exception:
    _CL_OK = False
    def _cl_get_label(x, **_): return x
    def _cl_set_folder(_): pass

# --- Probe POSITION 1-sigma (mm) for the independent-axis error bars ----------
# Shown as the X-error on the position axis (X/B or Y/B). NOTE: the value-axis
# (Y) uncertainty maps already fold position in via the local gradient, so this
# explicit position bar is mildly conservative (never understates) — see
# UNCERTAINTY_SPEC.md. Imported from utils.uncertainty; falls back to defaults.
try:
    from uncertainty import U_POS_X_MM as _U_POS_X_MM, U_POS_Y_MM as _U_POS_Y_MM
except Exception:
    _U_POS_X_MM, _U_POS_Y_MM = 1.0, 0.5

def _u0_reference(main_folder):
    """U0 velocity reference for profile normalisation = V_noz for BOTH free and
    cylinder cases. Prefers U0_ref_ms from Global_Velocity_Summary, else V_noz_ms
    from Flow_Data. (The idealised gap velocity V_gap = V_noz·B/(B-D) was dropped
    entirely — see mean_processor.) Returns None if nothing usable is found."""
    try:
        gsp = os.path.join(main_folder, "1D_Profiles_Results", "CSV",
                           "Global_Velocity_Summary.csv")
        if os.path.exists(gsp):
            d = pd.read_csv(gsp)
            if 'U0_ref_ms' in d.columns:
                v = float(d['U0_ref_ms'].iloc[0])
                if v > 0:
                    return v
    except Exception:
        pass
    try:
        fp = os.path.join(main_folder, "Flow_Data.csv")
        if os.path.exists(fp):
            d = pd.read_csv(fp)
            if 'V_noz_ms' in d.columns:
                v = float(d['V_noz_ms'].iloc[-1])
                if v > 0:
                    return v
    except Exception:
        pass
    return None


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

class Profile1DApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("1D Cross-Sectional Profiles")
        self.root.geometry("1100x850")
        self.folder_path = tk.StringVar(value=default_folder if default_folder else "")
        self.means_data = {}
        self.unc_data = {}          # pretty_name -> companion 1-sigma uncertainty map
        self._u0_val = None         # U0 normalisation reference (for the band U0 term)
        self._u0_unc = float('nan') # u_U0_ref_ms (reference uncertainty)
        self._u0norm = set()        # pretty_names that are U0-normalised velocities
        self._B_mm = 30.0           # slot height B (mm) — normalises the position error bars
        self.current_fig = None
        
        # --- Translation Dictionary for Clean UI Names ---
        # Keys are the raw_name extracted from the CSV filename (Map_<raw_name>.csv).
        self.name_mapping = {
            "Dominant_Freq_1st": "1st Dominant Frequency (Hz)",
            "Dominant_Freq_2nd": "2nd Dominant Frequency (Hz)",
            "Strouhal_Number": "Strouhal Number (St)",
            "Shedding_Coherence": "Shedding Coherence (%)",
            # Turbulence intensity — various naming conventions in the wild
            "I_u_%":     r"$I_u$ [%]",
            "I_u_Pct":   r"$I_u$ [%]",
            "I_u_(%)":   r"$I_u$ [%]",
            "I_v_%":     r"$I_v$ [%]",
            "I_v_Pct":   r"$I_v$ [%]",
            "I_v_(%)":   r"$I_v$ [%]",
            "I_w_%":     r"$I_w$ [%]",
            "I_w_Pct":   r"$I_w$ [%]",
            "I_w_(%)":   r"$I_w$ [%]",
            "I_uvw_%":   r"$I_{uvw}$ [%]",
            "I_uvw_Pct": r"$I_{uvw}$ [%]",
            "I_uvw_(%)": r"$I_{uvw}$ [%]",
        }

        # --- Variables skipped entirely (not shown or saved in pipeline) ---
        # Pressure and temperature are excluded from auto-save.
        # V_over_Umax is excluded because v_m_s is normalised inline when loaded.
        self._skip_raw_names = {
            # Pressure — data kept in CSV but not auto-plotted
            "Static_Pressure_Pa", "Total_Pressure_Pa", "Barometric_Pressure_Pa",
            # Temperature — belongs to thermal pipeline only
            "Temperature_C", "Temperature_degC", "Temperature_°C",
            # V_over_Umax column written by older mean_processor versions;
            # v_m_s is normalised by U_max inline when loaded, so this is redundant
            "V_over_Umax",
            # Raw-unit magnitude — Vel_Mag_m_s is normalised instead
            "Velocity_Magnitude_m_s",
            # Energy cascade slope removed from frequency profiles
            "Energy_Cascade_Slope", "Cascade_Slope",
        }

        # --- Translation Dictionary for Graph Axis Labels ---
        self.axis_labels = {
            "1st Dominant Frequency (Hz)": r'$f_{dom}$ [Hz]',
            "2nd Dominant Frequency (Hz)": r'$f_{dom,2}$ [Hz]',
            "Strouhal Number (St)": r'$St$ [-]',
            "Shedding Coherence (%)": r'Coherence [%]',
            r"$I_u$ [%]":     r'$I_u$ [%]',
            r"$I_v$ [%]":     r'$I_v$ [%]',
            r"$I_w$ [%]":     r'$I_w$ [%]',
            r"$I_{uvw}$ [%]": r'$I_{uvw}$ [%]',
        }

        self.setup_ui()
        if self.folder_path.get(): self.scan_folder()

    # -----------------------------------------------------------------------
    # Task 6 — Category assignment helper
    # -----------------------------------------------------------------------
    _FREQ_PRETTY_NAMES = {
        "1st Dominant Frequency (Hz)",
        "2nd Dominant Frequency (Hz)",
        "Strouhal Number (St)",
        "Shedding Coherence (%)",
    }

    def _get_category(self, pretty_name):
        """Return one of: 'Velocity', 'Turbulence Intensity', 'Frequency', 'Miscellaneous'."""
        # Velocity — names containing U/U, V/U, W/U, Vmag, Vel_Mag, or Velocity
        if any(tok in pretty_name for tok in ("U/U", "V/U", "W/U", "Vmag", "Vel_Mag", "Velocity")):
            return "Velocity"
        # Turbulence — LaTeX $I_ prefix or "Turbulence" in name
        if pretty_name.startswith(r"$I_") or "Turbulence" in pretty_name:
            return "Turbulence Intensity"
        # Frequency — fixed set from axis_labels frequency metrics
        if pretty_name in self._FREQ_PRETTY_NAMES:
            return "Frequency"
        return "Miscellaneous"

    def log_action(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Main Data Folder")
        if folder:
            self.folder_path.set(folder)
            self.scan_folder()

    def _load_uncertainty_companion(self, folder, raw_name, norm_ref=None):
        """Load the companion 1-sigma uncertainty map Map_u_<raw_name>.csv written
        by mean_processor (value in N, uncertainty in N+1 convention, as a parallel
        pivot). Applies the SAME normalisation reference as the value map
        (norm_ref) for velocity components. Returns a numeric DataFrame aligned
        like the value map, or None if no companion exists."""
        comp = os.path.join(folder, f"Map_u_{raw_name}.csv")
        if not os.path.exists(comp):
            return None
        try:
            du = pd.read_csv(comp, index_col=0)
            du.index = pd.to_numeric(du.index, errors='coerce')
            du = du[du.index.notnull()]
            du.columns = pd.to_numeric(du.columns, errors='coerce')
            du = du.reindex(sorted(du.columns), axis=1)
            if norm_ref and norm_ref > 0:
                du = du / norm_ref
            return du
        except Exception:
            return None

    def scan_folder(self):
        main_folder = self.folder_path.get()
        self.means_data.clear()
        self.unc_data.clear()
        var_names = []
        _u_max_ref = [None]   # mutable container so inner loop can write to it

        # U0 normalisation reference (replaces U_max). Prefer U0_ref_ms from
        # Global_Velocity_Summary (= V_noz for free AND cylinder cases); else
        # V_noz_ms from Flow_Data; else fall back later to the streamwise peak.
        _u0 = _u0_reference(main_folder)
        if _u0 and _u0 > 0:
            _u_max_ref[0] = _u0
        # U0-reference uncertainty (u_U0_ref_ms) for the normalised-velocity band.
        self._u0norm = set()
        self._u0_unc = float('nan')
        try:
            _gsp0 = os.path.join(main_folder, "1D_Profiles_Results", "CSV",
                                 "Global_Velocity_Summary.csv")
            if os.path.exists(_gsp0):
                _d0 = pd.read_csv(_gsp0)
                if 'u_U0_ref_ms' in _d0.columns:
                    self._u0_unc = float(_d0['u_U0_ref_ms'].iloc[0])
                if 'B_mm' in _d0.columns:
                    _b = float(_d0['B_mm'].iloc[0])
                    if _b > 0:
                        self._B_mm = _b
        except Exception:
            pass

        folders_to_scan = [
            os.path.join(main_folder, "1D_Profiles_Results", "CSV"),
            os.path.join(main_folder, "2D_Profiles_Results", "CSV"),
            os.path.join(main_folder, "Frequency_Results", "CSV"),
            os.path.join(main_folder, "Means", "CSV") 
        ]

        # Pre-scan: collect U_max from u_m_s so all velocity components use the
        # same reference even if their CSV files are processed before u_m_s.
        for folder in folders_to_scan:
            u_path = os.path.join(folder, "Map_u_m_s.csv")
            if os.path.exists(u_path):
                try:
                    _df_u = pd.read_csv(u_path, index_col=0)
                    _df_u.index = pd.to_numeric(_df_u.index, errors='coerce')
                    _df_u = _df_u[_df_u.index.notnull()]
                    _df_u.columns = pd.to_numeric(_df_u.columns, errors='coerce')
                    if not _u_max_ref[0]:   # keep U0 if already set above
                        _u_max_ref[0] = _df_u.max().max()
                except Exception:
                    pass
                break  # only need one

        for folder in folders_to_scan:
            if os.path.exists(folder):
                map_files = glob.glob(os.path.join(folder, "Map_*.csv"))
                # Raw names present in this folder, to detect companion uncertainty
                # maps: Map_u_<X>.csv is the 1-sigma companion of Map_<X>.csv when
                # "<X>" is itself a map here (distinguishes "u_X" companions from the
                # genuine u-velocity map "u_m_s", whose "m_s" stem is not a map).
                _raw_set = {os.path.basename(x).replace("Map_", "").replace(".csv", "")
                            for x in map_files}

                for f in map_files:
                    try:
                        raw_name = os.path.basename(f).replace("Map_", "").replace(".csv", "")

                        # --- Skip unwanted variables (pressure, temperature, duplicates) ---
                        if raw_name in self._skip_raw_names:
                            continue
                        # Skip companion uncertainty maps here; they are loaded
                        # alongside their value map below (not plotted on their own).
                        if raw_name.startswith("u_") and raw_name[2:] in _raw_set:
                            continue

                        norm_ref = None   # normalisation ref applied (velocity only)
                        df = pd.read_csv(f, index_col=0)

                        df.index = pd.to_numeric(df.index, errors='coerce')
                        df = df[df.index.notnull()]
                        df.columns = pd.to_numeric(df.columns, errors='coerce')

                        df = df.reindex(sorted(df.columns), axis=1)

                        # --- NORMALIZATION INTERCEPT FOR VELOCITY ---
                        # All velocity components are normalised by the U0
                        # reference (V_noz, or the gap velocity for cylinder
                        # cases), fetched above into _u_max_ref[0]. If U0 was not
                        # available we fall back to the streamwise peak U_max.
                        if raw_name == "u_m_s":
                            ref = _u_max_ref[0] if _u_max_ref[0] else df.max().max()
                            if ref and ref > 0:
                                if not _u_max_ref[0]:
                                    _u_max_ref[0] = ref
                                df = df / ref
                                norm_ref = ref
                            pretty_name = r"$U/U_{0}$"
                            self.axis_labels[pretty_name] = r'$U/U_{0}$ [-]'
                        elif raw_name == "v_m_s":
                            ref = _u_max_ref[0] if _u_max_ref[0] else df.max().max()
                            if ref > 0:
                                df = df / ref
                                norm_ref = ref
                            pretty_name = r"$V/U_{0}$"
                            self.axis_labels[pretty_name] = r'$V/U_{0}$ [-]'
                        elif raw_name == "w_m_s":
                            ref = _u_max_ref[0] if _u_max_ref[0] else df.max().max()
                            if ref > 0:
                                df = df / ref
                                norm_ref = ref
                            pretty_name = r"$W/U_{0}$"
                            self.axis_labels[pretty_name] = r'$W/U_{0}$ [-]'
                        elif raw_name == "Vel_Mag_m_s":
                            ref = _u_max_ref[0] if _u_max_ref[0] else df.max().max()
                            if ref > 0:
                                df = df / ref
                                norm_ref = ref
                            pretty_name = r"$V_{mag}/U_{0}$"
                            self.axis_labels[pretty_name] = r'$V_{mag}/U_{0}$ [-]'
                        else:
                            pretty_name = self.name_mapping.get(raw_name, raw_name.replace("_", " "))

                        # Track U0-normalised velocity vars (and the U0 used) so the
                        # band can add the reference-uncertainty term in plot_profile.
                        if norm_ref is not None:
                            self._u0norm.add(pretty_name)
                            if norm_ref > 0:
                                self._u0_val = norm_ref

                        # Only keep the first occurrence of each pretty_name to prevent
                        # duplicates when the same variable appears in multiple CSV folders.
                        if pretty_name not in self.means_data:
                            self.means_data[pretty_name] = df
                            # Load the matching 1-sigma uncertainty band, if present.
                            _u_df = self._load_uncertainty_companion(folder, raw_name, norm_ref)
                            if _u_df is not None:
                                self.unc_data[pretty_name] = _u_df
                        if pretty_name not in var_names:
                            var_names.append(pretty_name)
                    except Exception as e:
                        pass

        if var_names:
            var_names.sort()
            # Build category map: pretty_name -> category
            self._cat_map = {n: self._get_category(n) for n in var_names}
            # All unique categories, sorted, plus "All" at front
            cats = ["All"] + sorted(set(self._cat_map.values()))
            self.combo_cat['values'] = cats
            self.combo_cat.set("All")
            # Store full variable list for filtering
            self._all_var_names = var_names
            self.combo_var['values'] = var_names
            self.combo_var.current(0)
            self.update_slices()

        # Task 12 — human-readable case name in window title
        case_name = os.path.basename(os.path.normpath(self.folder_path.get()))
        if _CL_OK:
            _cl_set_folder(self.folder_path.get())
            display = _cl_get_label(case_name)
        else:
            display = case_name
        self.root.title(f"1D Profiles — {display}")

    def filter_vars_by_category(self, event=None):
        """Task 6 — filter combo_var to show only variables in the selected category."""
        cat = self.combo_cat.get()
        all_names = getattr(self, '_all_var_names', list(self.means_data.keys()))
        if cat == "All":
            filtered = all_names
        else:
            filtered = [n for n in all_names if self._cat_map.get(n, "Miscellaneous") == cat]
        self.combo_var['values'] = filtered
        if filtered:
            self.combo_var.current(0)
        else:
            self.combo_var.set("")
        self.update_slices()

    def update_slices(self, event=None):
        var = self.combo_var.get()
        if var in self.means_data:
            df = self.means_data[var]
            slice_type = self.slice_var.get()
            if slice_type == "X":
                vals = [f"{x:.3f}" for x in df.columns]
                self.lbl_slice.config(text="Select X/B Slice:")
            else:
                vals = [f"{y:.3f}" for y in df.index]
                self.lbl_slice.config(text="Select Y/B Slice:")
            self.combo_slice['values'] = vals
            if vals: self.combo_slice.current(0)

    def _get_global_means(self):
        """Fetches the pre-calculated global mean frequency and Strouhal from the summary CSV."""
        if not self.folder_path.get(): return None, None
        summary_path = os.path.join(self.folder_path.get(), "Frequency_Results", "CSV", "Global_Frequency_Summary.csv")
        if os.path.exists(summary_path):
            try:
                df = pd.read_csv(summary_path)
                return df['Global_Mean_Freq_Hz'].iloc[0], df['Global_Mean_St'].iloc[0]
            except Exception: pass
        return None, None

    def _get_velocity_metrics(self):
        """Fetches the Potential Core Length and Decay Rate from the summary CSV."""
        summary_path = os.path.join(self.folder_path.get(), "1D_Profiles_Results", "CSV", "Global_Velocity_Summary.csv")
        if os.path.exists(summary_path):
            try:
                df = pd.read_csv(summary_path)
                return df['L_c_XB'].iloc[0], df['K_decay'].iloc[0]
            except Exception: pass
        return None, None

    def plot_profile(self):
        var = self.combo_var.get()
        slice_val_str = self.combo_slice.get()
        slice_type = self.slice_var.get()
        
        if not var or not slice_val_str or var not in self.means_data: return

        if self.current_fig:
            plt.close(self.current_fig)

        df = self.means_data[var]
        val = float(slice_val_str)
        fig, ax = plt.subplots(figsize=(8, 6))

        label = self.axis_labels.get(var, var)

        drew_band = False
        if slice_type == "X":
            col = min(df.columns, key=lambda c: abs(c - val))
            y_vals = df.index.values
            profile = df[col].values
            # Per-point error bars (replaces the shaded band): x = value 1-sigma
            # (companion u_ map), y = transverse probe-position 1-sigma (u_pos_y/B).
            u_val = None
            if var in self.unc_data:
                try:
                    udf = self.unc_data[var]
                    ucol = min(udf.columns, key=lambda c: abs(c - col))
                    u_val = udf[ucol].reindex(df.index).values.astype(float)
                    # Add the U0-reference uncertainty for normalised velocities.
                    if var in self._u0norm and self._u0_unc == self._u0_unc and self._u0_val:
                        u_val = np.sqrt(u_val ** 2 +
                                        (np.asarray(profile, float) * (self._u0_unc / self._u0_val)) ** 2)
                except Exception:
                    u_val = None
            u_pos_yb = (_U_POS_Y_MM / self._B_mm) if self._B_mm else None
            ax.errorbar(profile, y_vals, xerr=u_val, yerr=u_pos_yb,
                        fmt='-o', color='#d9534f', lw=2, ms=5,
                        capsize=2, elinewidth=0.7, ecolor='#d9534f', label=r'$\pm 1\sigma$')
            drew_band = True
            # Compact title: formula, X/B value — no "Vertical Profile" prefix
            ax.set_title(rf'{label}, $X/B = {col:.2f}$')
            ax.set_xlabel(label)
            ax.set_ylabel(r'$Y/B$ [-]')
            ax.axhline(0, color='black', lw=1.5, zorder=1)

            # Global Mean Lines for Vertical Profiles
            mean_freq, mean_st = self._get_global_means()
            if var == "1st Dominant Frequency (Hz)" and mean_freq and mean_freq > 0:
                ax.axvline(mean_freq, color='red', linestyle='--', lw=1.5, alpha=0.7)
                ax.text(mean_freq, 0.95, f"{mean_freq:.1f} Hz\n(St: {mean_st:.2f})",
                        transform=ax.get_xaxis_transform(), color='red',
                        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            elif var == "Strouhal Number (St)" and mean_st and mean_st > 0:
                ax.axvline(mean_st, color='red', linestyle='--', lw=1.5, alpha=0.7)
                ax.text(mean_st, 0.95, f"Mean St: {mean_st:.3f}",
                        transform=ax.get_xaxis_transform(), color='red',
                        ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        else:
            row = min(df.index, key=lambda r: abs(r - val))
            x_vals = df.columns.values
            profile = df.loc[row].values
            # Per-point error bars (replaces the shaded band): x = streamwise
            # probe-position 1-sigma (u_pos_x/B), y = value 1-sigma (companion u_ map).
            u_val = None
            if var in self.unc_data:
                try:
                    udf = self.unc_data[var]
                    urow = min(udf.index, key=lambda r: abs(r - row))
                    u_val = udf.loc[urow].reindex(df.columns).values.astype(float)
                    if var in self._u0norm and self._u0_unc == self._u0_unc and self._u0_val:
                        u_val = np.sqrt(u_val ** 2 +
                                        (np.asarray(profile, float) * (self._u0_unc / self._u0_val)) ** 2)
                except Exception:
                    u_val = None
            u_pos_xb = (_U_POS_X_MM / self._B_mm) if self._B_mm else None
            ax.errorbar(x_vals, profile, xerr=u_pos_xb, yerr=u_val,
                        fmt='-o', color='#007acc', lw=2, ms=5,
                        capsize=2, elinewidth=0.7, ecolor='#007acc', label=r'$\pm 1\sigma$')
            drew_band = True
            # Compact title: formula, Y/B value — no "Horizontal Profile" prefix.
            # For turbulence intensity at centerline (Y/B ≈ 0) use dedicated label.
            _ti_vars = {r"$I_u$ [%]", r"$I_v$ [%]", r"$I_w$ [%]", r"$I_{uvw}$ [%]"}
            _ti_symbol_map = {
                r"$I_u$ [%]":     r"$I_u$",
                r"$I_v$ [%]":     r"$I_v$",
                r"$I_w$ [%]":     r"$I_w$",
                r"$I_{uvw}$ [%]": r"$I_{uvw}$",
            }
            if var in _ti_vars and abs(row) < 0.5:
                sym = _ti_symbol_map.get(var, label)
                ax.set_title(rf'{sym} centerline ($Y/B \approx 0$)')
            else:
                ax.set_title(rf'{label}, $Y/B = {row:.2f}$')
            ax.set_xlabel(r'$X/B$ [-]')
            ax.set_ylabel(label)
            ax.axvline(0, color='black', lw=1.5, zorder=1)

            # Draw Literature Metrics for Velocity (U/U0 centerline)
            lc, k = self._get_velocity_metrics()
            if var == r"$U/U_{0}$" and lc is not None and not np.isnan(lc):
                trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes)
                ax.axvline(lc, color='purple', linestyle=':', lw=1.5, alpha=0.7)
                ax.text(0.95, 0.95, f"Potential Core $L_c$: {lc:.2f} $B$\nDecay Rate $K$: {k:.3f}",
                        transform=trans, color='purple', ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple', pad=3))

            # Global Mean Lines for Horizontal Profiles
            mean_freq, mean_st = self._get_global_means()
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

            if var == "1st Dominant Frequency (Hz)" and mean_freq and mean_freq > 0:
                ax.axhline(mean_freq, color='red', linestyle='--', lw=1.5, alpha=0.7)
                ax.text(0.02, mean_freq, f"{mean_freq:.1f} Hz (St: {mean_st:.2f})",
                        transform=trans, color='red',
                        ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            elif var == "Strouhal Number (St)" and mean_st and mean_st > 0:
                ax.axhline(mean_st, color='red', linestyle='--', lw=1.5, alpha=0.7)
                ax.text(0.02, mean_st, f"Mean St: {mean_st:.3f}",
                        transform=trans, color='red',
                        ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        if drew_band:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, ls='--', alpha=0.6)
        self.current_fig = fig
        self.display_plot(fig)

    def save_plot(self):
        if not self.current_fig:
            self.log_action("⚠️ Error: Please generate a plot first before attempting to save.")
            return
        
        save_dir = os.path.join(self.folder_path.get(), "1D_Profiles_Results")
        png_dir = os.path.join(save_dir, "PNG")
        pdf_dir = os.path.join(save_dir, "PDF")
        
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Build a filesystem-safe base name from the pretty variable name.
        # Strip LaTeX markup characters before applying generic sanitisation.
        import re as _re
        var_raw = self.combo_var.get()
        var_safe = _re.sub(r'[\$\{\}\\]', '', var_raw)   # remove LaTeX $, {, }, \
        var_safe = (var_safe
                    .replace(" ", "_")
                    .replace("/", "_over_")
                    .replace("(", "").replace(")", "")
                    .replace("%", "Pct")
                    .replace("^", "")
                    .replace("[", "").replace("]", ""))
        slice_type = self.slice_var.get()
        val = self.combo_slice.get()
        filename_base = f"Profile_{var_safe}_{slice_type}_{val}"
        
        png_path = os.path.join(png_dir, f"{filename_base}.png")
        self.current_fig.savefig(png_path, dpi=300, bbox_inches='tight')
        
        original_titles = [ax.get_title() for ax in self.current_fig.axes]
        for ax in self.current_fig.axes:
            ax.set_title("")

        pdf_path = os.path.join(pdf_dir, f"{filename_base}.pdf")
        self.current_fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

        for ax, original_title in zip(self.current_fig.axes, original_titles):
            ax.set_title(original_title)
        self.log_action(f"✅ Saved: {filename_base} (Exported to PNG & PDF)")

    def display_plot(self, fig):
        for widget in self.canvas_frame.winfo_children(): widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, self.canvas_frame).update()

    def setup_ui(self):
        ctrl = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=10)
        tk.Label(ctrl, text="Global Workspace:", bg="#e0e0e0", font=("Arial", 10, "bold")).pack(side="left", padx=10, pady=10)
        tk.Entry(ctrl, textvariable=self.folder_path, state="readonly", width=50).pack(side="left", padx=5)
        tk.Button(ctrl, text="Browse...", command=self.select_folder).pack(side="left", padx=5)

        tools = tk.Frame(self.root, bg="#f4f4f4", pady=10)
        tools.pack(fill="x", padx=10)

        # Task 6 — Category selector (Level 1)
        tk.Label(tools, text="Category:", bg="#f4f4f4").pack(side="left", padx=5)
        self.combo_cat = ttk.Combobox(tools, state="readonly", width=20)
        self.combo_cat['values'] = ["All", "Velocity", "Turbulence Intensity", "Frequency", "Miscellaneous"]
        self.combo_cat.set("All")
        self.combo_cat.pack(side="left", padx=5)
        self.combo_cat.bind('<<ComboboxSelected>>', self.filter_vars_by_category)

        tk.Label(tools, text="Variable:", bg="#f4f4f4").pack(side="left", padx=5)
        self.combo_var = ttk.Combobox(tools, state="readonly", width=30)
        self.combo_var.pack(side="left", padx=5)
        self.combo_var.bind('<<ComboboxSelected>>', self.update_slices)

        self.slice_var = tk.StringVar(value="X")
        tk.Radiobutton(tools, text="Slice of X/B", variable=self.slice_var, value="X", bg="#f4f4f4", command=self.update_slices).pack(side="left", padx=10)
        tk.Radiobutton(tools, text="Slice of Y/B", variable=self.slice_var, value="Y", bg="#f4f4f4", command=self.update_slices).pack(side="left", padx=10)

        self.lbl_slice = tk.Label(tools, text="Select Slice:", bg="#f4f4f4")
        self.lbl_slice.pack(side="left", padx=5)
        self.combo_slice = ttk.Combobox(tools, state="readonly", width=10)
        self.combo_slice.pack(side="left", padx=5)

        tk.Button(tools, text="PLOT GRAPH", command=self.plot_profile, bg="#17a2b8", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=20)
        tk.Button(tools, text="💾 SAVE GRAPH", command=self.save_plot, bg="#343a40", fg="white", font=("Arial", 10, "bold")).pack(side="right", padx=20)

        self.canvas_frame = tk.Frame(self.root, bg="white", relief="sunken", bd=2)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        log_frame = tk.Frame(self.root, bg="#f4f4f4")
        log_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(log_frame, text="Save History:", bg="#f4f4f4", font=("Arial", 9, "bold"), fg="#555").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=5, state="disabled", bg="#2b2b2b", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill="x")

# save_lateral_profiles_for_pla was removed.
# Vertical slice figures are now generated by
# maps_2D_visualizer.save_vertical_slice_profiles().

if __name__ == "__main__":
    root = tk.Tk()
    app = Profile1DApp(root)
    root.mainloop()