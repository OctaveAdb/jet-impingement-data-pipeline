# =============================================================================
# THERMAL_VISUALIZER
# =============================================================================
# Purpose:
#   Generates publication-quality thermal result figures from the Nusselt
#   number mapping files produced by thermal_processor. Five plot types are
#   available, each selectable by Reynolds strategy suffix (the six
#   Meas/Theo combinations exported by thermal_processor): (1) Q_loss vs dT
#   — scatter plot of the measured thermal loss ratio q''_loss / q''_in
#   against wall-to-ambient temperature difference dT for free-jet baseline
#   cases, with linear trend fits; (2) Global Nu vs Re — experimental global
#   Nusselt number Nu_bar plotted against Re_noz for all cases, with the
#   Hofmann theoretical curve overlaid; (3) Enhancement Factor eta vs Re —
#   the Nusselt enhancement ratio eta = Nu_cyl / Nu_free (baseline) and
#   eta = Nu_cyl / Nu_theo (theoretical) for Cyl cases only; (4) Spatial
#   Nu mapping — local Nu_loc_exp as a function of normalised lateral distance
#   X/B for all cases, comparing Top and Bottom rows of thermocouples, with
#   Hofmann theoretical Nu_loc_the for free-jet cases; (5) Spatial dT mapping
#   — wall-to-ambient temperature difference dT across the plate for all cases.
#   The script supports headless (batch) operation when called without a GUI.
#
# Inputs:
#   - Per-case Nusselt mapping files from thermal_processor:
#       thermal/results/Map_Temp_<case_id>[<suffix>].csv
#   - Heat loss calibration points:
#       thermal/results/Qloss_Calibration_Points[<suffix>].csv
#   - Theoretical Nu curve:
#       thermal/results/Theoretical_Nu_Curve[<suffix>].csv
#
# Outputs:
#   - Thermal result figures (PNG and PDF) per Reynolds strategy suffix:
#       thermal/results/Figures/PNG/Qloss_vs_dT_Fit[<suffix>].png/.pdf
#       thermal/results/Figures/PNG/Global_Nu_vs_Re[<suffix>].png/.pdf
#       thermal/results/Figures/PNG/Enhancement_Factor_vs_Re[<suffix>].png/.pdf
#       thermal/results/Figures/PNG/Spatial_Nusselt_Maps[<suffix>].png/.pdf
#       thermal/results/Figures/PNG/Spatial_DeltaT_Maps[<suffix>].png/.pdf
#
# Dependencies:
#   - thermal_processor (produces the Map_Temp CSV inputs and calibration files)
#   - Imported by usage.py as "Thermal Results Visualizer"
#
# Usage:
#   - Standalone: python thermal_visualizer.py  (opens Tkinter GUI)
#   - Headless:   ThermalVisualizerApp(root=None, default_folder=<path>)
#   - Via hub:    launched from usage.py as "Thermal Results Visualizer"
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import gc

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import matplotlib.pyplot as plt
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

# --- case_labels integration ---
try:
    import sys as _sys, os as _os
    _labels_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils')
    if _labels_path not in _sys.path:
        _sys.path.insert(0, _labels_path)
    from case_labels import get_label as _get_label, get_labels_dict as _get_labels_dict, set_main_folder as _set_main_folder, _cached_re_map as _re_map
    _LABELS_AVAILABLE = True
except Exception:
    _LABELS_AVAILABLE = False
    def _get_label(cid, **_): return cid
    def _get_labels_dict(cids, **_): return {c: c for c in cids}
    def _set_main_folder(_): pass
    _re_map = {}

# Thermocouple dT uncertainty (UNCERTAINTY_SPEC.md §3): two TCs -> sqrt(2)*u_TC.
try:
    from uncertainty import U_TC_C as _U_TC
except Exception:
    _U_TC = 0.1
_U_DT = (2.0 ** 0.5) * _U_TC

# Thermocouple position 1-sigma -> X error bar on the plate profiles (Delta_T and
# Nu_loc vs Y/B). The plate axis is Y/B, so normalise the placement tolerance by
# the slot height B (fixed rig geometry, 30 mm). This is INDEPENDENT of the
# heat-flux/dT value uncertainty (which carries no position-gradient term), so —
# unlike the velocity profiles — it does NOT double-count position. Value from
# config_uncertainty.xlsx (u_pos_tc_mm).
try:
    from uncertainty import U_POS_TC_MM as _TC_POS_MM
except Exception:
    _TC_POS_MM = 1.0
_B_MM_THERMAL = 30.0
_TC_POS_YB = _TC_POS_MM / _B_MM_THERMAL


def _load_thermal_uncertainty(map_dir):
    """Build {Case_ID: {'u_re','u_nu','u_eta','rel_nu'}} from
    Thermal_Global_Summary.csv (the absolute u_*_abs columns added in Task 3).
    Returns {} if the summary or its uncertainty columns are unavailable."""
    out = {}
    path = os.path.join(map_dir, "Thermal_Global_Summary.csv")
    if not os.path.exists(path):
        return out
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            cid = str(r.get('Case_ID', '')).strip()
            if not cid:
                continue
            nu = r.get('Global_Nu_Exp', np.nan)
            u_nu = r.get('u_Nu_abs', np.nan)
            rel_nu = (u_nu / nu) if (pd.notna(nu) and nu not in (0, np.nan) and pd.notna(u_nu)) else np.nan
            out[cid] = {'u_re': r.get('u_Re_abs', np.nan),
                        'u_nu': u_nu,
                        'u_eta': r.get('u_eta_abs', np.nan),
                        'rel_nu': rel_nu}
    except Exception:
        pass
    return out


def _strip_re_suffix(case_id):
    """Remove the trailing _MeasTheo[E1/E2] strategy suffix from a case id."""
    return re.sub(r'(_(?:Meas|Theo){2}(?:E[12])?)$', '', case_id, flags=re.IGNORECASE)


PLOT_TYPES = ['qloss', 'nu_re', 'eta_re', 'dt_map', 'nu_loc']
PLOT_META = {
    'qloss':  ("Qloss vs dT",                    "#e83e8c"),
    'nu_re':  ("Global Nu vs Re",                "#6f42c1"),
    'eta_re': ("Enhancement Factor η",           "#d9534f"),
    'dt_map': ("Spatial ΔT Map",                 "#28a745"),
    'nu_loc': ("Local Nusselt Number Mapping",   "#ff7f0e"),
}
FILENAMES = {
    'qloss':  "Qloss_vs_dT_Fit",
    'nu_re':  "Global_Nu_vs_Re",
    'eta_re': "Enhancement_Factor_vs_Re",
    'dt_map': "Spatial_DeltaT_Maps",
    'nu_loc': "Local_Nu_loc_Map",
}

class ThermalVisualizerApp:
    def __init__(self, root, default_folder=None):
        self.headless = (root is None)
        self.root = root

        self.base_marker_map = {}
        self.config_color_map = {}
        self.bottom_x_vals = {6, 12, 18, 24, 30, 35, 40, 45, 65, 85, 100, 105, 125, 155}

        self.png_paths = {pt: None for pt in PLOT_TYPES}
        self._current_img_obj = None   
        self.available_suffixes = ["Default"]

        if self.headless:
            self._folder = default_folder or ""
            self._run_headless()
        else:
            self.folder_path = tk.StringVar(value=default_folder or "")
            self.root.title("Thermal Results Visualizer")
            self.root.geometry("1100x750")
            self._setup_ui()
            if self.folder_path.get():
                self._build_mappings_from_folder(self.folder_path.get())

    def _run_headless(self):
        map_dir = os.path.join(self._folder, "thermal", "results")
        if not os.path.exists(map_dir): return
        fig_dir = os.path.join(map_dir, "Figures")
        os.makedirs(fig_dir, exist_ok=True)
        self._build_mappings_from_folder(self._folder)

        for suffix in self.available_suffixes:
            for pt in PLOT_TYPES:
                try:
                    fig = self._build_figure(pt, map_dir, suffix)
                    if fig:
                        self._save_figure(fig, fig_dir, pt, suffix)
                        fig.clear()
                        del fig
                        gc.collect()
                except Exception as e: print(f"ERROR on {pt} [{suffix}]: {e}")

    def _setup_ui(self):
        ctrl = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=10)
        tk.Label(ctrl, text="Workspace:", bg="#e0e0e0", font=("Arial", 10, "bold")).pack(side="left", padx=10, pady=8)
        tk.Entry(ctrl, textvariable=self.folder_path, state="readonly", width=55).pack(side="left", padx=5)
        tk.Button(ctrl, text="Browse...", command=self._select_folder).pack(side="left", padx=5)

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        sidebar = tk.Frame(main, width=220, bg="#f4f4f4", bd=1, relief="groove")
        sidebar.pack(side="left", fill="y", padx=(0, 8))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="1. Dataset Mode:", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(15, 2))
        self.suffix_combo = ttk.Combobox(sidebar, state="readonly")
        self.suffix_combo.pack(fill="x", padx=10, pady=5)
        self.suffix_combo.bind('<<ComboboxSelected>>', self._on_suffix_change)

        tk.Frame(sidebar, height=2, bg="#ccc").pack(fill="x", padx=10, pady=10)

        tk.Label(sidebar, text="2. Thermal Plots", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(pady=(5, 5))

        for pt in PLOT_TYPES:
            label, color = PLOT_META[pt]
            tk.Button(sidebar, text=label, bg=color, fg="white", font=("Arial", 10), height=2,
                      command=lambda p=pt: self._trigger_plot(p)).pack(fill="x", padx=10, pady=3)

        tk.Frame(sidebar, height=2, bg="#ccc").pack(fill="x", padx=10, pady=12)
        tk.Button(sidebar, text="💾  Save Current Mode", bg="#343a40", fg="white", font=("Arial", 10, "bold"), height=2,
                  command=self._save_all).pack(fill="x", padx=10, pady=3)

        self.status_lbl = tk.Label(sidebar, text="Ready.", bg="#f4f4f4", fg="#555", wraplength=175, justify="left")
        self.status_lbl.pack(pady=10, padx=8)

        self.canvas_frame = tk.Frame(main, bg="white", relief="sunken", bd=2)
        self.canvas_frame.pack(side="right", fill="both", expand=True)

        self.img_label = tk.Label(self.canvas_frame, text="[ Thermal Render Area ]\n\nClick a button on the left to generate a plot.", bg="white", fg="#888", font=("Arial", 13))
        self.img_label.pack(expand=True, fill="both")
        self.canvas_frame.bind("<Configure>", self._on_canvas_resize)
        self._active_pt = None

    def _on_canvas_resize(self, event=None):
        if self._active_pt and self.png_paths[self._active_pt]:
            self._display_png(self.png_paths[self._active_pt])

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select Main Data Folder")
        if folder and folder != self.folder_path.get():
            self.folder_path.set(folder)
            self.png_paths = {pt: None for pt in PLOT_TYPES}
            self._active_pt = None
            self.img_label.config(image='', text="[ Thermal Render Area ]\n\nSelect a folder to load modes.")
            self._build_mappings_from_folder(folder)

    def _on_suffix_change(self, event=None):
        if self._active_pt:
            self._trigger_plot(self._active_pt)

    def _trigger_plot(self, plot_type):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder): return
        map_dir = os.path.join(folder, "thermal", "results")
        if not os.path.exists(map_dir): return
        fig_dir = os.path.join(map_dir, "Figures")
        os.makedirs(fig_dir, exist_ok=True)

        target_suffix = self.suffix_combo.get()

        self.status_lbl.config(text=f"Generating {PLOT_META[plot_type][0]}...")
        self.root.update_idletasks()
        try:
            fig = self._build_figure(plot_type, map_dir, target_suffix)
            if fig is None:
                self.status_lbl.config(text="No data found for this mode.")
                return
            png_path = self._save_figure(fig, fig_dir, plot_type, target_suffix)
            fig.clear()
            del fig
            gc.collect()
            self.png_paths[plot_type] = png_path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot:\n{e}")
            self.status_lbl.config(text="Error.")
            return

        self._active_pt = plot_type
        self._display_png(self.png_paths[plot_type])
        display_name = target_suffix if target_suffix != "Default" else "Standard"
        self.status_lbl.config(text=f"Showing: {PLOT_META[plot_type][0]}\nMode: {display_name}")

    def _display_png(self, png_path):
        if not PIL_AVAILABLE or not png_path or not os.path.exists(png_path): return
        w = max(self.canvas_frame.winfo_width(),  100)
        h = max(self.canvas_frame.winfo_height(), 100)
        img = Image.open(png_path)
        img.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._current_img_obj = photo  
        self.img_label.config(image=photo, text="")

    def _save_all(self):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder): return
        map_dir = os.path.join(folder, "thermal", "results")
        fig_dir = os.path.join(map_dir, "Figures")
        os.makedirs(fig_dir, exist_ok=True)

        target_suffix = self.suffix_combo.get()
        count = 0
        for pt in PLOT_TYPES:
            self.status_lbl.config(text=f"Saving {PLOT_META[pt][0]}...")
            self.root.update_idletasks()
            try:
                fig = self._build_figure(pt, map_dir, target_suffix)
                if fig:
                    self.png_paths[pt] = self._save_figure(fig, fig_dir, pt, target_suffix)
                    fig.clear()
                    del fig
                    gc.collect()
                    count += 1
            except Exception as e: print(f"Error saving {pt}: {e}")
        self.status_lbl.config(text=f"✓ {count} plots saved.", fg="green")
        messagebox.showinfo("Success", f"{count} plots saved to:\n{fig_dir}")

    # --- DYNAMIC REGEX & SUFFIX PARSING ---
    def _get_config_and_base(self, case_id):
        suffix_match = re.search(r'(_(?:Meas|Theo){2}(?:E[12])?)$', case_id, re.IGNORECASE)
        suffix = suffix_match.group(1) if suffix_match else ""
        clean_case = case_id.replace(suffix, "") if suffix else case_id

        match = re.match(r'(?i)^(Cyl\d+|Free)(\d+)PaPla$', clean_case.strip())
        if match:
            has_cyl = bool(re.search(r'(?i)Cyl', match.group(1)))
            base_id = match.group(2) + "Pa"
            config = "Cyl" if has_cyl else "Free"
            return config, base_id
        
        cleaned_id = re.sub(r'(?i)Cyl(10|12|14|15|16|18|20|25|30|35|40|50)', 'Cyl_', clean_case)
        match_pa = re.search(r'(\d+)\s*Pa', cleaned_id, re.IGNORECASE)
        base_id = match_pa.group(1) + "Pa" if match_pa else "Unknown"
        has_cyl = bool(re.search(r'(?i)Cyl', clean_case))
        config = "Cyl" if has_cyl else "Free"
        return config, base_id

    def _build_mappings_from_folder(self, main_folder):
        map_dir = os.path.join(main_folder, "thermal", "results")
        if not os.path.exists(map_dir): return
        map_files = glob.glob(os.path.join(map_dir, "Map_Temp_*.csv"))
        
        base_ids = set()
        suffixes = set()

        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            
            suffix_match = re.search(r'(_(?:Meas|Theo){2}(?:E[12])?)$', case_id, re.IGNORECASE)
            file_suffix = suffix_match.group(1) if suffix_match else "Default"
            suffixes.add(file_suffix)

            config, base_id = self._get_config_and_base(case_id)
            base_ids.add(base_id)
            
        marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']
        def sort_key(s):
            m = re.search(r'\d+', s)
            return int(m.group()) if m else 0
        sorted_bases = sorted(list(base_ids), key=sort_key)
        self.base_marker_map = {b: marker_list[i % len(marker_list)] for i, b in enumerate(sorted_bases)}

        self.available_suffixes = sorted(list(suffixes))
        if self.available_suffixes:
            self.suffix_combo['values'] = self.available_suffixes
            if self.suffix_combo.get() not in self.available_suffixes:
                self.suffix_combo.current(0)
        else:
            self.suffix_combo['values'] = ["Default"]
            self.suffix_combo.current(0)

    def _get_config_color(self, config):
        if config == 'Free': return '#d62728'
        if config not in self.config_color_map:
            palette = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#17becf']
            self.config_color_map[config] = palette[len(self.config_color_map) % len(palette)]
        return self.config_color_map[config]

    def _save_figure(self, fig, fig_dir, plot_type, target_suffix):
        png_dir, pdf_dir = os.path.join(fig_dir, "PNG"), os.path.join(fig_dir, "PDF")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        actual_suffix = "" if target_suffix == "Default" else target_suffix
        name = FILENAMES[plot_type] + actual_suffix

        png_path = os.path.join(png_dir, f"{name}.png")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')

        original_titles = [ax.get_title() for ax in fig.axes]
        for ax in fig.axes: ax.set_title("")
        fig.savefig(os.path.join(pdf_dir, f"{name}.pdf"), format='pdf', bbox_inches='tight')
        for ax, t in zip(fig.axes, original_titles): ax.set_title(t)
        return png_path

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE BUILDERS 
    # ══════════════════════════════════════════════════════════════════════
    def _build_figure(self, plot_type, map_dir, target_suffix):
        builders = {
            'qloss':  self._fig_qloss,
            'nu_re':  self._fig_nu_re,
            'eta_re': self._fig_eta_re,
            'dt_map': self._fig_dt_map,
            'nu_loc': self._fig_nu_loc,
        }
        return builders[plot_type](map_dir, target_suffix)

    def _filter_map_files(self, map_dir, target_suffix):
        all_files = glob.glob(os.path.join(map_dir, "Map_Temp_*.csv"))
        filtered = []
        for f in all_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            suffix_match = re.search(r'(_(?:Meas|Theo){2}(?:E[12])?)$', case_id, re.IGNORECASE)
            file_suffix = suffix_match.group(1) if suffix_match else "Default"
            if file_suffix == target_suffix:
                filtered.append(f)
        return filtered

    def _fig_qloss(self, map_dir, target_suffix):
        # Heat-loss constant: the bare-jet GLOBAL raw Nusselt number (full
        # electrical flux, NO loss applied) plotted against the Hofmann slot
        # correlation across the free-jet set-points. The single slope through the
        # origin r = Nu_Hofmann / Nu_bare,raw is the constant heat-RETENTION factor
        # (constant loss fraction = 1 - r) applied to every case. A tight line
        # (high R^2) is the direct justification that the loss fraction is constant.
        actual_suffix = "" if target_suffix == "Default" else target_suffix
        calib_file = os.path.join(map_dir, f"Qloss_Calibration_Points{actual_suffix}.csv")
        if not os.path.exists(calib_file):
            calib_files = glob.glob(os.path.join(map_dir, "Qloss_Calibration_Points*.csv"))
            if not calib_files: return None
            calib_file = calib_files[0]

        df = pd.read_csv(calib_file)
        if not {'Nu_bare_raw_global', 'Nu_Hofmann_global'}.issubset(df.columns):
            return None
        df = df.dropna(subset=['Nu_bare_raw_global', 'Nu_Hofmann_global'])
        if df.empty: return None

        x = df['Nu_bare_raw_global'].values.astype(float)
        y = df['Nu_Hofmann_global'].values.astype(float)
        nu_loss = (float(df['Nu_loss_const'].iloc[0]) if 'Nu_loss_const' in df.columns
                   else float((x - y).mean()))
        r2 = (float(df['Nu_loss_R2'].iloc[0]) if 'Nu_loss_R2' in df.columns else float('nan'))

        fig = Figure(figsize=(8, 6.5), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color='#1f77b4', s=60, zorder=3, label='Bare-jet cases (per set-point)')
        lo = float(min(x.min(), y.min())) * 0.96
        hi = float(x.max()) * 1.02
        x_line = np.array([lo, hi])
        ax.plot(x_line, x_line - nu_loss, color='#d62728', lw=1.8, zorder=2,
                label=(r'$\overline{Nu}_{\mathrm{Hof}} = \overline{Nu}_{\mathrm{bare,raw}} - %.1f$'
                       r'  ($R^2=%.3f$)' % (nu_loss, r2)))
        ax.set_xlabel(r"$\overline{Nu}_{\mathrm{bare,\,raw}}$  (full $q''_{in}$, no loss applied) [-]")
        ax.set_ylabel(r"$\overline{Nu}_{\mathrm{Hofmann}}$ [-]")
        ax.set_title(r"Heat-loss constant: bare-jet $\overline{Nu}$ vs Hofmann (additive offset $\overline{Nu}_{loss}$)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left')
        return fig

    # --- Change 2: _fig_nu_re --- three named curves ---
    def _fig_nu_re(self, map_dir, target_suffix):
        map_files = self._filter_map_files(map_dir, target_suffix)
        if not map_files: return None

        unc = _load_thermal_uncertainty(map_dir)
        records = []
        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            df = pd.read_csv(f)
            config, base_id = self._get_config_and_base(case_id)
            if 'Re_noz' in df.columns and 'Nu' in df.columns:
                if pd.notna(df['Re_noz'].iloc[0]) and pd.notna(df['Nu'].iloc[0]):
                    records.append({'Config': config, 'Base_ID': base_id,
                                    'Case_ID': _strip_re_suffix(case_id),
                                    'Re': df['Re_noz'].iloc[0], 'Nu': df['Nu'].iloc[0]})
        if not records: return None

        df_plot = pd.DataFrame(records).sort_values(by='Re')
        fig = Figure(figsize=(9, 6), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Fixed display names for the three curve types
        config_display = {'Free': 'Free Jet (baseline)', 'Cyl': 'Cylinder 12mm'}

        for config, c_group in df_plot.groupby('Config'):
            c = self._get_config_color(config)
            ls = '--' if 'Free' in config else '-'
            c_group_sorted = c_group.sort_values(by='Re')
            trend_label = config_display.get(config, config)
            ax.plot(c_group_sorted['Re'], c_group_sorted['Nu'], linestyle=ls, color=c,
                    linewidth=2, alpha=0.6, label=trend_label)
            for _, row in c_group.iterrows():
                m = self.base_marker_map.get(row['Base_ID'], 'o')
                _u = unc.get(row['Case_ID'], {})
                _ure, _unu = _u.get('u_re', np.nan), _u.get('u_nu', np.nan)
                ax.errorbar(row['Re'], row['Nu'],
                            xerr=(_ure if pd.notna(_ure) else None),
                            yerr=(_unu if pd.notna(_unu) else None),
                            marker=m, color=c, markersize=9, linestyle='None',
                            capsize=3, elinewidth=0.8, label=None)

        actual_suffix = "" if target_suffix == "Default" else target_suffix
        theo_curve_file = os.path.join(map_dir, f"Theoretical_Nu_Curve{actual_suffix}.csv")

        if os.path.exists(theo_curve_file):
            df_theo = pd.read_csv(theo_curve_file)
            ax.plot(df_theo['Re'], df_theo['Nu_the'], linestyle=':', color='black',
                    linewidth=2.5, alpha=0.4, zorder=1)
            ax.plot([], [], linestyle=':', color='black', linewidth=2.5, alpha=0.8,
                    label="Hofmann (theory)")

        ax.set_xlabel(r"$Re_{noz}$ [-]")
        ax.set_ylabel(r"$\overline{Nu}$ [-]")
        ax.set_title(r"$\overline{Nu}$ vs $Re_{noz}$")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        return fig

    # --- Change 1: _fig_eta_re --- single eta_baseline line only ---
    def _fig_eta_re(self, map_dir, target_suffix):
        map_files = self._filter_map_files(map_dir, target_suffix)
        if not map_files: return None

        unc = _load_thermal_uncertainty(map_dir)
        records = []
        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            df = pd.read_csv(f)
            config, base_id = self._get_config_and_base(case_id)

            if 'Cyl' in config and 'Re_noz' in df.columns:
                re = df['Re_noz'].iloc[0]
                eta_base = df['eta_baseline'].iloc[0] if 'eta_baseline' in df.columns else np.nan

                if pd.notna(re) and pd.notna(eta_base):
                    records.append({
                        'Config': config, 'Base_ID': base_id, 'Re': re,
                        'Case_ID': _strip_re_suffix(case_id),
                        'eta_base': eta_base,
                    })

        if not records: return None

        df_plot = pd.DataFrame(records).sort_values(by='Re')
        fig = Figure(figsize=(9, 6), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        c = self._get_config_color('Cyl')

        # Single trend line: eta = Nu_cyl / Nu_free_baseline
        ax.plot(df_plot['Re'], df_plot['eta_base'], linestyle='-', color=c, linewidth=2,
                alpha=0.6, label=r'$\eta = Nu_{cyl}/Nu_{free}$')

        # Scatter points (unlabelled) with 1-sigma eta error bars
        for _, row in df_plot.iterrows():
            m = self.base_marker_map.get(row['Base_ID'], 'o')
            if pd.notna(row['eta_base']):
                _u = unc.get(row['Case_ID'], {})
                _ue = _u.get('u_eta', np.nan)
                _ure = _u.get('u_re', np.nan)
                ax.errorbar(row['Re'], row['eta_base'],
                            xerr=(_ure if pd.notna(_ure) else None),
                            yerr=(_ue if pd.notna(_ue) else None),
                            marker=m, color=c, markersize=9, linestyle='None',
                            capsize=3, elinewidth=0.8, label=None)

        ax.set_xlabel(r'$Re_{noz}$ [-]')
        ax.set_ylabel(r'$\eta$ [-]')
        ax.set_title(r'$\eta$ vs $Re_{noz}$')
        eta_vals = [r['eta_base'] for r in records if not np.isnan(r['eta_base'])]
        if eta_vals:
            eta_max = max(eta_vals)
            ax.set_ylim(0, eta_max * 1.1)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
        return fig

    # --- Change 3: _fig_dt_map --- 1 case = 1 line, averaged over X/B, case_labels labels ---
    def _fig_dt_map(self, map_dir, target_suffix):
        map_files = self._filter_map_files(map_dir, target_suffix)
        if not map_files: return None

        fig = Figure(figsize=(11, 6), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Collect all case IDs for context-aware labelling
        all_case_ids = [
            os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            for f in map_files
        ]

        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            config, base_id = self._get_config_and_base(case_id)
            df = pd.read_csv(f)
            if 'X/B' not in df.columns or 'Delta_T_K' not in df.columns: continue

            # Robust Filter: Drop any rows with NaN
            df_clean = df.dropna(subset=['X/B', 'Delta_T_K']).copy()
            if df_clean.empty: continue

            c, m = self._get_config_color(config), self.base_marker_map.get(base_id, 'o')
            ls, alpha = ('--', 0.7) if 'Free' in config else ('-', 0.9)

            # Average Delta_T_K over all thermocouples at each X/B (merges Top & Bottom)
            df_avg = df_clean.groupby('X/B', as_index=False)['Delta_T_K'].mean().sort_values('X/B')

            # Use case_labels for clean label
            curve_label = _get_label(case_id, context_cases=all_case_ids)

            # Error bars: y = constant +/-sqrt(2)*u_TC on dT (UNCERTAINTY_SPEC.md §3);
            # x = thermocouple position 1-sigma (u_pos_tc/B) on the Y/B plate axis.
            ax.errorbar(df_avg['X/B'], df_avg['Delta_T_K'], xerr=_TC_POS_YB, yerr=_U_DT,
                        marker=m, linestyle=ls, color=c,
                        alpha=alpha, markersize=6, capsize=2, elinewidth=0.7,
                        label=curve_label)

        ax.set_xlabel(r'$Y/B$ [-]')
        ax.set_ylabel(r'$\Delta T$ [K]')
        ax.set_title(r'$\Delta T$ Mapping')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Legend BELOW the graph, two columns: left = cylinder cases (blue),
        # right = free-jet cases (red). matplotlib fills columns top-to-bottom
        # (column-major), so concatenating [cyl..., free...] puts each config in its
        # own column; pad the shorter group so the split lands cleanly.
        import re as _re
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        def _re_key(lbl):
            m = _re.search(r'(\d{3,6})', lbl.replace(' ', ''))
            return int(m.group(1)) if m else 0
        def _short(lbl):                       # drop the config prefix; keep the Re tag
            return lbl.split(', ')[-1] if ', ' in lbl else lbl
        cyl  = sorted([(_short(l), h) for l, h in by_label.items() if 'Free' not in l], key=lambda x: _re_key(x[0]))
        free = sorted([(_short(l), h) for l, h in by_label.items() if 'Free' in l],     key=lambda x: _re_key(x[0]))
        if cyl or free:
            blank = Line2D([], [], linestyle='none', marker='')
            cyl_col  = [('Cylinder 12 mm', blank)] + cyl
            free_col = [('Free jet', blank)]       + free
            n = max(len(cyl_col), len(free_col))
            cyl_col  += [('', blank)] * (n - len(cyl_col))
            free_col += [('', blank)] * (n - len(free_col))
            ordered = cyl_col + free_col
            leg = ax.legend([h for _, h in ordered], [l for l, _ in ordered],
                            loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
                            fontsize=8, frameon=False, columnspacing=6,
                            handletextpad=0.6, labelspacing=0.4)
            # bold the two column headers
            for t in leg.get_texts():
                if t.get_text() in ('Cylinder 12 mm', 'Free jet'):
                    t.set_fontweight('bold')
        return fig

    def _export_nu_loc_csv(self, map_dir, target_suffix):
        map_files = self._filter_map_files(map_dir, target_suffix)
        if not map_files: return
        rows = []
        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            df = pd.read_csv(f)
            if 'X/B' not in df.columns or 'Nu_loc_exp' not in df.columns: continue
            df_clean = df.dropna(subset=['X/B', 'Nu_loc_exp']).copy()
            if df_clean.empty: continue
            has_the = 'Nu_loc_the' in df_clean.columns
            for xb, grp in df_clean.groupby('X/B'):
                nu_exp_mean = grp['Nu_loc_exp'].mean()
                nu_the_mean = grp['Nu_loc_the'].mean() if has_the else np.nan
                rows.append({'Case_ID': case_id, 'Y/B': xb, 'Nu_loc_exp': nu_exp_mean, 'Nu_loc_the': nu_the_mean})
        if not rows: return
        actual_suffix = "" if target_suffix == "Default" else target_suffix
        out_path = os.path.join(map_dir, f"Local_Nu_loc_Map{actual_suffix}.csv")
        pd.DataFrame(rows).sort_values(by=['Case_ID', 'Y/B']).to_csv(out_path, index=False)

    def _fig_nu_loc(self, map_dir, target_suffix):
        # Prime the Re cache so labels use Re values instead of Pa values
        main_folder = os.path.dirname(os.path.dirname(map_dir))
        _set_main_folder(main_folder)

        map_files = self._filter_map_files(map_dir, target_suffix)
        if not map_files: return None

        unc = _load_thermal_uncertainty(map_dir)

        # Collect all case IDs for context-aware labelling
        all_case_ids = [
            os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            for f in map_files
        ]

        fig = Figure(figsize=(11, 6), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            config, base_id = self._get_config_and_base(case_id)
            df = pd.read_csv(f)
            if 'X/B' not in df.columns or 'Nu_loc_exp' not in df.columns: continue
            df_clean = df.dropna(subset=['X/B', 'Nu_loc_exp']).copy()
            if df_clean.empty: continue

            c = self._get_config_color(config)
            m = self.base_marker_map.get(base_id, 'o')
            ls = '--' if 'Free' in config else '-'
            alpha = 0.7 if 'Free' in config else 0.9

            curve_label = _get_label(case_id, context_cases=all_case_ids)
            df_avg = df_clean.groupby('X/B', as_index=False)['Nu_loc_exp'].mean().sort_values('X/B')
            # Per-point error bars (replaces the shaded band): y = case-relative Nu
            # 1-sigma applied to each local Nu (q'' term dominant/constant; dT
            # variation absorbed in the mean); x = thermocouple position 1-sigma
            # (u_pos_tc/B) on the Y/B plate axis.
            _rel = unc.get(_strip_re_suffix(case_id), {}).get('rel_nu', np.nan)
            _nu = df_avg['Nu_loc_exp'].values
            _ynu = (_nu * _rel) if (pd.notna(_rel) and _rel > 0) else None
            ax.errorbar(df_avg['X/B'], _nu, xerr=_TC_POS_YB, yerr=_ynu,
                        marker=m, linestyle=ls, color=c, alpha=alpha, markersize=6,
                        capsize=2, elinewidth=0.7, ecolor=c, label=curve_label)

        ax.set_xlabel(r"$Y/B$ [-]")
        ax.set_ylabel(r"$Nu_{loc}$ [-]")
        ax.set_title(r"$Nu_{loc}$ along the Plate ($Y/B$)")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Legend BELOW the graph, two columns: left = cylinder cases (blue),
        # right = free-jet cases (red). Same column-major scheme as the dT map.
        import re as _re
        from matplotlib.lines import Line2D
        handles_all, labels_all = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_all, handles_all))
        def _re_key(lbl):
            m = _re.search(r'(\d{3,6})', lbl.replace(' ', ''))
            return int(m.group(1)) if m else 0
        def _short(lbl):                       # drop the config prefix; keep the Re tag
            return lbl.split(', ')[-1] if ', ' in lbl else lbl
        cyl  = sorted([(_short(l), h) for l, h in by_label.items() if 'Free' not in l], key=lambda x: _re_key(x[0]))
        free = sorted([(_short(l), h) for l, h in by_label.items() if 'Free' in l],     key=lambda x: _re_key(x[0]))
        if cyl or free:
            blank = Line2D([], [], linestyle='none', marker='')
            cyl_col  = [('Cylinder 12 mm', blank)] + cyl
            free_col = [('Free jet', blank)]       + free
            n = max(len(cyl_col), len(free_col))
            cyl_col  += [('', blank)] * (n - len(cyl_col))
            free_col += [('', blank)] * (n - len(free_col))
            ordered = cyl_col + free_col
            leg = ax.legend([h for _, h in ordered], [l for l, _ in ordered],
                            loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
                            fontsize=8, frameon=False, columnspacing=6,
                            handletextpad=0.6, labelspacing=0.4)
            for t in leg.get_texts():
                if t.get_text() in ('Cylinder 12 mm', 'Free jet'):
                    t.set_fontweight('bold')

        self._export_nu_loc_csv(map_dir, target_suffix)
        return fig


def save_all_thermal_figures(main_folder: str) -> tuple:
    """
    Headless batch figure generation for all thermal plots.
    Reads the processed thermal CSVs from thermal/results/ and saves PNG+PDF
    for all available plot types and all detected Reynolds-strategy suffixes,
    without opening a GUI window.

    Returns (success: bool, message: str).
    """
    try:
        map_dir = os.path.join(main_folder, "thermal", "results")
        if not os.path.exists(map_dir):
            return False, f"Thermal results directory not found: {map_dir}"

        fig_dir = os.path.join(map_dir, "Figures")
        os.makedirs(fig_dir, exist_ok=True)

        # Build a minimal app instance without triggering the GUI or headless paths.
        # We use object.__new__ to bypass __init__, then manually initialise only
        # the attributes that _build_mappings_from_folder and the figure builders need.
        app = object.__new__(ThermalVisualizerApp)
        app.headless = True
        app.root = None
        app.base_marker_map = {}
        app.config_color_map = {}
        app.bottom_x_vals = {6, 12, 18, 24, 30, 35, 40, 45, 65, 85, 100, 105, 125, 155}
        app.png_paths = {pt: None for pt in PLOT_TYPES}
        app.available_suffixes = ["Default"]
        app._current_img_obj = None

        # Provide a stub for suffix_combo so _build_mappings_from_folder can write to
        # it safely without a real Tkinter widget.
        class _SuffixStub:
            def __init__(self):
                self._values = []
                self._current = 0
            def __setitem__(self, key, val):
                if key == 'values':
                    self._values = list(val)
            def __getitem__(self, key):
                return self._values if key == 'values' else None
            def get(self):
                return self._values[self._current] if self._values else "Default"
            def current(self, idx=None):
                if idx is not None:
                    self._current = idx

        app.suffix_combo = _SuffixStub()

        # Populate base_marker_map and available_suffixes from the results directory.
        app._build_mappings_from_folder(main_folder)

        # Use whichever suffixes were detected; fall back to ["Default"] if none found.
        suffixes = app.available_suffixes if app.available_suffixes else ["Default"]

        generated = []
        for suffix in suffixes:
            for plot_type in PLOT_TYPES:
                try:
                    fig = app._build_figure(plot_type, map_dir, suffix)
                    if fig is not None:
                        app._save_figure(fig, fig_dir, plot_type, suffix)
                        fig.clear()
                        del fig
                        gc.collect()
                        generated.append(f"{plot_type}[{suffix}]")
                except Exception as e:
                    print(f"  [thermal figures] skipped {plot_type}[{suffix}]: {e}")

        if generated:
            return True, f"Saved {len(generated)} thermal figure(s): {', '.join(generated)}"
        else:
            return False, "No thermal figures could be generated (check that thermal/results/ contains Map_Temp_*.csv files)."

    except Exception as e:
        return False, f"Thermal figure save failed: {e}"


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    root = tk.Tk()
    app = ThermalVisualizerApp(root)
    root.mainloop()