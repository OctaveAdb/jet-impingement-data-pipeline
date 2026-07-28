# =============================================================================
# PLOT_CALIBRATION_COMPARAISON
# =============================================================================
# Purpose:
#   Quantifies and visualises the impact of thermocouple calibration on the
#   measured wall-to-ambient temperature difference dT for all experimental
#   cases. The script loads the linear calibration coefficients from
#   calibration_TC.csv (slope and intercept per TC channel, derived from
#   Ebro reference calibration), then reads each Map_Temp_<case>.csv file
#   produced by thermal_processor. For each thermocouple reading, it inverts
#   the calibration transform to recover the raw (uncalibrated) temperature
#   and computes both the calibrated and uncalibrated dT. Both curves are
#   plotted against the normalised lateral position X/B on a shared axis for
#   all cases and configurations, allowing a direct assessment of the
#   systematic correction applied by the calibration procedure. The figure
#   and a companion data table are exported to the thermal/figures/ directory.
#
# Inputs:
#   - Thermocouple calibration coefficients:
#       thermal/raw_inputs/calibration_TC.csv (tab-separated, Ebro_True row)
#   - Per-case Nusselt/temperature mapping files (any matching):
#       **/Map_Temp_<case>.csv  (searched recursively under workspace)
#
# Outputs:
#   - Calibration impact comparison figure:
#       thermal/figures/PNG/Calib_Impact_AllCases.png
#       thermal/figures/PDF/Calib_Impact_AllCases.pdf
#   - Data table of calibrated and uncalibrated dT per case:
#       thermal/figures/CSV/Calib_Impact_AllCases.csv
#       thermal/figures/XLSX/Calib_Impact_AllCases.xlsx
#
# Dependencies:
#   - thermal_processor (produces the Map_Temp CSV inputs)
#   - Not called by pipeline.py; launched manually
#
# Usage:
#   - Standalone: python plot_calibration_comparaison.py  (opens Tkinter GUI)
#   - Via hub:    not integrated; run directly or via file browser
# =============================================================================

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import sys as _sys
from tkinter import filedialog, messagebox

# Thermocouple dT uncertainty (UNCERTAINTY_SPEC.md §3): two independent TCs ->
# u_dT = sqrt(2)*u_TC (constant ~0.141 K). Shown as error bars on the calibrated dT.
try:
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
    from uncertainty import U_TC_C as _U_TC
except Exception:
    _U_TC = 0.1
_U_DT = (2.0 ** 0.5) * _U_TC

# --- CONFIG LOADER ---
def _load_cfg(filename):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    _df = pd.read_excel(_p, index_col='parameter')
    return _df['value'] if 'value' in _df.columns else _df.iloc[:, 0]

try:
    _cfg_style = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_plot_styles.xlsx'))
    _cfg_cases = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cases.xlsx'))

    # Build dynamic cylinder diameter regex from config_cases.csv
    _cyl_patterns = _cfg_cases[_cfg_cases['has_cylinder'].astype(str).str.lower().isin(['true', '1', 'yes'])]['case_pattern'].tolist()
    _cyl_diams = '|'.join(str(p).replace('Cyl', '') for p in _cyl_patterns if str(p).startswith('Cyl') and str(p).replace('Cyl', '').isdigit())
    if not _cyl_diams:
        # Cylinder diameters — update config_cases.csv to change
        _cyl_diams = '10|12|14|15|16|18|20|25|30|35|40|50'
except Exception:
    _cfg_style = None
    _cfg_cases = None
    # Cylinder diameters — update config_cases.csv to change
    _cyl_diams = '10|12|14|15|16|18|20|25|30|35|40|50'

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

class CalibrationComparer:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibration Impact Viewer")
        self.root.geometry("400x200")

        self.main_folder = ""
        self.calib_dict = {}

        self.base_marker_map = {}
        self.config_color_map = {}

        tk.Button(self.root, text="Select Workspace Folder", command=self.select_folder,
                  font=("Arial", 10), height=2).pack(fill="x", padx=20, pady=(20, 5))

        tk.Button(self.root, text="GENERATE COMPARISON GRAPH", command=self.run,
                  bg="#17a2b8", fg="white", font=("Arial", 12, "bold"), height=2).pack(fill="x", padx=20, pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Main Data Folder")
        if folder:
            self.main_folder = folder

    def get_config_color(self, config_name, idx=0):
        """Returns color hex for a config label (Free/Cyl) from config_plot_styles.csv."""
        if _cfg_style is not None:
            prefix = 'Cyl' if ('Cyl' in config_name or 'cyl' in config_name) else 'Free'
            rows = _cfg_style[_cfg_style['case_prefix'] == prefix]
            if not rows.empty:
                return str(rows.iloc[idx % len(rows)]['color_hex'])
        # Fallback: original hardcoded logic
        if config_name == 'Free':
            return '#d62728'
        if config_name not in self.config_color_map:
            palette = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#17becf']
            self.config_color_map[config_name] = palette[len(self.config_color_map) % len(palette)]
        return self.config_color_map[config_name]

    def get_config_and_base(self, case_id):
        # Cylinder diameter list derived from config_cases.csv (fallback: hardcoded list)
        cleaned_id = re.sub(r'(?i)Cyl(' + _cyl_diams + r')', 'Cyl_', case_id)
        match_pa = re.search(r'(\d+)\s*Pa', cleaned_id, re.IGNORECASE)
        base_id = match_pa.group(1) + "Pa" if match_pa else "Unknown"
        has_cyl = bool(re.search(r'Cyl', case_id, re.IGNORECASE))
        config = "Cyl" if has_cyl else "Free"
        return config, base_id

    def build_global_mappings(self, map_files):
        base_ids = set()
        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            config, base_id = self.get_config_and_base(case_id)
            base_ids.add(base_id)
            
        marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']
        self.base_marker_map = {b: marker_list[i % len(marker_list)] for i, b in enumerate(sorted(base_ids))}

    def load_calibration(self):
        calib_file = os.path.join(self.main_folder, "thermal", "raw_inputs", "calibration_TC.csv")
        if not os.path.exists(calib_file):
            calib_file = os.path.join(self.main_folder, "thermal", "calibration_TC.csv")
        if not os.path.exists(calib_file):
            messagebox.showerror("Error", "calibration_TC.csv not found.")
            return False

        try:
            rows = []
            with open(calib_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip().strip('"')
                    if line:
                        rows.append(line.split('\t'))

            if not rows:
                messagebox.showerror("Error", "calibration_TC.csv est vide.")
                return False

            df_calib = pd.DataFrame(rows[1:], columns=rows[0])
            for col in df_calib.columns[1:]:
                df_calib[col] = df_calib[col].str.replace(',', '.', regex=False)

            tc_names_upper = df_calib.iloc[:, 0].astype(str).str.upper().values
            if 'EBRO_TRUE' not in tc_names_upper:
                messagebox.showerror("Error", "Ligne 'Ebro_True' introuvable dans calibration_TC.csv.")
                return False

            ebro_idx = np.where(tc_names_upper == 'EBRO_TRUE')[0][0]
            y_true = df_calib.iloc[ebro_idx, 1:].astype(float).values

            for index, row in df_calib.iterrows():
                tc_name = str(row.iloc[0]).strip().upper()
                if 'TC' in tc_name and tc_name != 'EBRO_TRUE':
                    match = re.search(r'TC(\d+)', tc_name)
                    if match:
                        tc_num = match.group(1)
                        x_measured = row.iloc[1:].astype(float).values
                        if len(x_measured) > 1 and not np.isnan(x_measured).all():
                            m, b = np.polyfit(x_measured, y_true, 1)
                            self.calib_dict[tc_num] = {'slope': m, 'intercept': b}
            return True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration: {e}")
            return False

    def run(self):
        if not self.main_folder:
            messagebox.showerror("Error", "Please select a workspace folder first.")
            return
        if not self.load_calibration():
            return

        fig_dir = os.path.join(self.main_folder, "thermal", "figures")
        png_dir = os.path.join(fig_dir, "PNG")
        pdf_dir = os.path.join(fig_dir, "PDF")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        map_files = []
        for root_dir, dirs, files in os.walk(self.main_folder):
            map_files.extend([
                os.path.join(root_dir, f)
                for f in files
                if f.startswith("Map_Temp_") and f.endswith(".csv")
            ])

        if not map_files:
            messagebox.showwarning("No Data", "No Map_Temp_*.csv files found.")
            return

        self.build_global_mappings(map_files)

        fig, ax = plt.subplots(figsize=(12, 7))
        legend_seen = set()
        export_rows = []

        for f in map_files:
            case_id = os.path.basename(f).replace("Map_Temp_", "").replace(".csv", "")
            config, base_id = self.get_config_and_base(case_id)
            df = pd.read_csv(f)

            if 'X/B' not in df.columns or 'Temp_C' not in df.columns or 'T_amb_C' not in df.columns:
                continue

            c  = self.get_config_color(config)
            m  = self.base_marker_map.get(base_id, 'o')
            ls = '--' if config == 'Free' else '-'

            uncalib_dt = []
            for idx, row in df.iterrows():
                tc = str(row['TC_Name']).replace('.0', '').strip()
                calib_temp = row['Temp_C']
                calib_amb  = row['T_amb_C']

                raw_temp = ((calib_temp - self.calib_dict[tc]['intercept']) / self.calib_dict[tc]['slope']
                            if tc in self.calib_dict else calib_temp)

                raw_amb  = ((calib_amb - self.calib_dict['30']['intercept']) / self.calib_dict['30']['slope']
                            if '30' in self.calib_dict else calib_amb)

                uncalib_dt.append(raw_temp - raw_amb)

            df['Uncalib_Delta_T_K'] = uncalib_dt
            df_sorted = df.sort_values(by='X/B')

            export_cols = [col for col in ['X/B', 'Delta_T_K', 'Uncalib_Delta_T_K'] if col in df_sorted.columns]
            df_export_chunk = df_sorted[export_cols].copy()
            df_export_chunk.insert(0, 'base_id', base_id)
            df_export_chunk.insert(0, 'config', config)
            df_export_chunk.insert(0, 'case_id', case_id)
            export_rows.append(df_export_chunk)

            label_calib = f"{config} ({base_id}) — Calibrated"
            ax.errorbar(df_sorted['X/B'], df_sorted['Delta_T_K'], yerr=_U_DT,
                        marker=m, linestyle=ls, color=c,
                        linewidth=2, markersize=6, capsize=2, elinewidth=0.8,
                        label=label_calib if label_calib not in legend_seen else "_nolegend_")
            legend_seen.add(label_calib)

            label_raw = f"{config} ({base_id}) — Raw"
            ax.plot(df_sorted['X/B'], df_sorted['Uncalib_Delta_T_K'],
                    marker=m, linestyle=ls, color=c,
                    markerfacecolor='white', linewidth=1.5, markersize=6, alpha=0.55,
                    label=label_raw if label_raw not in legend_seen else "_nolegend_")
            legend_seen.add(label_raw)

        ax.set_xlabel(r"Normalized Lateral Distance $X/B$ [-]")
        ax.set_ylabel(r"Temperature Difference $\Delta T$ [K]")
        ax.set_title(r"Impact of TC Calibration on $\Delta T$ — All Cases")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)

        plt.tight_layout()
        filename_base = "Calib_Impact_AllCases"
        fig.savefig(os.path.join(png_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')

        original_titles = [ax.get_title() for ax in fig.axes]
        for ax in fig.axes:
            ax.set_title("")

        fig.savefig(os.path.join(pdf_dir, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight')

        for ax, title in zip(fig.axes, original_titles):
            ax.set_title(title)

        plt.close(fig)

        if export_rows:
            df_all = pd.concat(export_rows, ignore_index=True)
            csv_dir = os.path.join(fig_dir, "CSV")
            xlsx_dir = os.path.join(fig_dir, "XLSX")
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(xlsx_dir, exist_ok=True)
            df_all.to_csv(os.path.join(csv_dir, "Calib_Impact_AllCases.csv"), index=False)
            df_all.to_excel(os.path.join(xlsx_dir, "Calib_Impact_AllCases.xlsx"), index=False)

        messagebox.showinfo("Success", f"Graph saved in:\n{fig_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationComparer(root)
    root.mainloop()