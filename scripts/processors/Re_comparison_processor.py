# =============================================================================
# RE_COMPARISON_PROCESSOR
# =============================================================================
# Purpose:
#   Performs a cross-validation of Reynolds numbers by comparing the
#   theoretical Re_noz derived from the tunnel differential pressure (via
#   Re_Ve_processor) against Re values measured directly from Cobra probe
#   velocity data at the two nozzle exits (.asA files). The script reads all
#   .asA and .thA files from the thermal/velocity_reference/ folder, organises
#   them by case (Pa level and Cyl vs Free), parses the mean flow speed at
#   each nozzle exit, computes Re_meas for exits 1 and 2, and calls the
#   Re_Ve_processor physics engine with automatically selected geometry
#   (nozzle width and cylinder diameter depend on configuration). Results are
#   displayed in a comparison table showing theoretical V_noz, both measured
#   exit velocities, theoretical and measured Re values, and percentage
#   deviations (E1 vs Theo, E2 vs Theo, E1 vs E2). The table can be exported
#   and appended to an existing Comparison_Results file in thermal/results/.
#
# Inputs:
#   - Cobra probe .asA reference files at nozzle exits:
#       thermal/velocity_reference/noz_exit<N>_<Pa>Pa_<config> (Ve).asA
#   - Cobra probe .thA files (if present, moved automatically to subfolder)
#
# Outputs:
#   - Comparison results table (merged with any existing file):
#       thermal/results/Comparison_Results.csv / .xlsx
#
# Dependencies:
#   - Re_Ve_processor (ReVeCalculatorApp.calculate_physics method)
#
# Usage:
#   - Standalone: python Re_comparison_processor.py  (opens Tkinter GUI)
#   - Via hub:    launched from usage.py as "Re & Velocity Comparison"
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import math
import os
import shutil
import re
import sys

# ---------------------------------------------------------
# IMPORTING THE FUNCTION DIRECTLY FROM YOUR SCRIPT
# ---------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from Re_Ve_processor import ReVeCalculatorApp, compute_flow_uncertainties
    HAS_RE_VE = True
except ImportError:
    HAS_RE_VE = False

# Shared uncertainty helper (Task 3): cobra raw-reading floor for the measured
# (.asA) velocities. The .asA "Mean flow speed" is the RAW Cobra mean, so its
# uncertainty is the per-mean floor (not the calibrated mean-speed value).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
try:
    from uncertainty import U_COBRA_FLOOR_MS as _U_COBRA_FLOOR
except Exception:
    _U_COBRA_FLOOR = 0.5

# --- Config loader ---
def _load_cfg(filename, value_col='value'):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    return pd.read_excel(_p, index_col='parameter')[value_col]

try:
    _cfg_geom  = _load_cfg('config_geometry.csv')   # geometry now uses the 'value' column
    _cfg_fluid = _load_cfg('config_fluid_properties.csv')
    _cfg_flow  = _load_cfg('config_flow_regime.csv')
    _cfg_cases = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cases.xlsx'))

    # Canonical geometry constants (read once)
    _D1_M   = float(_cfg_geom['tunnel_diameter_D1']) / 1000.0
    _B_M    = float(_cfg_geom['nozzle_height_B']) / 1000.0
    _W_CYL  = float(_cfg_geom['nozzle_width_cyl']) / 1000.0
    _W_FREE = float(_cfg_geom['nozzle_width_free']) / 1000.0
    _D2_M   = float(_cfg_geom['cylinder_diameter_D2']) / 1000.0
    _R_AIR  = float(_cfg_fluid['R_air'])
    _MU_REF = float(_cfg_fluid['mu_ref'])
    _T_SUT  = float(_cfg_fluid['sutherland_T_ref'])
    _S_SUT  = float(_cfg_fluid['sutherland_S'])
    _T_DEF  = float(_cfg_fluid['default_temp_fallback'])
    _P_DEF  = float(_cfg_fluid['default_patm_fallback'])
    _RE_LAM = float(_cfg_flow['re_laminar_upper'])
    _RE_TR  = float(_cfg_flow['re_transition_upper'])
    _C_LAM  = float(_cfg_flow['c_profile_laminar'])
    _N_COEF = float(_cfg_flow['c_profile_turbulent_log_coeff'])
    _N_OFF  = float(_cfg_flow['c_profile_turbulent_log_offset'])
except Exception as _cfg_err:
    print(f"[Re_comparison_processor] WARNING: Could not load config files: {_cfg_err}. Falling back to hardcoded defaults.")
    _cfg_geom  = None
    _cfg_fluid = None
    _cfg_flow  = None
    _cfg_cases = None
    _D1_M   = 200.0 / 1000.0
    _B_M    = 30.0 / 1000.0
    _W_CYL  = 150.0 / 1000.0
    _W_FREE = 170.0 / 1000.0
    _D2_M   = 12.0 / 1000.0
    _R_AIR  = 287.05
    _MU_REF = 1.716e-5
    _T_SUT  = 273.15
    _S_SUT  = 110.4
    _T_DEF  = 20.0
    _P_DEF  = 101325.0
    _RE_LAM = 2300
    _RE_TR  = 4000
    _C_LAM  = 0.50
    _N_COEF = 1.22
    _N_OFF  = 1.0

class ReComparisonApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("General Workspace & Reynolds Comparison Processor")
        self.root.geometry("1300x650")
        
        # Automatically use the folder passed from usage.py
        self.folder_path = tk.StringVar(value=default_folder if default_folder else "")
        self.setup_ui()

    def setup_ui(self):
        # 1. Folder Selection
        f_frame = tk.LabelFrame(self.root, text=" 1. Workspace (Select the General Folder containing 'Temperatures') ", padx=10, pady=10)
        f_frame.pack(padx=20, pady=10, fill="x")
        
        tk.Entry(f_frame, textvariable=self.folder_path, state="readonly", width=60).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(f_frame, text="Browse...", command=self.select_folder).pack(side="right", padx=5)

        # 2. Controls
        c_frame = tk.Frame(self.root)
        c_frame.pack(padx=20, pady=5, fill="x")
        
        btn_text = "ORGANIZE & PROCESS FILES" if HAS_RE_VE else "ERROR: Re_Ve_processor.py not found!"
        btn_state = "normal" if HAS_RE_VE else "disabled"
        btn_color = "#007acc" if HAS_RE_VE else "#dc3545"
        
        tk.Button(c_frame, text=btn_text, command=self.process_files, bg=btn_color, fg="white", font=("Arial", 11, "bold"), height=2, state=btn_state).pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_export = tk.Button(c_frame, text="EXPORT RESULTS (CSV & XLSX)", command=self.export_results, bg="#28a745", fg="white", font=("Arial", 11, "bold"), height=2, state="disabled")
        self.btn_export.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # 3. Results Table
        l_frame = tk.LabelFrame(self.root, text=" Comparison Results ", padx=10, pady=10)
        l_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.columns = (
            "Case", "Theo V_noz", "Exit 1 V", "Exit 2 V", 
            "Theo Re_noz", "E1 Re_noz", "E2 Re_noz", "Theo Re_obs", 
            "Δ% (E1 vs Theo)", "Δ% (E2 vs Theo)", "Δ% (E1 vs E2)"
        )
        self.tree = ttk.Treeview(l_frame, columns=self.columns, show="headings")
        
        for col in self.columns:
            self.tree.heading(col, text=col)
            w = 110
            if "Δ%" in col: w = 115
            elif "Case" in col: w = 100
            self.tree.column(col, width=w, anchor="center")
            
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(l_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select General Workspace Folder")
        if folder:
            self.folder_path.set(folder)

    def extract_metadata(self, filename):
        match_pa = re.search(r'(\d+)\s*Pa', filename, re.IGNORECASE)
        match_exit = re.search(r'exit\s*(\d+)', filename, re.IGNORECASE)
        match_cyl = re.search(r'Cyl', filename, re.IGNORECASE)
        
        pa_val = int(match_pa.group(1)) if match_pa else None
        exit_val = int(match_exit.group(1)) if match_exit else 1
        has_cyl = bool(match_cyl)
        
        return pa_val, exit_val, has_cyl

    def parse_asa_file(self, filepath):
        data = {'temp': _T_DEF, 'patm': _P_DEF, 'v_mean': 0.0}
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Mean temperature" in line:
                        m = re.search(r':\s*([\d\.]+)', line)
                        if m: data['temp'] = float(m.group(1))
                    elif "Barometric pressure" in line:
                        m = re.search(r':\s*([\d\.,]+)', line)
                        if m: data['patm'] = float(m.group(1).replace(',', ''))
                    elif "Mean flow speed" in line:
                        if i + 1 < len(lines):
                            vals = lines[i+1].split()
                            if len(vals) >= 1: data['v_mean'] = float(vals[0])
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
        return data

    def calc_measured_reynolds(self, v, temp_c, p_atm, length_mm):
        temp_k = temp_c + _T_SUT
        rho = p_atm / (_R_AIR * temp_k)
        mu = _MU_REF * (temp_k / _T_SUT)**1.5 * (_T_SUT + _S_SUT) / (temp_k + _S_SUT)
        re = (rho * v * (length_mm / 1000.0)) / mu
        return re

    def process_files(self):
        base_folder = self.folder_path.get()
        if not base_folder or not os.path.exists(base_folder):
            messagebox.showerror("Error", "Please select a valid general folder first.")
            return
            
        temp_folder = os.path.join(base_folder, "thermal")
        if not os.path.exists(temp_folder):
            messagebox.showerror("Error", f"Could not find a 'thermal' subfolder inside:\n{base_folder}")
            return

        tha_dir = os.path.join(temp_folder, "velocity_reference")
        asa_dir = os.path.join(temp_folder, "velocity_reference")
        
        os.makedirs(tha_dir, exist_ok=True)
        os.makedirs(asa_dir, exist_ok=True)

        # 1. Organize Files
        for f in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, f)
            if os.path.isfile(file_path):
                if f.endswith(".thA"):
                    shutil.move(file_path, os.path.join(tha_dir, f))
                elif f.endswith(".asA"):
                    shutil.move(file_path, os.path.join(asa_dir, f))

        # 2. Process Data
        measurements = {}
        asa_files = [f for f in os.listdir(asa_dir) if f.endswith(".asA")]
        
        # Geometry constants (from config)
        d1 = _D1_M * 1000.0
        B_mm = _B_M * 1000.0
        
        for f in asa_files:
            pa_val, exit_val, has_cyl = self.extract_metadata(f)
            if pa_val is None: continue
            
            # Unique Key to separate Cyl vs NoCyl if both exist
            case_key = f"{pa_val} Pa ({'Cyl' if has_cyl else 'No Cyl'})"
            
            data = self.parse_asa_file(os.path.join(asa_dir, f))
            re_measured_noz = self.calc_measured_reynolds(data['v_mean'], data['temp'], data['patm'], B_mm)
            
            if case_key not in measurements:
                measurements[case_key] = {
                    'pa': pa_val, 'has_cyl': has_cyl,
                    'temp': data['temp'], 'patm': data['patm'], 
                    'exit1_v': None, 'exit2_v': None, 
                    'exit1_re': None, 'exit2_re': None
                }
                
            if exit_val == 1:
                measurements[case_key]['exit1_v'] = data['v_mean']
                measurements[case_key]['exit1_re'] = re_measured_noz
            elif exit_val == 2:
                measurements[case_key]['exit2_v'] = data['v_mean']
                measurements[case_key]['exit2_re'] = re_measured_noz

        # Clear existing table
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 3. Populate Table and Apply Theoretical Calculation
        for case, m in measurements.items():
            
            # Automatic Geometry Logic (from config)
            w2 = _W_CYL * 1000.0 if m['has_cyl'] else _W_FREE * 1000.0
            d2 = _D2_M * 1000.0 if m['has_cyl'] else 0.0
            
            # Call theoretical calculator (returns 10 values; the idealised gap
            # velocity / gap-Reynolds outputs have been removed).
            res = ReVeCalculatorApp.calculate_physics(None, m['pa'], m['temp'], m['patm'], d1, w2, B_mm, d2)
            (v_tun_avg, v_noz, re_tun, re_noz, re_cyl, rho, mu, k_ratio, c_profile,
             v_noz_channel) = res

            e1_v = f"{m['exit1_v']:.2f}" if m['exit1_v'] else "N/A"
            e2_v = f"{m['exit2_v']:.2f}" if m['exit2_v'] else "N/A"
            
            e1_re = m['exit1_re']
            e2_re = m['exit2_re']
            e1_re_str = f"{int(e1_re)}" if e1_re else "N/A"
            e2_re_str = f"{int(e2_re)}" if e2_re else "N/A"
            
            # Theoretical Re_obs logic
            t_re_obs = f"{int(re_cyl)}" if m['has_cyl'] else "N/A"
            
            # Deltas
            d_e1_theo = f"{((e1_re - re_noz) / re_noz) * 100:.1f}%" if e1_re and re_noz > 0 else "N/A"
            d_e2_theo = f"{((e2_re - re_noz) / re_noz) * 100:.1f}%" if e2_re and re_noz > 0 else "N/A"
            d_e1_e2 = f"{((e1_re - e2_re) / e2_re) * 100:.1f}%" if e1_re and e2_re and e2_re > 0 else "N/A"
                
            self.tree.insert("", "end", values=(
                case, f"{v_noz:.2f}", e1_v, e2_v, 
                int(re_noz), e1_re_str, e2_re_str, t_re_obs, 
                d_e1_theo, d_e2_theo, d_e1_e2
            ))
            
        # 4. Build and save structured comparison DataFrame
        comparison_rows = []
        for case, m in measurements.items():
            w2 = _W_CYL * 1000.0 if m['has_cyl'] else _W_FREE * 1000.0
            d2 = _D2_M * 1000.0 if m['has_cyl'] else 0.0
            res = ReVeCalculatorApp.calculate_physics(None, m['pa'], m['temp'], m['patm'], d1, w2, B_mm, d2)
            (v_tun_avg, v_noz, re_tun, re_noz, re_cyl, rho, mu, k_ratio, c_profile,
             v_noz_channel) = res
            config_str = 'Cyl' if m['has_cyl'] else 'Free'

            # 1-sigma uncertainties (UNCERTAINTY_SPEC.md §3). Theoretical V_noz/Re
            # via the channel->pitot route; measured (.asA) values carry the raw
            # Cobra per-mean floor.
            u_theo = compute_flow_uncertainties(float(m['pa']), v_tun_avg, v_noz,
                                                v_noz_channel, re_tun, re_noz, re_cyl,
                                                d1, w2, B_mm, d2, c_profile=c_profile)
            u_v_noz = u_theo['u_V_noz_ms']
            u_re_noz = u_theo['u_Re_noz']
            for exit_num, (v_meas, re_meas) in [(1, (m['exit1_v'], m['exit1_re'])), (2, (m['exit2_v'], m['exit2_re']))]:
                if v_meas is None:
                    continue
                ve_dev = ((v_meas - v_noz) / v_noz * 100.0) if v_noz > 0 else float('nan')
                re_dev = ((re_meas - re_noz) / re_noz * 100.0) if re_noz > 0 else float('nan')
                u_v_meas = _U_COBRA_FLOOR                                   # raw Cobra mean floor
                u_re_meas = (re_meas * _U_COBRA_FLOOR / v_meas) if v_meas > 0 else float('nan')
                comparison_rows.append({
                    'Case_ID': f"{case}_Exit{exit_num}",
                    'Config': config_str,
                    'Pressure_Pa': m['pa'],
                    'Exit': exit_num,
                    'Ve_measured_ms': round(v_meas, 4),
                    'u_Ve_measured_ms': round(u_v_meas, 4),
                    'Re_measured': int(round(re_meas)),
                    'u_Re_measured': int(round(u_re_meas)) if u_re_meas == u_re_meas else '',
                    'Ve_theoretical_ms': round(v_noz, 4),
                    'u_Ve_theoretical_ms': round(u_v_noz, 4) if u_v_noz == u_v_noz else '',
                    'Re_theoretical': int(round(re_noz)),
                    'u_Re_theoretical': int(round(u_re_noz)) if u_re_noz == u_re_noz else '',
                    'Ve_deviation_pct': round(ve_dev, 2),
                    'Re_deviation_pct': round(re_dev, 2),
                })
        if comparison_rows:
            df_comp = pd.DataFrame(comparison_rows)
            results_dir = os.path.join(temp_folder, "results")
            os.makedirs(results_dir, exist_ok=True)
            csv_out = os.path.join(results_dir, "Comparison_Results.csv")
            xlsx_out = os.path.join(results_dir, "Comparison_Results.xlsx")
            df_comp.to_csv(csv_out, index=False)
            try:
                df_comp.to_excel(xlsx_out, index=False)
            except Exception:
                pass

        self.btn_export.config(state="normal")
        messagebox.showinfo("Success", "Files organized and processed with Automatic Geometries.")

    def export_results(self):
        base_folder = self.folder_path.get()
        temp_folder = os.path.join(base_folder, "thermal")

        # 1. Grab current table data
        rows = []
        for item in self.tree.get_children():
            rows.append(self.tree.item(item)["values"])

        if not rows: return

        new_df = pd.DataFrame(rows, columns=self.columns)
        new_df.set_index("Case", inplace=True)

        csv_path = os.path.join(temp_folder, "results", "Comparison_Results.csv")
        xlsx_path = os.path.join(temp_folder, "results", "Comparison_Results.xlsx")
        
        # 2. Check if file exists to update vs create
        if os.path.exists(csv_path):
            try:
                # Force pandas to read everything as string/object to avoid type inference issues
                existing_df = pd.read_csv(csv_path, dtype=str) 
                
                if "Case" in existing_df.columns:
                    existing_df.set_index("Case", inplace=True)
                    
                    # CLEAN FIX: Drop old rows that match the current cases, then concatenate
                    existing_df = existing_df[~existing_df.index.isin(new_df.index)]
                    final_df = pd.concat([existing_df, new_df]).reset_index()
                else:
                    final_df = new_df.reset_index()
            except Exception:
                final_df = new_df.reset_index()
        else:
            final_df = new_df.reset_index()
            
        # 3. Save
        final_df.to_csv(csv_path, index=False)
        try:
            final_df.to_excel(xlsx_path, index=False)
            messagebox.showinfo("Export Successful", f"Results safely merged and saved to:\n{csv_path}\n{xlsx_path}")
        except Exception as e:
            messagebox.showwarning("Export Warning", f"CSV saved, but Excel failed. Is the file currently open?\nError: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReComparisonApp(root)
    root.mainloop()