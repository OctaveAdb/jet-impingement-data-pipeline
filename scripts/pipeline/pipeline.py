# =============================================================================
# PIPELINE
# =============================================================================
# Purpose:
#   Executes the full 7-step aerodynamic processing and visualisation pipeline
#   for a single experimental case folder in a fixed sequential order. Each
#   step instantiates the relevant processor or visualiser class in a hidden
#   Tkinter toplevel, runs its core computation method in a background thread,
#   and passes control to the next step upon completion. The seven steps are:
#   (1) Re & Velocity Calculator — compute V_noz and Re from tunnel dP and
#   export to Flow_Data.csv; (2) Cobra Data Processor — parse .thA files,
#   apply probe rotation, and export per-point raw CSVs; (3) Spatial Averager
#   — compute time-averaged velocity maps and jet characteristic metrics;
#   (4) Frequency Processor — compute PSD, Strouhal, and coherence
#   maps; (5) 2D Flow Visualiser — save all 2D spatial field figures;
#   (6) Frequency Visualiser — save all FFT/PSD figures and centerline profiles;
#   (7) 1D Profile Viewer — save centerline 1D profile figures for all
#   variables. The pipeline can be launched as a standalone GUI (direct
#   execution) or headlessly from usage.py via subprocess with a JSON
#   parameter file passed on the command line.
#
# Inputs:
#   - User-supplied experimental parameters (via GUI or JSON config file):
#     step_x, step_y, x_start, y_start, az, el, roll (grid geometry and
#     probe alignment), B_mm (nozzle height mm), d_cyl (cylinder diameter mm),
#     w_noz (nozzle width mm), d1 (tunnel diameter mm), dp (tunnel dP Pa)
#   - Target case workspace folder containing raw .thA / .asA files
#
# Outputs:
#   - All outputs defined by each of the seven constituent scripts
#     (see their individual header blocks for details)
#
# Dependencies:
#   - Re_Ve_processor, read_th_file_processor, mean_processor,
#     frequency_processor, maps_2D_visualizer, frequency_visualizer,
#     graphs_1D_visualizer
#
# Usage:
#   - Standalone GUI: python pipeline.py  (opens configuration window)
#   - Headless (from usage.py):
#       python pipeline.py --folder <path> --config <params.json>
# =============================================================================

import sys
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import glob
import json
import argparse
import time
import threading
import tkinter.messagebox as mb
import re
import pandas as _pd

# --- CONFIG LOADER ---
def _load_cfg(filename):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    _df = _pd.read_excel(_p, index_col='parameter')
    return _df['value'] if 'value' in _df.columns else _df.iloc[:, 0]

try:
    _cfg_geom  = _load_cfg('config_geometry.csv')
    _cfg_fluid = _load_cfg('config_fluid_properties.csv')
except Exception:
    _cfg_geom  = None
    _cfg_fluid = None

# Import the existing processors
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processors'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'visualizers'))
from Re_Ve_processor import ReVeCalculatorApp
from read_th_file_processor import CobraDataProcessorApp
from mean_processor import DataMeansApp
from maps_2D_visualizer import DataVisualizerApp, save_vertical_slice_profiles
from graphs_1D_visualizer import Profile1DApp
from frequency_processor import FrequencyProcessorApp
from frequency_visualizer import FrequencyVisualizerApp

# --- DELTA-P FROM FOLDER NAME ---
def parse_dp_from_name(folder_name, d_cyl=None):
    """Extract the tunnel pressure drop ΔP (Pa) encoded in a case folder name.

    Case names embed the pressure as '<value>Pa', e.g.:
        Free10PaFree -> 10,  Free5PaPla -> 5,
        Cyl1210PaFree -> 10, Cyl125PaPla -> 5.

    For 'Cyl' cases the cylinder-diameter digits (D2, e.g. 12) are glued in
    front of the pressure ('Cyl12' + '10Pa' -> 'Cyl1210Pa'), so the known
    cylinder diameter is stripped from the captured digits to recover the
    pressure. Returns a float ΔP, or None if the name encodes no pressure.
    """
    m = re.search(r'(\d+)Pa', folder_name)
    if not m:
        return None
    digits = m.group(1)
    if folder_name.startswith('Cyl') and d_cyl:
        try:
            dstr = str(int(round(float(d_cyl))))
            if digits.startswith(dstr) and len(digits) > len(dstr):
                digits = digits[len(dstr):]
        except (TypeError, ValueError):
            pass
    try:
        return float(digits)
    except ValueError:
        return None


def find_reference_asa(folder):
    """Locate a representative .asA file for ambient temperature / barometric
    pressure.

    Real acquisition filenames carry suffixes (e.g. '0010 (Ve).asA',
    '0010 (Ve) (Ve).asA'), so an exact-name lookup of '0010.asA' fails. This
    tries the configured reference stem (default '0010', suffixes allowed) in
    the asA/ subfolder then the case root, and finally falls back to ANY .asA
    file — ambient conditions are essentially constant across a case, so any
    point is a valid source. Returns a path or None."""
    ref_name = "0010.asA"
    if _cfg_geom is not None:
        try:
            if 'reference_file_name' in _cfg_geom.index:
                ref_name = str(_cfg_geom['reference_file_name'])
        except Exception:
            pass
    stem = os.path.splitext(ref_name)[0]
    asa_dir = os.path.join(folder, "asA")
    for base in (asa_dir, folder):
        hits = sorted(glob.glob(os.path.join(base, stem + "*.asA")))
        if hits:
            return hits[0]
    for base in (asa_dir, folder):
        hits = sorted(glob.glob(os.path.join(base, "*.asA")))
        if hits:
            return hits[0]
    return None


# --- GLOBAL POPUP SUPPRESSION CACHE ---
original_showinfo = mb.showinfo
original_showwarning = mb.showwarning

def silence_popups():
    mb.showinfo = lambda *args, **kwargs: None
    mb.showwarning = lambda *args, **kwargs: None

def restore_popups():
    mb.showinfo = original_showinfo
    mb.showwarning = original_showwarning

# ==========================================
# THE CORE PIPELINE SEQUENCE ENGINE
# ==========================================
class PipelineSequence:
    def __init__(self, root, folder, params):
        self.root = root
        self.folder = folder
        self.folder_name = os.path.basename(os.path.normpath(folder))
        self.params = params
        self._keepalive = []
        
        # Start the sequence
        self.root.after(100, self.run_step_1)

    def run_step_1(self):
        print(f"--- PIPELINE 1/7: Running Re & Velocity Calculator | Folder: {self.folder_name} ---")
        w1 = tk.Toplevel(self.root)
        w1.withdraw() 
        app = ReVeCalculatorApp(w1, default_folder=self.folder)
        self._keepalive.append(app)
        
        def worker():
            try:
                # Dynamically fetch ambient Temp and Patm from a reference .asA
                # (glob-based, so the ' (Ve)' filename suffixes are matched).
                temp_val = float(_cfg_fluid['default_temp_fallback']) if _cfg_fluid is not None else 21.0
                patm_val = float(_cfg_fluid['default_patm_fallback']) if _cfg_fluid is not None else 101325.0

                asa_path = find_reference_asa(self.folder)

                if asa_path and os.path.exists(asa_path):
                    print(f"--- Step 1: ambient T/Patm from {os.path.basename(asa_path)} ---")
                    with open(asa_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tm = re.search(r'Mean temperature.*?:\s*([\d\.]+)', content)
                        pm = re.search(r'Barometric pressure.*?:\s*([\d\.,]+)', content)
                        if tm: temp_val = float(tm.group(1))
                        if pm: patm_val = float(pm.group(1).replace(',', ''))
                else:
                    print(f"--- Step 1: WARNING — no .asA found for ambient T/Patm; "
                          f"using defaults T={temp_val}, Patm={patm_val}. ---")
                
                # Determine ΔP: when none was supplied (dp == 0, e.g. the global
                # "Run Full Analysis" pipeline), fall back to the pressure encoded
                # in the case folder name so V_noz (and hence Strouhal) is computed
                # from the correct per-case tunnel pressure. An explicitly supplied
                # non-zero ΔP (single-case runs) always takes precedence.
                dp_val = self.params.get('dp', 0.0)
                try:
                    dp_is_zero = abs(float(dp_val)) < 1e-9
                except (TypeError, ValueError):
                    dp_is_zero = True
                if dp_is_zero:
                    parsed_dp = parse_dp_from_name(self.folder_name, self.params.get('d_cyl'))
                    if parsed_dp is not None:
                        dp_val = parsed_dp
                        print(f"--- Step 1: ΔP not provided; using ΔP = {dp_val:g} Pa "
                              f"parsed from folder name '{self.folder_name}'. ---")
                    else:
                        print(f"--- Step 1: WARNING — ΔP is 0 and no pressure found in "
                              f"folder name '{self.folder_name}'; V_noz will be 0. ---")

                app.delta_p_var.set(str(dp_val))
                app.temp_var.set(str(temp_val))
                app.p_atm_var.set(str(patm_val))
                app.d1_var.set(str(self.params['d1']))
                app.w2_var.set(str(self.params['w_noz']))
                app.h2_var.set(str(self.params['B_mm']))
                app.d2_var.set(str(self.params['d_cyl']))
                
                app.on_calculate()
                app.export_data()
            except Exception as e:
                print(f"Error in Step 1: {e}")
            self.root.after(0, lambda: self.run_step_2(w1))
            
        threading.Thread(target=worker, daemon=True).start()

    def run_step_2(self, prev_window):
        prev_window.destroy()
        print(f"--- PIPELINE 2/7: Running Cobra Data Processor | Folder: {self.folder_name} ---")
        w2 = tk.Toplevel(self.root)
        w2.withdraw() 
        app = CobraDataProcessorApp(w2, default_folder=self.folder)
        self._keepalive.append(app)
        
        def worker():
            try:
                app.process_files(
                    self.folder,
                    self.params['step_x'], self.params['step_y'], self.params['B_mm'],
                    self.params['x_start'], self.params['y_start'],
                    self.params['az'], self.params['el'], self.params['roll']
                )
            except Exception as e:
                print(f"Error in Step 2: {e}")
            self.root.after(0, lambda: self.run_step_3(w2))
            
        threading.Thread(target=worker, daemon=True).start()

    def run_step_3(self, prev_window):
        prev_window.destroy()
        print(f"--- PIPELINE 3/7: Running Spatial Averager | Folder: {self.folder_name} ---")
        w3 = tk.Toplevel(self.root)
        w3.withdraw()
        app = DataMeansApp(w3, default_folder=self.folder)
        self._keepalive.append(app)
        
        def worker():
            try:
                app.process_files(self.folder)
            except Exception as e:
                print(f"Error in Step 3: {e}")
            self.root.after(0, lambda: self.run_step_4(w3))
            
        threading.Thread(target=worker, daemon=True).start()

    def run_step_4(self, prev_window):
        prev_window.destroy()
        print(f"--- PIPELINE 4/7: Running Frequency Processor | Folder: {self.folder_name} ---")
        w4 = tk.Toplevel(self.root)
        w4.withdraw()
        app = FrequencyProcessorApp(w4, default_folder=self.folder)
        self._keepalive.append(app)
        
        def worker():
            try:
                app.process_fft_data(self.folder)
            except Exception as e:
                print(f"Error in Step 4: {e}")
            self.root.after(0, lambda: self.run_step_5(w4))
            
        threading.Thread(target=worker, daemon=True).start()

    def run_step_5(self, prev_window):
        prev_window.destroy()
        print(f"--- PIPELINE 5/7: Generating 2D Flow Profiles | Folder: {self.folder_name} ---")
        w5 = tk.Toplevel(self.root)
        w5.withdraw()
        app = DataVisualizerApp(w5, default_folder=self.folder)
        self._keepalive.append(app)

        def execute_step5():
            # --- 2D spatial profile plots (transverse-map cases) ---
            if self.folder_name.endswith("Pla"):
                try:
                    from maps_2D_visualizer import save_transverse_profiles
                    save_transverse_profiles(self.folder)
                except Exception as e:
                    print(f"[Step 5] Transverse profiles error: {e}")
                try:
                    save_vertical_slice_profiles(self.folder)
                except Exception as e:
                    print(f"[Step 5] Vertical slice profiles error: {e}")
            self.root.after(500, lambda: self.run_step_6(w5))

        self.root.after(1000, execute_step5)

    def run_step_6(self, prev_window):
        prev_window.destroy()
        print(f"--- PIPELINE 6/7: Generating Frequency FFTs & Profiles | Folder: {self.folder_name} ---")
        w6 = tk.Toplevel(self.root)
        w6.withdraw()
        app = FrequencyVisualizerApp(w6, default_folder=self.folder)
        self._keepalive.append(app)
        
        def worker():
            try:
                self.root.after(0, app.save_all_t1_points)
                
                time.sleep(1)
                while getattr(app, 'is_processing', False):
                    time.sleep(1)
                
                self.root.after(0, app.save_centerline_superposed_psd)
                time.sleep(2)
            except Exception as e:
                print(f"Error in Step 6: {e}")
                
            self.root.after(0, lambda: self.run_step_7(w6))

        threading.Thread(target=worker, daemon=True).start()

    def run_step_7(self, prev_window):
        prev_window.destroy()
        print(f"--- PIPELINE 7/7: Generating 1D Profiles | Folder: {self.folder_name} ---")
        w7 = tk.Toplevel(self.root)
        w7.withdraw()
        app = Profile1DApp(w7, default_folder=self.folder)
        self._keepalive.append(app)
        
        def execute_step7():
            # Variables that should NOT be auto-saved in the pipeline.
            # Pressure variables: data is kept in CSV but graphs are not useful
            # in the aerodynamic pipeline report.
            # Temperature: belongs to the thermal analysis pipeline only.
            PIPELINE_SKIP_VARS = {
                "Static Pressure (Pa)", "Total Pressure (Pa)",
                "Static_Pressure_Pa", "Total_Pressure_Pa",
                "Barometric Pressure Pa", "Barometric_Pressure_Pa",
            }

            try:
                if app.combo_var['values']:
                    for var in app.combo_var['values']:
                        # Skip pressure and temperature profiles
                        if var in PIPELINE_SKIP_VARS:
                            print(f"  [Step 7] Skipping {var} (excluded from pipeline)")
                            continue
                        # Skip temperature regardless of exact formatting
                        if "Temperature" in var or "Temp" in var:
                            print(f"  [Step 7] Skipping {var} (temperature — thermal pipeline only)")
                            continue

                        app.combo_var.set(var)
                        app.slice_var.set("Y")
                        app.update_slices()

                        if app.combo_slice['values']:
                            slice_floats = [float(val) for val in app.combo_slice['values']]
                            center_idx = min(range(len(slice_floats)), key=lambda i: abs(slice_floats[i] - 0.0))

                            app.combo_slice.current(center_idx)
                            app.plot_profile()
                            app.save_plot()
            except Exception as e:
                print(f"Error in Step 7: {e}")

            self.root.after(500, lambda: self.finish_pipeline(w7))
            
        self.root.after(1000, execute_step7)

    def finish_pipeline(self, prev_window):
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
            
        prev_window.destroy() 
        print(f"--- PIPELINE COMPLETE | Folder: {self.folder_name} ---")
        
        restore_popups()
        self.root.quit()

# ==========================================
# STANDALONE GUI (When double-clicked)
# ==========================================
class StandalonePipelineUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Data Pipeline Setup")
        self.root.geometry("550x700")
        
        self.folder_path = tk.StringVar()
        # Load geometry defaults from config (with fallback to old values)
        _step_x  = str(float(_cfg_geom['grid_step_x']))         if _cfg_geom is not None else "5.0"
        _step_y  = str(float(_cfg_geom['grid_step_y']))         if _cfg_geom is not None else "1.5875"
        _x_start = str(float(_cfg_geom['x_start_offset']))      if _cfg_geom is not None else "38.0"
        _h_noz   = str(float(_cfg_geom['nozzle_height_B']))     if _cfg_geom is not None else "30.0"
        _d_cyl   = str(float(_cfg_geom['cylinder_diameter_D2'])) if _cfg_geom is not None else "12.0"
        _w_noz   = str(float(_cfg_geom['nozzle_width_cyl']))    if _cfg_geom is not None else "150.0"
        _d1      = str(float(_cfg_geom['tunnel_diameter_D1']))  if _cfg_geom is not None else "200.0"
        self.pipe_step_x = tk.StringVar(value=_step_x)
        self.pipe_step_y = tk.StringVar(value=_step_y)
        self.pipe_x_start = tk.StringVar(value=_x_start)
        self.pipe_y_start = tk.StringVar(value="0.0")
        self.pipe_az = tk.StringVar(value="0.0")
        self.pipe_el = tk.StringVar(value="0.0")
        self.pipe_roll = tk.StringVar(value="0.0")
        self.pipe_h_noz = tk.StringVar(value=_h_noz)
        self.pipe_d_cyl = tk.StringVar(value=_d_cyl)
        self.pipe_w_noz = tk.StringVar(value=_w_noz)  # now uses config value (nozzle_width_cyl = 150.0 mm, PROTOCOL §2.1)
        self.pipe_d1 = tk.StringVar(value=_d1)
        self.pipe_dp = tk.StringVar(value="10.0")

        self.setup_ui()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Case Workspace")
        if folder:
            self.folder_path.set(folder)

    def start_pipeline(self):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder):
            messagebox.showwarning("Input Needed", "Please select a valid Workspace folder first.")
            return

        try:
            params = {
                'step_x': float(self.pipe_step_x.get()),
                'step_y': float(self.pipe_step_y.get()),
                'x_start': float(self.pipe_x_start.get()),
                'y_start': float(self.pipe_y_start.get()),
                'az': float(self.pipe_az.get()),
                'el': float(self.pipe_el.get()),
                'roll': float(self.pipe_roll.get()),
                'B_mm': float(self.pipe_h_noz.get()),
                'd_cyl': float(self.pipe_d_cyl.get()),
                'w_noz': float(self.pipe_w_noz.get()),
                'd1': float(self.pipe_d1.get()),
                'dp': float(self.pipe_dp.get())
            }
            if params['B_mm'] <= 0: raise ValueError("Nozzle height must be > 0.")
        except ValueError:
            messagebox.showerror("Format Error", "All parameters must be valid numbers.")
            return

        # Hide UI, silence popups, and run pipeline in the current process
        self.root.withdraw()
        silence_popups()
        PipelineSequence(self.root, folder, params)

    def setup_ui(self):
        tk.Label(self.root, text="Automated 7-Step Pipeline", font=("Arial", 16, "bold"), bg="#004e92", fg="white", pady=10).pack(fill="x")
        
        f_frame = tk.Frame(self.root, pady=10)
        f_frame.pack(fill="x", padx=20)
        tk.Label(f_frame, text="Target Folder:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        folder_row = tk.Frame(f_frame)
        folder_row.pack(fill="x")
        tk.Entry(folder_row, textvariable=self.folder_path, state="readonly").pack(side="left", fill="x", expand=True, padx=(0,10))
        tk.Button(folder_row, text="Browse...", command=self.select_folder).pack(side="right")

        form_frame = tk.Frame(self.root, padx=20, pady=5)
        form_frame.pack(fill="both", expand=True)

        def add_row(parent, label_text, string_var, row_idx):
            tk.Label(parent, text=label_text, font=("Arial", 10, "bold")).grid(row=row_idx, column=0, sticky="e", pady=5, padx=10)
            tk.Entry(parent, textvariable=string_var, width=15).grid(row=row_idx, column=1, sticky="w")

        tk.Label(form_frame, text="Grid & Offset Parameters", font=("Arial", 10, "italic"), fg="#555").grid(row=0, column=0, columnspan=2, sticky="w", pady=(5,5))
        add_row(form_frame, "Step X (mm):", self.pipe_step_x, 1)
        add_row(form_frame, "Step Y (mm):", self.pipe_step_y, 2)
        add_row(form_frame, "X Start Offset (mm):", self.pipe_x_start, 3)
        add_row(form_frame, "Y Manual Offset (mm):", self.pipe_y_start, 4)
        
        tk.Label(form_frame, text="Probe Rotations (Degrees)", font=("Arial", 10, "italic"), fg="#555").grid(row=5, column=0, columnspan=2, sticky="w", pady=(10,5))
        add_row(form_frame, "Azimuth:", self.pipe_az, 6)
        add_row(form_frame, "Elevation:", self.pipe_el, 7)
        add_row(form_frame, "Roll:", self.pipe_roll, 8)

        tk.Label(form_frame, text="Physical Geometry", font=("Arial", 10, "italic"), fg="#555").grid(row=9, column=0, columnspan=2, sticky="w", pady=(10,5))
        add_row(form_frame, "B (mm):", self.pipe_h_noz, 10)
        add_row(form_frame, "Nozzle Width (mm):", self.pipe_w_noz, 11)
        add_row(form_frame, "Cylinder Diameter 'D' (mm):", self.pipe_d_cyl, 12)
        add_row(form_frame, "Tunnel D1 (mm):", self.pipe_d1, 13)

        tk.Label(form_frame, text="Environment Variables", font=("Arial", 10, "italic"), fg="#555").grid(row=14, column=0, columnspan=2, sticky="w", pady=(10,5))
        add_row(form_frame, "Delta P Tunnel (Pa):", self.pipe_dp, 15)

        btn_frame = tk.Frame(self.root, pady=15)
        btn_frame.pack(fill="x", padx=20)
        tk.Button(btn_frame, text="🚀 RUN FULL PIPELINE", command=self.start_pipeline, bg="#28a745", fg="white", font=("Arial", 12, "bold"), height=2).pack(fill="x")

# ==========================================
# MAIN EXECUTION ROUTING
# ==========================================
if __name__ == "__main__":
    # Check if run via command line (Headless Subprocess from usage.py)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Run headless analysis pipeline.")
        parser.add_argument("--folder", required=True, help="Workspace folder")
        parser.add_argument("--config", required=True, help="Path to JSON parameters file")
        args = parser.parse_args()

        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                params = json.load(f)
        else:
            # Config file missing — build from the xlsx config files as fallback
            import pandas as _pd
            _cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config')
            def _rc(fname):
                fname = fname.replace('.csv', '.xlsx')
                _df = _pd.read_excel(os.path.join(_cfg_dir, fname), index_col='parameter')
                return _df['value'] if 'value' in _df.columns else _df.iloc[:, 0]
            try:
                _g = _rc('config_geometry.csv')
                params = {
                    'B_mm':   float(_g['nozzle_height_B']),
                    'step_x': float(_g['grid_step_x']),
                    'step_y': float(_g['grid_step_y']),
                    'x_start': float(_g['x_start_offset']),
                    'y_start': 0.0, 'az': 0.0, 'el': 0.0, 'roll': 0.0,
                    'd_cyl': float(_g['cylinder_diameter_D2']),
                    'w_noz': float(_g['nozzle_width_cyl']),
                    'd1':    float(_g['tunnel_diameter_D1']),
                    'dp': 0.0,
                }
            except Exception:
                params = {
                    'B_mm': 30.0, 'step_x': 5.0, 'step_y': 1.5875,
                    'x_start': 38.0, 'y_start': 0.0,
                    'az': 0.0, 'el': 0.0, 'roll': 0.0,
                    'd_cyl': 12.0, 'w_noz': 150.0, 'd1': 200.0, 'dp': 0.0,
                }

        # --- Per-case Step-Y override (config_cases.xlsx 'step_y_mm') ----------
        # The Cobra Y-traverse was sampled at a different index granularity in
        # some runs (e.g. Cyl 5 Pa steps 2 index-units per point), so the traverse
        # step is a PER-CASE quantity, not the single global config_geometry value.
        # Match the case folder to config_cases and override step_y when defined.
        try:
            import pandas as _pdc
            _cc = _pdc.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               '..', '..', 'config', 'config_cases.xlsx'))
            _cn = os.path.basename(os.path.normpath(args.folder))
            if 'step_y_mm' in _cc.columns:
                for _, _r in _cc.iterrows():
                    _pat = str(_r.get('case_pattern', ''))
                    if _pat and (_cn == _pat or _pat in _cn) and _pdc.notna(_r.get('step_y_mm', None)):
                        params['step_y'] = float(_r['step_y_mm'])
                        print(f"[pipeline] per-case Step-Y for {_cn}: {params['step_y']} mm (config_cases)")
                        break
        except Exception as _e:
            print(f"[pipeline] per-case Step-Y lookup skipped: {_e}")

        # Launch Headless
        root = tk.Tk()
        root.withdraw()
        silence_popups()
        app = PipelineSequence(root, args.folder, params)
        root.mainloop()
        
    # Or, if run directly (Standalone GUI Mode)
    else:
        root = tk.Tk()
        app = StandalonePipelineUI(root)
        root.mainloop()