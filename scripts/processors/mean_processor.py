# =============================================================================
# MEAN_PROCESSOR
# =============================================================================
# Purpose:
#   Performs time-averaging and spatial mapping of all per-grid-point CSV
#   files produced by read_th_file_processor. For each measurement location,
#   the script computes the temporal mean of all velocity components, pressure,
#   and auxiliary channels, and calculates turbulence intensities I_u, I_v,
#   I_w, and I_uvw (%). Results are assembled into spatial pivot maps
#   (Y/B × X/B) for each variable. For the streamwise velocity, jet-
#   characteristic metrics are also extracted: potential core length L_c,
#   plane-jet decay rate K, virtual origin intercept C, and the velocity ratio
#   C_profile relative to nozzle exit velocity V_noz.
#
#   Cases with a single transverse Y/B row (centreline-only traverse) store
#   results in 1D_Profiles_Results; cases with multiple Y/B rows (transverse
#   map traverse) store results in 2D_Profiles_Results. One CSV per variable
#   — no duplicate _Centerline extracts; all downstream code slices the map at
#   Y/B ≈ 0 when needed.
#
# Inputs:
#   - Per-grid-point raw CSV files:
#       experiments/<case>/Processed_CSVs/Raw_Data/*.csv
#   - Optional nozzle reference velocity V_noz from:
#       experiments/<case>/Flow_Data.csv (column: V_noz_ms)
#
# Outputs:
#   - Spatial maps as CSV and XLSX (one file per variable):
#       centreline-only → experiments/<case>/1D_Profiles_Results/CSV/Map_<variable>.csv
#       transverse-map  → experiments/<case>/2D_Profiles_Results/CSV/Map_<variable>.csv
#   - Jet characteristic summary (both case types):
#       experiments/<case>/1D_Profiles_Results/CSV/Global_Velocity_Summary.csv
#
# Dependencies:
#   - None (standalone processor; imported by pipeline.py and usage.py)
#
# Usage:
#   - Standalone: python mean_processor.py  (opens Tkinter GUI)
#   - Via pipeline: called programmatically by pipeline.py (Step 3)
#   - Via hub:      launched from usage.py as "Spatial Map Averager"
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import glob
import threading

# --- Config loader ---
def _load_cfg(filename):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    return pd.read_excel(_p, index_col='parameter')['value']

try:
    _cfg_flow = _load_cfg('config_flow_regime.csv')
    _IU_GUARD = float(_cfg_flow['iu_zero_guard'])
    _LC_THRESH = float(_cfg_flow['u_centerline_decay_threshold'])
except Exception as _cfg_err:
    print(f"[mean_processor] WARNING: Could not load config files: {_cfg_err}. Falling back to hardcoded defaults.")
    _cfg_flow = None
    _IU_GUARD = 1e-4
    _LC_THRESH = 0.98

# Velocity calibration (Cobra -> pitot affine law: V_cal = a*V_cobra + b).
# The per-point Raw_Data CSVs are already calibrated at read time, so the u_max
# computed below is the CALIBRATED peak; the inverse map V_raw = (V_cal - b)/a
# recovers the raw (uncalibrated) Cobra peak that PROTOCOL §10 reports.
# Calibration VALUES (a, b) from config_calibration.xlsx; the fit SE and the
# Cobra floor are uncertainty quantities -> config_uncertainty.xlsx (via the
# uncertainty module). Fallback identity (a=1, b=0).
try:
    _cal = pd.read_excel(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                     'config', 'config_calibration.xlsx')).set_index('parameter')['value']
    _CAL_A = float(_cal['cobra_slope_a'])
    _CAL_B = float(_cal['cobra_intercept_b'])
except Exception:
    _CAL_A, _CAL_B = 1.0, 0.0
try:
    import sys as _sys_unc
    _sys_unc.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
    from uncertainty import (COBRA_FIT_SE as _CAL_SE, U_COBRA_FLOOR_MS as _CAL_UFLOOR,
                             U_POS_X_MM as _U_POS_X, U_POS_Y_MM as _U_POS_Y)
except Exception:
    # _CAL_SE defaults to 0.0 -> no calibration fit residual (identity default).
    _CAL_SE, _CAL_UFLOOR, _U_POS_X, _U_POS_Y = 0.0, 0.5, 1.0, 0.5

# 1-sigma uncertainty of the CALIBRATED Cobra mean speed (UNCERTAINTY_SPEC.md §3):
# the raw per-mean floor pushed through the slope, plus the fit SE in quadrature.
#   u(U_mean) = sqrt[(a*u_floor)^2 + SE^2]; raw floor stays at u_floor.
_U_VEL_MEAN = ((_CAL_A * _CAL_UFLOOR) ** 2 + _CAL_SE ** 2) ** 0.5

class DataMeansApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("Spatial Mapping & Time Averaging")
        self.root.geometry("650x450")
        
        self.folder_path = tk.StringVar(value=default_folder if default_folder else "")
        self.setup_ui()

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder containing Data CSVs")
        if folder:
            self.folder_path.set(folder)
            self.log(f"Selected folder: {folder}")
            
            raw_dir = os.path.join(folder, "Processed_CSVs", "Raw_Data")
            old_processed_dir = os.path.join(folder, "Processed_CSVs")
            
            if os.path.exists(raw_dir):
                source_folder = raw_dir
            elif os.path.exists(old_processed_dir):
                source_folder = old_processed_dir
            else:
                source_folder = folder
            
            files = glob.glob(os.path.join(source_folder, "*.csv"))
            self.log(f"Found {len(files)} .csv files ready to process.")

    def start_processing(self):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Folder Error", "Please select a valid folder first.")
            return

        self.btn_process.config(state="disabled")
        self.log("\n--- STARTING MEAN CALCULATIONS & MAPPING ---")
        
        thread = threading.Thread(target=self.process_files, args=(folder,))
        thread.start()

    def process_files(self, folder):
        try:
            raw_dir = os.path.join(folder, "Processed_CSVs", "Raw_Data")
            old_processed_dir = os.path.join(folder, "Processed_CSVs")
            
            if os.path.exists(raw_dir):
                source_folder = raw_dir
            elif os.path.exists(old_processed_dir):
                source_folder = old_processed_dir
            else:
                source_folder = folder
            
            csv_files = glob.glob(os.path.join(source_folder, "*.csv"))
            total_files = len(csv_files)
            
            if total_files == 0:
                self.log(f"No .csv files found in {source_folder}. Aborting.")
                self.root.after(0, lambda: self.btn_process.config(state="normal"))
                return

            # Fetch V_noz for C_profile calculation
            v_noz_ref = 1.0
            u_v_noz_ref = np.nan          # 1-sigma uncertainty of V_noz (from Re_Ve)
            df_params = None
            # Re_Ve writes Flow_Data.csv to <case>/Processing_Parameters/, NOT the
            # case root. Use the same recursive lookup the other scripts use, else
            # V_noz is never found and U0_ref silently defaults to 1.0 (which breaks
            # every U/U0 normalisation and C_profile downstream).
            _fd = glob.glob(os.path.join(folder, "Flow_Data*.csv")) + \
                  glob.glob(os.path.join(folder, "**", "Flow_Data*.csv"), recursive=True)
            _fd = [f for f in _fd if os.path.isfile(f) and 'Synth' not in os.path.basename(f)]
            param_file = max(_fd, key=os.path.getctime) if _fd else None
            if param_file and os.path.exists(param_file):
                try:
                    df_params = pd.read_csv(param_file)
                    if 'V_noz_ms' in df_params.columns:
                        v_noz_ref = float(df_params['V_noz_ms'].iloc[-1])
                        self.log(f"Found reference V_noz: {v_noz_ref} m/s.")
                    if 'u_V_noz_ms' in df_params.columns:
                        u_v_noz_ref = float(df_params['u_V_noz_ms'].iloc[-1])
                except Exception as e:
                    self.log(f"Warning: Could not read V_noz from Flow_Data.csv: {e}.")

            # U0 reference for velocity-profile normalisation = V_noz (the nozzle
            # exit velocity, pitot-referenced) for BOTH free and cylinder cases.
            # The idealised gap velocity V_gap = V_noz·B/(B-D) was dropped entirely:
            # it assumed full closed-channel confinement, but the cylinder sits at an
            # OPEN jet exit, so it over-estimated the real throat speed (~1.7× too
            # large). Using V_noz makes the cylinder U/U0 directly comparable to the
            # free jet.
            _has_cyl = 'Cyl' in os.path.basename(os.path.normpath(folder))
            u0_ref = v_noz_ref            # V_noz for free AND cylinder cases
            u_u0_ref = u_v_noz_ref        # 1-sigma uncertainty of U0_ref
            self.log(f"U0 normalisation reference = {u0_ref:.3f} m/s (V_noz) "
                     f"[{'cylinder' if _has_cyl else 'free jet'}].")

            data_points = []
            for i, path in enumerate(csv_files):
                file_name = os.path.basename(path)
                try:
                    df = pd.read_csv(path)
                    if 'X/B' not in df.columns or 'Y/B' not in df.columns:
                        continue

                    x_nd = df['X/B'].iloc[0]
                    y_nd = df['Y/B'].iloc[0]

                    cols_to_exclude = ['X/B', 'Y/B', 'X_phys (mm)', 'Y_phys (mm)', 'Grid_X (mm)', 'Grid_Y (mm)', 'Time (s)']
                    cols_to_mean = [col for col in df.columns if col not in cols_to_exclude]
                    
                    file_means = df[cols_to_mean].mean().to_dict()
                    u_mean = file_means.get('u (m/s)', 0)
                    if 'u (m/s)' in df.columns:
                        u_std = df['u (m/s)'].std()
                        u_mean = file_means.get('u (m/s)', 0)
                        # Calculate I_u (preventing division by zero)
                        file_means['u_rms (m/s)'] = u_std
                        file_means['I_u (%)'] = (u_std / abs(u_mean) * 100.0) if abs(u_mean) > _IU_GUARD else 0.0

                    if 'v (m/s)' in df.columns:
                        v_std = df['v (m/s)'].std()
                        v_mean_val = file_means.get('v (m/s)', 0)
                        file_means['v_rms (m/s)'] = v_std
                        file_means['I_v (%)'] = (v_std / abs(u_mean) * 100.0) if abs(u_mean) > _IU_GUARD else 0.0

                    if 'w (m/s)' in df.columns:
                        w_std = df['w (m/s)'].std()
                        file_means['w_rms (m/s)'] = w_std
                        file_means['I_w (%)'] = (w_std / abs(u_mean) * 100.0) if abs(u_mean) > _IU_GUARD else 0.0

                    # Combined turbulence intensity Iuvw = sqrt(u² + v² + w²) / |U| × 100
                    _u_sq = file_means.get('u_rms (m/s)', 0.0)**2
                    _v_sq = file_means.get('v_rms (m/s)', 0.0)**2
                    _w_sq = file_means.get('w_rms (m/s)', 0.0)**2
                    if abs(u_mean) > _IU_GUARD:
                        file_means['I_uvw (%)'] = (np.sqrt(_u_sq + _v_sq + _w_sq) / abs(u_mean)) * 100.0
                    else:
                        file_means['I_uvw (%)'] = 0.0

                    # Velocity magnitude: sqrt(u² + v² + w²)
                    _um = file_means.get('u (m/s)', 0.0)
                    _vm = file_means.get('v (m/s)', 0.0)
                    _wm = file_means.get('w (m/s)', 0.0)
                    file_means['Vel_Mag (m/s)'] = np.sqrt(_um**2 + _vm**2 + _wm**2)

                    # --- 1-sigma uncertainties (UNCERTAINTY_SPEC.md §3) ---
                    # Mean velocity components / magnitude: the calibrated Cobra
                    # per-mean floor is combined with a POSITION term
                    # (probe X/Y placement uncertainty × local velocity gradient)
                    # in a post-pivot step below — these companion maps are NOT
                    # written here (they need neighbouring points for the gradient).
                    # Turbulence intensities I = rms/|u|: near-invariant under the
                    # affine calibration, so this is an INFORMATIONAL column
                    # (decision 2) — we propagate the dominant denominator term
                    # uI/I = u_umean/|umean| and neglect the rms statistical term.
                    if abs(u_mean) > _IU_GUARD:
                        _rel_umean = _U_VEL_MEAN / abs(u_mean)
                        file_means['u_I_u (%)']   = file_means.get('I_u (%)', 0.0)   * _rel_umean
                        file_means['u_I_v (%)']   = file_means.get('I_v (%)', 0.0)   * _rel_umean
                        file_means['u_I_w (%)']   = file_means.get('I_w (%)', 0.0)   * _rel_umean
                        file_means['u_I_uvw (%)'] = file_means.get('I_uvw (%)', 0.0) * _rel_umean
                    else:
                        file_means['u_I_u (%)'] = file_means['u_I_v (%)'] = 0.0
                        file_means['u_I_w (%)'] = file_means['u_I_uvw (%)'] = 0.0

                    file_means['X/B'] = x_nd
                    file_means['Y/B'] = y_nd
                    data_points.append(file_means)
                    
                except Exception as e:
                    self.log(f"  -> ERROR reading {file_name}: {e}")

            if not data_points:
                self.log("No valid data points extracted.")
                self.root.after(0, lambda: self.btn_process.config(state="normal"))
                return

            master_df = pd.DataFrame(data_points)

            # Centreline-only: single Y/B row measured (Free-jet traverse).
            # Transverse-map: multiple Y/B rows (plate-present traverse).
            centerline_only = master_df['Y/B'].nunique() == 1

            dir_1d_csv = os.path.join(folder, "1D_Profiles_Results", "CSV")
            dir_2d_csv = os.path.join(folder, "2D_Profiles_Results", "CSV")
            os.makedirs(dir_1d_csv, exist_ok=True)
            os.makedirs(dir_2d_csv, exist_ok=True)

            # Single output directory per case type — one CSV per variable, no
            # duplicate _Centerline extracts.
            out_csv_dir = dir_1d_csv if centerline_only else dir_2d_csv

            vars_to_map = [col for col in master_df.columns if col not in ['X/B', 'Y/B']]

            for var in vars_to_map:
                self.log(f"Mapping variable: {var}")
                pivot_map = pd.pivot_table(master_df, values=var, index='Y/B', columns='X/B', aggfunc="mean")
                pivot_map = pivot_map.sort_index(ascending=False)
                safe_var_name = var.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('(', '').replace(')', '')

                # --- JET LITERATURE METRICS EXTRACTION ---
                if var == 'u (m/s)':
                    center_y = min(pivot_map.index, key=lambda y: abs(y - 0.0))
                    u_center = pivot_map.loc[center_y].dropna()

                    u_max = u_center.max()
                    u_norm = u_center / u_max if u_max > 0 else u_center

                    # Calculate L_c (Potential Core Length)
                    idx_peak = u_center.idxmax()
                    u_post_peak = u_norm[u_norm.index >= idx_peak]
                    decay_points = u_post_peak[u_post_peak < _LC_THRESH]
                    L_c = decay_points.index[0] if not decay_points.empty else np.nan

                    # Calculate K (Plane Jet Decay Rate) and C (Intercept)
                    K_decay = np.nan
                    C_int = np.nan
                    if not pd.isna(L_c):
                        fit_data = u_norm[u_norm.index > L_c]
                        if len(fit_data) >= 2:
                            x_vals = fit_data.index.values
                            y_vals = (1.0 / fit_data.values)**2 # (U_max/U_c)^2
                            slope, intercept = np.polyfit(x_vals, y_vals, 1)
                            K_decay = slope
                            C_int = intercept

                    # --- Propagate the velocity uncertainty to the decay metrics ---
                    # Per centerline point the 1-sigma velocity uncertainty is the
                    # Cobra mean floor + the streamwise-gradient POSITION term
                    # (uX·dU/dx). Monte-Carlo: perturb the centerline U profile by
                    # that, recompute (L_c, K, C) each draw, take the spread.
                    def _decay_metrics(uc_vals, x_idx):
                        _um = np.nanmax(uc_vals)
                        if not (_um > 0):
                            return (np.nan, np.nan, np.nan)
                        _un = uc_vals / _um
                        _ip = x_idx[int(np.nanargmax(uc_vals))]
                        _m = x_idx >= _ip
                        _below = x_idx[_m][_un[_m] < _LC_THRESH]
                        _lc = float(_below[0]) if len(_below) else np.nan
                        _k = _c = np.nan
                        if not np.isnan(_lc):
                            _fm = x_idx > _lc
                            if int(_fm.sum()) >= 2:
                                _k, _c = np.polyfit(x_idx[_fm], (1.0 / _un[_fm]) ** 2, 1)
                        return (_lc, _k, _c)

                    u_L_c = u_K = u_C = np.nan
                    # Virtual origin X0/B = -C/K and its propagated 1-sigma.
                    X0_B = u_X0_B = np.nan
                    try:
                        _B_dec = (float(df_params['B_mm'].iloc[-1])
                                  if (df_params is not None and 'B_mm' in df_params.columns) else 30.0)
                        _xidx = u_center.index.values.astype(float)
                        _uc_vals = u_center.values.astype(float)
                        _xphys = _xidx * _B_dec
                        _dudx = np.gradient(_uc_vals, _xphys) if len(_xphys) >= 2 else np.zeros_like(_uc_vals)
                        _u_uc = np.sqrt(_U_VEL_MEAN ** 2 + (_dudx * _U_POS_X) ** 2)  # per-point 1-sigma
                        _rng = np.random.default_rng(0)
                        _Ls, _Ks, _Cs = [], [], []
                        for _ in range(300):
                            _l, _k, _c = _decay_metrics(_uc_vals + _rng.normal(0.0, _u_uc), _xidx)
                            _Ls.append(_l); _Ks.append(_k); _Cs.append(_c)
                        _Ls = np.array(_Ls, float); _Ks = np.array(_Ks, float); _Cs = np.array(_Cs, float)
                        if np.isfinite(_Ls).any(): u_L_c = float(np.nanstd(_Ls))
                        if np.isfinite(_Ks).any(): u_K = float(np.nanstd(_Ks))
                        if np.isfinite(_Cs).any(): u_C = float(np.nanstd(_Cs))
                        # Virtual origin X0/B = -C/K. Prefer the per-draw spread of
                        # X0/B (it captures the K,C covariance from the shared fit,
                        # which the quotient formula cannot); fall back to the
                        # quotient rule on the stored (K,C,u_K,u_C) only if the MC
                        # draws are unusable. (UNCERTAINTY_SPEC.md §9.1)
                        if not pd.isna(K_decay) and K_decay != 0 and not pd.isna(C_int):
                            X0_B = -C_int / K_decay
                        with np.errstate(divide='ignore', invalid='ignore'):
                            _X0s = -_Cs / _Ks
                        _X0s = _X0s[np.isfinite(_X0s)]
                        if _X0s.size:
                            u_X0_B = float(np.nanstd(_X0s))
                        elif not pd.isna(X0_B) and not pd.isna(u_K) and not pd.isna(u_C):
                            u_X0_B = abs(X0_B) * np.sqrt((u_C / C_int) ** 2 + (u_K / K_decay) ** 2)
                    except Exception:
                        pass

                    # Calculate C_profile
                    c_profile = v_noz_ref / u_max if u_max > 0 else np.nan

                    # Velocity summary always goes to 1D_Profiles_Results so that
                    # superposition.py and graphs_1D_visualizer find it consistently.
                    # u_max comes from already-calibrated Raw_Data, so it is the
                    # CALIBRATED peak (pitot reference). Store both states
                    # (PROTOCOL §10): U_max_ms is the canonical (calibrated) value
                    # used for downstream normalisation; U_max_raw_ms =
                    # (U_max_ms - b)/a is the raw Cobra peak; U_max_calibrated_ms
                    # is an explicit alias.
                    u_max_raw = (u_max - _CAL_B) / _CAL_A if _CAL_A else u_max
                    # 1-sigma uncertainties (UNCERTAINTY_SPEC.md §3): the calibrated
                    # peak carries the calibrated Cobra floor; the raw peak the raw
                    # floor; U0_ref the propagated V_noz/V_gap uncertainty from Re_Ve.
                    df_vel_sum = pd.DataFrame([{'L_c_XB': L_c, 'u_L_c_XB': u_L_c,
                                                'K_decay': K_decay, 'u_K_decay': u_K,
                                                'C_intercept': C_int, 'u_C_intercept': u_C,
                                                'X0_B': X0_B, 'u_X0_B': u_X0_B,
                                                'U_max_ms': u_max, 'u_U_max_ms': _U_VEL_MEAN,
                                                'U_max_calibrated_ms': u_max, 'u_U_max_calibrated_ms': _U_VEL_MEAN,
                                                'U_max_raw_ms': u_max_raw, 'u_U_max_raw_ms': _CAL_UFLOOR,
                                                'cal_slope_a': _CAL_A, 'cal_intercept_b': _CAL_B,
                                                'cal_fit_SE_ms': _CAL_SE,
                                                'U0_ref_ms': u0_ref, 'u_U0_ref_ms': u_u0_ref,
                                                'C_profile': c_profile}])
                    df_vel_sum.to_csv(os.path.join(dir_1d_csv, "Global_Velocity_Summary.csv"), index=False)
                    self.log(f" -> Metrics: L_c={L_c:.2f}B | K={K_decay:.3f} | C_profile={c_profile:.2f}")

                # --- STANDARD SAVING LOGIC ---
                # One map CSV + XLSX per variable; folder determined by case type.
                out_xlsx_subdir = "1D_Profiles_Results" if centerline_only else "2D_Profiles_Results"
                out_xlsx_dir = os.path.join(folder, out_xlsx_subdir, "XLSX")
                os.makedirs(out_xlsx_dir, exist_ok=True)
                pivot_map.to_csv(os.path.join(out_csv_dir, f"Map_{safe_var_name}.csv"))
                pivot_map.to_excel(os.path.join(out_xlsx_dir, f"Map_{safe_var_name}.xlsx"))

            # --- Velocity VALUE-uncertainty companion maps (Map_u_*) ------------
            # The companion map stores the POSITION-FREE value 1-sigma (the Cobra
            # mean floor ⊕ calibration SE = _U_VEL_MEAN, spatially
            # uniform). Probe-placement uncertainty is deliberately NOT folded in
            # here: the plotters show it as a separate HORIZONTAL (position) error
            # bar (u_pos_x/B, u_pos_y/B). Folding it into the value as well — via the
            # local gradient (dV/dx·uX) — would double-count position, since the same
            # ±placement would then appear both as the X bar and inside this Y value.
            # The decay Monte-Carlo (above) keeps the position-gradient term in its
            # OWN scalar propagation of (L_c, K, C) — that is a derived metric, not a
            # plot axis, so it is unaffected by this map.
            try:
                _u_xlsx_dir = os.path.join(folder,
                    "1D_Profiles_Results" if centerline_only else "2D_Profiles_Results", "XLSX")
                os.makedirs(_u_xlsx_dir, exist_ok=True)
                _vel_unc = [('u (m/s)', 'u_u_m_s'), ('v (m/s)', 'u_v_m_s'),
                            ('w (m/s)', 'u_w_m_s'), ('Vel_Mag (m/s)', 'u_Vel_Mag_m_s')]
                for _vcol, _ustem in _vel_unc:
                    if _vcol not in master_df.columns:
                        continue
                    _pv = pd.pivot_table(master_df, values=_vcol, index='Y/B',
                                         columns='X/B', aggfunc='mean').sort_index(ascending=False)
                    # Position-free value 1-sigma (uniform = the calibrated mean floor).
                    _uval = np.full(_pv.values.shape, _U_VEL_MEAN, dtype=float)
                    _u_pv = pd.DataFrame(_uval, index=_pv.index, columns=_pv.columns)
                    _u_pv.to_csv(os.path.join(out_csv_dir, f"Map_{_ustem}.csv"))
                    try:
                        _u_pv.to_excel(os.path.join(_u_xlsx_dir, f"Map_{_ustem}.xlsx"))
                    except Exception:
                        pass
                self.log(" -> Velocity value-uncertainty maps written (position-free; "
                         "placement is shown separately as the X error bar).")
            except Exception as _e:
                self.log(f"  -> WARNING: velocity value-uncertainty maps skipped: {_e}")

            self.log("\n--- PROCESSING COMPLETE ---")
            
        except Exception as e:
            self.log(f"\nCRITICAL ERROR during processing: {e}")
        finally:
            self.root.after(0, lambda: self.btn_process.config(state="normal"))

    def setup_ui(self):
        f_frame = tk.LabelFrame(self.root, text=" 1. Select Workspace ", padx=10, pady=10)
        f_frame.pack(padx=20, pady=15, fill="x")
        
        tk.Entry(f_frame, textvariable=self.folder_path, state="readonly", width=60).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(f_frame, text="Browse...", command=self.select_folder).pack(side="right", padx=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=20, pady=5, fill="x")
        self.btn_process = tk.Button(btn_frame, text="CALCULATE MEANS & GENERATE MAPS", command=self.start_processing, bg="#28a745", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_process.pack(fill="x")

        tk.Label(self.root, text="Console Log:").pack(anchor="w", padx=20, pady=(10, 0))
        self.log_text = tk.Text(self.root, height=12, state="disabled", bg="#f4f4f4")
        self.log_text.pack(padx=20, pady=(0, 20), fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataMeansApp(root)
    root.mainloop()