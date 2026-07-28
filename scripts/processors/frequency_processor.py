# =============================================================================
# FREQUENCY_PROCESSOR
# =============================================================================
# Purpose:
#   Performs spectral analysis on the per-grid-point velocity time series
#   produced by read_th_file_processor. For each measurement location, the
#   transverse velocity fluctuation v' is decomposed into its power spectral
#   density (PSD) using Welch's method (fs = 2000 Hz, nperseg = 2048). From
#   each PSD, three aerodynamic metrics are extracted: (A) the dominant
#   shedding frequency (Hz), identified as the peak above a case-dependent
#   threshold (75 Hz for free jet, 400 Hz for Cyl12); (B) the local Strouhal
#   number St = f * L / U_ref, where L = D (cylinder diameter) for forced cases
#   / B (nozzle height) for the free jet, and U_ref is the nozzle exit velocity
#   V_noz derived from the tunnel dP measurement; and (C) the vortex
#   shedding coherence, defined as the fraction
#   of total PSD energy concentrated near the dominant peak. Results are saved
#   as spatial maps and global summary tables.
#
# Inputs:
#   - Per-grid-point raw CSV files:
#       experiments/<case>/Processed_CSVs/Raw_Data/*.csv
#   - Flow parameters (D2_mm, B_mm, V_noz_ms) from:
#       experiments/<case>/Flow_Data.csv
#
# Outputs:
#   - Per-point PSD data:
#       experiments/<case>/Processed_CSVs/FFT/<grid>_FFT.csv
#   - Flat metrics table (frequency, St, coherence per grid point):
#       experiments/<case>/1D_Profiles_Results/Frequency_Metrics_Flat.csv
#         or 2D_Profiles_Results/ depending on measurement dimensionality
#   - Spatial maps (CSV and XLSX) for each metric:
#       experiments/<case>/Frequency_Results/CSV/Map_Dominant_Freq_1st.csv
#       experiments/<case>/Frequency_Results/CSV/Map_Strouhal_Number.csv
#       experiments/<case>/Frequency_Results/CSV/Map_Shedding_Coherence.csv
#   - Global mean frequency and Strouhal summary:
#       experiments/<case>/Frequency_Results/CSV/Global_Frequency_Summary.csv
#
# Dependencies:
#   - None (standalone processor; imported by pipeline.py and usage.py)
#
# Usage:
#   - Standalone: python frequency_processor.py  (opens Tkinter GUI)
#   - Via pipeline: called programmatically by pipeline.py (Step 4)
#   - Via hub:      launched from usage.py as "FFT & Aerodynamics Processor"
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
import glob
import threading
from scipy import signal

# --- Config loader ---
def _load_cfg(filename, value_col='value'):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    return pd.read_excel(_p, index_col='parameter')[value_col]

try:
    _cfg_acq   = _load_cfg('config_acquisition.csv')
    _cfg_flow  = _load_cfg('config_flow_regime.csv')
    _cfg_lit   = _load_cfg('config_literature_nu.csv')
    _cfg_geom  = _load_cfg('config_geometry.csv')   # geometry now uses the 'value' column
    _cfg_cases = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cases.xlsx'))
except Exception as _cfg_err:
    print(f"[frequency_processor] WARNING: Could not load config files: {_cfg_err}. Falling back to hardcoded defaults.")
    _cfg_acq   = None
    _cfg_flow  = None
    _cfg_lit   = None
    _cfg_geom  = None
    _cfg_cases = None

# --- ENERGY-CASCADE / TRANSITION-CRITERION CONSTANTS (Task 3) ---
# A measurement point is flagged as organised vortex shedding when BOTH:
#   (1) its shedding coherence (fraction of PSD energy in the dominant peak)
#       exceeds _ORG_COH_MIN_PCT, AND
#   (2) the high-frequency energy-cascade slope sits near the Kolmogorov
#       inertial value of -5/3, i.e. |slope - (-5/3)| <= _CASCADE_SLOPE_TOL.
# Note on the coherence threshold: a literal "coherence >= ~0.5" is far too high
# for this metric, whose per-point values are typically much smaller (the
# turbulent shear-layer edges sit near the noise floor, the organised core well
# above it). _ORG_COH_MIN_PCT is therefore set to a moderate percentage that
# cleanly separates the organised core (Y/B ~ 0) from the shear-layer edges;
# tune it to your own dataset.
_ORG_COH_MIN_PCT    = 15.0          # [%] min shedding coherence for "organised"
_CASCADE_SLOPE_REF  = -5.0 / 3.0    # Kolmogorov inertial-subrange slope
_CASCADE_SLOPE_TOL  = 0.7           # tolerance band around the -5/3 reference

# Shared RSS uncertainty-propagation helper (Task 5)
import sys as _sys_unc
_sys_unc.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
try:
    from uncertainty import (freq_resolution_hz as _u_freq_res, rel_strouhal as _u_rel_st,
                             rel_strouhal_from_relV as _u_rel_st_relv,
                             U_B_MM as _U_B, U_D2_MM as _U_D2)
    _HAS_UNC = True
except Exception:
    _HAS_UNC = False
    _U_B, _U_D2 = 0.1, 0.1

try:
    from air_properties import calc_air_properties as _calc_air_props
    _HAS_AIR = True
except Exception:
    _HAS_AIR = False

class FrequencyProcessorApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("Aerodynamic FFT Processor")
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
        folder = filedialog.askdirectory(title="Select Main Data Folder")
        if folder:
            self.folder_path.set(folder)
            self.log(f"Selected workspace: {folder}")

    def start_processing(self):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder first.")
            return

        self.btn_process.config(state="disabled")
        threading.Thread(target=self.process_fft_data, args=(folder,)).start()

    def process_fft_data(self, main_folder):
        folder_name = os.path.basename(os.path.normpath(main_folder))
        if _cfg_cases is not None:
            _match = _cfg_cases[_cfg_cases['case_pattern'].apply(lambda p: str(p) in folder_name)]
            peak_threshold = float(_match['peak_threshold_hz'].iloc[0]) if not _match.empty else 75.0
            if not _match.empty:
                has_cyl = str(_match['has_cylinder'].iloc[0]).strip().lower() in ('true', '1', 'yes')
            else:
                has_cyl = 'Cyl' in folder_name
        else:
            peak_threshold = 400.0 if 'Cyl12' in folder_name else 75.0
            has_cyl = 'Cyl' in folder_name

        self.log("\n--- FETCHING FLOW PARAMETERS ---")
        self.log(f"Detected case: {folder_name} -> Setting Peak Detection Threshold to {peak_threshold} Hz")
        
        d_mm = None
        B_mm = None
        v_noz_global = None
        u_v_noz_global = None        # 1-sigma uncertainty of V_noz (from Re_Ve)
        dp_global = None
        temp_global = None
        patm_global = None

        search_paths = [
            os.path.join(main_folder, "Flow_Data*.csv"),
            os.path.join(main_folder, "**", "Flow_Data*.csv")
        ]
        
        param_file = None
        for sp in search_paths:
            files = glob.glob(sp, recursive=True)
            if files:
                param_file = max(files, key=os.path.getctime)
                break
                
        if param_file:
            try:
                df_params = pd.read_csv(param_file)
                if 'D2_mm' in df_params.columns:
                    d_mm = float(df_params['D2_mm'].dropna().iloc[-1])
                    self.log(f"Success: Found Cylinder Diameter D2 = {d_mm} mm")
                if 'B_mm' in df_params.columns:
                    B_mm = float(df_params['B_mm'].dropna().iloc[-1])
                    self.log(f"Success: Found Nozzle Height B = {B_mm} mm")
                elif 'H_noz_mm' in df_params.columns: # Fallback for older exports
                    B_mm = float(df_params['H_noz_mm'].dropna().iloc[-1])
                    self.log(f"Success: Found Nozzle Height B = {B_mm} mm")
                elif 'Nozzle_h_mm' in df_params.columns: # Fallback for read_th param merge
                    B_mm = float(df_params['Nozzle_h_mm'].dropna().iloc[-1])
                    self.log(f"Success: Found Nozzle Height B = {B_mm} mm")
                    
                if 'V_noz_ms' in df_params.columns:
                    v_noz_global = float(df_params['V_noz_ms'].dropna().iloc[-1])
                    self.log(f"Success: Found Nozzle Velocity V_noz = {v_noz_global} m/s")
                # 1-sigma velocity uncertainty propagated by Re_Ve_processor
                # (UNCERTAINTY_SPEC.md §3) — used for the Strouhal/Re uncertainty.
                if 'u_V_noz_ms' in df_params.columns:
                    u_v_noz_global = float(df_params['u_V_noz_ms'].dropna().iloc[-1])
                if 'dP_Pa' in df_params.columns:
                    dp_global = float(df_params['dP_Pa'].dropna().iloc[-1])
                if 'Temp_C' in df_params.columns:
                    temp_global = float(df_params['Temp_C'].dropna().iloc[-1])
                if 'P_atm_Pa' in df_params.columns:
                    patm_global = float(df_params['P_atm_Pa'].dropna().iloc[-1])
            except Exception as e:
                self.log(f"Error reading Flow_Data.csv: {e}")
                
        if d_mm is None:
            self.log("Failed to auto-fetch cylinder diameter. Defaulting to 0.0 mm.")
            d_mm = 0.0
        if B_mm is None:
            _B_default = float(_cfg_geom['nozzle_height_B']) if _cfg_geom is not None else 18.0
            self.log(f"Failed to auto-fetch nozzle height. Defaulting to {_B_default} mm.")
            B_mm = _B_default
        # --- STROUHAL CHARACTERISTIC LENGTH ---
        # The Strouhal number uses the physically appropriate length scale per
        # configuration: for the cylinder-FORCED cases the relevant instability
        # is the cylinder vortex shedding, so L = D (cylinder diameter, 12 mm);
        # for the FREE jet there is no cylinder and the relevant scale is the
        # nozzle height, so L = B. The REFERENCE VELOCITY is V_noz (nozzle exit
        # velocity from the tunnel dP measurement), read from Flow_Data.csv and
        # applied in the Strouhal block as St = f_dom * L / V_noz.
        if has_cyl:
            _D_default = float(_cfg_geom['cylinder_diameter_D2']) if _cfg_geom is not None else 12.0
            L_char_mm = d_mm if (d_mm is not None and d_mm > 0.0) else _D_default
            self.log(f"Cylinder-forced case: using L = D = {L_char_mm} mm for Strouhal calculation.")
        else:
            L_char_mm = B_mm
            self.log(f"Free-jet case: using L = B = {L_char_mm} mm for Strouhal calculation.")

        raw_dir = os.path.join(main_folder, "Processed_CSVs", "Raw_Data")
        old_processed_dir = os.path.join(main_folder, "Processed_CSVs")
        
        source_dir = raw_dir if os.path.exists(raw_dir) else (old_processed_dir if os.path.exists(old_processed_dir) else main_folder)
        csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
        total_files = len(csv_files)

        if total_files == 0:
            self.log(f"No CSV files found in {source_dir}. Aborting.")
            self.root.after(0, lambda: self.btn_process.config(state="normal"))
            return

        self.log("\n--- STARTING FREQUENCY ANALYSIS ---")
        
        fft_dir = os.path.join(main_folder, "Processed_CSVs", "FFT")
        freq_dir = os.path.join(main_folder, "Frequency_Results")
        os.makedirs(fft_dir, exist_ok=True)
        os.makedirs(os.path.join(freq_dir, "CSV"), exist_ok=True)
        os.makedirs(os.path.join(freq_dir, "XLSX"), exist_ok=True)

        # --- REFERENCE VELOCITY (V_noz from pressure measurement) ---
        # V_noz is the theoretical nozzle exit velocity from Flow_Data.csv
        # (tunnel dP + area ratio + C_profile, PROTOCOL §7.1) — the operating
        # point, consistent with the Re used in thermal_processor.
        _dp_str = f"dP={dp_global:.1f} Pa" if dp_global is not None else "dP unknown"
        U_ref = v_noz_global if (v_noz_global is not None and v_noz_global > 0.01) else np.nan

        # Shedding reference velocity for the cylinder Strouhal number.
        # The idealised gap velocity V_gap = V_noz·B/(B-D) (full-confinement
        # continuity) OVER-estimates the real throat speed: the cylinder sits at an
        # OPEN jet exit, so the flow spreads/entrains rather than being fully forced
        # through the gap, and V_gap drives St below the canonical value. We instead
        # use the MEASURED peak Cobra velocity U_max (the empirical throat/convection
        # velocity, from mean_processor's Global_Velocity_Summary.csv), which recovers
        # a Strouhal number close to the canonical von Karman value; it was shown
        # to be far more robust than the idealised gap or a near-nozzle profile mean.
        # For the free jet there is no cylinder shedding, so the reference is V_noz.
        # Read the MEASURED peak Cobra velocity U_max (and its 1-sigma) from
        # mean_processor's Global_Velocity_Summary.csv.
        u_max_meas = u_max_unc = None
        for _gvs in glob.glob(os.path.join(main_folder, "**", "Global_Velocity_Summary.csv"),
                              recursive=True):
            try:
                _dfv = pd.read_csv(_gvs)
                _uc = next((c for c in ("U_max_ms", "U_max_calibrated_ms")
                            if c in _dfv.columns), None)
                if _uc:
                    _uv = float(pd.to_numeric(_dfv[_uc], errors="coerce").dropna().iloc[-1])
                    if np.isfinite(_uv) and _uv > 0.01:
                        u_max_meas = _uv
                        if 'u_U_max_ms' in _dfv.columns:
                            _uu = float(pd.to_numeric(_dfv['u_U_max_ms'], errors="coerce").dropna().iloc[-1])
                            u_max_unc = _uu if np.isfinite(_uu) else None
                        break
            except Exception:
                pass
        # TWO Strouhal reference velocities are reported (the idealised gap velocity
        # V_gap is no longer used at all):
        #   * THROAT (corrected)  = measured U_max for the cylinder, V_noz for the free
        #     jet -> a Strouhal number comparable to the canonical von Karman value.
        #   * APPROACH (nominal)  = V_noz, the textbook cylinder-St convention
        #     (confinement-inflated at this blockage ratio -> a higher St).
        if has_cyl and not np.isnan(U_ref) and u_max_meas is not None:
            U_shed, u_U_shed, _shed_lbl = u_max_meas, u_max_unc, "U_max (measured throat)"
        else:
            U_shed, u_U_shed = U_ref, u_v_noz_global
            _shed_lbl = "V_noz" if not has_cyl else "V_noz (U_max unavailable)"
        U_approach, u_U_approach = U_ref, u_v_noz_global   # nominal cylinder-St convention
        if not np.isnan(U_ref):
            self.log(f"Reference velocity V_noz = {U_ref:.3f} m/s ({_dp_str}); "
                     f"St throat ref = {U_shed:.3f} m/s [{_shed_lbl}]; "
                     f"St approach ref = {U_approach:.3f} m/s [V_noz].")
        else:
            self.log("WARNING: V_noz not found in Flow_Data.csv; Strouhal and Re will be NaN.")

        # Reynolds numbers built on the shedding reference velocity U_shed (the
        # measured U_max for cylinder cases, V_noz for the free jet). Re_noz on B,
        # Re_cyl on D. NOTE: the thermal operating Reynolds number uses V_noz (the
        # pitot-referenced nozzle velocity); these spectral Re are on U_shed and are
        # used only for the Strouhal/spectral diagnostics, not the thermal baseline.
        re_noz = re_cyl = np.nan
        if not np.isnan(U_shed) and U_shed > 0.01 and _HAS_AIR \
                and temp_global is not None and patm_global is not None:
            try:
                _rho, _mu, _, _ = _calc_air_props(temp_global, patm_global)
                re_noz = (_rho * U_shed * (B_mm / 1000.0)) / _mu
                if has_cyl and L_char_mm > 0:
                    re_cyl = (_rho * U_shed * (L_char_mm / 1000.0)) / _mu
                self.log(f"Re_noz{'(gap)' if has_cyl else ''} = {re_noz:.0f}"
                         + (f" | Re_cyl(gap) = {re_cyl:.0f}" if has_cyl else ""))
            except Exception as _e:
                self.log(f"  -> could not compute Re: {_e}")

        map_data = []

        for i, path in enumerate(csv_files):
            file_name = os.path.basename(path)
            col_name = os.path.splitext(file_name)[0]
            self.log(f"[{i+1}/{total_files}] Crunching aerodynamics for: {file_name}")

            try:
                df = pd.read_csv(path)
                if not all(k in df.columns for k in ['Time (s)', 'u (m/s)', 'v (m/s)', 'X/B', 'Y/B']):
                    continue

                x_nd, y_nd = df['X/B'].iloc[0], df['Y/B'].iloc[0]
                time = df['Time (s)'].values
                u_vel, v_vel = df['u (m/s)'].values, df['v (m/s)'].values
                
                u_mean = np.mean(u_vel)
                v_mean = np.mean(v_vel)
                U_mag = np.sqrt(u_mean**2 + v_mean**2)
                
                fs = float(_cfg_acq['sampling_frequency_Fs']) if _cfg_acq is not None else 2000.0
                v_fluct = v_vel - v_mean
                _nperseg = int(_cfg_acq['welch_nperseg']) if _cfg_acq is not None else 2048
                _nperseg = min(len(v_fluct), _nperseg)
                # Hann window + 50% overlap, per PROTOCOL §9.1 (these are also
                # scipy's defaults; stated explicitly here for reproducibility).
                freqs, psd = signal.welch(v_fluct, fs=fs, window='hann',
                                          nperseg=_nperseg, noverlap=_nperseg // 2)
                
                df_fft = pd.DataFrame({'Frequency (Hz)': freqs, 'PSD (m^2/s^2/Hz)': psd})
                df_fft.to_csv(os.path.join(fft_dir, f"{col_name}_FFT.csv"), index=False)

                # A. Mean Resonance — dominant shedding peak ABOVE the case
                # noise-floor threshold (peak_threshold_hz from config: 400 Hz for
                # Cyl12, 75 Hz for free jet). The shedding peak genuinely sits above
                # this threshold, so gating the argmax search to f > peak_threshold
                # both (a) locks onto the correct shedding peak and (b) avoids the
                # spurious low-frequency picks on the energetic Kolmogorov rolloff.
                # This is the SAME detection used per-point in frequency_visualizer,
                # so the individual PSD figures and the global/superposed means agree.
                min_shed_freq = peak_threshold
                valid_idx = np.where(freqs > min_shed_freq)[0]
                if len(valid_idx) > 0:
                    p_i = valid_idx[np.argmax(psd[valid_idx])]
                    mean_resonance = freqs[p_i]
                else:
                    mean_resonance = np.nan

                # B. Strouhal Number (per-point map) on the THROAT velocity U_shed:
                #    Cylinder: L = D, U_shed = U_max (measured throat velocity).
                #    Free jet: L = B, U_shed = V_noz.
                #    The nominal approach-velocity (V_noz) St is reported globally too.
                # The reference velocity is V_noz (nozzle exit velocity from dP,
                # computed once above), and the characteristic length L is the
                # cylinder diameter D for forced cases / the nozzle height B for
                # the free jet (L_char_mm). If V_noz could not be determined, or
                # no shedding peak was detected, the Strouhal number is left as NaN.
                u_ref = U_shed if (not np.isnan(U_shed) and U_shed > 0.01) else np.nan
                if np.isnan(mean_resonance) or np.isnan(u_ref):
                    strouhal = np.nan
                else:
                    strouhal = (mean_resonance * (L_char_mm / 1000.0)) / u_ref

                # C. Shedding Coherence (% of total energy in the peak)
                # If no peak was detected (mean_resonance = NaN), coherence is 0 (no tonal energy).
                total_energy = np.sum(psd)
                _coh_win = float(_cfg_lit['strouhal_min_coherence_window']) if _cfg_lit is not None else 10.0
                if not np.isnan(mean_resonance) and total_energy > 0:
                    peak_window_idx = np.where((freqs > mean_resonance - _coh_win) & (freqs < mean_resonance + _coh_win))[0]
                    peak_energy = np.sum(psd[peak_window_idx])
                    coherence = peak_energy / total_energy * 100.0
                else:
                    coherence = 0.0

                # D. Energy-cascade (Kolmogorov) slope — log-log fit of PSD over
                # the inertial subrange, i.e. frequencies above the dominant
                # shedding peak (f > f_dom * cascade_multiplier) up to Nyquist.
                # A fully developed inertial cascade gives a slope near -5/3.
                _casc_mult = float(_cfg_lit['strouhal_cascade_multiplier']) if _cfg_lit is not None else 1.5
                _small = float(_cfg_flow['kolmogorov_small_value']) if _cfg_flow is not None else 1e-12
                _f_lo = (mean_resonance * _casc_mult) if not np.isnan(mean_resonance) else (min_shed_freq * _casc_mult)
                casc_mask = (freqs >= _f_lo) & (freqs > 0) & (psd > 0)
                if int(np.count_nonzero(casc_mask)) >= 5:
                    cascade_slope = float(np.polyfit(np.log10(freqs[casc_mask]),
                                                     np.log10(psd[casc_mask] + _small), 1)[0])
                else:
                    cascade_slope = np.nan

                # E. Organised-shedding transition flag — 1 where the point is a
                # coherent shedding core AND its high-frequency cascade is
                # consistent with the -5/3 inertial law. When the Cobra sampling
                # rate (fs = 2000 Hz, Nyquist = 1000 Hz) leaves only a narrow band
                # between the shedding tone and Nyquist, the inertial band ABOVE the
                # tone is too narrow to resolve a -5/3 slope, so the slope is NaN at the
                # organised core. In that under-resolved case the cascade test is
                # treated as non-disqualifying and the coherence governs the
                # flag; where a slope IS resolvable it must lie within tolerance
                # of -5/3. This yields the intended delineation: the coherent
                # core (Y/B ~ 0) is flagged, the turbulent shear-layer edges
                # (Y/B ~ +/-0.3, low coherence) are not.
                _cascade_ok = (np.isnan(cascade_slope)
                               or abs(cascade_slope - _CASCADE_SLOPE_REF) <= _CASCADE_SLOPE_TOL)
                organized_flag = int((coherence >= _ORG_COH_MIN_PCT) and _cascade_ok)

                map_data.append({
                    'X/B': x_nd, 'Y/B': y_nd,
                    'Mean_Resonance_Hz': mean_resonance,
                    'Strouhal_Number': strouhal,
                    'Shedding_Coherence (%)': coherence,
                    'Energy_Cascade_Slope': cascade_slope,
                    'Organized_Shedding_Flag': organized_flag
                })

            except Exception as e:
                self.log(f"  -> ERROR processing {file_name}: {e}")

        if map_data:
            df_map = pd.DataFrame(map_data)

            # --- OUTPUT 2: Global Mean Summary ---
            self.log("\nCalculating Global Means...")
            # Filter to centerline rows only (Y/B ≈ 0, within ±0.1B) so that
            # cases with multiple cross-jet rows (e.g. Cyl1210PaPla with Y/B
            # from -0.32 to +0.32) do not dilute the shedding frequency estimate
            # with off-axis measurements.
            centerline_mask = df_map['Y/B'].abs() < 0.1
            df_centerline = df_map[centerline_mask] if centerline_mask.sum() > 0 else df_map
            if centerline_mask.sum() == 0:
                self.log("  WARNING: No Y/B ≈ 0 rows found; falling back to all rows for global means.")

            # Average only rows where a genuine shedding peak was detected (non-NaN, > 0).
            # NaN entries are broadband points with no real peak — they must not bias the mean.
            valid_freqs = df_centerline['Mean_Resonance_Hz'].dropna()
            valid_freqs = valid_freqs[valid_freqs > 0]
            global_mean_freq = valid_freqs.mean() if not valid_freqs.empty else 0.0

            valid_st = df_centerline['Strouhal_Number'].dropna()
            valid_st = valid_st[valid_st > 0]
            global_mean_st = valid_st.mean() if not valid_st.empty else 0.0

            # --- Task 3b: width-averaged near-plate coherent frequency ---
            # At the measurement station nearest the plate (maximum X/B), average
            # the dominant frequency over the coherent core, i.e. only the Y/B
            # points whose shedding coherence exceeds _ORG_COH_MIN_PCT. This
            # confirms whether organised shedding survives to the impingement
            # plate. For purely centerline (free-jet) cases the "band" collapses
            # to the single Y/B = 0 point, which is the expected behaviour.
            nearplate_xb = df_map['X/B'].max()
            _np_station = df_map[df_map['X/B'] == nearplate_xb]
            _np_core = _np_station[_np_station['Shedding_Coherence (%)'] >= _ORG_COH_MIN_PCT]
            _np_freqs = _np_core['Mean_Resonance_Hz'].dropna()
            _np_freqs = _np_freqs[_np_freqs > 0]
            nearplate_coherent_freq = _np_freqs.mean() if not _np_freqs.empty else np.nan
            nearplate_band_count = int(_np_freqs.shape[0])
            self.log(f" -> Near-plate (X/B={nearplate_xb:.2f}) coherent f_dom: "
                     f"{nearplate_coherent_freq:.1f} Hz over {nearplate_band_count} coherent Y/B point(s)."
                     if not np.isnan(nearplate_coherent_freq)
                     else f" -> Near-plate (X/B={nearplate_xb:.2f}): no coherent shedding band detected.")

            # Near-plate energy-cascade (Kolmogorov) transition slope: the width-
            # averaged inertial-range PSD slope across the station nearest the
            # plate (mean of the resolvable, non-NaN per-point slopes at that
            # X/B). Characterises how far the flow has cascaded toward developed
            # turbulence by the time it reaches the plate.
            _np_slopes = _np_station['Energy_Cascade_Slope'].dropna()
            nearplate_cascade_slope = _np_slopes.mean() if not _np_slopes.empty else np.nan
            nearplate_cascade_count = int(_np_slopes.shape[0])
            self.log(f" -> Near-plate (X/B={nearplate_xb:.2f}) energy-cascade slope: "
                     f"{nearplate_cascade_slope:.3f} over {nearplate_cascade_count} resolvable Y/B point(s)."
                     if not np.isnan(nearplate_cascade_slope)
                     else f" -> Near-plate (X/B={nearplate_xb:.2f}): cascade slope unresolved (band too narrow).")

            # --- RSS uncertainty propagation (Task 5) ---
            # f uncertainty = Welch bin width (fs/nperseg). St uncertainty
            # propagates the dP-based velocity uncertainty through V_noz:
            #   u_St/St = sqrt((u_f/f)^2 + (0.5*u_dP/dP)^2).
            _fs_sum = float(_cfg_acq['sampling_frequency_Fs']) if _cfg_acq is not None else 2000.0
            _nperseg_sum = int(_cfg_acq['welch_nperseg']) if _cfg_acq is not None else 2048
            freq_resolution_hz = u_st_pct = np.nan
            u_st_abs = u_re_noz = u_re_cyl = np.nan
            # APPROACH-velocity Strouhal (nominal convention, on V_noz). St scales as
            # 1/U, so St_approach = St_throat * U_shed/U_approach (same frequencies).
            global_mean_st_approach = u_st_app_abs = u_st_app_pct = np.nan
            if (not np.isnan(global_mean_st) and U_approach is not None
                    and not np.isnan(U_approach) and U_approach > 0.01
                    and not np.isnan(U_shed) and U_shed > 0.01):
                global_mean_st_approach = global_mean_st * U_shed / U_approach
            if _HAS_UNC:
                freq_resolution_hz = _u_freq_res(_fs_sum, _nperseg_sum)
                # Velocity relative uncertainty for the THROAT St/Re: the 1-sigma of
                # the shedding reference velocity (measured U_max for the cylinder,
                # V_noz for the free jet); fall back to the bare dP term if absent.
                _u_shed = u_U_shed
                _rel_v = (_u_shed / U_shed) if (_u_shed is not None and not np.isnan(U_shed) and U_shed > 0.01) else np.nan
                if global_mean_freq > 0:
                    if not np.isnan(_rel_v):
                        u_st_pct = _u_rel_st_relv(global_mean_freq, _rel_v, _fs_sum, _nperseg_sum) * 100.0
                    elif dp_global is not None and dp_global > 0:
                        u_st_pct = _u_rel_st(global_mean_freq, dp_global, _fs_sum, _nperseg_sum) * 100.0
                if not np.isnan(global_mean_st) and not np.isnan(u_st_pct):
                    u_st_abs = global_mean_st * u_st_pct / 100.0
                # APPROACH-St uncertainty (on V_noz)
                _rel_v_app = (u_U_approach / U_approach) if (u_U_approach is not None
                    and U_approach is not None and not np.isnan(U_approach) and U_approach > 0.01) else np.nan
                if global_mean_freq > 0 and not np.isnan(_rel_v_app):
                    u_st_app_pct = _u_rel_st_relv(global_mean_freq, _rel_v_app, _fs_sum, _nperseg_sum) * 100.0
                if not np.isnan(global_mean_st_approach) and not np.isnan(u_st_app_pct):
                    u_st_app_abs = global_mean_st_approach * u_st_app_pct / 100.0
                if not np.isnan(_rel_v):
                    # Reynolds characteristic-length geometry (UNCERTAINTY_SPEC §3):
                    # Re_noz on slot B -> B cancels (V∝1/B, Re∝V·B), remove uB/B;
                    # Re_cyl on cylinder D=L_char -> D independent, add uD/D.
                    _rel_B = (_U_B / B_mm) if B_mm else 0.0
                    if not np.isnan(re_noz):
                        u_re_noz = re_noz * (max(0.0, _rel_v**2 - _rel_B**2) ** 0.5)
                    if not np.isnan(re_cyl) and L_char_mm:
                        _rel_D = _U_D2 / L_char_mm
                        u_re_cyl = re_cyl * ((_rel_v**2 + _rel_D**2) ** 0.5)

            df_global_summary = pd.DataFrame([{
                'Global_Mean_Freq_Hz': global_mean_freq,
                'u_Global_Mean_Freq_Hz': freq_resolution_hz,
                'Global_Mean_St': global_mean_st,                  # THROAT (U_max) — primary
                'u_Global_Mean_St': u_st_abs,
                'Global_Mean_St_approach': global_mean_st_approach,  # NOMINAL (V_noz)
                'u_Global_Mean_St_approach': u_st_app_abs,
                'St_throat_Velocity_ms': U_shed,                    # = U_max (measured)
                'St_approach_Velocity_ms': U_approach,              # = V_noz
                'Ref_Velocity_ms': U_ref,
                'Re_noz': re_noz,
                'u_Re_noz': u_re_noz,
                'Re_cyl': re_cyl,
                'u_Re_cyl': u_re_cyl,
                'NearPlate_XB': nearplate_xb,
                'NearPlate_Coherent_Freq_Hz': nearplate_coherent_freq,
                'NearPlate_Coherent_Band_Count': nearplate_band_count,
                'NearPlate_Cascade_Slope': nearplate_cascade_slope,
                'NearPlate_Cascade_Count': nearplate_cascade_count,
                'Freq_Resolution_Hz': freq_resolution_hz,
                'u_St_pct': u_st_pct
            }])
            df_global_summary.to_csv(os.path.join(freq_dir, "CSV", "Global_Frequency_Summary.csv"), index=False)
            try:
                df_global_summary.to_excel(os.path.join(freq_dir, "XLSX", "Global_Frequency_Summary.xlsx"), index=False)
            except: pass
            self.log(f" -> Global Mean Frequency: {global_mean_freq:.1f} Hz | Global Mean St: {global_mean_st:.3f}")

            # --- OUTPUT 3: Per-Y/B measurement line profiles ---
            self.log("\nSaving per-measurement-line frequency profiles...")
            df_all = pd.DataFrame(map_data)

            # Always route flat metrics to 1D_Profiles_Results
            out_1d_dir = os.path.join(main_folder, "1D_Profiles_Results", "CSV")
            os.makedirs(out_1d_dir, exist_ok=True)
            out_1d_xlsx = os.path.join(main_folder, "1D_Profiles_Results", "XLSX")
            os.makedirs(out_1d_xlsx, exist_ok=True)

            df_all.to_csv(os.path.join(out_1d_dir, "Frequency_Metrics_Flat.csv"), index=False)
            try:
                df_all.to_excel(os.path.join(out_1d_xlsx, "Frequency_Metrics_Flat.xlsx"), index=False)
            except Exception: pass

            # Save one CSV per unique Y/B value
            unique_y = sorted(df_all['Y/B'].unique())
            center_y = min(unique_y, key=lambda y: abs(y - 0.0))

            for y_val in unique_y:
                df_y = df_all[df_all['Y/B'] == y_val].sort_values('X/B')
                safe_y = f"{y_val:.3f}".replace('.', 'p').replace('-', 'm')
                fname = f"Freq_Profile_Y{safe_y}.csv"
                df_y.to_csv(os.path.join(out_1d_dir, fname), index=False)
                if y_val == center_y:
                    df_y.to_csv(os.path.join(out_1d_dir, "Centerline_Freq_Profile.csv"), index=False)
                    try:
                        df_y.to_excel(os.path.join(out_1d_xlsx, "Centerline_Freq_Profile.xlsx"), index=False)
                    except Exception: pass

            self.log(f" -> Saved {len(unique_y)} Y/B profile(s) to 1D_Profiles_Results.")

            # --- OUTPUT 4: 2D spatial pivot maps (Y/B index x X/B columns) ---
            # Written to Frequency_Results/CSV so the 2D-map visualiser (Tab 2),
            # the near-plate superposed plots, and the transverse freq/St profiles
            # all have their pivot-table inputs. The module header documented
            # these outputs but they were never actually being written.
            self.log("\nSaving 2D frequency pivot maps...")
            _freq_csv_dir = os.path.join(freq_dir, "CSV")
            _freq_xlsx_dir = os.path.join(freq_dir, "XLSX")
            os.makedirs(_freq_csv_dir, exist_ok=True)
            os.makedirs(_freq_xlsx_dir, exist_ok=True)
            _pivot_specs = [
                ('Mean_Resonance_Hz',       'Map_Dominant_Freq_1st'),
                ('Strouhal_Number',         'Map_Strouhal_Number'),
                ('Shedding_Coherence (%)',  'Map_Shedding_Coherence'),
                ('Energy_Cascade_Slope',    'Map_Energy_Cascade_Slope'),
                ('Organized_Shedding_Flag', 'Map_Transition_Flag'),
            ]
            _n_maps = 0
            for _col, _stem in _pivot_specs:
                if _col not in df_all.columns:
                    continue
                try:
                    _pivot = df_all.pivot_table(index='Y/B', columns='X/B', values=_col)
                    _pivot = _pivot.sort_index().reindex(sorted(_pivot.columns), axis=1)
                    _pivot.to_csv(os.path.join(_freq_csv_dir, f"{_stem}.csv"))
                    try:
                        _pivot.to_excel(os.path.join(_freq_xlsx_dir, f"{_stem}.xlsx"))
                    except Exception:
                        pass
                    _n_maps += 1
                except Exception as _e:
                    self.log(f"  -> WARNING: could not build {_stem}: {_e}")
            self.log(f" -> Saved {_n_maps} 2D frequency map(s) to Frequency_Results/CSV.")

        self.log(f"\n--- PROCESSING COMPLETE ---")
        self.root.after(0, lambda: self.btn_process.config(state="normal"))

    def setup_ui(self):
        ctrl = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=10)
        tk.Label(ctrl, text="Global Workspace:", bg="#e0e0e0", font=("Arial", 10, "bold")).pack(side="left", padx=10, pady=10)
        tk.Entry(ctrl, textvariable=self.folder_path, state="readonly", width=35).pack(side="left", padx=5)
        tk.Button(ctrl, text="Browse...", command=self.select_folder).pack(side="left", padx=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=20, pady=5, fill="x")
        self.btn_process = tk.Button(btn_frame, text="CALCULATE FFT & MAPS", command=self.start_processing, bg="#6f42c1", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_process.pack(fill="x")

        tk.Label(self.root, text="Console Log:").pack(anchor="w", padx=20, pady=(5, 0))
        self.log_text = tk.Text(self.root, height=14, state="disabled", bg="#f4f4f4")
        self.log_text.pack(padx=20, pady=(0, 20), fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = FrequencyProcessorApp(root)
    root.mainloop()