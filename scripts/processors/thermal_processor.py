# =============================================================================
# THERMAL_PROCESSOR
# =============================================================================
# Purpose:
#   Computes local and global Nusselt numbers from thermocouple temperature
#   measurements and electrical heat flux data for all experimental cases
#   (Free jet and Cyl configurations at multiple differential pressures). The
#   processor first applies linear calibration corrections to each
#   thermocouple channel using coefficients derived from calibration_TC.csv.
#   For each case, it calculates air thermophysical properties (rho, mu, k, Pr)
#   via Sutherland's law, derives the nozzle Reynolds number Re_noz from the
#   tunnel differential pressure (Bernoulli + area ratio + C_profile), and
#   computes the local experimental Nusselt number Nu_loc_exp = (q'' / dT) *
#   (D_h / k). A single constant ADDITIVE heat-loss term Nu_loss is derived from
#   the free-jet baseline cases as the mean offset of global Nu_bare (raw, no
#   loss) above the Hofmann Nu, then SUBTRACTED from all cases (Nu_corrected =
#   Nu_raw - Nu_loss). The script exports spatial Nu profiles, global summary
#   tables, the heat-loss calibration (Nu_bare vs Hofmann), and a
#   theoretical Nu curve (Hofmann) over the experimental Re range.
#
# Inputs:
#   - Thermocouple temperature CSV files:
#       thermal/raw_data/temp_<case>.csv  (processed copies of raw data)
#       thermal/raw_data/temp_raw_<case>.csv  (originals from thermal/ root)
#   - Thermocouple spatial positions:
#       thermal/raw_inputs/position_TC.csv
#   - TC calibration coefficients:
#       thermal/raw_inputs/calibration_TC.csv
#   - Electrical power supply parameters:
#       thermal/raw_inputs/power_supply.csv
#   - Cobra probe .asA files (used only for barometric pressure):
#       thermal/velocity_reference/noz_exit<N>_<Pa>Pa_<config> (Ve).asA
#
# Outputs:
#   - Per-case spatially resolved Nu mapping files:
#       thermal/results/Map_Temp_<case_id>.csv
#   - Thermal Global Summary (Re, Nu, eta for all cases):
#       thermal/results/Thermal_Global_Summary.csv / .xlsx
#   - Heat loss calibration points for free-jet baseline:
#       thermal/results/Qloss_Calibration_Points.csv
#   - Theoretical Nu curve (Hofmann) over the experimental Re range:
#       thermal/results/Theoretical_Nu_Curve.csv
#   - Copies of per-case results to individual experiment folders:
#       experiments/<case_id>/Temperatures/Map_Temp_<case_id>.csv
#
# Dependencies:
#   - Re_Ve_processor (calc_theo_re replicates its physics internally)
#
# Usage:
#   - Standalone: python thermal_processor.py  (opens Tkinter GUI)
#   - Via hub:    launched from usage.py as "Thermal & Power Processor"
#   - Note: not included in the automated 7-step pipeline; run separately
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import os
import glob
import shutil
import threading
import re
import math
import sys as _sys

# --- CONFIG LOADER ---
def _load_cfg(filename):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    _df = pd.read_excel(_p, index_col='parameter')
    return _df['value'] if 'value' in _df.columns else _df.iloc[:, 0]

try:
    _cfg_geom  = _load_cfg('config_geometry.csv')
    _cfg_flow  = _load_cfg('config_flow_regime.csv')
    _cfg_lit   = _load_cfg('config_literature_nu.csv')
    _cfg_acq   = _load_cfg('config_acquisition.csv')
    _cfg_cases = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cases.xlsx'))

    _D1_M   = float(_cfg_geom['tunnel_diameter_D1']) / 1000.0
    _B_M    = float(_cfg_geom['nozzle_height_B']) / 1000.0
    _W_CYL  = float(_cfg_geom['nozzle_width_cyl']) / 1000.0
    _W_FREE = float(_cfg_geom['nozzle_width_free']) / 1000.0
    _D2_M   = float(_cfg_geom['cylinder_diameter_D2']) / 1000.0
    _RE_LAM = float(_cfg_flow['re_laminar_upper'])
    _RE_TR  = float(_cfg_flow['re_transition_upper'])
    _C_LAM  = float(_cfg_flow['c_profile_laminar'])
    _N_COEF = float(_cfg_flow['c_profile_turbulent_log_coeff'])
    _N_OFF  = float(_cfg_flow['c_profile_turbulent_log_offset'])
    _HOF_PR = float(_cfg_lit['hofmann_pr_exponent'])
    _HOF_A  = float(_cfg_lit['hofmann_spatial_coeff'])
    _HOF_B  = float(_cfg_lit['hofmann_spatial_decay'])
    _HOF_C  = float(_cfg_lit['hofmann_int_normalization'])
    _TEMP_ROWS = int(_cfg_acq['temperature_series_rows'])
except Exception:
    _cfg_geom  = None; _cfg_flow  = None; _cfg_lit   = None
    _cfg_acq   = None; _cfg_cases = None
    _D1_M   = 0.200; _B_M    = 0.030; _W_CYL  = 0.150; _W_FREE = 0.170; _D2_M = 0.012
    _RE_LAM = 2300.0; _RE_TR  = 4000.0; _C_LAM  = 0.50
    _N_COEF = 1.22;   _N_OFF  = 1.0
    _HOF_PR = 0.42;   _HOF_A  = 0.042; _HOF_B  = 0.052; _HOF_C  = 1.24
    _TEMP_ROWS = 300

# Import shared air properties utility
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
try:
    from air_properties import calc_air_properties as _calc_air, R_AIR, MU_REF, T_SUT, S_SUT, T_FALLBACK, P_FALLBACK
except Exception:
    R_AIR = 287.05; MU_REF = 1.716e-5; T_SUT = 273.15; S_SUT = 110.4
    T_FALLBACK = 20.0; P_FALLBACK = 101325.0
    def _calc_air(temp_c, p_atm):
        T = temp_c + 273.15
        rho = p_atm / (R_AIR * T)
        mu  = MU_REF * (T / T_SUT)**1.5 * (T_SUT + S_SUT) / (T + S_SUT)
        k   = 0.0242 + 0.00007 * temp_c
        Pr  = (mu * 1005.0) / k
        return rho, mu, k, Pr

# Shared RSS uncertainty-propagation helper (Task 3, see ../../docs/UNCERTAINTY_SPEC.md)
try:
    from uncertainty import (rel_area_ratio as _u_rel_area,
                             rel_velocity_channel as _u_rel_vchan,
                             apply_calibration_u as _u_apply_cal,
                             rel_nusselt as _u_rel_nu, rel_eta as _u_rel_eta,
                             U_VOLT_V as _U_VOLT, U_CURR_A as _U_CURR,
                             CHANNEL_FIT_SE as _CH_SE, U_B_MM as _U_B)
    _HAS_UNC = True
except Exception:
    _HAS_UNC = False
    _U_VOLT = 0.1; _U_CURR = 0.1; _CH_SE = 0.0; _U_B = 0.1  # _CH_SE=0 -> no calib. residual

# Channel -> pitot velocity calibration VALUES (affine): V_noz = a*V_channel + b.
# Applied in calc_theo_re so the thermal Reynolds number is expressed in the SAME
# pitot reference as Re_Ve_processor (config_calibration.xlsx; identity fallback).
# The fit SE (_CH_SE) is an uncertainty -> config_uncertainty.xlsx (imported above).
try:
    _cal_th = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'config', 'config_calibration.xlsx')).set_index('parameter')['value']
    _CH_A = float(_cal_th['channel_slope'])
    _CH_B = float(_cal_th['channel_intercept'])
except Exception:
    _CH_A, _CH_B = 1.0, 0.0

# Build dynamic cylinder diameter regex from config_cases.csv
try:
    _cyl_patterns = _cfg_cases[_cfg_cases['has_cylinder'].astype(str).str.lower().isin(['true', '1', 'yes'])]['case_pattern'].tolist()
    _cyl_diams = '|'.join(str(p).replace('Cyl', '') for p in _cyl_patterns if str(p).startswith('Cyl') and str(p).replace('Cyl', '').isdigit())
    if not _cyl_diams:
        _cyl_diams = '10|12|14|15|16|18|20|25|30|35|40|50'  # fallback
except Exception:
    _cyl_diams = '10|12|14|15|16|18|20|25|30|35|40|50'  # fallback hardcoded list

class ThermalProcessorApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("Thermal & Power Input Processor")
        self.root.geometry("850x750") 
        
        self.folder_path = tk.StringVar(value=default_folder if default_folder else "")
        self.t2_patm_var = tk.StringVar(value="102500")
        self.slot_width_var = tk.StringVar(value=str(float(_cfg_geom['nozzle_height_B'])) if _cfg_geom is not None else "30.0")

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

    def calc_air_properties(self, temp_c, p_atm):
        """Thin wrapper delegating to shared air_properties utility (config-driven)."""
        return _calc_air(temp_c, p_atm)

    # -------------------------------------------------------------------------
    # Hofmann, Kind & Martin (2007), Int. J. Heat Mass Transfer 50, 3957-3965.
    # Slot-nozzle (2-D) local / surface-averaged Nusselt correlations, Eq.(13)/(14):
    #     Nu_loc = Pr^0.42 (Re_S^3 + 10 Re_S^2)^0.25 * 0.042 * exp(-0.052 * x/S)
    #     Nu_int = Pr^0.42 (Re_S^3 + 10 Re_S^2)^0.25 * (1 - exp(-0.052 x/S))/(1.24 x/S)
    # Both the characteristic length S = 2*B (slot hydraulic diameter, s_mm = 60 mm)
    # AND the Reynolds number Re_S = u*S/nu = 2*Re_noz are used, exactly per the
    # paper's nomenclature. These functions receive the project Re_noz (built on B)
    # and convert internally to Re_S = 2*Re_noz. PRIMARY baseline (Hofmann); Martin
    # (1977) SSN is computed on the same S as a cross-check (calc_nu_int_martin).
    # Validity (slot): Re_S 14000-230000, H/S 0.5-10 — confirm the operating
    # Reynolds range and H/S of your rig fall inside this band before applying it.
    # -------------------------------------------------------------------------
    def calc_nu_loc_hofmann(self, pr, re, x_mm, s_mm):
        re_s = 2.0 * re   # Hofmann slot correlation is defined on Re_S = 2*Re_noz
        x_over_s = np.abs(x_mm) / s_mm
        term_pr = pr ** _HOF_PR
        term_re = (re_s**3 + 10 * re_s**2) ** 0.25
        term_spatial = _HOF_A * np.exp(-_HOF_B * x_over_s)
        return term_pr * term_re * term_spatial

    def calc_nu_int_hofmann(self, pr, re, x_mm, s_mm):
        re_s = 2.0 * re
        x_over_s = np.abs(x_mm) / s_mm
        term_pr = pr ** _HOF_PR
        term_re = (re_s**3 + 10 * re_s**2) ** 0.25
        if x_over_s == 0: return term_pr * term_re * _HOF_A
        term_spatial = (1.0 - np.exp(-_HOF_B * x_over_s)) / (_HOF_C * x_over_s)
        return term_pr * term_re * term_spatial

    # Martin (1977), Adv. Heat Transfer 13, Eq.(4.14) — single slot nozzle (SSN),
    # surface-averaged Nu (over 0..x) on the slot hydraulic diameter S = 2B and
    # Re_S = 2*Re_noz. Used as a CROSS-CHECK on the Hofmann baseline.
    #   Nu/Pr^0.42 = 1.53/(x/S + H/S + 1.39) * Re_S^m
    #   m = 0.695 - (x/S + (H/S)^1.33 + 3.06)^-1
    # Validity: 3000 < Re_S < 90000, 2 < x/S < 25, 2 < H/S < 10.
    def calc_nu_int_martin(self, pr, re, x_mm, s_mm, h_mm):
        re_s = 2.0 * re
        xs = np.abs(x_mm) / s_mm
        if xs <= 0.0:
            xs = 1e-9
        hs = h_mm / s_mm
        m = 0.695 - 1.0 / (xs + hs ** 1.33 + 3.06)
        return (pr ** _HOF_PR) * 1.53 / (xs + hs + 1.39) * (re_s ** m)

    def calc_theo_re(self, dp_pa, temp_c, p_atm, B_mm, has_cyl):
        """Mathematical Reynolds Calculation mimicking Re_Ve_processor (config-driven).

        The heat-transfer Reynolds number uses the nozzle exit velocity V_noz on
        the slot height B, for both free and cylinder cases. V_noz is expressed in
        the PITOT reference via the channel->pitot affine calibration
        (V_noz = a*V_channel + b), exactly as Re_Ve_processor does, so thermal and
        aero Re share one reference. NO gap/blockage factor is applied (the thermal
        Re is on the bulk nozzle velocity). `has_cyl` selects the cylinder-config
        nozzle width (150 vs 170 mm) in the area ratio."""
        rho, mu, _, _ = self.calc_air_properties(temp_c, p_atm)
        d1    = _D1_M
        w_noz = _W_CYL if has_cyl else _W_FREE
        h_noz = B_mm / 1000.0

        a_tun = math.pi * (d1 / 2.0)**2
        a_noz = w_noz * h_noz
        k_ratio = a_tun / a_noz

        v_tun_center = math.sqrt((2 * dp_pa) / rho) if dp_pa > 0 else 0.0

        # Profile factor C_profile DISABLED (fixed at 1.0), matching
        # Re_Ve_processor: the empirical channel->pitot calibration already
        # subsumes the centerline-vs-mean profile, so applying the power law would
        # double-correct. (See Re_Ve_processor.calculate_physics / README.)
        c_profile = 1.0
        v_noz_channel = (v_tun_center * c_profile) * k_ratio   # raw channel route
        # Express in the pitot reference (channel->pitot affine), matching
        # Re_Ve_processor. No gap/blockage factor (bulk nozzle Re basis).
        v_noz = (_CH_A * v_noz_channel + _CH_B) if v_noz_channel > 0 else 0.0
        return (rho * v_noz * h_noz) / mu

    def parse_asa_file(self, filepath):
        """Extracts properties directly from TFI .asA files."""
        data = {'patm': P_FALLBACK, 'v_mean': 0.0}
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Barometric pressure" in line:
                        m = re.search(r':\s*([\d\.,]+)', line)
                        if m: data['patm'] = float(m.group(1).replace(',', ''))
                    elif "Mean flow speed" in line:
                        if i + 1 < len(lines):
                            vals = lines[i+1].split()
                            if len(vals) >= 1: data['v_mean'] = float(vals[0])
        except Exception: pass
        return data

    def build_calibration_curves(self, main_folder):
        calib_file = os.path.join(main_folder, "thermal", "raw_inputs", "calibration_TC.csv")
        self.calib_dict = {}
        if not os.path.exists(calib_file): return

        try:
            rows = []
            with open(calib_file, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip().strip('"')  
                    if line: rows.append(line.split('\t'))

            if not rows: return
            df_calib = pd.DataFrame(rows[1:], columns=rows[0])
            for col in df_calib.columns[1:]: df_calib[col] = df_calib[col].str.replace(',', '.', regex=False)

            tc_names_upper = df_calib.iloc[:, 0].astype(str).str.upper().values
            if 'EBRO_TRUE' not in tc_names_upper: return
            y_true = df_calib.iloc[np.where(tc_names_upper == 'EBRO_TRUE')[0][0], 1:].astype(float).values

            for index, row in df_calib.iterrows():
                tc_name = str(row.iloc[0]).strip().upper()
                if 'TC' in tc_name and tc_name != 'EBRO_TRUE':
                    match = re.search(r'TC(\d+)', tc_name)
                    if match:
                        x_measured = row.iloc[1:].astype(float).values
                        if len(x_measured) > 1 and not np.isnan(x_measured).all():
                            m, b = np.polyfit(x_measured, y_true, 1)
                            self.calib_dict[match.group(1)] = {'slope': m, 'intercept': b}
        except Exception: pass

    def process_temperature_series(self, cf):
        df_temp = pd.read_csv(cf).iloc[:_TEMP_ROWS]
        cols_to_drop = [c for c in df_temp.columns if 'time' in str(c).lower() or 'unnamed' in str(c).lower()]
        mean_series = df_temp.drop(columns=cols_to_drop, errors='ignore').apply(pd.to_numeric, errors='coerce').mean(numeric_only=True)
        
        t_amb = np.nan
        for col in mean_series.index:
            if '30' in str(col).lower() or 'inf' in str(col).lower():
                t_amb = mean_series[col]
                break
                
        avg_temps = mean_series.reset_index()
        avg_temps.columns = ['Raw_Name', 'Temp_C']
        avg_temps['TC_Name'] = avg_temps['Raw_Name'].str.extract(r'(?i)TC(\d+)')
        avg_temps['TC_Name'] = avg_temps['TC_Name'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        calibrated_temps = []
        for idx, row in avg_temps.iterrows():
            tc = str(row['TC_Name'])
            raw_temp = row['Temp_C']
            if pd.notna(raw_temp) and hasattr(self, 'calib_dict') and tc in self.calib_dict:
                m, b = self.calib_dict[tc]['slope'], self.calib_dict[tc]['intercept']
                calibrated_temps.append((raw_temp * m) + b)
            else:
                calibrated_temps.append(raw_temp) 
                
        avg_temps['Temp_C'] = calibrated_temps
        
        if pd.notna(t_amb) and hasattr(self, 'calib_dict') and '30' in self.calib_dict:
             m_amb, b_amb = self.calib_dict['30']['slope'], self.calib_dict['30']['intercept']
             t_amb = (t_amb * m_amb) + b_amb
        
        return avg_temps, t_amb

    def get_case_info(self, case_id):
        # Cylinder diameter list derived from config_cases.csv (fallback: hardcoded list)
        cleaned_id = re.sub(r'(?i)Cyl(' + _cyl_diams + r')', 'Cyl_', case_id)
        match_pa = re.search(r'(\d+)\s*Pa', cleaned_id, re.IGNORECASE)
        pa_val = match_pa.group(1) + "Pa" if match_pa else "Unknown"
        has_cyl = bool(re.search(r'(?i)Cyl', case_id))
        return pa_val, has_cyl

    def start_processing(self):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder): return
        self.btn_process.config(state="disabled")
        threading.Thread(target=self.run_master_pipeline, args=(folder,)).start()

    def run_master_pipeline(self, main_folder):
        self.log("\n=========================================")
        self.log(" STARTING THERMAL ANALYSIS")
        self.log("=========================================")
        self.build_calibration_curves(main_folder)
        
        temp_root = os.path.join(main_folder, "thermal")
        asa_dir = os.path.join(temp_root, "velocity_reference")
        raw_dir = os.path.join(temp_root, "raw_data")
        data_dir = os.path.join(temp_root, "processed_data")
        map_dir = os.path.join(temp_root, "results")
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(map_dir, exist_ok=True)

        B_mm = float(self.slot_width_var.get())
        s_mm = 2.0 * B_mm
        dh_m = s_mm / 1000.0

        # 1. POWER SUPPLY
        power_file = os.path.join(temp_root, "raw_inputs", "power_supply.csv")
        q_in_flux = 0.0
        q_volt, q_curr = np.nan, np.nan
        if os.path.exists(power_file):
            df_p = pd.read_csv(power_file, sep=';', decimal=',')
            v_val, i_val = float(df_p['V'].iloc[0]), float(df_p['A'].iloc[0])
            w_mm, l_mm = float(df_p['W'].iloc[0]), float(df_p['L'].iloc[0]) if 'W' in df_p.columns else (150.0, 400.0)
            q_in_flux = (v_val * i_val) / ((w_mm / 1000.0) * (l_mm / 1000.0))
            q_volt, q_curr = v_val, i_val
            self.log(f"Heat Flux Loaded: {q_in_flux:.2f} W/m^2")

        # 2. FILE REORG
        for f in os.listdir(temp_root):
            if f.startswith("temp") and f.endswith(".csv"):
                shutil.move(os.path.join(temp_root, f), os.path.join(raw_dir, f.replace("temp_", "temp_raw_") if "temp_raw_" not in f else f))
            
        for rf in glob.glob(os.path.join(raw_dir, "temp_raw_*.csv")):
            cal_name = os.path.basename(rf).replace("temp_raw_", "temp_")
            cal_path = os.path.join(data_dir, cal_name)
            pd.read_csv(rf).iloc[:_TEMP_ROWS].to_csv(cal_path, index=False)

        df_pos = pd.read_csv(os.path.join(temp_root, "raw_inputs", "position_TC.csv"), sep='\t', decimal=',')
        df_pos.rename(columns={'Position (mm from Center)': 'X_mm', 'Thermocouple Number': 'TC_Name'}, inplace=True)
        df_pos['TC_Name'] = df_pos['TC_Name'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_pos['X/B'] = df_pos['X_mm'] / B_mm

        cal_files = glob.glob(os.path.join(data_dir, "temp_*.csv"))
        
        # -------------------------------------------------------------
        # 3. PRE-CALCULATE ALL REYNOLDS NUMBERS (Meas & Theo)
        # -------------------------------------------------------------
        self.log("\n[PRE-LOADING DATA & CALCULATING BOTH REYNOLDS]")
        preloaded_cases = []
        for cf in cal_files:
            case_id = os.path.basename(cf).replace("temp_", "").replace(".csv", "")
            pa_val, has_cyl = self.get_case_info(case_id)
            
            avg_temps, t_amb = self.process_temperature_series(cf)
            if pd.isna(t_amb): continue
            
            pa_num = re.search(r'(\d+)Pa', pa_val).group(1)
            re_meas_e1, re_meas_e2 = np.nan, np.nan
            p_atm = P_FALLBACK
            
            if has_cyl:
                path_e1 = os.path.join(asa_dir, f"noz_exit1_{pa_num}Pa_Cyl (Ve).asA")
                path_e2 = os.path.join(asa_dir, f"noz_exit2_{pa_num}Pa_Cyl (Ve).asA")
                if os.path.exists(path_e1):
                    dat = self.parse_asa_file(path_e1)
                    p_atm = dat['patm']
                    rho, mu, _, _ = self.calc_air_properties(t_amb, p_atm)
                    re_meas_e1 = (rho * dat['v_mean'] * (B_mm / 1000.0)) / mu
                if os.path.exists(path_e2):
                    dat = self.parse_asa_file(path_e2)
                    p_atm = dat['patm']
                    rho, mu, _, _ = self.calc_air_properties(t_amb, p_atm)
                    re_meas_e2 = (rho * dat['v_mean'] * (B_mm / 1000.0)) / mu
            else:
                path_free = os.path.join(asa_dir, f"noz_exit_{pa_num}Pa (Ve).asA")
                if os.path.exists(path_free):
                    dat = self.parse_asa_file(path_free)
                    p_atm = dat['patm']
                    rho, mu, _, _ = self.calc_air_properties(t_amb, p_atm)
                    re_meas_e1 = (rho * dat['v_mean'] * (B_mm / 1000.0)) / mu
                    re_meas_e2 = re_meas_e1 # Free has only 1 exit
            
            re_theo = self.calc_theo_re(float(pa_num), t_amb, p_atm, B_mm, has_cyl)
            rho_air, mu_air, k_air, pr_air = self.calc_air_properties(t_amb, p_atm)
            df_map = pd.merge(df_pos, avg_temps, on='TC_Name', how='left')

            # V_noz back-computed from re_theo: V_noz = Re * mu / (rho * B)
            v_noz = re_theo * mu_air / (rho_air * (B_mm / 1000.0)) if re_theo > 0 else np.nan

            preloaded_cases.append({
                'case_id': case_id, 'pa_val': pa_val, 'has_cyl': has_cyl,
                're_meas_e1': re_meas_e1, 're_meas_e2': re_meas_e2, 're_theo': re_theo,
                'v_noz': v_noz,
                'df_map': df_map, 't_amb': t_amb, 'k_air': k_air, 'pr_air': pr_air
            })
            m1_txt = f"{re_meas_e1:.0f}" if pd.notna(re_meas_e1) else "NaN"
            m2_txt = f"{re_meas_e2:.0f}" if pd.notna(re_meas_e2) else "NaN"
            self.log(f"Loaded {case_id} -> Re_meas(E1): {m1_txt} | Re_meas(E2): {m2_txt} | "
                     f"Re_theo(V_noz): {re_theo:.0f} | V_noz: {v_noz:.3f} m/s")

        # -------------------------------------------------------------
        # 3b. FINALISE THE REYNOLDS BASIS PER CASE — always V_noz (dP),
        #     uncorrected nozzle velocity for both free and cylinder cases.
        # -------------------------------------------------------------
        for d in preloaded_cases:
            d['re_active'] = d['re_theo']
            d['re_basis'] = 'theoretical(V_noz)'
            self.log(f"  {d['case_id']}: Re_used = {d['re_active']:.0f}  [theoretical(V_noz)]")

        # -------------------------------------------------------------
        # 4. EXECUTE PIPELINE (Re from V_noz, dP-based, uncorrected)
        # -------------------------------------------------------------
        self.log("\n[RUNNING THERMAL ANALYSIS — Re from V_noz (dP-based), Hofmann reference]")

        # ----------------------------------------------------------------------
        # HEAT-LOSS CONSTANT (ADDITIVE) from the GLOBAL Nu_bare-vs-Hofmann offset.
        # For each bare (free-jet) plate case form the GLOBAL (surface-averaged)
        # RAW Nusselt number Nu_raw (full electrical flux q_in, i.e. NO loss
        # applied) and the GLOBAL Hofmann slot-correlation Nu. Across the bare
        # cases Nu_raw exceeds Hofmann by a roughly CONSTANT additive amount; the
        # single constant  Nu_loss = mean(Nu_raw - Nu_Hofmann)  is the heat-loss
        # term (in Nu units), SUBTRACTED from ALL cases (free and cylinder):
        #   Nu_corrected = Nu_raw - Nu_loss.
        # Its constancy across the sweep (small scatter) justifies one constant.
        bare_raw_list, bare_hof_list, bare_cal_rows = [], [], []
        for data in preloaded_cases:
            if data['has_cyl']: continue
            active_re = data['re_active']
            if pd.isna(active_re): continue
            _nu_raw_pts, _nu_hof_pts = [], []
            for idx, row in data['df_map'].iterrows():
                dT = row['Temp_C'] - data['t_amb']
                if pd.notna(dT) and dT > 0:
                    _nu_raw_pts.append((q_in_flux / dT) * (dh_m / data['k_air']))
                    _nu_hof_pts.append(self.calc_nu_loc_hofmann(data['pr_air'], active_re, float(row['X_mm']), s_mm))
            if _nu_raw_pts and _nu_hof_pts:
                _nu_raw_g = float(np.mean(_nu_raw_pts))
                _nu_hof_g = float(np.mean(_nu_hof_pts))
                bare_raw_list.append(_nu_raw_g); bare_hof_list.append(_nu_hof_g)
                bare_cal_rows.append({'Base_ID': data['pa_val'], 'Re_used': active_re,
                                      'Nu_bare_raw_global': _nu_raw_g,
                                      'Nu_Hofmann_global': _nu_hof_g,
                                      'Nu_diff_raw_minus_Hof': _nu_raw_g - _nu_hof_g})

        _xr = np.array(bare_raw_list, dtype=float); _yr = np.array(bare_hof_list, dtype=float)
        _diffs = _xr - _yr
        if _diffs.size:
            nu_loss_const = float(_diffs.mean())                          # additive constant (Nu units)
            nu_loss_std = float(_diffs.std(ddof=1)) if _diffs.size > 1 else 0.0
            _ss_res = float(((_xr - (_yr + nu_loss_const)) ** 2).sum())    # slope-1, offset model
            _ss_tot = float(((_xr - _xr.mean()) ** 2).sum())
            nu_loss_r2 = float(1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float('nan')
        else:
            nu_loss_const, nu_loss_std, nu_loss_r2 = 0.0, 0.0, float('nan')
        for _row in bare_cal_rows:
            _row['Nu_loss_const'] = nu_loss_const
            _row['Nu_loss_std'] = nu_loss_std
            _row['Nu_loss_R2'] = nu_loss_r2
        self.log(f"\n[HEAT-LOSS CONSTANT additive] Nu_loss = {nu_loss_const:.3f} "
                 f"+/- {nu_loss_std:.3f} (Nu units),  R^2 = {nu_loss_r2:.4f},  "
                 f"from {_xr.size} bare cases")

        case_results, baseline_nus, calibration_export = [], {}, []
        baseline_dt = {}   # mean dT of each free-jet baseline, for eta uncertainty
        for data in preloaded_cases:
            has_cyl = data['has_cyl']
            active_re = data['re_active']

            df_map = data['df_map'].copy()

            nu_exp_list, nu_the_list, nu_int_the_list = [], [], []
            q_loss_list, q_loss_ratio_list = [], []
            delta_t_list, delta_t_the_list = [], []

            for idx, row in df_map.iterrows():
                dT = row['Temp_C'] - data['t_amb'] if pd.notna(row['Temp_C']) else np.nan
                if pd.notna(dT) and dT > 0:
                    nu_raw_loc = (q_in_flux / dT) * (dh_m / data['k_air'])

                    # Apply the constant ADDITIVE heat-loss term Nu_loss
                    nu_loc_exp = nu_raw_loc - nu_loss_const

                    if not has_cyl and pd.notna(active_re):
                        nu_the = self.calc_nu_loc_hofmann(data['pr_air'], active_re, row['X_mm'], s_mm)
                        nu_int = self.calc_nu_int_hofmann(data['pr_air'], active_re, row['X_mm'], s_mm)
                    else:
                        nu_the, nu_int = np.nan, np.nan

                    q_conv_exp = nu_loc_exp * (data['k_air'] / dh_m) * dT
                    q_loss = q_in_flux - q_conv_exp
                    q_loss_ratio = q_loss / q_in_flux

                    nu_exp_list.append(nu_loc_exp)
                    nu_the_list.append(nu_the)
                    nu_int_the_list.append(nu_int)
                    q_loss_list.append(q_loss)
                    q_loss_ratio_list.append(q_loss_ratio)
                    delta_t_list.append(dT)
                    delta_t_the_list.append(q_conv_exp / ((nu_the * data['k_air']) / dh_m) if pd.notna(nu_the) and nu_the > 0 else np.nan)
                else:
                    nu_exp_list.append(np.nan); nu_the_list.append(np.nan); nu_int_the_list.append(np.nan)
                    q_loss_list.append(np.nan); q_loss_ratio_list.append(np.nan)
                    delta_t_list.append(np.nan); delta_t_the_list.append(np.nan)

            df_map["Nu_loc_exp"], df_map["Nu_loc_the"], df_map["Nu_int_the"] = nu_exp_list, nu_the_list, nu_int_the_list
            df_map["q''_loss_W/m2"], df_map["q''_loss_ratio"] = q_loss_list, q_loss_ratio_list
            df_map["Delta_T_K"], df_map["Delta_T_the_K"] = delta_t_list, delta_t_the_list

            nu_global_exp = df_map['Nu_loc_exp'].mean(skipna=True)
            # The additive heat-loss term is the same constant for every case.
            global_delta_nu = nu_loss_const
            nu_the_global_list = []
            if pd.notna(active_re):
                nu_the_global_list = [self.calc_nu_loc_hofmann(data['pr_air'], active_re, float(row['X_mm']), s_mm) for _, row in df_map.iterrows() if pd.notna(row['X_mm'])]
            nu_global_the = np.mean(nu_the_global_list) if nu_the_global_list else np.nan

            mean_dt = df_map['Delta_T_K'].mean(skipna=True)
            if not has_cyl:
                baseline_nus[data['pa_val']] = nu_global_exp
                baseline_dt[data['pa_val']] = mean_dt
            df_map['T_amb_C'], df_map['Re_noz'], df_map['Pr'] = data['t_amb'], active_re, data['pr_air']

            case_results.append({
                'case_id': data['case_id'], 'pa_val': data['pa_val'], 'df_map': df_map,
                'nu_global_exp': nu_global_exp, 'nu_global_the': nu_global_the,
                'pr': data['pr_air'], 're': active_re, 'delta_nu_loss': global_delta_nu,
                'mean_dt': mean_dt, 're_basis': data['re_basis'], 'v_noz': data['v_noz']
            })

        pd.DataFrame(bare_cal_rows).to_csv(os.path.join(map_dir, "Qloss_Calibration_Points.csv"), index=False)

        # Calculate 100-point Hofmann theoretical Nu curve over Re range
        all_re = [r['re'] for r in case_results if pd.notna(r['re'])]
        if all_re:
            re_min, re_max = min(all_re), max(all_re)
            if re_min == re_max: re_min, re_max = re_min * 0.8, re_max * 1.2
            re_range = np.linspace(re_min, re_max, 100)
            avg_pr = np.mean([r['pr'] for r in case_results if pd.notna(r['pr'])])

            theo_curve_data = []
            for r in re_range:
                nu_loc_list = [self.calc_nu_loc_hofmann(avg_pr, r, float(x), s_mm) for x in df_pos['X_mm'] if pd.notna(x)]
                theo_curve_data.append({'Re': r, 'Nu_the': np.mean(nu_loc_list)})
            pd.DataFrame(theo_curve_data).to_csv(os.path.join(map_dir, "Theoretical_Nu_Curve.csv"), index=False)

        # Export per-case results and aggregate global summary
        self.log("\n  -> Global Metrics:")
        global_summary_logs = []
        for res in case_results:
            case_id, pa_val, df_map = res['case_id'], res['pa_val'], res['df_map']
            nu_global_exp, nu_global_the = res['nu_global_exp'], res['nu_global_the']
            global_delta_nu = res['delta_nu_loss']

            # eta_baseline  = matched-dP enhancement (Nu_exp,cyl / Nu_exp,free at same dP)
            # eta_theo      = matched-Re literature enhancement (Nu_exp / Nu_Hofmann at the
            #                 case's own Re_S) -- primary literature reference (Hofmann).
            eta_baseline = nu_global_exp / baseline_nus[pa_val] if pa_val in baseline_nus else np.nan
            eta_theo = nu_global_exp / nu_global_the if pd.notna(nu_global_the) and nu_global_the > 0 else np.nan

            # Martin (1977) SSN cross-check: surface-averaged Nu over the measured
            # plate extent (x = max|X|), on the same S=2B and Re_S=2*Re_noz.
            x_ext = df_map['X_mm'].abs().max() if 'X_mm' in df_map.columns else np.nan
            nu_global_martin = (self.calc_nu_int_martin(res['pr'], res['re'], x_ext, s_mm, 4.0 * B_mm)
                                if pd.notna(res['re']) and pd.notna(x_ext) and x_ext > 0 else np.nan)
            eta_theo_martin = (nu_global_exp / nu_global_martin
                               if pd.notna(nu_global_martin) and nu_global_martin > 0 else np.nan)

            df_map['Nu'], df_map['Nu_the'] = nu_global_exp, nu_global_the
            df_map['Global_Delta_Nu_Loss'] = global_delta_nu
            df_map['eta_baseline'], df_map['eta_theo'] = eta_baseline, eta_theo

            re_str = f"{res['re']:.0f}" if pd.notna(res['re']) else "NaN"
            nu_str = f"{nu_global_exp:.1f}" if pd.notna(nu_global_exp) else "NaN"
            nu_the_str = f"{nu_global_the:.1f}" if pd.notna(nu_global_the) else "NaN"
            dnu_str = f"{global_delta_nu:.1f}" if pd.notna(global_delta_nu) else "NaN"
            eta_str = f"{eta_theo:.3f}" if pd.notna(eta_theo) else "NaN"

            self.log(f"     [{case_id}] Re: {re_str} | Nu: {nu_str} | Nu_theo: {nu_the_str} | dNu_Loss: {dnu_str} | eta (Nu/Nu_theo): {eta_str}")

            final_cols = ['TC_Name', 'X_mm', 'X/B', 'Temp_C', 'T_amb_C', 'Delta_T_K', 'Delta_T_the_K', 'Pr', 'Re_noz', 'Nu_loc_exp', 'Nu_loc_the', 'Nu_int_the', "q''_loss_W/m2", "q''_loss_ratio", 'Global_Delta_Nu_Loss', 'Nu', 'Nu_the', 'eta_baseline', 'eta_theo']
            df_map_final = df_map[[c for c in final_cols if c in df_map.columns]]

            map_filepath = os.path.join(map_dir, f"Map_Temp_{case_id}.csv")
            df_map_final.to_csv(map_filepath, index=False)

            # Copy to individual experiment folder
            target_case = os.path.join(main_folder, "experiments", case_id)
            if os.path.exists(target_case):
                _temp_dst = os.path.join(target_case, "Temperatures")
                os.makedirs(_temp_dst, exist_ok=True)
                shutil.copy2(map_filepath, os.path.join(_temp_dst, f"Map_Temp_{case_id}.csv"))

            # --- RSS uncertainty propagation (UNCERTAINTY_SPEC.md §3) ---
            # Re: channel->pitot route (Bernoulli dP sqrt-law + area-ratio term),
            # pushed through the channel->pitot affine calibration (now applied in
            # calc_theo_re) -> u(V_noz)=sqrt[(a_ch*u_channel)^2 + SE_ch^2]; the
            # relative Re uncertainty equals u(V_noz)/V_noz. Nu: q'' from (V,I) and
            # dT from two TCs; eta: Nu ratio. Stored as percent AND absolute.
            _has_cyl_case = 'Cyl' in case_id
            u_re_pct = u_nu_pct = u_eta_pct = np.nan
            u_re_abs = u_nu_abs = u_eta_abs = np.nan
            if _HAS_UNC and pd.notna(q_volt) and pd.notna(q_curr):
                _m_dp = re.search(r'(\d+)\s*Pa', pa_val)
                _dp = float(_m_dp.group(1)) if _m_dp else np.nan
                _dt = res.get('mean_dt', np.nan)
                _w_mm = (_W_CYL if _has_cyl_case else _W_FREE) * 1000.0
                if pd.notna(_dp) and _dp > 0:
                    _rel_vchan = _u_rel_vchan(_dp, _u_rel_area(_D1_M * 1000.0, _w_mm, _B_M * 1000.0))
                    # (No power-law / C_profile term: C_profile fixed at 1.0.)
                    _vnoz_cal = res.get('v_noz', np.nan)
                    if pd.notna(_vnoz_cal) and _vnoz_cal > 0 and abs(_CH_A) > 1e-9:
                        _vchan = (_vnoz_cal - _CH_B) / _CH_A          # invert the affine map
                        _u_vnoz = _u_apply_cal(_vchan * _rel_vchan, _CH_A, _CH_SE)
                        _rel_re = _u_vnoz / _vnoz_cal
                    else:
                        _rel_re = _rel_vchan
                    # The thermal Re is built on the slot height B, which CANCELS the
                    # area-ratio B (Re_noz ∝ V_noz·B, V_noz ∝ 1/B). Remove uB/B
                    # (UNCERTAINTY_SPEC §3); the affine residual is ~0.2 %.
                    _rel_B = _U_B / (_B_M * 1000.0)
                    _rel_re = max(0.0, _rel_re**2 - _rel_B**2) ** 0.5
                    u_re_pct = _rel_re * 100.0
                    if pd.notna(res['re']):
                        u_re_abs = res['re'] * _rel_re
                if pd.notna(_dt) and _dt > 0:
                    _rel_nu = _u_rel_nu(q_volt, q_curr, _dt)
                    u_nu_pct = _rel_nu * 100.0
                    if pd.notna(nu_global_exp):
                        u_nu_abs = nu_global_exp * _rel_nu
                    _dt_base = baseline_dt.get(pa_val, np.nan)
                    _rel_nu_base = _u_rel_nu(q_volt, q_curr, _dt_base) if pd.notna(_dt_base) and _dt_base > 0 else _rel_nu
                    _rel_eta = _u_rel_eta(_rel_nu, _rel_nu_base)
                    u_eta_pct = _rel_eta * 100.0
                    if pd.notna(eta_theo):
                        u_eta_abs = eta_theo * _rel_eta

            global_summary_logs.append({
                'Case_ID': case_id,
                'Base_Pressure': pa_val,
                'Has_Cylinder': 'Yes' if 'Cyl' in case_id else 'No',
                'Re_used': res['re'],
                'Re_S': (2.0 * res['re']) if pd.notna(res['re']) else np.nan,
                'Re_basis': res.get('re_basis', 'theoretical(dP)'),
                'Ref_Velocity_ms': res.get('v_noz', np.nan),
                'Global_Nu_Exp': nu_global_exp,
                'Global_Nu_Theo': nu_global_the,            # Hofmann (Re_S), primary
                'Global_Nu_Theo_Martin': nu_global_martin,  # Martin SSN (Re_S), cross-check
                'Global_Delta_Nu_Loss': global_delta_nu,
                'Nu_Loss_additive': nu_loss_const,
                'Nu_Loss_std': nu_loss_std,
                'Nu_Loss_R2': nu_loss_r2,
                'eta_baseline': eta_baseline,
                'eta_theo': eta_theo,
                'eta_theo_martin': eta_theo_martin,
                # explicit, self-documenting aliases of the two requested enhancements:
                'eta_matched_dP': eta_baseline,
                'eta_matched_Re_lit': eta_theo,
                'u_Re_pct': u_re_pct,
                'u_Re_abs': u_re_abs,
                'u_Nu_pct': u_nu_pct,
                'u_Nu_abs': u_nu_abs,
                'u_eta_pct': u_eta_pct,
                'u_eta_abs': u_eta_abs
            })

        # Export global summary to CSV and XLSX
        if global_summary_logs:
            df_summary = pd.DataFrame(global_summary_logs)
            csv_path = os.path.join(temp_root, "results", "Thermal_Global_Summary.csv")
            xlsx_path = os.path.join(temp_root, "results", "Thermal_Global_Summary.xlsx")
            df_summary.to_csv(csv_path, index=False)
            try:
                df_summary.to_excel(xlsx_path, index=False)
            except Exception:
                pass
            self.log(f"\nSaved Global Summary to:\n{csv_path}")

        self.log("\n--- PROCESSING COMPLETE ---")
        self.root.after(0, lambda: self.btn_process.config(state="normal"))

    def setup_ui(self):
        ctrl = tk.Frame(self.root, bg="#e0e0e0", bd=2, relief="groove")
        ctrl.pack(fill="x", padx=10, pady=(10, 5))
        tk.Label(ctrl, text="Workspace:", bg="#e0e0e0", font=("Arial", 10, "bold")).pack(side="left", padx=10)
        tk.Entry(ctrl, textvariable=self.folder_path, state="readonly", width=45).pack(side="left", padx=5)
        tk.Button(ctrl, text="Browse", command=self.select_folder).pack(side="left", padx=5)

        env_frame = tk.LabelFrame(self.root, text=" Geometry & Comparison Settings ", padx=10, pady=10, font=("Arial", 9, "bold"))
        env_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(env_frame, text="Slot Height 'B' (mm):").grid(row=0, column=0, sticky="e", pady=5)
        tk.Entry(env_frame, textvariable=self.slot_width_var, width=10).grid(row=0, column=1, sticky="w", padx=5)

        self.btn_process = tk.Button(self.root, text="RUN AUTONOMOUS THERMAL MAPPING", command=self.start_processing, bg="#ff8c00", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_process.pack(fill="x", padx=20, pady=10)

        self.log_text = tk.Text(self.root, height=18, state="disabled", bg="#f4f4f4", font=("Consolas", 9), wrap="none")
        self.log_text.pack(padx=20, pady=10, fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalProcessorApp(root)
    root.mainloop()