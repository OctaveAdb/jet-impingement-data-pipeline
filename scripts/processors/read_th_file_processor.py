# =============================================================================
# READ_TH_FILE_PROCESSOR
# =============================================================================
# Purpose:
#   Parses binary Cobra probe data files (.thA) and their companion summary
#   files (.asA) produced by the TFI Cobra probe acquisition system. For each
#   measurement grid point, this script decodes the binary time-series of the
#   three velocity components (u, v, w), static pressure, and optional
#   reference pressure, then applies a three-stage coordinate rotation:
#   (1) a 90° axis switch fixing the probe Y/Z convention, (2) a fixed
#   -arctan(3/30) tilt about Z, and (3) alignment of the centerline
#   "middle" flow (the middle-X, mean-Y reference file, where U is strongest)
#   with the +X axis. Physical
#   coordinates (X/B, Y/B) are assigned from the filename's grid index using
#   user-supplied step sizes and nozzle height B. Quality-control metadata
#   (yield percentage, temperature, barometric pressure) is extracted from the
#   companion .asA files. Results are exported as one CSV per grid point.
#
#   VELOCITY CALIBRATION (Cobra -> pitot, affine):
#   After all per-point geometry and rotations are computed, the velocity is
#   calibrated onto the co-located PITOT reference with an AFFINE law applied
#   per time-sample on the speed:
#       V_cal = a * |V| + b        (a, b, SE, validity limit in
#                                   config/config_calibration.xlsx)
#   The three components are rescaled by V_cal/|V| to preserve direction; pitch,
#   yaw and static pressure are unchanged. The law is valid for raw |V| up to
#   the Cobra fold-back (~45 m/s); above that it is extrapolated with a per-point
#   warning. This REPLACES the former multiplicative k_correction (which was
#   anchored to the channel velocity, which reads systematically low vs the
#   pitot). The channel
#   velocity V_noz is calibrated separately in the Re/Velocity calculator with
#   the channel->pitot law. decay_check() regenerates the post-calibration decay
#   graph (a verification only). See "calibration/" and
#   VELOCITY_CALIBRATION.md.
#
# Inputs:
#   - Raw Cobra probe acquisition files: experiments/<case>/thA/*.thA
#   - Companion summary files:           experiments/<case>/asA/*.asA
#   - User-supplied GUI parameters: grid step X/Y (mm), nozzle height B (mm),
#     X/Y start offsets, optional rotation angles (azimuth, elevation, roll)
#
# Outputs:
#   - Per-grid-point CSV files containing time-series of u, v, w (m/s),
#     velocity magnitude, pitch and yaw angles (deg), static pressure (Pa),
#     temperature (deg C), barometric pressure (Pa), and QC yield (%):
#       experiments/<case>/Processed_CSVs/Raw_Data/<grid>.csv
#   - Processing parameters merged into or creating:
#       experiments/<case>/Flow_Data.csv / .xlsx
#       experiments/<case>/Processing_Parameters/
#
# Dependencies:
#   - None (standalone processor; imported by pipeline.py and usage.py)
#
# Usage:
#   - Standalone: python read_th_file_processor.py  (opens Tkinter GUI)
#   - Via pipeline: called programmatically by pipeline.py (Step 2)
#   - Via hub:      launched from usage.py as "Cobra Data Processor"
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import os
import glob
import threading
import shutil
import re
from datetime import datetime

# --- CONFIG LOADER ---
def _load_cfg(filename):
    filename = filename.replace('.csv', '.xlsx')
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', filename)
    df = pd.read_excel(_p, index_col='parameter')
    return df['value'] if 'value' in df.columns else df.iloc[:, 0]

try:
    _cfg_geom  = _load_cfg('config_geometry.csv')
    _cfg_fluid = _load_cfg('config_fluid_properties.csv')
    _cfg_cases = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config', 'config_cases.xlsx'))
except Exception:
    _cfg_geom  = None
    _cfg_fluid = None
    _cfg_cases = None

# --- 1. COORDINATE TRANSFORMATIONS & ROTATIONS ---
def velocity(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)

def vel_pitch_yaw(u, v, w):
    vel = velocity(u, v, w)
    pitch = np.degrees(np.arcsin(np.divide(w, vel, out=np.zeros_like(w), where=vel!=0)))
    yaw = np.degrees(np.atan2(v, u))
    return vel, pitch, yaw

def uvw(vel, pitch, yaw):
    p_rad, y_rad = np.radians(pitch), np.radians(yaw)
    u = vel * np.cos(p_rad) * np.cos(y_rad)
    v = vel * np.cos(p_rad) * np.sin(y_rad)
    w = vel * np.sin(p_rad)
    return u, v, w

def transform_axes_az_el(u, v, w, az_deg, el_deg, roll_deg, direction=1):
    az, el, roll = np.radians(az_deg), np.radians(el_deg), np.radians(roll_deg)
    Raz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rel = np.array([[np.cos(el), 0, np.sin(el)], [0, 1, 0], [-np.sin(el), 0, np.cos(el)]])
    Rroll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    
    R = Rroll @ Rel @ Raz
    if direction == 2: R = R.T

    v_orig = np.vstack((u, v, w))
    v_rot = R @ v_orig
    return v_rot[0, :], v_rot[1, :], v_rot[2, :]


# --- VELOCITY CALIBRATION (Cobra -> pitot, AFFINE) -------------------------
# The Cobra velocity is calibrated onto the co-located pitot reference with an
# AFFINE law:  V_cal = a * V_cobra + b   (slope a, intercept b, validity limit
# and fit standard error SE come from config/config_calibration.xlsx, derived
# in "calibration/Calibration_velocities.xlsx").  Applied per-sample
# in process_files AFTER the rotations.  This REPLACES the former multiplicative
# k_correction (which was anchored to the channel velocity, reading low vs pitot).
def _load_calibration():
    a, b, vmax = 1.0, 0.0, float('inf')
    try:
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'config', 'config_calibration.xlsx')
        _df = pd.read_excel(_p).set_index('parameter')['value']
        a    = float(_df['cobra_slope_a'])
        b    = float(_df['cobra_intercept_b'])
        vmax = float(_df['cobra_valid_max_ms'])
    except Exception as _e:
        print(f"[calibration] config_calibration.xlsx not loaded ({_e}); "
              f"falling back to identity V_cal = V_cobra.")
    # The fit SE is an uncertainty quantity -> config_uncertainty.xlsx (single source).
    try:
        import sys as _s
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
        from uncertainty import COBRA_FIT_SE as se
    except Exception:
        se = 0.0   # no calibration fit residual by default (identity map)
    return a, b, vmax, se

_CAL_A, _CAL_B, _CAL_VMAX, _CAL_SE = _load_calibration()

# Plane-jet centerline decay law (Schauer & Eustis, via Beitalmal 2006 Eq.3;
# Martin 1977 gives the same constant 2.38 and core ~4 slot widths).  With the
# project convention x0 = 0 and D = B, (x - x0)/(D/2) = 2*(X/B):
#     U_c / U0 = min(1, 2.35 / sqrt(2 * X/B))
def plane_jet_decay(xb):
    """Literature centerline velocity ratio U_c/U0 at X/B = xb."""
    xb = np.asarray(xb, dtype=float)
    return np.minimum(1.0, 2.35 / np.sqrt(np.clip(2.0 * xb, 1e-9, None)))


# --- AFFINE COBRA->PITOT CALIBRATION ---------------------------------------
def apply_cobra_calibration(speed):
    """Apply the affine Cobra->pitot calibration  V_cal = a*V_cobra + b  to a
    speed value or array (a, b from config_calibration.xlsx). Returns
    (V_cal, n_over_vmax); n_over_vmax counts samples above the validity limit,
    where the law is EXTRAPOLATED past the probe fold-back (caller logs a
    warning).  Direction is preserved by the caller (components rescaled by
    V_cal/V_cobra)."""
    speed = np.asarray(speed, dtype=float)
    n_over = int(np.count_nonzero(speed > _CAL_VMAX))
    return _CAL_A * speed + _CAL_B, n_over


def _centerline_calibrated(case_dir):
    """Return (XB, U_c, V_noz): the CALIBRATED mean centerline speed |V| at each
    X station, the X/B coordinate, and V_noz from Flow_Data (normalisation for
    the post-calibration decay check)."""
    fd = glob.glob(os.path.join(case_dir, "**", "Flow_Data*.csv"), recursive=True)
    if not fd:
        return None
    df = pd.read_csv(max(fd, key=os.path.getctime))
    V_noz = float(df['V_noz_ms'].iloc[-1]); B = float(df['B_mm'].iloc[-1])
    x_start = float(df['X_Start_Offset_mm'].iloc[-1]); step_x = float(df['Step_X_mm'].iloc[-1])
    rows = {}
    for f in glob.glob(os.path.join(case_dir, "thA", "*.thA")):
        m = re.match(r'(\d{2})(\d{2})', os.path.basename(f))
        if not m:
            continue
        u, v, w, _ps, _pr, _meta = read_th_file(f)
        cal, _ = apply_cobra_calibration(np.sqrt(u**2 + v**2 + w**2))
        rows[(int(m.group(1)), int(m.group(2)))] = float(cal.mean())
    if not rows:
        return None
    yis = sorted(set(k[1] for k in rows)); ymid = yis[len(yis) // 2]
    xis = sorted(x for (x, y) in rows if y == ymid)
    X = np.array(xis, dtype=float)
    Uc = np.array([rows[(int(x), ymid)] for x in xis])
    XB = (x_start + X * step_x) / B
    return XB, Uc, V_noz


def decay_check(experiments_dir, plot_path=None):
    """POST-CALIBRATION decay CHECK (a verification, NOT a calibration).  For
    every case, plot the calibrated centerline U_c/V_noz vs X/B together with the
    literature plane-jet decay law, to confirm the calibrated field follows the
    expected decay.  Derives/modifies no calibration constant."""
    if not plot_path:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        all_cases = sorted(
            d for d in os.listdir(experiments_dir)
            if os.path.isdir(os.path.join(experiments_dir, d, "thA")))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(all_cases), 1)))
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        xb = np.linspace(0.8, 5.0, 200)
        ax.plot(xb, plane_jet_decay(xb), 'k--', lw=2, label='Literature 2.35/sqrt(2 X/B)')
        for c, col in zip(all_cases, cmap):
            res = _centerline_calibrated(os.path.join(experiments_dir, c))
            if res is None:
                continue
            XB, Uc, V_noz = res
            ax.plot(XB, Uc / V_noz, 'o-', ms=3, color=col, label=c)
        ax.set_title('Post-calibration decay check: calibrated U_c / V_noz')
        ax.set_xlabel('X / B'); ax.set_ylabel('U_c / U0')
        ax.axhline(1, ls=':', c='gray'); ax.grid(alpha=.3); ax.legend(fontsize=7)
        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=130); plt.close(fig)
        print(f"[decay_check] wrote {plot_path}")
    except Exception as _e:
        print(f"[decay_check] skipped: {_e}")


# --- THREE-STAGE FLOW ROTATION HELPERS -------------------------------------
# The Cobra velocity field is brought into the physical, flow-aligned frame by
# three sequential rotations (applied in this order):
#   1) a 90° axis switch that fixes the probe Y/Z convention,
#   2) a fixed geometric tilt of -arctan(rise/run) about the Z axis, and
#   3) a final alignment that rotates the centerline "middle" flow onto +X.
# Stages 1 and 2 are fixed; stage 3 is data-driven (see process_files).

def axis_switch_90(u, v, w):
    """ROTATION 1/3 — 90° axis switch.

    A -90° roll about the streamwise X axis maps (u, v, w) -> (u, w, -v),
    restoring the physical convention (V spanwise, W wall-normal) from the
    probe output which had Y/Z swapped. Returns copies to avoid aliasing."""
    return u, w.copy(), -v.copy()


def z_tilt_deg_from_config():
    """ROTATION 2/3 angle — the fixed Z-axis tilt, -arctan(rise/run) in degrees.

    rise/run default to 3/30 (configurable via config_geometry.csv:
    flow_z_tilt_rise_mm / flow_z_tilt_run_mm)."""
    rise, run = 3.0, 30.0
    if _cfg_geom is not None:
        try:
            if 'flow_z_tilt_rise_mm' in _cfg_geom.index:
                rise = float(_cfg_geom['flow_z_tilt_rise_mm'])
            if 'flow_z_tilt_run_mm' in _cfg_geom.index:
                run = float(_cfg_geom['flow_z_tilt_run_mm'])
        except Exception:
            rise, run = 3.0, 30.0
    return -np.degrees(np.arctan2(rise, run))

# --- 2. FILE PARSERS ---

def parse_asa_file(filepath):
    """Parses a specific TFI .asA file to extract its local constants."""
    _temp_fb  = float(_cfg_fluid['default_temp_fallback'])  if _cfg_fluid is not None else 20.0
    _patm_fb  = float(_cfg_fluid['default_patm_fallback'])  if _cfg_fluid is not None else 101325.0
    data = {'yield_pct': 100.0, 'temp': _temp_fb, 'patm': _patm_fb, 'pitch': 0.0, 'yaw': 0.0}
    if not os.path.exists(filepath):
        return data
        
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Number of good samples" in line:
                    match = re.search(r'\((\d+\.?\d*)%\)', line)
                    if match: data['yield_pct'] = float(match.group(1))
                elif "Mean temperature" in line:
                    match = re.search(r':\s*([\d\.]+)', line)
                    if match: data['temp'] = float(match.group(1))
                elif "Barometric pressure" in line:
                    match = re.search(r':\s*([\d\.,]+)', line)
                    if match: data['patm'] = float(match.group(1).replace(',', ''))
                elif "Mean flow speed, pitch angle, yaw angle" in line:
                    # Values are explicitly printed on the line directly below this text
                    if i + 1 < len(lines):
                        vals = lines[i+1].split()
                        if len(vals) >= 3:
                            data['pitch'] = float(vals[1])
                            data['yaw'] = float(vals[2])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return data

def _read_header_strings(fid):
    d = np.fromfile(fid, dtype=np.int16, count=4) 
    t = np.fromfile(fid, dtype=np.int16, count=4) 
    date_str = f"{d[3]:02d}/{d[1]:02d}/{d[0]}" if len(d)==4 else "N/A"
    time_str = f"{t[0]:02d}:{t[1]:02d}:{t[2]:02d}" if len(t)==4 else "N/A"
    return date_str, time_str

def read_th_file(file_path):
    with open(file_path, 'rb') as fid:
        file_format = np.fromfile(fid, dtype=np.int32, count=1)[0]
        meta = {'format': file_format}

        if file_format == 3:
            fid.seek(4, 1)
            meta['device_id'] = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
            fid.seek(4, 1)
            meta['date'], meta['time'] = _read_header_strings(fid)
            meta['num_samples'] = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
            meta['block_size'] = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
            meta['data_rate'] = float(np.fromfile(fid, dtype=np.float64, count=1)[0])
            meta['p_baro'], meta['t_mean'] = np.fromfile(fid, dtype=np.float64, count=2)
            meta['has_pref'] = bool(np.fromfile(fid, dtype=np.uint8, count=1)[0])
        else:
            meta['device_id'] = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
            meta['num_samples'] = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
            meta['block_size'] = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
            meta['data_rate'] = float(np.fromfile(fid, dtype=np.float64, count=1)[0])
            if file_format == 2:
                meta['p_baro'], meta['t_mean'] = np.fromfile(fid, dtype=np.float64, count=2)
            meta['has_pref'] = bool(np.fromfile(fid, dtype=np.uint8, count=1)[0])

        n, b = meta['num_samples'], meta['block_size']
        n_blocks = n // b
        u_all, v_all, w_all, ps_all, pref_all = [], [], [], [], []

        for _ in range(n_blocks):
            u_all.append(np.fromfile(fid, dtype=np.float32, count=b))
            v_all.append(np.fromfile(fid, dtype=np.float32, count=b))
            w_all.append(np.fromfile(fid, dtype=np.float32, count=b))
            ps_all.append(np.fromfile(fid, dtype=np.float32, count=b))
            if meta['has_pref']:
                pref_all.append(np.fromfile(fid, dtype=np.float32, count=b))

        # Raw firmware velocity components (no correction here).  The affine
        # Cobra->pitot calibration (V_cal = a*|V| + b) is applied later in
        # process_files, AFTER all rotations and per-point calculations, on the
        # velocity magnitude; u, v, w are then rescaled by V_cal/|V|.
        u = np.concatenate(u_all)
        v = np.concatenate(v_all)
        w = np.concatenate(w_all)
        ps = np.concatenate(ps_all)
        pref = np.concatenate(pref_all) if meta['has_pref'] else None

        return (u, v, w, ps, pref, meta)


# --- 3. GUI APPLICATION ---
class CobraDataProcessorApp:
    def __init__(self, root, default_folder=None):
        self.root = root
        self.root.title("Cobra Probe Data Processor")
        self.root.geometry("750x700")
        
        self.folder_path = tk.StringVar(value=default_folder if default_folder else "")

        # Load geometry defaults from config (with fallback to old values)
        _step_x_def  = str(float(_cfg_geom['grid_step_x']))        if _cfg_geom is not None else "15.0"
        _step_y_def  = str(float(_cfg_geom['grid_step_y']))        if _cfg_geom is not None else "1.5875"
        _h_noz_def   = str(float(_cfg_geom['nozzle_height_B']))    if _cfg_geom is not None else "18.0"
        _x_start_def = str(float(_cfg_geom['x_start_offset']))     if _cfg_geom is not None else "40.0"
        _ref_file    = str(_cfg_geom['reference_file_name'])        if _cfg_geom is not None else "0010.asA"

        # Grid parameters
        self.step_x = tk.StringVar(value=_step_x_def)
        self.step_y = tk.StringVar(value=_step_y_def)

        # Physical & Non-dimensional parameters
        self.B_mm = tk.StringVar(value=_h_noz_def)
        self.x_start = tk.StringVar(value=_x_start_def)
        self.y_start = tk.StringVar(value="0.0") 
        
        # Rotation
        self.rot_az = tk.StringVar(value="0.0")
        self.rot_el = tk.StringVar(value="0.0")
        self.rot_roll = tk.StringVar(value="0.0")
        
        self.setup_ui()

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Case Study Folder")
        if folder:
            self.folder_path.set(folder)
            self.log(f"Selected workspace: {folder}")

    def start_processing(self):
        folder = self.folder_path.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder first.")
            return
            
        try:
            dx = float(self.step_x.get())
            dy = float(self.step_y.get())
            h = float(self.B_mm.get())
            x_s = float(self.x_start.get())
            y_s = float(self.y_start.get())
            az = float(self.rot_az.get())
            el = float(self.rot_el.get())
            roll = float(self.rot_roll.get())

            if h <= 0: raise ValueError("Nozzle height 'B' must be greater than 0.")
        except ValueError:
            messagebox.showerror("Error", "Please verify all numeric inputs are valid.")
            return

        self.btn_process.config(state="disabled")
        thread = threading.Thread(target=self.process_files, args=(folder, dx, dy, h, x_s, y_s, az, el, roll))
        thread.start()

    def process_files(self, main_folder, dx, dy, h, x_start, y_start, az, el, roll):
        tha_dir = os.path.join(main_folder, "thA")
        asa_dir = os.path.join(main_folder, "asA")
        
        root_tha = glob.glob(os.path.join(main_folder, "*.thA"))
        root_asa = glob.glob(os.path.join(main_folder, "*.asA"))
        
        if root_tha or root_asa:
            self.log("Organizing mixed raw files into /thA and /asA subfolders...")
            if root_tha: os.makedirs(tha_dir, exist_ok=True)
            if root_asa: os.makedirs(asa_dir, exist_ok=True)
            
            for f in root_tha: shutil.move(f, os.path.join(tha_dir, os.path.basename(f)))
            for f in root_asa: shutil.move(f, os.path.join(asa_dir, os.path.basename(f)))
            self.log(" -> Organization complete.")

        source_dir = tha_dir if os.path.exists(tha_dir) else main_folder
        files_to_process = sorted(glob.glob(os.path.join(source_dir, "*.thA")))
        total_files = len(files_to_process)
        
        if total_files == 0:
            self.log("No .thA files found in workspace or /thA subfolder. Aborting.")
            self.root.after(0, lambda: self.btn_process.config(state="normal"))
            return

        self.log("Scanning files to auto-calculate Y midpoint...")
        y_grids = []
        y_indices = []
        for path in files_to_process:
            file_name = os.path.basename(path)
            name_no_ext = os.path.splitext(file_name)[0]
            if len(name_no_ext) >= 4 and name_no_ext[:4].isdigit():
                idx_y = int(name_no_ext[2:4])
                y_grids.append(idx_y * dy)
                y_indices.append(idx_y)

        if y_grids:
            y_midpoint = (min(y_grids) + max(y_grids)) / 2.0
            self.log(f" -> Min Y: {min(y_grids)} mm | Max Y: {max(y_grids)} mm")
            self.log(f" -> Auto-centering all Y values around midpoint: {y_midpoint} mm")
        else:
            y_midpoint = 0.0
            self.log(" -> Warning: Could not detect valid 'abxy' grids. Center set to 0.")

        # Centerline Y-index = middle of the measured Y range (mean of total Y
        # count). Centerline-only cases give 10 (files XX10); full 2-D cases
        # give e.g. 06 (Y range 00..12) or 05 (00..10).
        if y_indices:
            centerline_y_idx = int(round((min(y_indices) + max(y_indices)) / 2.0))
        else:
            centerline_y_idx = 10
        self.log(f" -> Centerline Y-index (mean of Y range): {centerline_y_idx:02d}")

        csv_out_dir = os.path.join(main_folder, "Processed_CSVs", "Raw_Data")
        param_out_dir = os.path.join(main_folder, "Processing_Parameters")
        os.makedirs(csv_out_dir, exist_ok=True)
        os.makedirs(param_out_dir, exist_ok=True)

        # =========================================================
        # FLOW ALIGNMENT (ROTATION 3/3): align the centerline middle-flow
        # with +X.  The reference is the MIDDLE point of the centerline row —
        # the middle X index at the centerline Y index — applying the same
        # "mean of the range" logic to X as to Y.  U is strongest mid-domain,
        # so it gives the most reliable flow-direction reference (better than
        # the first X=00 station).  E.g. centerline-only cases -> 1010, 2-D
        # cases -> 0506 / 0505.  We read that reference, apply rotations 1
        # (axis switch) and 2 (Z tilt), measure the residual mean pitch/yaw of
        # its flow, and rotation 3 zeroes those so the centerline runs along X.
        # =========================================================
        z_tilt_deg = z_tilt_deg_from_config()
        self.log(f"\n[ROTATION 2/3] Fixed Z-axis tilt = {z_tilt_deg:.4f}° "
                 f"(-arctan(rise/run)).")

        align_yaw, align_pitch = 0.0, 0.0
        ref_basename = "(none)"
        ref_x_idx = -1

        if az == 0.0 and el == 0.0:
            # Auto: among the centerline-row files (Y == centerline_y_idx),
            # pick the one whose X index is closest to the middle of the X
            # range. Picking the closest existing file keeps the reference
            # valid even when the exact middle station was not measured.
            centerline_row = []
            for _p in files_to_process:
                _nm = os.path.splitext(os.path.basename(_p))[0]
                if len(_nm) >= 4 and _nm[:4].isdigit() and int(_nm[2:4]) == centerline_y_idx:
                    centerline_row.append((int(_nm[:2]), _p))
            if centerline_row:
                _x_idx = [xi for xi, _ in centerline_row]
                _mid_x = (min(_x_idx) + max(_x_idx)) / 2.0
                ref_x_idx, _ref_path = min(centerline_row, key=lambda t: abs(t[0] - _mid_x))
                try:
                    _ur, _vr, _wr, _, _, _ = read_th_file(_ref_path)
                    _u1, _v1, _w1 = axis_switch_90(_ur, _vr, _wr)
                    _u2, _v2, _w2 = transform_axes_az_el(
                        _u1, _v1, _w1, az_deg=z_tilt_deg, el_deg=0.0, roll_deg=0.0)
                    _mu, _mv, _mw = float(np.mean(_u2)), float(np.mean(_v2)), float(np.mean(_w2))
                    _vmag = (_mu**2 + _mv**2 + _mw**2) ** 0.5
                    if _vmag > 1e-9:
                        align_yaw = float(np.degrees(np.arctan2(_mv, _mu)))
                        align_pitch = float(np.degrees(np.arcsin(np.clip(_mw / _vmag, -1.0, 1.0))))
                    ref_basename = os.path.basename(_ref_path)
                    self.log(f"[ROTATION 3/3] Centerline middle reference: {ref_basename} "
                             f"(X-idx={ref_x_idx:02d}, Y-idx={centerline_y_idx:02d})")
                    self.log(f" -> Residual mean flow after rotations 1+2: "
                             f"yaw={align_yaw:.3f}°, pitch={align_pitch:.3f}°  (will be zeroed)")
                except Exception as e:
                    self.log(f"[ROTATION 3/3] Could not read reference "
                             f"{os.path.basename(_ref_path)}: {e} — alignment = 0°.")
            else:
                self.log(f"[ROTATION 3/3] Warning: no centerline-row files at "
                         f"Y-idx {centerline_y_idx:02d} — alignment = 0°.")
        else:
            # Manual override: az/el are taken as the centerline misalignment
            # (yaw/pitch) to be zeroed.
            align_yaw, align_pitch = az, el
            ref_basename = "(manual override)"
            self.log(f"[ROTATION 3/3] Manual override — yaw={align_yaw}°, pitch={align_pitch}°.")

        self.log("\nSaving Processing Parameters summary...")
        sum_data = [{
            "Step_X_mm": dx,
            "Step_Y_mm": dy,
            "B_mm": h,
            "X_Start_Offset_mm": x_start, 
            "Y_Auto_Midpoint_mm": y_midpoint,
            "Y_Manual_Offset_mm": y_start,
            "Azimuth_deg": align_yaw,
            "Elevation_deg": align_pitch,
            "Roll_deg": roll,
            "Z_Tilt_deg": z_tilt_deg,
            "Centerline_Y_idx": centerline_y_idx,
            "Reference_X_idx": ref_x_idx,
            "Alignment_Ref_File": ref_basename,
            "Total_Files_Processed": total_files
        }]
        df_sum = pd.DataFrame(sum_data)

        self.log("Searching for Flow_Data.csv to merge processing parameters...")
        search_patterns = [
            os.path.join(main_folder, "Flow_Data*.csv"),
            os.path.join(main_folder, "**", "Flow_Data*.csv")
        ]
        
        target_csv = None
        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                target_csv = max(matches, key=os.path.getctime)
                break

        if target_csv:
            self.log(f"Found master file: {os.path.basename(target_csv)}")
            try:
                existing_df = pd.read_csv(target_csv)
                if not existing_df.empty:
                    for col in df_sum.columns:
                        existing_df[col] = df_sum[col].iloc[0]
                    updated_df = existing_df
                else:
                    updated_df = df_sum

                updated_df.to_csv(target_csv, index=False)
                target_xlsx = target_csv.replace('.csv', '.xlsx')
                try:
                    updated_df.to_excel(target_xlsx, index=False)
                except Exception:
                    pass
                self.log(" -> Successfully merged processing parameters into Flow_Data files.")
            except Exception as e:
                self.log(f" -> Error updating file: {e}")
        else:
            self.log("Flow_Data.csv not found! Creating a new master file with parameters.")
            new_csv = os.path.join(main_folder, "Flow_Data.csv")
            new_xlsx = os.path.join(main_folder, "Flow_Data.xlsx")
            df_sum.to_csv(new_csv, index=False)
            try:
                df_sum.to_excel(new_xlsx, index=False)
            except Exception:
                pass
            self.log(f" -> Saved parameters to: {new_csv}")

        self.log(
            f"\n[VELOCITY CALIBRATION] Cobra -> pitot affine law: "
            f"V_cal = {_CAL_A:.4f} * V_cobra + {_CAL_B:.4f}  (SE={_CAL_SE:.2f} m/s, "
            f"valid V <= {_CAL_VMAX:.0f} m/s; config_calibration.xlsx). Applied per-sample "
            "to |V| AFTER rotations, components rescaled by V_cal/|V| (pitch/yaw and static "
            "pressure unchanged). Raw |V| above the limit is extrapolated with a per-point "
            "warning."
            if abs(_CAL_A - 1.0) > 1e-9 or abs(_CAL_B) > 1e-9 else
            "\n[VELOCITY CALIBRATION] identity (a=1, b=0): config_calibration.xlsx not found; "
            "velocities left uncalibrated."
        )

        self.log("\n--- STARTING BATCH PROCESSING ---")
        
        for i, path in enumerate(files_to_process):
            file_name = os.path.basename(path)
            name_no_ext = os.path.splitext(file_name)[0]
            self.log(f"[{i+1}/{total_files}] Processing: {file_name}")
            
            coord_x_grid, coord_y_grid = 0.0, 0.0
            if len(name_no_ext) >= 4 and name_no_ext[:4].isdigit():
                idx_x = int(name_no_ext[:2]) 
                idx_y = int(name_no_ext[2:4]) 
                coord_x_grid = idx_x * dx
                coord_y_grid = idx_y * dy
            else:
                self.log(f"  -> Warning: Filename '{name_no_ext}' is not 'abxy'. Grid set to 0.")

            x_phys = coord_x_grid + x_start
            y_phys = coord_y_grid - y_midpoint + y_start
            
            x_nd = x_phys / h
            y_nd = y_phys / h

            # =========================================================
            # MATCHING EACH .thA WITH ITS EXACT .asA COUNTERPART
            # =========================================================
            current_asa = path.replace('.thA', '.asA')
            if not os.path.exists(current_asa):
                current_asa = os.path.join(asa_dir, os.path.basename(path).replace('.thA', '.asA'))
            
            qc_data = parse_asa_file(current_asa)

            try:
                # Raw (uncorrected) firmware velocities; the affine Cobra->pitot
                # calibration is applied below, after the rotations.
                u_raw, v_raw, w_raw, ps, p_ref, meta = read_th_file(path)

                # Log a QC-yield warning if below the 80 % Cobra acceptance threshold.
                if qc_data['yield_pct'] < 80.0:
                    self.log(f"  -> QC-YIELD WARNING [{file_name} | X/B={x_nd:.3f} Y/B={y_nd:.3f}]: yield {qc_data['yield_pct']:.1f}% < 80%.")

                time_s = np.arange(len(u_raw)) / meta['data_rate']

                # ============================================================
                # THREE-STAGE FLOW ROTATION (applied in order)
                # ------------------------------------------------------------
                # 1/3 — 90° axis switch: probe Y/Z were swapped; map
                #       (u, v, w) -> (u, w, -v) to restore V=spanwise, W=wall-normal.
                u1, v1, w1 = axis_switch_90(u_raw, v_raw, w_raw)

                # 2/3 — fixed Z-axis tilt of -arctan(rise/run) (geometric
                #       correction), applied AFTER the axis switch.
                u2, v2, w2 = transform_axes_az_el(
                    u1, v1, w1, az_deg=z_tilt_deg, el_deg=0.0, roll_deg=0.0)

                # 3/3 — align the centerline middle-flow with +X by zeroing the
                #       reference residual angles (az=-yaw, el=+pitch convention).
                u_rot, v_rot, w_rot = transform_axes_az_el(
                    u2, v2, w2, az_deg=-align_yaw, el_deg=align_pitch, roll_deg=0.0)

                # Calculate the exact corrected instantaneous angles and magnitude
                vel_mag, pitch_corr, yaw_corr = vel_pitch_yaw(u_rot, v_rot, w_rot)

                # --- VELOCITY CALIBRATION: affine Cobra -> pitot, per-sample ---
                # Applied LAST (after every rotation and the magnitude/angle
                # calculation).  The calibration is defined on the SPEED |V|:
                #     V_cal = a * |V| + b      (a, b from config_calibration.xlsx)
                # so each instantaneous speed is mapped and the three components
                # are rescaled by V_cal/|V| to preserve direction (pitch, yaw and
                # the static pressure are left untouched).  Because the law is
                # affine, the mean -> a*mean+b and the RMS fluctuations -> a*RMS.
                vel_cal, _n_over = apply_cobra_calibration(vel_mag)
                _scale = np.divide(vel_cal, vel_mag,
                                   out=np.ones_like(vel_mag), where=vel_mag > 1e-9)
                u_rot = u_rot * _scale
                v_rot = v_rot * _scale
                w_rot = w_rot * _scale
                vel_mag = vel_cal
                # Out-of-range warning: raw |V| above the calibration validity
                # limit (~45 m/s, the Cobra fold-back) is EXTRAPOLATED, not valid.
                if _n_over > 0:
                    self.log(
                        f"  -> CALIBRATION RANGE WARNING [{file_name} | "
                        f"X/B={x_nd:.3f} Y/B={y_nd:.3f}]: {_n_over}/{vel_mag.size} "
                        f"samples ({100.0*_n_over/vel_mag.size:.1f}%) had raw |V| > "
                        f"{_CAL_VMAX:.0f} m/s (past fold-back) — calibration extrapolated.")

                # Append everything to the CSV
                df = pd.DataFrame({
                    'X/B': x_nd,
                    'Y/B': y_nd,
                    'X_phys (mm)': x_phys,
                    'Y_phys (mm)': y_phys,
                    'Time (s)': time_s,
                    'u (m/s)': u_rot, 
                    'v (m/s)': v_rot, 
                    'w (m/s)': w_rot,
                    'Velocity_Magnitude (m/s)': vel_mag,
                    'Pitch_Angle (deg)': pitch_corr,
                    'Yaw_Angle (deg)': yaw_corr,
                    'Static_Pressure (Pa)': ps,
                    'Temperature (°C)': qc_data['temp'],
                    'Barometric_Pressure (Pa)': qc_data['patm'],
                    'QC_Yield (%)': qc_data['yield_pct']
                })

                if p_ref is not None:
                    df['Reference_Pressure (Pa)'] = p_ref

                out_name = file_name.replace('.thA', '.csv')
                out_path = os.path.join(csv_out_dir, out_name)
                df.to_csv(out_path, index=False) 
                
            except Exception as e:
                self.log(f"  -> ERROR processing {file_name}: {e}")

        self.log("\n--- PROCESSING COMPLETE ---")
        self.log(f"All CSVs saved in: {csv_out_dir}")
        self.root.after(0, lambda: self.btn_process.config(state="normal"))

    def setup_ui(self):
        f_frame = tk.LabelFrame(self.root, text=" 1. Workspace / Case Study Folder ", padx=10, pady=10)
        f_frame.pack(padx=20, pady=10, fill="x")
        
        tk.Entry(f_frame, textvariable=self.folder_path, state="readonly", width=50).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(f_frame, text="Browse...", command=self.select_folder).pack(side="right", padx=5)

        g_frame = tk.LabelFrame(self.root, text=" 2. Coordinate Mapping & Geometry ", padx=10, pady=10)
        g_frame.pack(padx=20, pady=5, fill="x")
        
        tk.Label(g_frame, text="Step X (mm):").grid(row=0, column=0, sticky="e", pady=5)
        tk.Entry(g_frame, textvariable=self.step_x, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        tk.Label(g_frame, text="Step Y (mm):").grid(row=0, column=2, sticky="e", pady=5, padx=(10,0))
        tk.Entry(g_frame, textvariable=self.step_y, width=10).grid(row=0, column=3, sticky="w", padx=5)
        
        tk.Label(g_frame, text="B (mm):").grid(row=0, column=4, sticky="e", pady=5, padx=(10,0))
        tk.Entry(g_frame, textvariable=self.B_mm, width=10).grid(row=0, column=5, sticky="w", padx=5)

        tk.Label(g_frame, text="X Start Offset (mm):").grid(row=1, column=0, sticky="e", pady=5)
        tk.Entry(g_frame, textvariable=self.x_start, width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        tk.Label(g_frame, text="Y Start Offset (mm):").grid(row=1, column=2, sticky="e", pady=5, padx=(10,0))
        tk.Entry(g_frame, textvariable=self.y_start, width=10).grid(row=1, column=3, sticky="w", padx=5)

        r_frame = tk.LabelFrame(self.root, text=" 3. Probe Rotation Override (Degrees) ", padx=10, pady=10)
        r_frame.pack(padx=20, pady=5, fill="x")
        
        tk.Label(r_frame, text="(Leave at 0.0 to auto-align the flow using the middle-X centerline reference point)", fg="#888", font=("Arial", 9, "italic")).pack(side="top", anchor="w", pady=(0, 5))
        
        rot_inputs = tk.Frame(r_frame)
        rot_inputs.pack(side="top", anchor="w")
        tk.Label(rot_inputs, text="Azimuth (Yaw):").pack(side="left")
        tk.Entry(rot_inputs, textvariable=self.rot_az, width=8).pack(side="left", padx=5)
        tk.Label(rot_inputs, text="Elevation (Pitch):").pack(side="left", padx=(15,0))
        tk.Entry(rot_inputs, textvariable=self.rot_el, width=8).pack(side="left", padx=5)
        tk.Label(rot_inputs, text="Roll:").pack(side="left", padx=(15,0))
        tk.Entry(rot_inputs, textvariable=self.rot_roll, width=8).pack(side="left", padx=5)

        self.btn_process = tk.Button(self.root, text="PROCESS ALL .thA TO .CSV", command=self.start_processing, bg="#007acc", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_process.pack(padx=20, pady=15, fill="x")

        tk.Label(self.root, text="Process Log:").pack(anchor="w", padx=20)
        self.log_text = tk.Text(self.root, height=12, state="disabled", bg="#f4f4f4")
        self.log_text.pack(padx=20, pady=(0,20), fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = CobraDataProcessorApp(root)
    root.mainloop()
