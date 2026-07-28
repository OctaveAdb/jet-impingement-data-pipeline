"""
data_intake.py — New Case Data Intake Validator & Registrar
============================================================

Standalone Tkinter GUI that:
  1. Scans the new_data/ staging folder for candidate case folders.
  2. Validates each folder against naming convention and detected file types.
  3. Auto-sorts flat folders: .asA → asA/, .thA → thA/ subfolders.
  4. Handles thermal supplements: temperature CSVs and PicoLog files.
  5. Moves aerodynamic folders to experiments/ and appends a row to config_cases.csv.

Run directly:
    python scripts/pipeline/data_intake.py

Or launch from usage.py sidebar ("Import New Case Data").

Path resolution is always relative to this script's location — no hardcoded
absolute paths. The project root is two directories up from this file:
    <root>/scripts/pipeline/data_intake.py  ->  <root>/
"""

import os
import re
import csv
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ---------------------------------------------------------------------------
# Path resolution — always derived from __file__, never hardcoded
# ---------------------------------------------------------------------------

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_BASE          = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_NEW_DATA      = os.path.join(_BASE, 'new_data')
_EXPERIMENTS   = os.path.join(_BASE, 'experiments')
_CONFIG_CASES  = os.path.join(_BASE, 'config', 'config_cases.xlsx')
_THERMAL_BASE  = os.path.join(_BASE, 'thermal')

# Naming convention: (Cyl|Free) + integer + Pa + (Free|Pla)
_NAME_REGEX = re.compile(r'^(Cyl|Free)\d+Pa(Free|Pla)$')

# ---------------------------------------------------------------------------
# Core validation and sorting functions
# ---------------------------------------------------------------------------

def validate_and_sort_case_folder(folder_path):
    """
    Inspects a flat intake folder, sorts files, and returns:
    (is_valid, error_message, case_type, sorted_counts)
    case_type: 'aerodynamic', 'thermal', 'mixed', 'invalid'
    sorted_counts: dict with counts of files found per type
    """
    name = os.path.basename(folder_path)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    asa_files = [f for f in files if f.lower().endswith('.asa')]
    tha_files = [f for f in files if f.lower().endswith('.tha')]
    # Temperature CSVs: named temp_*.csv or similar (NOT already having 'raw' in name)
    temp_csvs_raw = [f for f in files if f.lower().endswith('.csv') and 'raw' not in f.lower() and any(kw in f.lower() for kw in ['temp', 'temperature'])]
    temp_csvs_already_raw = [f for f in files if f.lower().endswith('.csv') and 'raw' in f.lower()]
    # PicoLog files: .pl2, .pls, .picolog, or any non-csv/asa/tha files that look like logger data
    picolog_files = [f for f in files if f.lower().endswith(('.pl2', '.pls', '.picolog', '.dat', '.log'))]
    other_csvs = [f for f in files if f.lower().endswith('.csv') and f not in temp_csvs_raw and f not in temp_csvs_already_raw]

    # Also check existing subfolders (pre-sorted folders are still accepted)
    asa_dir = os.path.join(folder_path, 'asA')
    tha_dir = os.path.join(folder_path, 'thA')
    if os.path.isdir(asa_dir):
        asa_files += [f for f in os.listdir(asa_dir) if f.lower().endswith('.asa')]
    if os.path.isdir(tha_dir):
        tha_files += [f for f in os.listdir(tha_dir) if f.lower().endswith('.tha')]

    has_aero = bool(asa_files or tha_files)
    has_thermal = bool(temp_csvs_raw or temp_csvs_already_raw or picolog_files)

    if not has_aero and not has_thermal:
        return False, "No recognizable files (.asA, .thA, temperature CSVs, PicoLog files)", 'invalid', {}

    case_type = 'mixed' if (has_aero and has_thermal) else ('aerodynamic' if has_aero else 'thermal')

    # Validate naming for aerodynamic component
    if has_aero:
        if not re.match(r'^(Cyl|Free)\d+Pa(Free|Pla)$', name):
            return False, f"Folder name '{name}' does not match convention [Cyl|Free][N]Pa[Free|Pla]", case_type, {}

    sorted_counts = {
        'asA': len(asa_files), 'thA': len(tha_files),
        'temp_csv_new': len(temp_csvs_raw),
        'temp_csv_existing_raw': len(temp_csvs_already_raw),
        'picolog': len(picolog_files),
        'other': len(other_csvs)
    }
    return True, "", case_type, sorted_counts


def sort_files_in_place(folder_path):
    """
    Moves files within the folder into asA/ and thA/ subfolders.
    Creates subfolders if they don't exist.
    Returns (success, message).
    """
    asa_dir = os.path.join(folder_path, 'asA')
    tha_dir = os.path.join(folder_path, 'thA')
    os.makedirs(asa_dir, exist_ok=True)
    os.makedirs(tha_dir, exist_ok=True)

    moved = {'asA': 0, 'thA': 0}
    for f in os.listdir(folder_path):
        src = os.path.join(folder_path, f)
        if not os.path.isfile(src):
            continue
        if f.lower().endswith('.asa'):
            shutil.move(src, os.path.join(asa_dir, f))
            moved['asA'] += 1
        elif f.lower().endswith('.tha'):
            shutil.move(src, os.path.join(tha_dir, f))
            moved['thA'] += 1
    return True, f"Sorted {moved['asA']} .asA and {moved['thA']} .thA files"


def handle_thermal_files(folder_path, thermal_base_path):
    """
    Handles temperature CSVs and PicoLog files from an intake folder.
    - temp_*.csv without 'raw' → copy as temp_raw_*.csv to thermal/raw_data/,
      move original to thermal/raw_data/
    - PicoLog files → move to thermal/raw_inputs/Data_Picolog/
    Returns (success, message)
    """
    raw_data_dir = os.path.join(thermal_base_path, 'raw_data')
    picolog_dir  = os.path.join(thermal_base_path, 'raw_inputs', 'Data_Picolog')
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(picolog_dir, exist_ok=True)

    msgs = []
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for f in files:
        src = os.path.join(folder_path, f)
        fl = f.lower()
        if fl.endswith('.csv') and 'raw' not in fl and any(kw in fl for kw in ['temp', 'temperature']):
            # Create raw copy
            base, ext = os.path.splitext(f)
            raw_name = f"temp_raw_{base.replace('temp_', '').replace('temperature_', '')}{ext}"
            shutil.copy2(src, os.path.join(raw_data_dir, raw_name))
            shutil.move(src, os.path.join(raw_data_dir, f))
            msgs.append(f"  {f} -> raw_data/ (+ raw copy as {raw_name})")
        elif fl.endswith('.csv') and 'raw' in fl:
            shutil.move(src, os.path.join(raw_data_dir, f))
            msgs.append(f"  {f} -> raw_data/")
        elif fl.endswith(('.pl2', '.pls', '.picolog', '.dat', '.log')):
            shutil.move(src, os.path.join(picolog_dir, f))
            msgs.append(f"  {f} -> raw_inputs/Data_Picolog/")

    return True, "Thermal files handled:\n" + "\n".join(msgs) if msgs else "No thermal files found"


def scan_intake_folder(new_data_path: str) -> list:
    """
    Scan the new_data/ staging folder and validate every immediate subfolder.

    Parameters
    ----------
    new_data_path : str
        Absolute path to the new_data/ directory.

    Returns
    -------
    list of (folder_path, is_valid, error_message, case_type, sorted_counts) tuples.

    Raises
    ------
    FileNotFoundError
        If new_data_path does not exist.
    """
    if not os.path.isdir(new_data_path):
        raise FileNotFoundError(
            f"Staging folder not found: {new_data_path}\n"
            "Create new_data/ at the project root before running intake."
        )

    results = []
    for entry in sorted(os.listdir(new_data_path)):
        full_path = os.path.join(new_data_path, entry)
        if not os.path.isdir(full_path):
            continue  # skip loose files (e.g. README_INTAKE.txt)
        is_valid, msg, case_type, sorted_counts = validate_and_sort_case_folder(full_path)
        results.append((full_path, is_valid, msg, case_type, sorted_counts))

    return results


# ---------------------------------------------------------------------------
# Registration function
# ---------------------------------------------------------------------------

def _derive_config_row(case_name: str) -> dict:
    """
    Derive default config_cases.csv row values from a validated case name.

    Logic (can be overridden manually in config_cases.csv afterwards):
      - case_pattern     : "Cyl12" if name starts with "Cyl12",
                           "Free"  if name starts with "Free",
                           else the raw case name.
      - has_cylinder     : True if name starts with "Cyl".
      - peak_threshold_hz: 400.0 for Cyl12x families, else 75.0.
      - nozzle_width_mm  : 150.0 for cylinder cases, 170.0 for free-jet.
      - x_start_offset_mm: 38.0 (default; user should verify before pipeline).
      - description      : auto-generated summary string.
    """
    has_cyl = case_name.startswith('Cyl')

    if case_name.startswith('Cyl12'):
        case_pattern     = 'Cyl12'
        peak_threshold   = 400.0
    elif has_cyl:
        case_pattern     = case_name           # keep full name for unique Cyl cases
        peak_threshold   = 75.0
    else:
        case_pattern     = 'Free'
        peak_threshold   = 75.0

    nozzle_width     = 150.0 if has_cyl else 170.0  # cylinder slot span per PROTOCOL §2.1/§3 (config_geometry: nozzle_width_cyl)
    x_start_offset   = 38.0

    seeding = 'with heated plate' if case_name.endswith('PaPla') else 'aerodynamic only'
    config  = 'Cylinder obstacle' if has_cyl else 'Free jet'
    description = f"{config}; {seeding}; auto-registered by data_intake.py"

    return {
        'case_pattern'      : case_pattern,
        'has_cylinder'      : str(has_cyl),
        'peak_threshold_hz' : str(peak_threshold),
        'nozzle_width_mm'   : str(nozzle_width),
        'x_start_offset_mm' : str(x_start_offset),
        'description'       : description,
    }


def register_case(
    case_name: str,
    new_data_path: str,
    experiments_path: str,
    config_path: str,
    case_type: str = 'aerodynamic',
    thermal_base_path: str = None,
) -> tuple:
    """
    Register a validated case: sort files in place, move aero folder to
    experiments/, route thermal files to thermal/, and append to CSV.

    Steps
    -----
    1. Check source folder is still present.
    2. Sort flat files into asA/ and thA/ subfolders (aerodynamic component).
    3. Handle thermal files (thermal/mixed components).
    4. Move aerodynamic folder to experiments/ (aerodynamic/mixed only).
    5. Append a new row to config/config_cases.csv (aerodynamic/mixed only).

    Parameters
    ----------
    case_name          : str — bare folder name (e.g. "Cyl125PaFree").
    new_data_path      : str — absolute path to new_data/ directory.
    experiments_path   : str — absolute path to experiments/ directory.
    config_path        : str — absolute path to config_cases.csv.
    case_type          : str — 'aerodynamic', 'thermal', or 'mixed'.
    thermal_base_path  : str — absolute path to thermal/ directory.

    Returns
    -------
    (success, message) : tuple[bool, str]
    """
    if thermal_base_path is None:
        thermal_base_path = _THERMAL_BASE

    src = os.path.join(new_data_path, case_name)
    dst = os.path.join(experiments_path, case_name)

    # Guard: source must still be present
    if not os.path.isdir(src):
        return False, f"Source folder no longer found: {src}"

    messages = []
    has_aero = case_type in ('aerodynamic', 'mixed')
    has_thermal = case_type in ('thermal', 'mixed')

    # --- Step 1: sort flat files into asA/ and thA/ ---
    if has_aero:
        ok, sort_msg = sort_files_in_place(src)
        messages.append(sort_msg)

    # --- Step 2: handle thermal files ---
    if has_thermal:
        ok, thermal_msg = handle_thermal_files(src, thermal_base_path)
        messages.append(thermal_msg)

    # --- Step 3: move aerodynamic folder to experiments/ ---
    if has_aero:
        # Guard: refuse to overwrite an existing experiment
        if os.path.exists(dst):
            return False, (
                f"Case '{case_name}' already exists in experiments/. "
                "Registration aborted — no files were moved."
            )

        # Ensure experiments/ exists
        os.makedirs(experiments_path, exist_ok=True)

        try:
            shutil.move(src, dst)
            messages.append(f"Moved to experiments/{case_name}")
        except (OSError, shutil.Error) as exc:
            return False, f"Failed to move folder to experiments/: {exc}"

    # --- Step 4: append row to config_cases.xlsx (aero cases only) ---
    if has_aero:
        row = _derive_config_row(case_name)
        fieldnames = [
            'case_pattern',
            'has_cylinder',
            'peak_threshold_hz',
            'nozzle_width_mm',
            'x_start_offset_mm',
            'description',
        ]

        try:
            import pandas as _pd
            # Read existing rows (xlsx) if present, else start a fresh frame.
            if os.path.isfile(config_path) and os.path.getsize(config_path) > 0:
                df_cases = _pd.read_excel(config_path)
            else:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                df_cases = _pd.DataFrame(columns=fieldnames)
            # Preserve existing columns, add any of ours that are missing.
            for col in fieldnames:
                if col not in df_cases.columns:
                    df_cases[col] = None
            df_cases = _pd.concat([df_cases, _pd.DataFrame([row])], ignore_index=True)
            df_cases.to_excel(config_path, index=False)
            messages.append("Row appended to config_cases.xlsx")

        except Exception as exc:
            return True, (
                f"Folder moved to experiments/{case_name}, but failed to update "
                f"config_cases.xlsx: {exc}\n"
                "Please add the row manually.\n" + "\n".join(messages)
            )

    summary = "\n".join(messages)
    if has_aero:
        return True, (
            f"Case '{case_name}' registered successfully.\n"
            f"  Moved  : {dst}\n"
            f"  Config : row appended to config_cases.csv\n"
            f"  Next   : run the pipeline from Step 1 (usage.py)\n"
            f"{summary}"
        )
    else:
        # Thermal-only: source folder remains after files were moved out — delete it
        try:
            shutil.rmtree(src)
            messages.append(f"Cleaned up new_data/{case_name}")
        except OSError:
            messages.append(f"Warning: could not remove new_data/{case_name} — delete manually")
        summary = "\n".join(messages)
        return True, (
            f"Thermal supplement '{case_name}' processed.\n"
            f"  Files routed to thermal/ subfolders.\n"
            f"{summary}"
        )


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class DataIntakeApp:
    """
    Tkinter GUI for scanning, validating, and registering new experimental cases.

    Layout
    ------
    - Header: title + paths panel
    - Middle left: Listbox of discovered folders with VALID/INVALID status and case type
    - Bottom: Log area (scrolled text widget)
    - Buttons: Scan, Register Selected, Register All Valid
    """

    # Status icons (plain text; avoid font issues on all platforms)
    _ICON_VALID   = "V VALID"
    _ICON_INVALID = "X INVALID"

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.root.title("New Data Intake Validator")
        self.root.geometry("880x620")
        self.root.resizable(True, True)

        # Internal state: list of (folder_path, is_valid, error_message, case_type, sorted_counts)
        self._scan_results: list = []

        self._build_ui()
        self._log("Ready. Drop a flat folder into new_data/, then click 'Scan new_data/'.")
        self._log("Files will be auto-sorted: .asA -> asA/, .thA -> thA/ on registration.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble all UI widgets."""
        PAD = 8

        # ---- Top frame: title + path display -------------------------
        top = tk.Frame(self.root, bd=1, relief=tk.GROOVE, padx=PAD, pady=PAD)
        top.pack(fill=tk.X, padx=PAD, pady=(PAD, 0))

        tk.Label(
            top,
            text="New Data Intake — Flat Folder Drop",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        tk.Label(
            top,
            text="Drop any folder (flat or pre-sorted) into new_data/. "
                 ".asA/.thA files are auto-sorted; temperature CSVs and PicoLog files "
                 "are routed to thermal/ automatically.",
            fg="#333333",
            wraplength=750,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 6))

        tk.Label(top, text="Staging folder:", anchor=tk.W).grid(
            row=2, column=0, sticky=tk.W, pady=(2, 0)
        )
        tk.Label(
            top,
            text=_NEW_DATA,
            fg="#555555",
            wraplength=650,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=2, column=1, sticky=tk.W, pady=(2, 0))

        tk.Label(top, text="Experiments folder:", anchor=tk.W).grid(
            row=3, column=0, sticky=tk.W
        )
        tk.Label(
            top,
            text=_EXPERIMENTS,
            fg="#555555",
            wraplength=650,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=3, column=1, sticky=tk.W)

        tk.Label(top, text="Thermal folder:", anchor=tk.W).grid(
            row=4, column=0, sticky=tk.W
        )
        tk.Label(
            top,
            text=_THERMAL_BASE,
            fg="#555555",
            wraplength=650,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=4, column=1, sticky=tk.W)

        tk.Label(top, text="Config CSV:", anchor=tk.W).grid(
            row=5, column=0, sticky=tk.W
        )
        tk.Label(
            top,
            text=_CONFIG_CASES,
            fg="#555555",
            wraplength=650,
            anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=5, column=1, sticky=tk.W)

        # ---- Middle frame: listbox + scrollbar -----------------------
        mid = tk.Frame(self.root, padx=PAD, pady=PAD)
        mid.pack(fill=tk.BOTH, expand=True, padx=PAD, pady=(PAD, 0))

        tk.Label(
            mid,
            text="Folders found in new_data/:",
            font=("TkDefaultFont", 10, "bold"),
            anchor=tk.W,
        ).pack(anchor=tk.W)

        list_frame = tk.Frame(mid)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        scrollbar_y = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = tk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self._listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
            selectmode=tk.SINGLE,
            font=("Courier", 10),
            activestyle="none",
        )
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.config(command=self._listbox.yview)
        scrollbar_x.config(command=self._listbox.xview)

        # Colour tags for valid/invalid display
        self._listbox.configure(selectbackground="#0078d7", selectforeground="white")

        # ---- Button row ----------------------------------------------
        btn_frame = tk.Frame(self.root, padx=PAD, pady=4)
        btn_frame.pack(fill=tk.X, padx=PAD)

        tk.Button(
            btn_frame,
            text="Scan new_data/",
            width=18,
            command=self._on_scan,
            bg="#e8f4fd",
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame,
            text="Register Selected",
            width=18,
            command=self._on_register_selected,
            bg="#e8fde8",
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame,
            text="Register All Valid",
            width=18,
            command=self._on_register_all,
            bg="#fdf8e8",
        ).pack(side=tk.LEFT, padx=(0, 6))

        # ---- Log area -----------------------------------------------
        log_label = tk.Label(
            self.root,
            text="Log:",
            font=("TkDefaultFont", 10, "bold"),
            anchor=tk.W,
        )
        log_label.pack(anchor=tk.W, padx=PAD)

        self._log_text = scrolledtext.ScrolledText(
            self.root,
            height=8,
            state=tk.DISABLED,
            font=("Courier", 9),
            bg="#f8f8f8",
        )
        self._log_text.pack(fill=tk.X, padx=PAD, pady=(0, PAD))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Append a line to the log text area."""
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, message.rstrip() + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _refresh_listbox(self) -> None:
        """Re-populate the listbox from self._scan_results."""
        self._listbox.delete(0, tk.END)
        for folder_path, is_valid, error_msg, case_type, sorted_counts in self._scan_results:
            name = os.path.basename(folder_path)
            type_tag = f"[{case_type.upper()}]" if case_type != 'invalid' else "[INVALID]"
            if is_valid:
                label = f"  {self._ICON_VALID:<10}  {type_tag:<14}  {name}"
            else:
                label = f"  {self._ICON_INVALID:<10}  {type_tag:<14}  {name}  --  {error_msg}"
            self._listbox.insert(tk.END, label)
            # Colour: green for valid, red for invalid
            idx = self._listbox.size() - 1
            fg = "#1a7a1a" if is_valid else "#b30000"
            self._listbox.itemconfig(idx, fg=fg)

    def _get_selected_index(self) -> int | None:
        """Return the currently selected listbox index, or None."""
        sel = self._listbox.curselection()
        return sel[0] if sel else None

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_scan(self) -> None:
        """Scan new_data/ and refresh the listbox."""
        self._log("-" * 60)
        self._log(f"Scanning: {_NEW_DATA}")
        try:
            results = scan_intake_folder(_NEW_DATA)
        except FileNotFoundError as exc:
            self._log(f"ERROR: {exc}")
            messagebox.showerror("Folder Not Found", str(exc))
            return

        self._scan_results = results
        self._refresh_listbox()

        n_valid   = sum(1 for _, v, _, _, _ in results if v)
        n_invalid = len(results) - n_valid

        if not results:
            self._log("No subfolders found in new_data/. Nothing to validate.")
        else:
            self._log(
                f"Found {len(results)} folder(s): "
                f"{n_valid} valid, {n_invalid} invalid."
            )
            for folder_path, is_valid, error_msg, case_type, sorted_counts in results:
                name = os.path.basename(folder_path)
                if is_valid:
                    counts_str = ", ".join(f"{k}={v}" for k, v in sorted_counts.items() if v > 0)
                    self._log(f"  {name}: {case_type} ({counts_str})")
                else:
                    self._log(f"  {name}: INVALID — {error_msg}")

    def _register_one(self, index: int) -> bool:
        """
        Register a single case by its listbox index.

        Returns True on success, False on failure.
        Logs results internally.
        """
        if index >= len(self._scan_results):
            self._log("ERROR: index out of range.")
            return False

        folder_path, is_valid, error_msg, case_type, sorted_counts = self._scan_results[index]
        case_name = os.path.basename(folder_path)

        if not is_valid:
            self._log(
                f"Skipped '{case_name}': folder is invalid — {error_msg}"
            )
            return False

        self._log(f"Registering '{case_name}' (type: {case_type}) ...")
        success, message = register_case(
            case_name,
            new_data_path=_NEW_DATA,
            experiments_path=_EXPERIMENTS,
            config_path=_CONFIG_CASES,
            case_type=case_type,
            thermal_base_path=_THERMAL_BASE,
        )

        for line in message.splitlines():
            self._log(f"  {line}")

        if success:
            # Remove from internal list so it doesn't appear after re-scan
            self._scan_results.pop(index)
            self._refresh_listbox()

        return success

    def _on_register_selected(self) -> None:
        """Register the folder currently selected in the listbox."""
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showinfo(
                "Nothing Selected",
                "Please click a folder in the list, then click Register Selected.",
            )
            return
        self._log("-" * 60)
        self._register_one(idx)

    def _on_register_all(self) -> None:
        """Register all currently valid folders in sequence."""
        valid_indices = [
            i for i, (_, is_valid, _, _, _) in enumerate(self._scan_results) if is_valid
        ]
        if not valid_indices:
            messagebox.showinfo(
                "No Valid Folders",
                "No valid folders to register. Run 'Scan new_data/' first.",
            )
            return

        self._log("-" * 60)
        self._log(f"Registering {len(valid_indices)} valid folder(s) ...")
        n_ok = 0
        # Iterate in reverse index order so pops don't shift remaining indices
        for idx in sorted(valid_indices, reverse=True):
            ok = self._register_one(idx)
            if ok:
                n_ok += 1

        self._log(
            f"Done. {n_ok}/{len(valid_indices)} case(s) registered successfully."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = DataIntakeApp(root)
    root.mainloop()
