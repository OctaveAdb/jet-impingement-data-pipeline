# =============================================================================
# CASE_LABELS
# =============================================================================
# Purpose:
#   Central dictionary + helpers that map raw case IDs (e.g. "Cyl1210PaPla")
#   to clean human-readable labels for use in figures and tables.
#
# Inputs:  thermal/results/Thermal_Global_Summary.csv  (optional, for Re)
# Outputs: Functions imported by all visualisers.
#
# Usage:
#   import sys, os
#   sys.path.insert(0, os.path.join(<root>, 'scripts', 'utils'))
#   from case_labels import get_label, get_labels_dict, set_main_folder
# =============================================================================
import os
import re as _re

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_cached_re_map   = {}   # {pa_val (e.g. "5Pa"): Re_used (int)}  — populated by load_re_map()
_main_folder     = None # root folder, set via set_main_folder()

# ---------------------------------------------------------------------------
# 1. Static base labels (format strings filled at render time)
# ---------------------------------------------------------------------------
# Each value is a (config_tag, pa_tag, plate_tag) tuple derived from the ID;
# the dict below gives the long description template before context stripping.
# This mapping acts as the canonical list of known cases.

CASE_BASE_LABELS = {
    # Cylinder 12 mm ─────────────────────────────────────────────────────────
    "Cyl1210PaFree": "Cylinder 12mm, {Re} (no plate)",
    "Cyl1210PaPla":  "Cylinder 12mm, {Re}",
    "Cyl125PaFree":  "Cylinder 12mm, {Re} (no plate)",
    "Cyl125PaPla":   "Cylinder 12mm, {Re}",
    # Free Jet ────────────────────────────────────────────────────────────────
    "Free5PaFree":   "Free Jet, {Re} (no plate)",
    "Free5PaPla":    "Free Jet, {Re}",
    "Free10PaFree":  "Free Jet, {Re} (no plate)",
    "Free10PaPla":   "Free Jet, {Re}",
}

# ---------------------------------------------------------------------------
# 2. ID parser
# ---------------------------------------------------------------------------
_ID_PATTERN = _re.compile(
    r'^(?P<config>Cyl12|Free)'   # configuration: Cyl12 or Free
    r'(?P<pressure>\d+)Pa'       # pressure in Pa  (no leading zeros expected)
    r'(?P<seeding>Pla|Free)$'    # seeding: Pla (with plate) or Free (no plate)
)

def _parse_case_id(case_id: str) -> dict | None:
    """Return parsed components or None if the ID doesn't match."""
    m = _ID_PATTERN.match(case_id)
    if m is None:
        return None
    config   = m.group('config')           # "Cyl12" | "Free"
    pressure = m.group('pressure')         # "10" | "5"
    seeding  = m.group('seeding')          # "Pla" | "Free"
    pa_val   = pressure + "Pa"             # "10Pa" | "5Pa"
    has_cyl  = (config == "Cyl12")
    has_pla  = (seeding == "Pla")
    return dict(config=config, pressure=pressure, pa_val=pa_val,
                seeding=seeding, has_cyl=has_cyl, has_pla=has_pla)

# ---------------------------------------------------------------------------
# 3. Re map loader
# ---------------------------------------------------------------------------
def load_re_map(main_folder: str) -> dict:
    """
    Read thermal/results/Thermal_Global_Summary.csv and return
    {pa_val: Re_used (int)}.  Falls back to {} if file is absent.

    The summary contains multiple runs per pressure level; we take the Free-jet
    rows (Has_Cylinder == "No" / False) as the canonical Re per pressure since
    the Free-jet is the undisturbed reference.  If no Free-jet row exists for a
    given pressure, the first available row for that pressure is used.
    """
    global _cached_re_map

    csv_path = os.path.join(main_folder, 'thermal', 'results',
                            'Thermal_Global_Summary.csv')
    if not os.path.isfile(csv_path):
        return {}

    try:
        import csv
        re_map = {}
        with open(csv_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            # Detect Re column name flexibly
            fieldnames = reader.fieldnames or []
            re_col = None
            for candidate in ('Re_used', 'Re', 're_used', 'RE_used'):
                if candidate in fieldnames:
                    re_col = candidate
                    break
            if re_col is None:
                # Try first column that looks like a Reynolds number
                for fn in fieldnames:
                    if 're' in fn.lower():
                        re_col = fn
                        break
            if re_col is None:
                return {}

            # pa_col
            pa_col = None
            for candidate in ('Base_Pressure', 'base_pressure', 'pa_val', 'Pa'):
                if candidate in fieldnames:
                    pa_col = candidate
                    break
            if pa_col is None:
                return {}

            # has_cyl col
            cyl_col = None
            for candidate in ('Has_Cylinder', 'has_cylinder', 'HasCylinder'):
                if candidate in fieldnames:
                    cyl_col = candidate
                    break

            rows = list(reader)

        # Build pa_val → Re, preferring Free-jet rows
        for row in rows:
            pa_val = row.get(pa_col, '').strip()
            if not pa_val:
                continue
            try:
                re_val = int(round(float(row[re_col])))
            except (ValueError, TypeError):
                continue

            is_free_jet = False
            if cyl_col:
                cyl_str = row.get(cyl_col, '').strip().lower()
                is_free_jet = cyl_str in ('no', 'false', '0')

            # Only set if not already set by a Free-jet row
            if pa_val not in re_map or is_free_jet:
                re_map[pa_val] = re_val

        _cached_re_map = re_map
        return re_map

    except Exception:
        return {}

# ---------------------------------------------------------------------------
# 4. set_main_folder — lets callers prime the cache once
# ---------------------------------------------------------------------------
def set_main_folder(path: str) -> None:
    """Cache the project root so subsequent get_label() calls auto-load Re."""
    global _main_folder, _cached_re_map
    _main_folder = path
    _cached_re_map = load_re_map(path)

# ---------------------------------------------------------------------------
# 5. get_label — main labelling function
# ---------------------------------------------------------------------------
def get_label(case_id: str,
              re_val: int | None = None,
              context_cases: list | None = None) -> str:
    """
    Return a clean human-readable label for *case_id*.

    Parameters
    ----------
    case_id       : Raw case identifier, e.g. "Cyl1210PaPla".
    re_val        : Reynolds number to embed. If None, the cached Re map is
                    consulted; if still absent, the Pa value is used as proxy.
    context_cases : Full list of case_ids being plotted together.  Used to
                    suppress redundant parts of the label.

    Returns
    -------
    str, e.g. "Cylinder 12mm, Re≈NNNNN" or "Re≈NNNNN (no plate)" or "Re≈NNNNN"
    """
    parsed = _parse_case_id(case_id)
    if parsed is None:
        # Unknown ID — return as-is
        return case_id

    # ── Resolve Re ────────────────────────────────────────────────────────
    if re_val is None:
        # Try cache
        if not _cached_re_map and _main_folder:
            load_re_map(_main_folder)
        re_val = _cached_re_map.get(parsed['pa_val'])

    if re_val is not None:
        re_rounded = round(int(re_val) / 100) * 100
        re_str = f"Re≈{re_rounded:d}"
    else:
        re_str = parsed['pa_val']   # fallback: "10Pa"

    # ── Context analysis ──────────────────────────────────────────────────
    all_pla   = True   # all cases have plate?
    all_free_seed = True   # all cases are aero (no plate)?
    all_cyl   = True   # all cases are Cylinder?
    all_free_cfg  = True   # all cases are Free Jet?

    if context_cases:
        for cid in context_cases:
            p = _parse_case_id(cid)
            if p is None:
                continue
            if p['has_pla']:
                all_free_seed = False
            else:
                all_pla = False
            if p['has_cyl']:
                all_free_cfg = False
            else:
                all_cyl = False
    else:
        # Single-case context: no suppression
        all_pla = all_free_seed = all_cyl = all_free_cfg = False

    # ── Assemble label parts ──────────────────────────────────────────────
    parts = []

    # Config prefix
    if parsed['has_cyl']:
        if not all_cyl:       # only show "Cylinder 12mm" if other configs exist
            parts.append("Cylinder 12mm")
    else:
        if not all_free_cfg:  # only show "Free Jet" if other configs exist
            parts.append("Free Jet")

    # Re value
    parts.append(re_str)

    label = ", ".join(parts)

    # Plate annotation
    is_aero = not parsed['has_pla']
    if is_aero and not all_free_seed:
        # Aero case among a mixed set — tag it
        label += " (no plate)"
    # Pla cases: plate is the thermal default — no annotation needed

    return label

# ---------------------------------------------------------------------------
# 6. get_labels_dict — convenience wrapper
# ---------------------------------------------------------------------------
def get_labels_dict(case_ids: list,
                    re_dict: dict | None = None) -> dict:
    """
    Return {case_id: label} for every ID in *case_ids*.

    Parameters
    ----------
    case_ids : list of case ID strings.
    re_dict  : Optional {pa_val: Re_used} mapping.  If not supplied, the
               module's cached map (populated via set_main_folder or a prior
               load_re_map call) is used.
    """
    # Merge supplied re_dict into cache if provided
    effective_re = dict(_cached_re_map)
    if re_dict:
        effective_re.update(re_dict)

    result = {}
    for cid in case_ids:
        parsed = _parse_case_id(cid)
        if parsed is None:
            result[cid] = cid
            continue
        re_val = effective_re.get(parsed['pa_val'])
        result[cid] = get_label(cid,
                                re_val=re_val,
                                context_cases=case_ids)
    return result


# ---------------------------------------------------------------------------
# Self-test (run as __main__)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    # Try to locate project root relative to this file's position
    _here   = os.path.dirname(os.path.abspath(__file__))
    _root   = os.path.normpath(os.path.join(_here, '..', '..'))
    set_main_folder(_root)

    print("Re map loaded:", _cached_re_map)
    print()

    ALL_CASES = [
        'Cyl1210PaFree', 'Cyl1210PaPla',
        'Cyl125PaFree',  'Cyl125PaPla',
        'Free5PaFree',   'Free5PaPla',
        'Free10PaFree',  'Free10PaPla',
    ]

    print("=== All 8 cases — full context (all shown together) ===")
    labels = get_labels_dict(ALL_CASES)
    for cid, lbl in labels.items():
        print(f"  {cid:20s} -> {lbl}")

    print()
    print("=== Cylinder-only context ===")
    cyl_cases = [c for c in ALL_CASES if c.startswith('Cyl')]
    labels_cyl = get_labels_dict(cyl_cases)
    for cid, lbl in labels_cyl.items():
        print(f"  {cid:20s} -> {lbl}")

    print()
    print("=== Free-Jet-only context ===")
    free_cases = [c for c in ALL_CASES if c.startswith('Free')]
    labels_free = get_labels_dict(free_cases)
    for cid, lbl in labels_free.items():
        print(f"  {cid:20s} -> {lbl}")

    print()
    print("=== Thermal (Pla) only context ===")
    pla_cases = [c for c in ALL_CASES if c.endswith('Pla')]
    labels_pla = get_labels_dict(pla_cases)
    for cid, lbl in labels_pla.items():
        print(f"  {cid:20s} -> {lbl}")

    print()
    print("=== Single-case labels (no context) ===")
    for cid in ALL_CASES:
        print(f"  {cid:20s} -> {get_label(cid)}")

    sys.exit(0)
