#!/usr/bin/env python
# =============================================================================
# PEC_PROCESSOR  —  Performance Evaluation Criterion (pumping-penalty) pipeline
# =============================================================================
# Computes the thermal-hydraulic PEC of the cylinder vortex generator at the
# CONSTANT-PUMPING-POWER condition (Webb):
#
#     PEC(Re) = (Nu/Nu0) / (f/f0)^(1/3)
#
#   Nu/Nu0  = eta_theo  = measured cylinder Nu  /  Hofmann correlation Nu0,
#             both at the cylinder's own Re (already Re-matched; the correlation
#             is treated as exact, so u(Nu/Nu0) = u_Nu of the measurement).
#   f       = with-cylinder loss  (Delta_p_loss), read DIRECTLY at the cylinder
#             Nu set-points — the pumping runs share the cylinder nozzle
#             (W = 150 mm), so set-point -> Re is identical to the thermal Re.
#   f0      = no-cylinder loss (Delta_p_loss0), measured on the FREE nozzle
#             (W = 170 mm); because the 150/170 area ratio shifts the free Re at
#             a given Delta_p, f0 is INTERPOLATED onto the cylinder Re.
#
#   f/f0 is the loss-pressure ratio AT MATCHED Re. At matched Re_noz the nozzle
#   dynamic head (1/2 rho V_noz^2) is common to both cases and cancels, so the
#   Delta_p_loss ratio IS the nozzle-referenced loss-coefficient ratio — the
#   physically correct friction-factor ratio for PEC. (We deliberately do NOT
#   ratio the channel-referenced K = Delta_p_loss/Delta_p_dyn across geometries:
#   at matched Re the channel dynamic heads differ by (170/150)^2 and would
#   inject a spurious geometric factor.)
#
# Uncertainty (calibration framework, UNCERTAINTY_SPEC.md):
#   u(Delta_p_tot) = pitot 0.1 mBar (Type-B)  ⊕  half-difference of the 2 runs
#                    (Type-A, ascending vs descending sweep)
#   Delta_p_loss = Delta_p_tot - Delta_p_dyn   (u_dP on the dynamic term)
#   u(f/f0)/(f/f0) = sqrt[(u_f/f)^2 + (u_f0/f0)^2]
#   u((f/f0)^1/3)/(f/f0)^1/3 = (1/3) u(f/f0)/(f/f0)
#   u_PEC/PEC = sqrt[(u_Nu/Nu)^2 + (1/3 u_{f/f0}/(f/f0))^2]
#   Re carries the channel->pitot calibration uncertainty (x error bar).
#
# Set-point -> velocity/Re uses the SAME channel->pitot calibration as the rest
# of the pipeline (Re_Ve_processor.calculate_physics), per geometry.
#
# Inputs:
#   pumping_penalty/deltaP and loss factor.xlsx   (2 runs per set-point)
#   thermal/results/Thermal_Global_Summary.csv    (Nu, Nu0=Hofmann, eta_theo, u_Nu)
#   config/config_geometry.xlsx, config_uncertainty.xlsx, config_calibration.xlsx
#
# Outputs:
#   pumping_penalty/deltaP and loss factor.xlsx    (+ 'Loss_curves', 'PEC_vs_Re')
#   pumping_penalty/PEC_Summary.csv
#   pumping_penalty/PEC_vs_Re.{png,pdf}
#
# NOTE: the pumping run's air state (T, P_atm) was not logged; Re uses the
# pipeline fallback (20 degC, 101325 Pa) — it reproduces the thermal Re to
# ~0.2 %, and any residual scale cancels in the matched-Re ratio.
# =============================================================================

import os
import sys
import math
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # processors/
sys.path.insert(0, os.path.join(_HERE, '..', 'utils'))      # utils/

from Re_Ve_processor import (ReVeCalculatorApp as _RV,
                             compute_flow_uncertainties as _u_flow)

try:
    from uncertainty import U_PITOT_MBAR as _U_PITOT_MBAR, U_DP_PA as _U_DP_PA
except Exception:
    _U_PITOT_MBAR, _U_DP_PA = 0.1, 0.5

# Project paths (Data/Data root is three levels up from scripts/processors/).
_DATA = os.path.normpath(os.path.join(_HERE, '..', '..'))
_PUMP_XLSX = os.path.join(_DATA, 'pumping_penalty', 'deltaP and loss factor.xlsx')
_THERMAL   = os.path.join(_DATA, 'thermal', 'results', 'Thermal_Global_Summary.csv')
_GEOM      = os.path.join(_DATA, 'config', 'config_geometry.xlsx')

_T_C, _P_ATM = 20.0, 101325.0     # pumping-run air state fallback (see header note)


def _geom():
    """Return (D1, B, W_free, W_cyl, D2) in mm from config_geometry.xlsx."""
    s = pd.read_excel(_GEOM).set_index('parameter')['value']
    return (float(s['tunnel_diameter_D1']), float(s['nozzle_height_B']),
            float(s['nozzle_width_free']), float(s['nozzle_width_cyl']),
            float(s['cylinder_diameter_D2']))


def _re_noz(dp_pa, w_mm, d1, b, d2):
    """(re_noz, v_noz, u_re_noz) at channel set-point dp_pa for nozzle width w_mm,
    via the shared channel->pitot calibration (identical to the rest of the
    pipeline). u_re_noz is the SAME relative Re uncertainty used for the Nu points;
    it is a horizontal (x) error bar only — it does not enter the vertical bars."""
    r = _RV.calculate_physics(None, float(dp_pa), _T_C, _P_ATM, d1, w_mm, b, d2)
    v_tun, v_noz, re_tun, re_noz, re_cyl = r[0], r[1], r[2], r[3], r[4]
    u = _u_flow(dp_pa, v_tun, v_noz, r[9], re_tun, re_noz, re_cyl, d1, w_mm, b, d2)
    return re_noz, v_noz, u['u_Re_noz']


def _load_pumping():
    """Parse the 2-runs-per-set-point pumping table. Returns a DataFrame indexed
    by integer set-point with the mean and Type-A (half-range) of the with- and
    no-cylinder loss pressures.

    Loss pressures are computed from the RAW measured columns (set-point Delta_p_dyn
    [Pa], and the two pitot totals in mBar) — NOT from the workbook's formula
    columns, whose cached values do not survive an openpyxl re-save:
        Delta_p_loss = Delta_p_tot[mBar]*100 - Delta_p_dyn[Pa]."""
    raw = pd.read_excel(_PUMP_XLSX, sheet_name='Tabelle1', header=None)
    data = raw.iloc[2:].copy()                      # rows 2.. are data (16 up + 16 down)
    sp    = pd.to_numeric(data.iloc[:, 0], errors='coerce')    # Delta_p_dyn [Pa] (set-point)
    tot0  = pd.to_numeric(data.iloc[:, 1], errors='coerce')    # Delta_p_tot0 [mBar] (no cyl)
    tot   = pd.to_numeric(data.iloc[:, 4], errors='coerce')    # Delta_p_tot  [mBar] (cyl)
    loss0 = tot0 * 100.0 - sp                                  # Delta_p_loss0 [Pa]
    loss  = tot  * 100.0 - sp                                  # Delta_p_loss  [Pa]
    df = pd.DataFrame({'setpoint': sp, 'loss0': loss0, 'loss': loss}).dropna()
    df['setpoint'] = df['setpoint'].round().astype(int)
    out = []
    for s, g in df.groupby('setpoint'):
        l0, l = g['loss0'].values, g['loss'].values
        out.append({
            'setpoint': s,
            'loss_cyl': l.mean(),   'uA_cyl':  0.5 * (l.max() - l.min()),
            'loss0':    l0.mean(),  'uA_free': 0.5 * (l0.max() - l0.min()),
            'n': len(g),
        })
    return pd.DataFrame(out).sort_values('setpoint').reset_index(drop=True)


def _u_loss(value_independent_uA):
    """Type-B (pitot 0.1 mBar -> Pa, ⊕ dynamic-term u_dP) combined with Type-A."""
    uB = math.hypot(_U_PITOT_MBAR * 100.0, _U_DP_PA)    # mBar->Pa; ⊕ u(Delta_p_dyn)
    return math.hypot(uB, value_independent_uA)


def run():
    d1, b, w_free, w_cyl, d2 = _geom()
    pump = _load_pumping()

    # --- Set-point -> Re (+ horizontal u_Re) per geometry, and the vertical
    #     Type-B⊕Type-A loss uncertainties -------------------------------------
    _rc = [_re_noz(s, w_cyl,  d1, b, d2) for s in pump['setpoint']]
    _rf = [_re_noz(s, w_free, d1, b, d2) for s in pump['setpoint']]
    pump['Re_cyl'],  pump['u_Re_cyl']  = [v[0] for v in _rc], [v[2] for v in _rc]
    pump['Re_free'], pump['u_Re_free'] = [v[0] for v in _rf], [v[2] for v in _rf]
    pump['u_loss_cyl'] = [_u_loss(u) for u in pump['uA_cyl']]
    pump['u_loss0']    = [_u_loss(u) for u in pump['uA_free']]

    # --- Thermal cylinder operating points (Nu/Nu0 = eta_theo, Hofmann) --------
    th = pd.read_csv(_THERMAL)
    cyl = th[th['Has_Cylinder'].astype(str).str.lower().eq('yes')].copy()
    cyl['setpoint'] = cyl['Base_Pressure'].str.replace('Pa', '', regex=False).astype(int)
    cyl = cyl.sort_values('Re_used').reset_index(drop=True)

    # Free-loss interpolation table (monotonic in Re_free).
    rf = pump.sort_values('Re_free')
    Re_free, loss0, u_loss0 = rf['Re_free'].values, rf['loss0'].values, rf['u_loss0'].values

    rows = []
    for _, c in cyl.iterrows():
        sp = int(c['setpoint'])
        Re_i = float(c['Re_used'])
        pc = pump[pump['setpoint'] == sp]
        if pc.empty:
            continue
        f  = float(pc['loss_cyl'].iloc[0]);  u_f = float(pc['u_loss_cyl'].iloc[0])   # direct match
        f0 = float(np.interp(Re_i, Re_free, loss0))                                  # interpolated
        u_f0 = float(np.interp(Re_i, Re_free, u_loss0))
        extrap = not (Re_free.min() <= Re_i <= Re_free.max())

        ff0  = f / f0
        rel_ff0 = math.hypot(u_f / f, u_f0 / f0)
        cube = ff0 ** (1.0 / 3.0)
        rel_cube = rel_ff0 / 3.0

        eta = float(c['eta_theo']); rel_eta = float(c['u_Nu_pct']) / 100.0
        pec = eta / cube
        rel_pec = math.hypot(rel_eta, rel_cube)

        rows.append({
            'Case_ID': c['Case_ID'], 'Setpoint_Pa': sp, 'Re': Re_i,
            'u_Re': float(c['u_Re_abs']),
            'Nu': float(c['Global_Nu_Exp']), 'Nu0_Hofmann': float(c['Global_Nu_Theo']),
            'Nu_over_Nu0': eta, 'u_Nu_over_Nu0': eta * rel_eta,
            'f_loss_cyl_Pa': f, 'u_f': u_f,
            'f0_loss_free_Pa': f0, 'u_f0': u_f0, 'f0_extrapolated': extrap,
            'f_over_f0': ff0, 'u_f_over_f0': ff0 * rel_ff0,
            'ff0_pow_1_3': cube, 'u_ff0_pow_1_3': cube * rel_cube,
            'PEC': pec, 'u_PEC': pec * rel_pec, 'u_PEC_pct': 100.0 * rel_pec,
        })
    pec_df = pd.DataFrame(rows)

    curves = pump[['setpoint', 'Re_cyl', 'u_Re_cyl', 'loss_cyl', 'u_loss_cyl',
                   'Re_free', 'u_Re_free', 'loss0', 'u_loss0', 'n']].rename(columns={
        'setpoint': 'Setpoint_Pa', 'loss_cyl': 'f_loss_cyl_Pa',
        'u_loss_cyl': 'u_f', 'loss0': 'f0_loss_free_Pa', 'u_loss0': 'u_f0',
        'n': 'n_runs'})

    _write_outputs(curves, pec_df)
    return curves, pec_df


def _rebuild_tabelle1():
    """Return Tabelle1 with every derived column filled as a VALUE (recomputed
    from the raw measured columns). The original workbook stored these as Excel
    formulas, whose cached results are lost on an openpyxl save; storing values
    makes the sheet self-consistent and the pipeline idempotent."""
    g = pd.read_excel(_PUMP_XLSX, sheet_name='Tabelle1', header=None)
    for i in range(2, len(g)):
        dyn = g.iat[i, 0]
        if not isinstance(dyn, (int, float)) or pd.isna(dyn):
            continue
        loss0 = g.iat[i, 1] * 100.0 - dyn          # Delta_p_loss0 [Pa]
        loss  = g.iat[i, 4] * 100.0 - dyn          # Delta_p_loss  [Pa]
        g.iat[i, 2] = loss0
        g.iat[i, 3] = loss0 / dyn                  # K0
        g.iat[i, 5] = loss
        g.iat[i, 6] = loss / dyn                   # K
        g.iat[i, 7] = loss / loss0                 # f/f0
        g.iat[i, 8] = (loss / loss0) ** (1.0 / 3.0)
    return g


def _write_outputs(curves, pec_df):
    outdir = os.path.join(_DATA, 'pumping_penalty')
    pec_df.to_csv(os.path.join(outdir, 'PEC_Summary.csv'), index=False)

    # Rewrite the workbook FRESH (mode='w') with Tabelle1 restored to values plus
    # the two computed sheets. A fresh write avoids the openpyxl append that strips
    # the formula caches, and keeps the file consistent on every re-run.
    tab = _rebuild_tabelle1()
    target = _PUMP_XLSX
    try:
        with pd.ExcelWriter(target, engine='openpyxl') as xw:
            tab.to_excel(xw, sheet_name='Tabelle1', index=False, header=False)
            curves.to_excel(xw, sheet_name='Loss_curves', index=False)
            pec_df.to_excel(xw, sheet_name='PEC_vs_Re', index=False)
        print(f"  -> wrote {os.path.basename(target)} (Tabelle1 restored + Loss_curves, PEC_vs_Re)")
    except Exception as e:
        alt = os.path.join(outdir, 'deltaP and loss factor_PEC.xlsx')
        with pd.ExcelWriter(alt, engine='openpyxl') as xw:
            tab.to_excel(xw, sheet_name='Tabelle1', index=False, header=False)
            curves.to_excel(xw, sheet_name='Loss_curves', index=False)
            pec_df.to_excel(xw, sheet_name='PEC_vs_Re', index=False)
        print(f"  -> ORIGINAL LOCKED ({e}); wrote {os.path.basename(alt)} instead")

    _plot(curves, pec_df, outdir)


def _save(fig, stem, outdir):
    """Save fig to pumping_penalty/ (PNG+PDF). The usage.py gallery scans this
    folder directly and groups the PEC figures under their own category
    "Pumping Penalty / PEC", so no copy into summary/ is needed."""
    fig.savefig(os.path.join(outdir, f'{stem}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(outdir, f'{stem}.pdf'), bbox_inches='tight')


def _plot(curves, pec_df, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 9, 'figure.dpi': 300})
    # Horizontal Re bars are a TRANSLATION (common calibration systematic); they do
    # NOT enter the vertical bars (which stay as the loss / (f/f0)^1/3 / PEC 1-sigma).

    # --- Figure 1: loss pressure f, f0 vs Re ---------------------------------
    f1, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.errorbar(curves['Re_cyl'], curves['f_loss_cyl_Pa'],
                xerr=curves['u_Re_cyl'], yerr=curves['u_f'],
                fmt='-o', color='#005b96', capsize=2, elinewidth=0.7, ms=4,
                label=r'$f$: with cylinder ($W{=}150$)')
    ax.errorbar(curves['Re_free'], curves['f0_loss_free_Pa'],
                xerr=curves['u_Re_free'], yerr=curves['u_f0'],
                fmt='--s', color='#e34a33', capsize=2, elinewidth=0.7, ms=4,
                label=r'$f_0$: no cylinder ($W{=}170$)')
    ax.set_xlabel(r'$Re_{noz}$ [-]'); ax.set_ylabel(r'$\Delta p_{loss}$ [Pa]')
    ax.set_title('Loss pressure vs Reynolds'); ax.grid(True, ls='--', alpha=0.4)
    ax.legend(loc='upper left')
    _save(f1, 'PEC_loss_vs_Re', outdir); plt.close(f1)

    # --- Figure 2: (f/f0)^(1/3) vs Re ----------------------------------------
    f2, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.errorbar(pec_df['Re'], pec_df['ff0_pow_1_3'],
                xerr=pec_df['u_Re'], yerr=pec_df['u_ff0_pow_1_3'],
                fmt='-o', color='#6a0572', capsize=3, elinewidth=0.8, ms=7)
    ax.set_xlabel(r'$Re_{noz}$ [-]'); ax.set_ylabel(r'$(f/f_0)^{1/3}$ [-]')
    ax.set_title(r'Pumping factor $(f/f_0)^{1/3}$ vs Reynolds')
    ax.grid(True, ls='--', alpha=0.4)
    _save(f2, 'PEC_ff0cube_vs_Re', outdir); plt.close(f2)

    # --- Figure 3: PEC vs Re --------------------------------------------------
    f3, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.errorbar(pec_df['Re'], pec_df['PEC'],
                xerr=pec_df['u_Re'], yerr=pec_df['u_PEC'],
                fmt='-o', color='#2a8c9e', capsize=3, elinewidth=0.8, ms=7)
    ax.axhline(1.0, color='k', lw=1, ls=':', alpha=0.7)
    ax.set_xlabel(r'$Re_{noz}$ [-]'); ax.set_ylabel('PEC [-]')
    ax.set_title(r'PEC $=(Nu/Nu_0)/(f/f_0)^{1/3}$ vs Reynolds')
    ax.grid(True, ls='--', alpha=0.4)
    _save(f3, 'PEC_PEC_vs_Re', outdir); plt.close(f3)

    print('  -> wrote PEC_loss_vs_Re, PEC_ff0cube_vs_Re, PEC_PEC_vs_Re (.png/.pdf '
          'in pumping_penalty/)')


if __name__ == '__main__':
    print('--- PEC processor ---')
    c, p = run()
    pd.set_option('display.width', 200, 'display.max_columns', None)
    print('\nPEC vs Re:')
    print(p[['Setpoint_Pa', 'Re', 'Nu_over_Nu0', 'f_over_f0', 'ff0_pow_1_3',
             'PEC', 'u_PEC_pct']].round(3).to_string(index=False))
