# =============================================================================
# UNCERTAINTY  (shared RSS error-propagation helper)
# =============================================================================
# Purpose:
#   Small, dependency-light helper that propagates the known experimental input
#   uncertainties through to the derived quantities (Re, V_e, f, St, Nu, eta)
#   using root-sum-of-squares (RSS, first-order Gaussian) propagation. Imported
#   by thermal_processor (Nu / eta / Re uncertainty columns) and
#   frequency_processor (frequency-resolution and Strouhal uncertainty columns).
#
#   All functions return RELATIVE uncertainties (fractions) unless the name ends
#   in `_abs`. Multiply a relative uncertainty by the value to get the absolute
#   1-sigma uncertainty; multiply by 100 for a percentage.
#
# Default input uncertainties (1-sigma) — EXAMPLE rig-metrology values. Replace
# them with your own instrument specifications via config_uncertainty.xlsx (see
# "Calibration & configuration" in the README):
#   - tunnel differential pressure dP : +/- 1 Pa
#   - heater supply voltage V         : +/- 0.1 V
#   - heater supply current I         : +/- 0.1 A
#   - thermocouples (each)            : +/- 0.1 degC
#   - barometric pressure P_atm       : +/- 0.5 hPa (= 50 Pa)
#   - Cobra probe mean-velocity floor : +/- 0.5 m/s (per-mean, does NOT average
#                                       down; pushed through the affine slope by
#                                       u_cobra_mean_abs)
#   - lengths D1/W/B/D2                : +/- 1 mm each
#   - Cobra QC yield threshold        : >= 80 % (points below flagged as unreliable)
#
# Calibration-aware additions: the affine Cobra->pitot and channel->pitot
# calibrations are propagated with
#   u(V_cal) = sqrt[(a*u_meas)^2 + SE^2]   (apply_calibration_u)
# and the channel velocity carries the area-ratio term (rel_area_ratio,
# rel_velocity_channel). The affine constants (slopes/intercepts) come from
# config_calibration.xlsx (load_calibration); the fit standard errors from
# config_uncertainty.xlsx. With NO calibration file present the code defaults to
# the identity map (a=1, b=0) with zero fit residual, i.e. raw = calibrated.
#
# Worked example of rel_heat_flux at a nominal operating point V = 12 V, I = 5 A:
#   rel_heat_flux = sqrt((0.1/12)^2 + (0.1/5)^2) ~ 2.1 %
#
# Usage:
#   from uncertainty import (rel_velocity_from_dp, rel_reynolds_from_dp,
#                            rel_heat_flux, rel_nusselt, rel_eta,
#                            freq_resolution_hz, rel_strouhal,
#                            U_DP_PA, U_VOLT_V, U_CURR_A, U_TC_C,
#                            U_PATM_PA, QC_YIELD_MIN)
# =============================================================================

import math

# =============================================================================
# SINGLE SOURCE OF TRUTH: config/config_uncertainty.xlsx
# =============================================================================
# ALL uncertainty knobs live in config_uncertainty.xlsx (parameter/value/unit/
# description). Edit that ONE file to iterate. This module loads it once at import
# into the constants below; every processor imports the constants from here rather
# than reading the config itself. The hard-coded values below are only fallbacks
# used if the file or pandas is unavailable.
_UNC_DEFAULTS = dict(
    u_dP_Pa=1.0, u_pitot_mbar=0.1, u_TC_C=0.1, u_volt_V=0.1, u_curr_A=0.1,
    # cobra_fit_SE / channel_fit_SE default to 0.0 -> NO calibration applied
    # (identity map, no fit residual). Supply your own fitted values via
    # config_uncertainty.xlsx once you have run your velocity calibration.
    u_patm_Pa=50.0, cobra_u_floor_ms=0.5, cobra_fit_SE=0.0, channel_fit_SE=0.0,
    u_D1_mm=2.0, u_W_mm=0.5, u_B_mm=0.1, u_D2_mm=0.1,
    u_pos_x_mm=1.0, u_pos_y_mm=0.5, u_pos_tc_mm=1.0,
)


def load_uncertainty_config(config_dir=None):
    """Read every uncertainty value from config/config_uncertainty.xlsx into a
    dict (lazy pandas import keeps this module light for callers that don't need
    it). Falls back to _UNC_DEFAULTS for any missing key or if the file is
    unavailable, so the pipeline never hard-fails on a config issue."""
    out = dict(_UNC_DEFAULTS)
    try:
        import os
        import pandas as pd
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', 'config')
        s = pd.read_excel(os.path.join(config_dir,
                                       'config_uncertainty.xlsx')).set_index('parameter')['value']
        for k in out:
            if k in s.index:
                out[k] = float(s[k])
    except Exception:
        pass
    return out


_UNC = load_uncertainty_config()

# --- 1-sigma input uncertainties (from config_uncertainty.xlsx) --------------
U_DP_PA   = _UNC['u_dP_Pa']      # tunnel differential pressure, Pa
U_VOLT_V  = _UNC['u_volt_V']     # heater supply voltage, V
U_CURR_A  = _UNC['u_curr_A']     # heater supply current, A
U_TC_C    = _UNC['u_TC_C']       # single thermocouple, degC
U_PATM_PA = _UNC['u_patm_Pa']    # barometric pressure (0.5 hPa = 50 Pa)
U_PITOT_MBAR = _UNC['u_pitot_mbar']

# Per-dimension length 1-sigma uncertainties (mm)
U_D1_MM = _UNC['u_D1_mm']        # tunnel diameter (= 1% of 200 mm)
U_W_MM  = _UNC['u_W_mm']         # nozzle spanwise width
U_B_MM  = _UNC['u_B_mm']         # slot height B
U_D2_MM = _UNC['u_D2_mm']        # cylinder diameter D
U_LEN_MM = U_B_MM                # legacy alias (kept for back-compat)

# Cobra probe POSITION 1-sigma (mm): translates into a velocity uncertainty via
# the local spatial gradient,  u_V_pos = sqrt[(dV/dx·uX)² + (dV/dy·uY)²].
U_POS_X_MM = _UNC['u_pos_x_mm']  # streamwise (X)
U_POS_Y_MM = _UNC['u_pos_y_mm']  # transverse (Y)
# Thermocouple plate POSITION 1-sigma (mm): X error bar on the Delta_T / Nu_loc
# vs Y/B plate profiles. Independent of the Cobra traverse position above.
U_POS_TC_MM = _UNC['u_pos_tc_mm']

# Cobra per-mean velocity floor (does NOT average down; autocorrelated flow).
U_COBRA_FLOOR_MS = _UNC['cobra_u_floor_ms']   # m/s, raw reading
# Affine-calibration fit standard errors (STEYX, Type-A)
COBRA_FIT_SE   = _UNC['cobra_fit_SE']    # m/s, Cobra->pitot
CHANNEL_FIT_SE = _UNC['channel_fit_SE']  # m/s, channel->pitot

# Cobra QC yield minimum acceptance threshold (PROTOCOL §11)
QC_YIELD_MIN = 80.0  # per cent — measurement points below this are unreliable


def _safe_rel(num, den):
    """Return |num/den|, or NaN if den is zero / non-finite."""
    try:
        if den == 0 or not math.isfinite(den):
            return float('nan')
        return abs(num / den)
    except Exception:
        return float('nan')


def rel_velocity_from_dp(dp_pa, u_dp_pa=U_DP_PA):
    """Relative uncertainty of a Bernoulli velocity V ~ sqrt(dP).

    Since V proportional to dP^(1/2), u_V/V = 0.5 * u_dP/dP.
    """
    return 0.5 * _safe_rel(u_dp_pa, dp_pa)


def rel_reynolds_from_dp(dp_pa, u_dp_pa=U_DP_PA):
    """Relative uncertainty of Re (~ V) driven by the dP measurement.

    Re proportional to V at fixed geometry / fluid properties, so to first order
    u_Re/Re ~ u_V/V. Temperature / pressure effects on rho and mu are second
    order for the +/-1 Pa, +/-0.1 degC tolerances here and are neglected.
    """
    return rel_velocity_from_dp(dp_pa, u_dp_pa)


def rel_heat_flux(volt, curr, u_volt=U_VOLT_V, u_curr=U_CURR_A):
    """Relative uncertainty of electrical heat flux q'' = V*I/A.

    Heater area A is treated as exact, so u_q/q = sqrt((u_V/V)^2 + (u_I/I)^2).
    """
    rv = _safe_rel(u_volt, volt)
    ri = _safe_rel(u_curr, curr)
    return math.sqrt(rv * rv + ri * ri)


def rel_delta_t(delta_t_c, u_tc_c=U_TC_C):
    """Relative uncertainty of a temperature difference dT = T_plate - T_amb.

    Two independent thermocouples, so u_dT = sqrt(2) * u_TC.
    """
    u_dt = math.sqrt(2.0) * u_tc_c
    return _safe_rel(u_dt, delta_t_c)


def rel_nusselt(volt, curr, delta_t_c,
                u_volt=U_VOLT_V, u_curr=U_CURR_A, u_tc_c=U_TC_C):
    """Relative uncertainty of Nu = (q''/dT)*(Dh/k).

    Dh and k carry negligible uncertainty relative to q'' and dT, so
    u_Nu/Nu = sqrt((u_q/q)^2 + (u_dT/dT)^2).
    """
    rq = rel_heat_flux(volt, curr, u_volt, u_curr)
    rt = rel_delta_t(delta_t_c, u_tc_c)
    return math.sqrt(rq * rq + rt * rt)


def rel_eta(rel_nu_case, rel_nu_baseline):
    """Relative uncertainty of a ratio eta = Nu / Nu_baseline.

    u_eta/eta = sqrt((u_Nu/Nu)^2 + (u_Nu_base/Nu_base)^2).
    """
    a = rel_nu_case if rel_nu_case == rel_nu_case else 0.0          # NaN-safe
    b = rel_nu_baseline if rel_nu_baseline == rel_nu_baseline else 0.0
    return math.sqrt(a * a + b * b)


def freq_resolution_hz(fs_hz, nperseg):
    """Absolute frequency uncertainty = Welch bin width = fs / nperseg [Hz]."""
    try:
        if nperseg and nperseg > 0:
            return float(fs_hz) / float(nperseg)
    except Exception:
        pass
    return float('nan')


def rel_strouhal(f_hz, dp_pa, fs_hz, nperseg,
                 u_dp_pa=U_DP_PA):
    """Relative uncertainty of St = f*L/V (L exact, V from dP).

    u_St/St = sqrt((u_f/f)^2 + (u_V/V)^2).
    """
    u_f = freq_resolution_hz(fs_hz, nperseg)
    rf = _safe_rel(u_f, f_hz)
    rv = rel_velocity_from_dp(dp_pa, u_dp_pa)
    return math.sqrt(rf * rf + rv * rv)


# =============================================================================
# CALIBRATION-AWARE PROPAGATION (Task 3 — see ../../docs/UNCERTAINTY_SPEC.md)
# =============================================================================

def rel_area_ratio(d1_mm, w_mm, b_mm, u_d1=None, u_w=None, u_b=None):
    """Relative uncertainty of the area ratio R = A_tun / A_noz, with
    A_tun = pi*(D1/2)^2 and A_noz = W*B:

        uR/R = sqrt[ (2*uD1/D1)^2 + (uW/W)^2 + (uB/B)^2 ]

    Per-dimension 1-sigma lengths default to the config_uncertainty.xlsx values
    (U_D1_MM, U_W_MM, U_B_MM). ~2.05 % for the rig (D1 term dominant)."""
    u_d1 = U_D1_MM if u_d1 is None else u_d1
    u_w  = U_W_MM if u_w is None else u_w
    u_b  = U_B_MM if u_b is None else u_b
    t = (2.0 * _safe_rel(u_d1, d1_mm)) ** 2
    t += _safe_rel(u_w, w_mm) ** 2
    t += _safe_rel(u_b, b_mm) ** 2
    return math.sqrt(t)


def rel_velocity_channel(dp_pa, rel_area, u_dp_pa=U_DP_PA):
    """Relative uncertainty of the channel-derived velocity BEFORE the affine
    calibration:  uV/V = sqrt[ (1/2 * u_dP/dP)^2 + (uR/R)^2 ]  (Bernoulli sqrt
    term + area ratio; C_profile is treated as a deterministic model)."""
    rv = rel_velocity_from_dp(dp_pa, u_dp_pa)
    return math.sqrt(rv * rv + rel_area * rel_area)


def apply_calibration_u(u_meas_abs, slope_a, fit_se):
    """1-sigma uncertainty after applying an affine calibration
    V_cal = a*V_meas + b:

        u(V_cal) = sqrt[ (a*u_meas)^2 + SE^2 ]

    `u_meas_abs` is the ABSOLUTE measurement uncertainty in input units; `fit_se`
    is the fit standard error (STEYX) in output units and is Type-A, so the input
    measurement spec is NOT added again here (no double-counting)."""
    return math.sqrt((slope_a * u_meas_abs) ** 2 + fit_se * fit_se)


def u_cobra_mean_abs(slope_a, fit_se, u_floor_ms=U_COBRA_FLOOR_MS):
    """Absolute 1-sigma uncertainty of the CALIBRATED Cobra mean speed: the
    per-mean probe floor (does not average down) pushed through the slope, plus
    the fit SE in quadrature -> sqrt[(a*u_floor)^2 + SE^2]."""
    return apply_calibration_u(u_floor_ms, slope_a, fit_se)


def rel_strouhal_from_relV(f_hz, rel_v, fs_hz, nperseg):
    """Relative uncertainty of St = f*L/U using a directly supplied velocity
    relative uncertainty `rel_v` (e.g. the calibrated V_gap uncertainty read from
    Flow_Data), instead of the bare dP term:

        u_St/St = sqrt[ (u_f/f)^2 + rel_v^2 ]."""
    u_f = freq_resolution_hz(fs_hz, nperseg)
    rf = _safe_rel(u_f, f_hz)
    return math.sqrt(rf * rf + rel_v * rel_v)


def load_calibration(config_dir=None):
    """Load the affine calibration VALUES from config_calibration.xlsx (slopes,
    intercepts). The fit standard errors and the Cobra floor are uncertainty
    quantities and now live in config_uncertainty.xlsx, so they are taken from the
    module constants (COBRA_FIT_SE, CHANNEL_FIT_SE, U_COBRA_FLOOR_MS). Identity
    fallback if the file or pandas is unavailable."""
    ident = dict(cobra_a=1.0, cobra_b=0.0, cobra_SE=COBRA_FIT_SE,
                 cobra_u_floor=U_COBRA_FLOOR_MS,
                 channel_a=1.0, channel_b=0.0, channel_SE=CHANNEL_FIT_SE)
    try:
        import os
        import pandas as pd
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..', 'config')
        s = pd.read_excel(os.path.join(config_dir,
                                       'config_calibration.xlsx')).set_index('parameter')['value']
        return dict(
            cobra_a=float(s['cobra_slope_a']), cobra_b=float(s['cobra_intercept_b']),
            cobra_SE=COBRA_FIT_SE, cobra_u_floor=U_COBRA_FLOOR_MS,
            channel_a=float(s['channel_slope']), channel_b=float(s['channel_intercept']),
            channel_SE=CHANNEL_FIT_SE)
    except Exception:
        return ident
