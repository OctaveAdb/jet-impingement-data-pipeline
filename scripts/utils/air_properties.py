# =============================================================================
# AIR_PROPERTIES
# =============================================================================
# Purpose:
#   Shared utility providing air thermophysical property calculations
#   (density, dynamic viscosity, thermal conductivity, Prandtl number)
#   using Sutherland's law. All constants are loaded from config.
#
# Inputs:  config/config_fluid_properties.csv
# Outputs: Functions and module-level constants imported by processors.
# Usage:   import sys, os; sys.path.insert(0, os.path.join(..., 'utils'));
#          from air_properties import calc_air_properties
# =============================================================================
import os
import pandas as pd

def _load_fluid_cfg():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'config', 'config_fluid_properties.xlsx')
    df = pd.read_excel(cfg_path, index_col='parameter')
    # The 'value' column may be named 'value' or be the first numeric column
    return df['value'] if 'value' in df.columns else df.iloc[:, 0]

try:
    _CFG       = _load_fluid_cfg()
    R_AIR      = float(_CFG['R_air'])
    MU_REF     = float(_CFG['mu_ref'])
    T_SUT      = float(_CFG['sutherland_T_ref'])
    S_SUT      = float(_CFG['sutherland_S'])
    CP_AIR     = float(_CFG['cp_air'])
    K_BASE     = float(_CFG['k_air_base'])
    K_SLOPE    = float(_CFG['k_air_slope'])
    T_FALLBACK = float(_CFG['default_temp_fallback'])
    P_FALLBACK = float(_CFG['default_patm_fallback'])
except Exception:
    # Fallback hardcoded values if config is missing
    R_AIR      = 287.05
    MU_REF     = 1.716e-5
    T_SUT      = 273.15
    S_SUT      = 110.4
    CP_AIR     = 1005.0
    K_BASE     = 0.0242
    K_SLOPE    = 0.00007
    T_FALLBACK = 20.0
    P_FALLBACK = 101325.0


def calc_air_properties(temp_c, p_atm):
    """Returns (rho, mu, k, Pr) for dry air at given temperature (°C) and pressure (Pa)."""
    T = temp_c + 273.15
    rho = p_atm / (R_AIR * T)
    mu  = MU_REF * (T / T_SUT)**1.5 * (T_SUT + S_SUT) / (T + S_SUT)
    k   = K_BASE + K_SLOPE * temp_c
    Pr  = (mu * CP_AIR) / k
    return rho, mu, k, Pr
