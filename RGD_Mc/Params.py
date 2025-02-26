import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# 1. Configuration of the NEHLA case study

PERIOD_min = 15 # time resolution of the planner
PERIOD_hour = PERIOD_min / 60  # (hours)
PWL_NUM = 10
U_DG = np.ones(96)

# ------------------------------------------------------------------------------------------------------------------
# 2. DG PARAMETERS

DG_MIN = 0 # (kW)
DG_MAX = 750 # (kW)
DG_RAMP_UP = 100 # (kWh)
DG_RAMP_DOWN = 100 # (kWh)
DG_RESERVE_UP = 750 # (kWh)
DG_RESERVE_DOWN = 750 # (kWh)
DG_P_RATE = 80 # (kW)

# ------------------------------------------------------------------------------------------------------------------
# 3. ESS PARAMETERS

BATTERY_CAPACITY = 567 # (kWh)
BATTERY_POWER = 250 # (kW)
SOC_INI = 187.5 # (kWh)
SOC_END = SOC_INI # (kWh)

SOC_MAX = 396.9
SOC_MIN = 170.1
SOC_MAX_RE = 453.6 # (kWh)
SOC_MIN_RE = 113.4 # (kWh)

CHARGE_EFFICIENCY = 0.93 #
DISCHARGE_EFFICIENCY = 0.93 #
CHARGING_POWER = BATTERY_CAPACITY # (kW)
DISCHARGING_POWER = BATTERY_CAPACITY # (kW)

# ------------------------------------------------------------------------------------------------------------------
# 4. RE PARAMETERS

PV_MIN = 0 # (kW)
PV_MAX = 600 # (kW)
PV_RAMP_UP = 150 # (kWh)
PV_RAMP_DOWN = 150 # (kWh)
PV_CAPACITY = 600 # (kW)

# ------------------------------------------------------------------------------------------------------------------
# 4. LOAD PARAMETERS

LOAD_RAMP_UP = 280
LOAD_RAMP_DOWN = 280

# ------------------------------------------------------------------------------------------------------------------
# 5. COST PARAMETERS

C_DG_A = 0.001
C_DG_B = 0.015
C_DG_C = 0.059
C_DG_POS = 0.2
C_DG_NEG = 0.15
C_ESS_OM_PRE = 0.3
C_PV_CUT_PRE = 0.001
C_DG_POS_RE = 0.2
C_DG_NEG_RE = 0.1
C_ESS_OM_RE = 0.03
C_PV_CUT_RE = 0.04
C_PV_ADD_RE = 0.01

# ------------------------------------------------------------------------------------------------------------------


DG_params = {"DG_min": DG_MIN,
             "DG_max": DG_MAX,
             "DG_ramp_up": DG_RAMP_UP,
             "DG_ramp_down": DG_RAMP_DOWN,
             "DG_reserve_up": DG_RESERVE_UP,
             "DG_reserve_down": DG_RESERVE_DOWN,
             "DG_p_rate": DG_P_RATE}

ESS_params = {"capacity": BATTERY_CAPACITY,
             "soc_min": SOC_MIN,
             "soc_max": SOC_MAX,
             "soc_min_re": SOC_MIN_RE,
             "soc_max_re": SOC_MAX_RE,
             "soc_ini": SOC_INI,
             "soc_end": SOC_END,
             "charge_eff": CHARGE_EFFICIENCY,
             "discharge_eff": DISCHARGE_EFFICIENCY,
             "charge_power": BATTERY_POWER,
             "discharge_power": BATTERY_POWER}

RE_params = {"PV_min": PV_MIN,
             "PV_max": PV_MAX,
             "PV_capacity": PV_CAPACITY}

cost_params = {"DG_a": C_DG_A,
               "DG_b": C_DG_B,
               "DG_c": C_DG_C,
               "C_DG_pos": C_DG_POS,
               "C_DG_neg": C_DG_NEG,
               "C_ESS_OM": C_ESS_OM_PRE,
               "C_PV_cut_pre": C_PV_CUT_PRE,
               "C_DG_pos_re": C_DG_POS_RE,
               "C_DG_neg_re": C_DG_NEG_RE,
               "C_ESS_OM_re": C_ESS_OM_RE,
               "C_PV_cut_re": C_PV_CUT_RE,
               "C_PV_add_re": C_PV_ADD_RE}

load_params = {"ramp_up": LOAD_RAMP_UP,
               "ramp_down": LOAD_RAMP_DOWN}


PARAMETERS = {}
PARAMETERS["period_hours"] = PERIOD_hour
PARAMETERS['RE'] = RE_params
PARAMETERS['cost'] = cost_params
PARAMETERS['DG'] = DG_params
PARAMETERS['ESS'] = ESS_params
PARAMETERS['load'] = load_params
PARAMETERS['PWL'] = PWL_NUM
PARAMETERS['u_DG'] = U_DG
