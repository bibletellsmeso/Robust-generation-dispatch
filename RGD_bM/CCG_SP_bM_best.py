import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt

from utils import read_file
from SP_primal_LP import *
from Params import PARAMETERS
from root_project import ROOT_DIR
from Data_read import *

class CCG_SP_best():
    """
    CCGD = Column and Constraint Gneration Dual
    SP = Sub Problem of the CCG dual cutting plane algorithm.
    SP = Max-min problem that is reformulated as a single max problem by taking the dual.
    The resulting maximization problem is bilinear and is linearized using McCormick not big-M method.
    The final reformulated SP is a MILP due to binary variables related to the uncertainty set.
    
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)

    :ivar model: a Gurobi model (-)

    """

    def __init__(self, PV_forecast:np.array, load_forecast:np.array, power:np.array, reserve_pos:np.array, reserve_neg:np.array,
                 charge:np.array, discharge:np.array, SOC:np.array, curtailment:np.array, GAMMA:float=0, PI:float=0, M:float=0):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours'] # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)
        self.seg_num = 10

        self.PV_forecast = PV_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        self.PV_lb = data.PV_lb
        self.PV_ub = data.PV_ub
        self.load_lb = data.load_lb
        self.load_ub = data.load_ub
        self.x = power # (kW) The power of diesel generator
        self.x_pos = reserve_pos # (kw) The reserve rate of diesel generator
        self.x_neg = reserve_neg
        self.x_chg = charge
        self.x_dis = discharge
        self.x_S = SOC
        self.x_PV_cut = curtailment # (kW) The curtailment of
        self.PV_pos = data.PV_pos # (kW) The maximal deviation betwwen the min and forecast PV uncertainty set bounds
        self.PV_neg = data.PV_neg # (kW) The maximal deviation between the max and forecast PV uncertainty set bounds
        self.load_pos = data.load_pos # (kw) The maximal deviation between the min and forecast load uncertainty set bounds
        self.load_neg = data.load_neg # (kW) The maximal deviation between the max and forecast load uncertainty set bounds
        self.GAMMA = GAMMA # uncertainty budget <= self.nb_periods, gamma = 0: no uncertainty
        self.PI = PI
        self.M = M

        # Parameters required for the MP in the CCG algorithm
        self.u_DG = PARAMETERS['u_DG'] # on/off
        self.DG_min = PARAMETERS['DG']['DG_min'] # (kW)
        self.DG_max = PARAMETERS['DG']['DG_max'] # (kW)
        self.DG_ramp_up = PARAMETERS['DG']['DG_ramp_up'] # (kW)
        self.DG_ramp_down = PARAMETERS['DG']['DG_ramp_down'] # (kW)
        self.DG_reserve_up = PARAMETERS['DG']['DG_reserve_up']
        self.DG_reserve_down = PARAMETERS['DG']['DG_reserve_down']
        self.DG_p_rate = PARAMETERS['DG']['DG_p_rate']

        # ESS parameters
        self.ESS_capacity = PARAMETERS['ESS']['capacity']  # (kWh)
        self.soc_ini = PARAMETERS['ESS']['soc_ini']  # (kWh)
        self.soc_end = PARAMETERS['ESS']['soc_end']  # (kWh)
        self.soc_min = PARAMETERS['ESS']['soc_min']  # (kWh)
        self.soc_max = PARAMETERS['ESS']['soc_max']  # (kWh)
        self.soc_min_re = PARAMETERS['ESS']['soc_min_re']  # (kWh)
        self.soc_max_re = PARAMETERS['ESS']['soc_max_re']  # (kWh)
        self.charge_eff = PARAMETERS['ESS']['charge_eff']  # (/)
        self.discharge_eff = PARAMETERS['ESS']['discharge_eff']  # (/)
        self.charge_power = PARAMETERS['ESS']['charge_power']  # (kW)
        self.discharge_power = PARAMETERS['ESS']['discharge_power']  # (kW)

        # RE parameters
        self.PV_min = PARAMETERS['RE']['PV_min']
        self.PV_max = PARAMETERS['RE']['PV_max']
        self.PV_ramp_up = PARAMETERS['RE']['PV_ramp_up']
        self.PV_ramp_down = PARAMETERS['RE']['PV_ramp_down']

        # load parameters
        self.load_ramp_up = PARAMETERS['load']['ramp_up']
        self.load_ramp_down = PARAMETERS['load']['ramp_down']

        # Cost parameters
        self.cost_DG_a = PARAMETERS['cost']['DG_a']
        self.cost_DG_b = PARAMETERS['cost']['DG_b']
        self.cost_DG_c = PARAMETERS['cost']['DG_c']
        self.cost_DG_pos = PARAMETERS['cost']['C_DG_pos']
        self.cost_DG_neg = PARAMETERS['cost']['C_DG_neg']
        self.cost_DG_pos_re = PARAMETERS['cost']['C_DG_pos_re']
        self.cost_DG_neg_re = PARAMETERS['cost']['C_DG_neg_re']
        self.cost_ESS_OM_pre = PARAMETERS['cost']['C_ESS_OM']
        self.cost_ESS_OM_re = PARAMETERS['cost']['C_ESS_OM_re']
        self.cost_PV_cut_pre = PARAMETERS['cost']['C_PV_cut_pre']
        self.cost_PV_cut_re = PARAMETERS['cost']['C_PV_cut_re']
        self.cost_PV_add_re = PARAMETERS['cost']['C_PV_add_re']

        # Piecewise linearlization parameters
        self.seg_num = PARAMETERS['PWL']

        self.time_building_model = None
        self.time_solving_model = None

        # Create model
        self.model = self.create_model()

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. create model
        model = gp.Model("SP_dual_MILP")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create dual variables -> phi

        # 2.1 Continuous variables
        # primal constraints <= b -> dual variables <= 0, primal constraints = b -> dual varialbes are free, (primal constraints >= b -> dual variables >= 0)
        phi_pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_pos")
        phi_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_neg")
        phi_chg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_chg")
        phi_dis = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_dis")
        phi_ini = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_ini") # free of dual variable 
        phi_S = model.addVars(self.nb_periods - 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_S") # num: 95, free dual of ESS dynamics (=)
        phi_end = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_end") # free of dual variable
        phi_Smin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smin")
        phi_Smax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smax")
        phi_PV = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_PV") # free of dual variable
        phi_load = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_load") # free of dual variable
        phi_cut = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_cut")
        phi = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi") # free dual of power balance
        phi_add = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_add")

        # 2.2 Continuous variables related to the uncertainty set
        epsilon_pos = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.BINARY, obj=0, name="epsilon_pos")
        epsilon_neg = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.BINARY, obj=0, name="epsilon_neg")
        delta_pos = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.BINARY, obj=0, name="delta_pos")
        delta_neg = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.BINARY, obj=0, name="delta_neg")

        # 2.3 Continuous variables use for the linearization of the bilinear terms
        alpha_pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_pos')
        alpha_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_neg')
        beta_pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_pos')
        beta_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_neg')       
        gamma_pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_pos')
        gamma_neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_neg')

        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        obj_exp = 0
        obj_exp += phi_ini * self.soc_ini + phi_end * self.soc_end
        for i in range(self.nb_periods - 1):
            obj_exp += phi_S[i] * (self.charge_eff * self.x_chg[i+1] - self.x_dis[i+1] / self.discharge_eff) * self.period_hours
        for i in self.t_set:
            obj_exp += phi_pos[i] * self.x_pos[i] + phi_neg[i] * self.x_neg[i]
            obj_exp += phi_chg[i] * self.charge_power + phi_dis[i] * self.discharge_power 
            obj_exp += - phi_Smin[i] * self.soc_min_re + phi_Smax[i] * self.soc_max_re
            obj_exp += phi_PV[i] * self.PV_forecast[i] + alpha_pos[i] * self.PV_pos[i] - alpha_neg[i] * self.PV_neg[i]
            obj_exp += phi_cut[i] * self.PV_forecast[i] + beta_pos[i] * self.PV_pos[i] - beta_neg[i] * self.PV_neg[i] - phi_cut[i] * self.x_PV_cut[i] + phi_add[i] * self.x_PV_cut[i]
            obj_exp += phi_load[i] * self.load_forecast[i] + gamma_pos[i] * self.load_pos[i] - gamma_neg[i] * self.load_neg[i]
            obj_exp += - phi[i] * (self.x[i] - self.x_chg[i] + self.x_dis[i] - self.x_PV_cut[i])
        # a constant
            obj_exp += PWL_val(self.seg_num, self.DG_min, self.DG_max, FC, self.x[i])
            obj_exp += PWL_val(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV, self.x_PV_cut[i])
            obj_exp += self.cost_DG_pos * self.x_pos[i] + self.cost_DG_neg * self.x_neg[i]
            obj_exp += self.cost_ESS_OM_pre * (self.x_chg[i] + self.x_dis[i])
 
        model.setObjective(obj_exp, GRB.MAXIMIZE)
        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints
        # primal variables >= 0 -> dual constraints <= c, primal variables are free -> dual constraints = c, (primal variables <= 0 -> dual constraints >= c)
        # Constraints related to DG
        model.addConstrs((phi_pos[i] + phi[i] <= self.cost_DG_pos_re for i in self.t_set), name='c_DG_pos')
        model.addConstrs((phi_neg[i] - phi[i] <= self.cost_DG_neg_re for i in self.t_set), name='c_DG_neg')
        # Constraints related to the ESS charge and discharge
        model.addConstr((phi_chg[0] - phi_ini * self.charge_eff * self.period_hours - phi[0] <= self.cost_ESS_OM_re), name='c_ESS_chg_first')
        model.addConstrs((phi_chg[i] - phi_S[i - 1] * self.charge_eff * self.period_hours - phi[i] <= self.cost_ESS_OM_re for i in range(1, self.nb_periods)), name='c_ESS_chg')
        model.addConstr((phi_dis[0] + phi_ini / self.discharge_eff * self.period_hours + phi[0] <= self.cost_ESS_OM_re), name='c_ESS_dis_first')
        model.addConstrs((phi_dis[i] + phi_S[i - 1] / self.discharge_eff * self.period_hours + phi[i] <= self.cost_ESS_OM_re for i in range(1, self.nb_periods)), name='c_ESS_dis')
        # Constraints related to the ESS SOC
        model.addConstr((- phi_Smin[0] + phi_Smax[0] + phi_ini - phi_S[0] <= 0), name='c_S_first') # time period 1 for phi_Smin/phi_Smax and time period 2 for phi_S
        model.addConstrs((- phi_Smin[i] + phi_Smax[i] + phi_S[i - 1] - phi_S[i] <= 0 for i in range(1, self.nb_periods - 1)), name='c_S') # time period 3 to nb_periods - 1
        model.addConstr((- phi_Smin[self.nb_periods - 1] + phi_Smax[self.nb_periods - 1] + phi_end + phi_S[self.nb_periods - 2] <= 0), name='c_S_end') # last time period
        # Constraints related to PV and load
        model.addConstrs((phi_PV[i] + phi[i] <= 0 for i in self.t_set), name='c_PV')
        model.addConstrs((phi_load[i] - phi[i] <= 0 for i in self.t_set), name='c_load')
        # Constraints related to PV curtailment
        model.addConstrs((phi_cut[i] - phi[i] <= self.cost_PV_cut_re for i in self.t_set), name='c_PV_cut')
        model.addConstrs((phi_add[i] + phi[i] <= self.cost_PV_add_re for i in self.t_set), name='c_PV_add')
        # -------------------------------------------------------------------------------------------------------------

        # Constraints related to the uncertainty budget
        model.addConstr(gp.quicksum(epsilon_pos[i] + epsilon_neg[i] for i in self.t_set) <= self.GAMMA, name='c_GAMMA') # PV uncertainty budget
        model.addConstr(gp.quicksum(delta_pos[i] + delta_neg[i] for i in self.t_set) <= self.PI, name='c_PI') # load uncertainty budget

        # model.addConstrs(((self.PV_forecast[i] + self.PV_pos[i] * epsilon_pos[i]) - (self.PV_forecast[i-1] - self.PV_neg[i-1] * epsilon_neg[i-1]) <= self.PV_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_PV_ramp_1')
        # model.addConstrs((- (self.PV_forecast[i] + self.PV_pos[i] * epsilon_pos[i]) + (self.PV_forecast[i-1] - self.PV_neg[i-1] * epsilon_neg[i-1]) <= self.PV_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_PV_ramp_2')
        # model.addConstrs(((self.PV_forecast[i] - self.PV_neg[i] * epsilon_neg[i]) - (self.PV_forecast[i-1] + self.PV_pos[i-1] * epsilon_pos[i-1]) <= self.PV_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_PV_ramp_3')
        # model.addConstrs((- (self.PV_forecast[i] - self.PV_neg[i] * epsilon_neg[i]) + (self.PV_forecast[i-1] + self.PV_pos[i-1] * epsilon_pos[i-1]) <= self.PV_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_PV_ramp_4')

        # model.addConstrs(((self.load_forecast[i] + self.load_pos[i] * delta_pos[i]) - (self.load_forecast[i-1] - self.load_neg[i-1] * delta_neg[i-1]) <= self.load_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_load_ramp_1')
        # model.addConstrs((- (self.load_forecast[i] + self.load_pos[i] * delta_pos[i]) + (self.load_forecast[i-1] - self.load_neg[i-1] * delta_neg[i-1]) <= self.load_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_load_ramp_2')
        # model.addConstrs(((self.load_forecast[i] - self.load_neg[i] * delta_neg[i]) - (self.load_forecast[i-1] + self.load_pos[i-1] * delta_pos[i-1]) <= self.load_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_load_ramp_3')
        # model.addConstrs((- (self.load_forecast[i] - self.load_neg[i] * delta_neg[i]) + (self.load_forecast[i-1] + self.load_pos[i-1] * delta_pos[i-1]) <= self.load_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_load_ramp_4')

        # Constraints related to the big-M method----------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # # Constraints related uncertainty variable simultaneous
        model.addConstrs((epsilon_pos[i] + epsilon_neg[i] <= 1 for i in self.t_set), "epsilon_simultaneous")
        model.addConstrs((delta_pos[i] + delta_neg[i] <= 1 for i in self.t_set), "delta_simultaenous")

        model.addConstrs((alpha_pos[i] >= - self.M * epsilon_pos[i] for i in self.t_set), name='c_alpha_pos__1')
        model.addConstrs((alpha_pos[i] >= phi_PV[i] + self.M * epsilon_pos[i] - self.M for i in self.t_set), name='c_alpha_pos__2')
        model.addConstrs((alpha_pos[i] <= self.M * epsilon_pos[i] for i in self.t_set), name='c_alpha_pos__3')
        model.addConstrs((alpha_pos[i] <= phi_PV[i] - self.M * epsilon_pos[i] + self.M for i in self.t_set), name='c_alpha_pos__4')

        model.addConstrs((alpha_neg[i] >= - self.M * epsilon_neg[i] for i in self.t_set), name='c_alpha_neg__1')
        model.addConstrs((alpha_neg[i] >= phi_PV[i] + self.M * epsilon_neg[i] - self.M for i in self.t_set), name='c_alpha_neg__2')
        model.addConstrs((alpha_neg[i] <= self.M * epsilon_neg[i] for i in self.t_set), name='c_alpha_neg__3')
        model.addConstrs((alpha_neg[i] <= phi_PV[i] - self.M * epsilon_neg[i] + self.M for i in self.t_set), name='c_alpha_neg__4')

        # beta_pos/neg
        model.addConstrs((beta_pos[i] >= - self.M * epsilon_pos[i] for i in self.t_set), name='c_beta_pos_1')
        model.addConstrs((beta_pos[i] >= phi_cut[i]  for i in self.t_set), name='c_beta_pos_2')
        model.addConstrs((beta_pos[i] <= 0 for i in self.t_set), name='c_beta_pos_3')
        model.addConstrs((beta_pos[i] <= phi_cut[i] - self.M * epsilon_pos[i] + self.M for i in self.t_set), name='c_beta_pos_4')

        model.addConstrs((beta_neg[i] >= - self.M * epsilon_neg[i] for i in self.t_set), name='c_beta_neg_1')
        model.addConstrs((beta_neg[i] >= phi_cut[i] for i in self.t_set), name='c_beta_neg_2')
        model.addConstrs((beta_neg[i] <= 0 for i in self.t_set), name='c_beta_neg_3')
        model.addConstrs((beta_neg[i] <= phi_cut[i] - self.M * epsilon_neg[i] + self.M for i in self.t_set), name='c_beta_neg_4')

        # gamma_pos/neg
        model.addConstrs((gamma_pos[i] >= - self.M * delta_pos[i] for i in self.t_set), name='c_gamma_pos__1')
        model.addConstrs((gamma_pos[i] >= phi_load[i] + self.M * delta_pos[i] - self.M for i in self.t_set), name='c_gamma_pos__2')
        model.addConstrs((gamma_pos[i] <= self.M * delta_pos[i] for i in self.t_set), name='c_gamma_pos__3')
        model.addConstrs((gamma_pos[i] <= phi_load[i] - self.M * delta_pos[i] + self.M for i in self.t_set), name='c_gamma_pos__4')

        model.addConstrs((gamma_neg[i] >= - self.M * delta_neg[i] for i in self.t_set), name='c_gamma_neg__1')
        model.addConstrs((gamma_neg[i] >= phi_load[i] + self.M * delta_neg[i] - self.M for i in self.t_set), name='c_gamma_neg__2')
        model.addConstrs((gamma_neg[i] <= self.M * delta_neg[i] for i in self.t_set), name='c_gamma_neg__3')
        model.addConstrs((gamma_neg[i] <= phi_load[i] - self.M * delta_neg[i] + self.M for i in self.t_set), name='c_gamma_neg__4')

        # model.addConstrs((- phi_PV[i] <= self.M for i in self.t_set), name='c_phi_PV_lb')
        # model.addConstrs((phi_PV[i] <= self.M for i in self.t_set), name='c_phi_PV_MC_ub')
        # model.addConstrs((- phi_cut[i] <= self.M for i in self.t_set), name='c_phi_cut_lb')
        # model.addConstrs((phi_cut[i] <= 0 for i in self.t_set), name='c_phi_cut_MC_ub')
        # model.addConstrs((- phi_load[i] <= self.M for i in self.t_set), name='c_phi_load_lb')
        # model.addConstrs((phi_load[i] <= self.M for i in self.t_set), name='c_phi_load_MC_ub')

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['phi_pos'] = phi_pos
        self.allvar['phi_neg'] = phi_neg
        self.allvar['phi_chg'] = phi_chg
        self.allvar['phi_dis'] = phi_dis
        self.allvar['phi'] = phi
        self.allvar['phi_ini'] = phi_ini
        self.allvar['phi_S'] = phi_S
        self.allvar['phi_end'] = phi_end
        self.allvar['phi_Smin'] = phi_Smin
        self.allvar['phi_Smax'] = phi_Smax
        self.allvar['phi_PV'] = phi_PV
        self.allvar['phi_load'] = phi_load
        self.allvar['phi_cut'] = phi_cut
        self.allvar['phi_add'] = phi_add

        self.allvar['epsilon_pos'] = epsilon_pos
        self.allvar['epsilon_neg'] = epsilon_neg
        self.allvar['delta_pos'] = delta_pos
        self.allvar['delta_neg'] = delta_neg

        self.allvar['alpha_pos'] = alpha_pos
        self.allvar['alpha_neg'] = alpha_neg
        self.allvar['beta_pos'] = beta_pos
        self.allvar['beta_neg'] = beta_neg
        self.allvar['gamma_pos'] = gamma_pos
        self.allvar['gamma_neg'] = gamma_neg

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program %gs" % self.time_building_model)
        
        return model
    
    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):
        """
        :param LogToConsole: no log in the console if set to False.
        :param logfile: no log in file if set to ""
        :param Threads: Default value = 0 -> use all threads
        :param MIPFocus: If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
                        If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
                        If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.
        :param TimeLimit: in seconds.
        """

        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        self.model.setParam('LogFile', logfile)
        self.model.setParam('Threads', Threads)

        self.model.optimize()

        # if self.model.status == 2 or self.model.status == 9:
        #     pass
        # else:
        #     self.model.computeIIS()
        #     self.model.write("infeasible_model.ilp")
        #     print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')

        # if self.model.status == gp.GRB.Status.UNBOUNDED:
        #     self.model.computeIIS()
        #     self.model.write("unbounded_model_ilp")

        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        # 0 dimensional variables
        for var in ['phi_ini', 'phi_end']:
            solution[var] = self.allvar[var].X

        # 1 dimensional variables
        for var in ['phi_pos', 'phi_neg', 'phi_chg', 'phi_dis', 'phi_Smin', 'phi_Smax',
                    'phi', 'phi_PV', 'phi_load', 'phi_cut', 'phi_add',
                    'epsilon_pos', 'epsilon_neg', 'delta_pos', 'delta_neg',
                    'alpha_pos', 'alpha_neg', 'beta_pos', 'beta_neg', 'gamma_pos', 'gamma_neg']:
            solution[var] = [self.allvar[var][t].X for t in self.t_set]

        for var in ['phi_S']:
            solution[var] = [self.allvar[var][t].X for t in range(self.nb_periods - 1)]

        # 6. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution
    
    def export_model(self, filename):
        """
        Export the model into a lp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/RGD_bM_best'
    day = '2018-07-04'

    PV_forecast = data.PV_pred
    load_forecast = data.load_pred
    
    power = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_power')
    reserve_pos = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_reserve_pos')
    reserve_neg = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_reserve_neg')
    charge = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_charge')
    discharge = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_discharge')
    SOC = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_SOC')
    curtailment = read_file(dir='/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/export_MILP/', name='sol_MILP_curtailment')

    PV_lb = PV_forecast - data.PV_neg
    PV_ub = PV_forecast + data.PV_pos
    load_lb = load_forecast - data.load_neg
    load_ub = load_forecast + data.load_pos
    PV_pos = data.PV_pos
    PV_neg = data.PV_neg
    load_pos = data.load_pos
    load_neg = data.load_neg

    GAMMA = 0
    PI = 0
    M = 1

    SP_dual = CCG_SP_best(PV_forecast=PV_forecast, load_forecast=load_forecast, power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg,
                     charge=charge, discharge=discharge, SOC=SOC, curtailment=curtailment, GAMMA=GAMMA, PI=PI, M=M)

    SP_dual.export_model(dirname + 'SP_dual_MILP')
    MIPFocus = 0
    TimeLimit = 15
    logname = 'SP_dual_MILP_start_' + 'MIPFocus_' + str(MIPFocus) + '.log'
    SP_dual.solve(LogToConsole=True, logfile=dirname + logname, Threads=1, MIPFocus=MIPFocus, TimeLimit=TimeLimit)
    solution = SP_dual.store_solution()

    print('Robust objective %.2f' % (solution['obj']))
    plt.style.use(['science'])
    
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(solution['phi_PV'], label='phi_PV')
    plt.legend()
    plt.title('phi_PV')

    plt.subplot(2, 1, 2)
    plt.plot(solution['phi_load'], label='phi_load')
    plt.legend()
    plt.title('phi_load')

    plt.tight_layout()
    plt.show()

    # 2. epsilon_pos, epsilon_neg, delta_pos, delta_neg을 한 플롯에 나눠서
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(solution['epsilon_pos'], label='epsilon_pos')
    plt.plot(solution['epsilon_neg'], label='epsilon_neg')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('epsilon_pos and epsilon_neg')

    plt.subplot(2, 1, 2)
    plt.plot(solution['delta_pos'], label='delta_pos')
    plt.plot(solution['delta_neg'], label='delta_neg')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('delta_pos and delta_neg')

    plt.tight_layout()
    plt.savefig(dirname + day + '_epsilon-gamma.png', dpi=300)
    plt.show()

    # 3. alpha, beta, gamma를 한 플롯에 나눠서
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(solution['alpha_pos'], label='alpha_pos')
    plt.plot(solution['alpha_neg'], label='alpha_neg')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('alpha_pos and alpha_neg')

    plt.subplot(3, 1, 2)
    plt.plot(solution['beta_pos'], label='beta_pos')
    plt.plot(solution['beta_neg'], label='beta_neg')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('beta_pos and beta_neg')

    plt.subplot(3, 1, 3)
    plt.plot(solution['gamma_pos'], label='gamma_pos')
    plt.plot(solution['gamma_neg'], label='gamma_neg')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('gamma_pos and gamma_neg')

    plt.tight_layout()
    plt.ylabel('Power (kW)')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)

    plt.show()

    # 4. PV와 load의 bestcase를 한 플롯에 나눠서
    PV_best_case = [PV_forecast[i] + PV_pos[i] * solution['epsilon_pos'][i] - PV_neg[i] * solution['epsilon_neg'][i] for i in range(96)]
    load_best_case = [load_forecast[i] + load_pos[i] * solution['delta_pos'][i] - load_neg[i] * solution['delta_neg'][i] for i in range(96)]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(PV_best_case, marker='.', color='k', label='PV best case')
    plt.plot(PV_forecast, label='PV forecast')
    plt.plot(PV_forecast - PV_neg, ':', label='PV min')
    plt.plot(PV_forecast + PV_pos, ':', label='PV max')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('PV best case and forecast')

    plt.subplot(2, 1, 2)
    plt.plot(load_best_case, marker='.', color='k', label='Load best case')
    plt.plot(load_forecast, label='Load forecast')
    plt.plot(load_forecast - load_neg, ':', label='Load min')
    plt.plot(load_forecast + load_pos, ':', label='Load max')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('Load best case and forecast')

    plt.tight_layout()

    plt.ylabel('Power (kW)')
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.savefig(dirname + day + '_best-case.png', dpi=300)
    plt.show()
    # plt.close('all')