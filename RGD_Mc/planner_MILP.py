import math
import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from utils import *
from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from Data_read import *
from Params import PARAMETERS

class Planner_MILP():
    """
    MILP capacity firming formulation: binary variables to avoid simultaneous charge and discharge.
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar RG_forecast: RG point forecasts (kW)
    :ivar load_forecast: load forecast (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array, load_forecast:np.array, PV_trajectory:np.array, load_trajectory:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours'] # 1/4 hour
        self.nb_periods = int(24 / self.period_hours) # 96
        self.t_set = range(self.nb_periods)
        
        self.PV_lb = data.PV_lb
        self.PV_ub = data.PV_ub
        self.PV_trajectory = PV_trajectory
        self.load_trajectory = load_trajectory
        self.PV_forecast = PV_forecast
        self.load_forecast = load_forecast

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
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. Create model
        model = gp.Model("planner_MILP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create vairables
        # 2.1 Create First-stage variables -> x
        x = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x') # comtemporary output of DG
        x_pos = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_pos') # reserve capacity of DG
        x_neg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_neg')
        x_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_b') # on/off of ESS charge
        x_chg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_chg') # charge power
        x_dis = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_dis') # discharge power
        x_S = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_S') # ESS SOC
        x_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_PV') # RE output
        x_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_cut') # pre-dispatch curtailment of RE
        x_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='x_load') # load demand

        y_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b")
        y_pos = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_pos")
        y_neg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_neg")
        y_chg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_chg")
        y_dis = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_dis")
        y_S = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_S")
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")
        y_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cut")
        y_add = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_add")
        y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")

        # # -------------------------------------------------------------------------------------------------------------
        x_cost_fuel = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel")
        x_cost_res = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_res")
        x_cost_ESS = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_ESS")
        x_cost_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_cut")
        y_cost_fuel = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_fuel")      
        y_cost_ESS = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_ESS")
        y_cost_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cut")
        y_cost_add = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_add")

        # -------------------------------------------------------------------------------------------------------------
        # 3. Create objective
        obj = gp.quicksum(x_cost_fuel[i] + x_cost_res[i] + x_cost_cut[i] + x_cost_ESS[i]
                          + y_cost_fuel[i] + y_cost_ESS[i] + y_cost_cut[i] + y_cost_add[i] for i in self.t_set)
        model.setObjective(obj, GRB.MINIMIZE)


        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints
        # 4.1 Fisrt stage constraints
        model.addConstrs((- x[i] <= - self.u_DG[i] * self.DG_min for i in self.t_set), name='c_x_min')
        model.addConstrs((x[i] <= self.u_DG[i] * self.DG_max for i in self.t_set), name='c_x_max')
        model.addConstrs(((x[i] + x_pos[i]) - (x[i-1] - x_neg[i-1]) <= self.u_DG[i-1] * self.DG_ramp_up + (1 - self.u_DG[i-1]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP1')
        model.addConstrs((-(x[i] + x_pos[i]) + (x[i-1] - x_neg[i-1]) <= self.u_DG[i] * self.DG_ramp_down + (1 - self.u_DG[i]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP2')
        model.addConstrs(((x[i] - x_neg[i]) - (x[i-1] + x_pos[i-1]) <= self.u_DG[i-1] * self.DG_ramp_up + (1 - self.u_DG[i-1]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP3')
        model.addConstrs((-(x[i] - x_neg[i]) + (x[i-1] + x_pos[i-1]) <= self.u_DG[i] * self.DG_ramp_down + (1 - self.u_DG[i]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP4')
        model.addConstrs((- x[i] + x_neg[i] <= - self.u_DG[i] * self.DG_min for i in self.t_set), name='c_xr_neg_min')
        model.addConstrs((x[i] - x_neg[i] <= self.u_DG[i] * self.DG_max for i in self.t_set), name='c_xr_neg_max')
        model.addConstrs((- x[i] - x_pos[i] <= - self.u_DG[i] * self.DG_min for i in self.t_set), name='c_xr_pos_min' )
        model.addConstrs((x[i] + x_pos[i] <= self.u_DG[i] * self.DG_max for i in self.t_set), name='c_xr_pos_max')
        model.addConstrs((x_neg[i] <= self.u_DG[i] * self.DG_reserve_down for i in self.t_set), name='c_r_neg_min')
        model.addConstrs((x_pos[i] <= self.u_DG[i] * self.DG_reserve_up for i in self.t_set), name='c_r_pos_min')

        model.addConstrs((x_chg[i] <= x_b[i] * self.charge_power for i in self.t_set), name='c_chgarge_max') # LP
        model.addConstrs((x_dis[i] <= (1 - x_b[i]) * self.discharge_power for i in self.t_set), name='c_discharge_max') # LP
        model.addConstrs((- x_S[i] <= - self.soc_min for i in self.t_set), name='c_SOC_min')
        model.addConstrs((x_S[i] <= self.soc_max for i in self.t_set), name='c_SOC_max')
        model.addConstr((x_S[0] - (x_chg[0] * self.charge_eff - x_dis[0] / self.discharge_eff) * self.period_hours == self.soc_ini), name='c_SOC_first')
        model.addConstrs((x_S[i] - x_S[i - 1] - ((self.charge_eff * x_chg[i]) - (x_dis[i] / self.discharge_eff)) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_SOC_dynamic')
        model.addConstr((x_S[self.nb_periods - 1] == self.soc_end), name='c_SOC_last')

        model.addConstrs((x[i] - x_chg[i] + x_dis[i] + x_PV[i] - x_cut[i] - x_load[i] == 0 for i in self.t_set), name='c_power_balance')

        model.addConstrs((x_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_output')
        model.addConstrs((x_cut[i] <= self.PV_lb[i] for i in self.t_set), name='c_x_cut')
        model.addConstrs((x_load[i] == self.load_forecast[i] for i in self.t_set), name='c_x_load_demand')

        model.addConstrs((x_cost_res[i] == self.cost_DG_pos * x_pos[i] + self.cost_DG_neg * x_neg[i] for i in self.t_set), name='c_cost_fuel_res')
        model.addConstrs((x_cost_ESS[i] == self.cost_ESS_OM_pre * (x_chg[i] + x_dis[i]) for i in self.t_set), name='c_cost_OM_ESS')

        for i in self.t_set:
            model.addGenConstrPWL(x[i], x_cost_fuel[i], PWL(self.seg_num, self.DG_min, self.DG_max, FC)[0],
                                  PWL(self.seg_num, self.DG_min, self.DG_max, FC)[1])
            model.addGenConstrPWL(x_cut[i], x_cost_cut[i], PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[0],
                                  PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[1])
            
        # 4.2 Second stage constraints
        model.addConstrs((y_pos[i] <= x_pos[i] for i in self.t_set), name='c_reserve_pos_DG')
        model.addConstrs((y_neg[i] <= x_neg[i] for i in self.t_set), name='c_reserve_neg_DG')
        model.addConstrs((y_dis[i] <= y_b[i] * self.charge_power for i in self.t_set), name='c_discharge_re')
        model.addConstrs((y_chg[i] <= (1 - y_b[i]) * self.discharge_power for i in self.t_set), name='c_charge_re')
        model.addConstrs((- y_S[i] <= - self.soc_min_re for i in self.t_set), name='c_min_S')
        model.addConstrs((y_S[i] <= self.soc_max_re for i in self.t_set), name='c_max_S')
        model.addConstr((y_S[0] - (y_chg[0] * self.charge_eff - y_dis[0] / self.discharge_eff) * self.period_hours == self.soc_ini), name='c_ESS_first_period')
        model.addConstrs((y_S[i] - y_S[i - 1] - (self.charge_eff * x_chg[i] - x_dis[i] / self.discharge_eff) * self.period_hours - (self.charge_eff * y_chg[i] - y_dis[i] / self.discharge_eff) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_ESS_re-dispatch')
        model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_ESS_last_period')
        model.addConstrs((y_PV[i] == self.PV_trajectory[i] for i in self.t_set), name='c_PV_re-dispatch')
        model.addConstrs((y_load[i] == self.load_trajectory[i] for i in self.t_set), name='c_load_re-dispatch')
        model.addConstrs((y_cut[i] <= self.PV_trajectory[i] - x_cut[i] for i in self.t_set), name='c_y_cut')
        model.addConstrs((y_add[i] <= x_cut[i] for i in self.t_set), name='c_y_add')

        model.addConstrs((y_cost_fuel[i] == self.cost_DG_pos_re * y_pos[i] + self.cost_DG_neg_re * y_neg[i] for i in self.t_set), name='c_cost_reg_DG')
        model.addConstrs((y_cost_ESS[i] == self.cost_ESS_OM_re * (y_chg[i] + y_dis[i]) for i in self.t_set), name='c_cost_reg_ES')
        model.addConstrs((y_cost_cut[i] == self.cost_PV_cut_re * y_cut[i] for i in self.t_set), name='c_cost_PV_cut_re')
        model.addConstrs((y_cost_add[i] == self.cost_PV_add_re * y_add[i] for i in self.t_set), name='c_cost_PV_add_re')

        # 4.2.2 power balance equation
        model.addConstrs((x[i] + y_pos[i] - y_neg[i] - x_chg[i] + x_dis[i] - y_chg[i] + y_dis[i] + y_PV[i] - x_cut[i] - y_cut[i] + y_add[i] - y_load[i] == 0 for i in self.t_set))

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['x'] = x
        self.allvar['x_pos'] = x_pos
        self.allvar['x_neg'] = x_neg
        self.allvar['x_b'] = x_b
        self.allvar['x_chg'] = x_chg
        self.allvar['x_dis'] = x_dis
        self.allvar['x_S'] = x_S
        self.allvar['x_PV'] = x_PV
        self.allvar['x_cut'] = x_cut
        self.allvar['x_load'] = x_load
        self.allvar['x_cost_fuel'] = x_cost_fuel
        self.allvar['x_cost_res'] = x_cost_res
        self.allvar['x_cost_ESS'] = x_cost_ESS
        self.allvar['x_cost_cut'] = x_cost_cut

        self.allvar['y_pos'] = y_pos
        self.allvar['y_neg'] = y_neg
        self.allvar['y_b'] = y_b
        self.allvar['y_chg'] = y_chg
        self.allvar['y_dis'] = y_dis
        self.allvar['y_S'] = y_S
        self.allvar['y_PV'] = y_PV
        self.allvar['y_cut'] = y_cut
        self.allvar['y_add'] = y_add
        self.allvar['y_load'] = y_load
        self.allvar['y_cost_fuel'] = y_cost_fuel
        self.allvar['y_cost_ESS'] = y_cost_ESS
        self.allvar['y_cost_cut'] = y_cost_cut
        self.allvar['y_cost_add'] = y_cost_add

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):

        t_solve = time.time()

        self.model.setParam('LogToConsole', LogToConsole) # no log in the console if set to False
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        # self.model.setParam('DualReductions', 0) # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.

        # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

        self.model.setParam('LogFile', logfile) # no log in file if set to ""
        self.model.setParam('Threads', Threads) # Default value = 0 -> use all threads

        self.model.optimize()

        if self.model.status == 2 or self.model.status == 9:
            pass
        else:
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")

        if self.model.status == gp.GRB.Status.UNBOUNDED:
            self.model.computeIIS()
            self.model.write("unbounded_model_ilp")

        self.solver_status = self.model.status
        self.time_solving_model = time.time() - t_solve

        # self.model.computeIIS()
        # self.model.write("infeasible_model.ilp")

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            solution['obj'] = m.objVal

            # 1 dimensional variables
            for var in ['x', 'x_pos', 'x_neg', 'x_b', 'x_chg', 'x_dis', 'x_S', 'x_PV', 'x_cut', 'x_load', 'x_cost_fuel', 'x_cost_ESS', 'x_cost_cut',
                        'y_pos', 'y_neg', 'y_b', 'y_chg', 'y_dis', 'y_PV', 'y_cut', 'y_add', 'y_load', 'y_cost_fuel', 'y_cost_ESS', 'y_cost_cut', 'y_cost_add']:
                solution[var] = [self.allvar[var][t].X for t in self.t_set]
        else:
            print('WARNING planner MILP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            solution['obj'] = math.nan

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)
        # self.model.write("%s.mps" % filename)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/RGD_Mc/export_MILP/'

    
    PV_forecast = data.PV_pred
    load_forecast = data.load_pred
    PV_trajectory = read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/RGD_Mc/export_worst/PV_Sandia/', name='2018-07-04_PV_worst_case')
    load_trajectory = read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/RGD_Mc/export_worst/PV_Sandia/', name='2018-07-04_load_worst_case')
    PV_lb = data.PV_lb
    PV_ub = data.PV_ub
    load_lb = data.load_lb
    load_ub = data.load_ub

    day = '2018-07-04'

    # Plot point forecasts vs observations
    FONTSIZE = 20
    plt.style.use(['science'])

    plt.figure(figsize=(16,9))
    plt.plot(PV_trajectory, label='forecast')
    plt.plot(PV_lb, linestyle='--', color='darkgrey')
    plt.plot(PV_ub, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig(dirname + day + '_PV_comparison' + '.png')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(load_trajectory, label='forecast')
    plt.plot(load_lb, linestyle='--', color='darkgrey')
    plt.plot(load_ub, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig(dirname + day + '_load_comparison' + '.png')
    plt.close('all')

    # MILP planner with forecasts
    planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_trajectory=PV_trajectory, load_trajectory=load_trajectory)
    planner.export_model(dirname + 'MILP')
    planner.solve()
    solution = planner.store_solution()

    dump_file(dir=dirname, name='sol_MILP_power', file=solution['x'])
    dump_file(dir=dirname, name='sol_MILP_reserve_pos', file=solution['x_pos'])
    dump_file(dir=dirname, name='sol_MILP_reserve_neg', file=solution['x_neg'])
    dump_file(dir=dirname, name='sol_MILP_charge', file=solution['x_chg'])
    dump_file(dir=dirname, name='sol_MILP_discharge', file=solution['x_dis'])
    dump_file(dir=dirname, name='sol_MILP_SOC', file=solution['x_S'])
    dump_file(dir=dirname, name='sol_MILP_curtailment', file=solution['x_cut'])

    print('objective point forecasts %.2f' % (solution['obj']))

    plt.figure(figsize=(16,9))
    plt.plot(solution['x'], label='DG output')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('MILP formulation')
    plt.savefig(dirname + day + '_DG_units_output_' + '.png')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(solution['x_chg'], label='x_chg')
    plt.plot(solution['x_dis'], label='x_dis')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('pre-ESS schedule')
    plt.savefig(dirname + day + '_pre-ESS_schedule_' + '.png')
    plt.close('all')    

    plt.figure(figsize=(16,9))
    plt.plot(solution['x_pos'], label='x_pos')
    plt.plot(solution['x_neg'], label='x_neg')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('pre-reserve schedule')
    plt.savefig(dirname + day + '_pre-reserve_schedule_' + '.png')
    plt.close('all')    

    plt.figure(figsize=(16,9))
    plt.plot(solution['y_pos'], label='y_pos')
    plt.plot(solution['y_neg'], label='y_neg')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('re-reserve schedule')
    plt.savefig(dirname + day + '_re-reserve_schedule_' + '.png')
    plt.close('all')    


    plt.figure(figsize=(16,9))
    plt.plot(solution['y_chg'], label='y_chg')
    plt.plot(solution['y_dis'], label='y_dis')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('re-ESS schedule')
    plt.savefig(dirname + day + '_re-ESS_schedule_' + '.png')
    plt.close('all')    

    plt.figure(figsize=(16,9))
    plt.plot(solution['x_cut'], label='PV curt predict')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig(dirname + day + 'x_curt_' + '.png')
    plt.close('all')
    
    plt.figure(figsize=(16,9))
    plt.plot(solution['x'], linewidth=2, label='DG output')
    plt.plot(solution['x_load'], color='darkorange', linewidth=2, label='load')
    plt.plot(solution['x_S'], color='green', linewidth=2, label='SOC')
    plt.plot(solution['x_PV'], color='royalblue', linewidth=2, label='PV generation')
    plt.plot(solution['x_cut'], color='dimgrey', linestyle='-.', linewidth=2, label='PV curtailment')
    plt.plot(([hs - eg for hs, eg in zip(solution['x_PV'], solution['x_cut'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    plt.legend()
    plt.savefig(dirname + day + 'MILP_day ahead' + '.png')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(solution['y_cut'], label='PV curtailment real time')
    plt.plot(solution['y_add'], label='PV addition real time')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.savefig(dirname + day + 'y_curt_' + '.png')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(([hs + egg - eg for hs, egg, eg in zip(solution['x'], solution['y_pos'], solution['y_neg'])]), linewidth=2, label='DG output')
    plt.plot(solution['y_load'], color='darkorange', linewidth=2, label='load')
    # plt.plot(solution['y_S'], color='royalblue', linewidth=2, label='PV generation')
    plt.plot(([hs - egg + eg for hs, egg, eg in zip(solution['y_PV'], solution['y_cut'], solution['y_add'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    plt.legend()
    plt.savefig(dirname + day + 'MILP_real time' + '.png')
    plt.close('all')