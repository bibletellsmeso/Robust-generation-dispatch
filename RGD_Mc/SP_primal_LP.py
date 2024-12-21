import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
import matplotlib.pyplot as plt

from utils import *
from Params import PARAMETERS
from Data_read import *

class SP_primal_LP():
    """
    SP primal of the benders decomposition using gurobi.
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar PV_trajectory: PV trajectory (kW)
    :ivar load_trajectory: load trajectory (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_trajectory:np.array, load_trajectory:np.array, power:np.array, reserve_pos:np.array, reserve_neg:np.array,
                 charge:np.array, discharge:np.array, SOC:np.array, curtailment:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)
        self.PV_forecast = data.PV_pred
        self.PV_trajectory = PV_trajectory # (kW)
        self.load_trajectory = load_trajectory # (kW)
        self.PV_lb = data.PV_lb
        self.PV_ub = data.PV_ub
        self.load_lb = data.load_lb
        self.load_ub = data.load_ub

        self.x = power # (kW) The power of diesel generator
        self.x_pos = reserve_pos # (kw) The reserve rate of diesel generator
        self.x_neg = reserve_neg
        self.x_chg = charge # (kW) The power of ESS charge
        self.x_dis = discharge
        self.x_S = SOC
        self.x_cut = curtailment # (kW) The curtailment of PV
    
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
        # 1. create model
        model = gp.Model("SP_primal_LP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create second-stage decision variables -> y
        y_pos = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_pos")
        y_neg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_neg")
        y_chg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_chg")
        y_dis = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_dis")
        y_S = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y=S")
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")
        y_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cut")
        y_add = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_add")
        y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")

        # -------------------------------------------------------------------------------------------------------------
        # 2.1 Create second-stage cost variables -> x (off-set) and y
        x_cost_ESS = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_ESS')
        x_cost_fuel_PWL = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_fuel_PWL')
        x_cost_res = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_res')
        x_cost_cut_PWL = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_cut_PWL')
        y_cost_fuel = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='y_cost_fuel')
        y_cost_ESS = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_ESS")
        y_cost_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cut")
        y_cost_add = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_add")

        # -------------------------------------------------------------------------------------------------------------
        # 3. Create objective function
        objective = gp.quicksum(x_cost_fuel_PWL[i] + x_cost_res[i] + x_cost_cut_PWL[i] + x_cost_ESS[i]
                                + y_cost_fuel[i] + y_cost_ESS[i] + y_cost_cut[i] + y_cost_add[i] for i in self.t_set)
        model.setObjective(objective, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create second-stage constraints
        model.addConstrs((x_cost_fuel_PWL[i] == PWL_val(self.seg_num, self.DG_min, self.DG_max, FC, self.x[i]) for i in self.t_set), name='c_cost_fuel_PWL')
        model.addConstrs((x_cost_res[i] == self.cost_DG_pos * self.x_pos[i] + self.cost_DG_neg * self.x_neg[i] for i in self.t_set), name='c_cost_fuel_res')
        model.addConstrs((x_cost_ESS[i] == self.cost_ESS_OM_pre * (self.x_chg[i] + self.x_dis[i]) for i in self.t_set), name='c_cost_ESS_OM_pre')
        model.addConstrs((x_cost_cut_PWL[i] == PWL_val(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV, self.x_cut[i]) for i in self.t_set), name='c_cost_PV_cut_PWL')
        model.addConstrs((y_cost_fuel[i] == self.cost_DG_pos_re * y_pos[i] + self.cost_DG_neg_re * y_neg[i] for i in self.t_set), name='c_cost_fuel_res_re')
        model.addConstrs((y_cost_ESS[i] == self.cost_ESS_OM_re * (y_dis[i] + y_chg[i]) for i in self.t_set), name='c_cost_ESS_OM_re')
        model.addConstrs((y_cost_cut[i] == self.cost_PV_cut_re * y_cut[i] for i in self.t_set), name='c_cost_PV_cut_re')
        model.addConstrs((y_cost_add[i] == self.cost_PV_add_re * y_add[i] for i in self.t_set), name='c_cost_PV_add_re')

            
        model.addConstrs((y_pos[i] <= self.x_pos[i] for i in self.t_set), name='c_reserve_pos')
        model.addConstrs((y_neg[i] <= self.x_neg[i] for i in self.t_set), name='c_reserve_neg')
        model.addConstrs((y_chg[i] <= self.charge_power for i in self.t_set), name='c_charge_re')
        model.addConstrs((y_dis[i] <= self.discharge_power for i in self.t_set), name='c_discharge_re')
        model.addConstrs((- y_S[i] <= - self.soc_min_re for i in self.t_set), name='c_SOC_min')
        model.addConstrs((y_S[i] <= self.soc_max_re for i in self.t_set), name='c_S_max')
        model.addConstr((y_S[0] - (y_chg[0] * self.charge_eff - y_dis[0] / self.discharge_eff) * self.period_hours == self.soc_ini), name='c_ESS_first_period')
        model.addConstrs((y_S[i] - y_S[i - 1] - (self.charge_eff * self.x_chg[i] - self.x_dis[i] / self.discharge_eff) * self.period_hours - (self.charge_eff * y_chg[i] -  y_dis[i] / self.discharge_eff) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_SOC_re')
        model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_ESS_last_period')
        model.addConstrs((y_PV[i] == self.PV_trajectory[i] for i in self.t_set), name='c_y_PV')
        model.addConstrs((y_load[i] == self.load_trajectory[i] for i in self.t_set), name='c_y_load')
        model.addConstrs((y_cut[i] <= self.PV_trajectory[i] - self.x_cut[i] for i in self.t_set), name='c_y_cut')
        model.addConstrs((y_add[i] <= self.x_cut[i] for i in self.t_set), name='c_y_add')
        model.addConstrs((self.x[i] + y_pos[i] - y_neg[i] - self.x_chg[i] + self.x_dis[i] - y_chg[i] + y_dis[i] + y_PV[i] - self.x_cut[i] - y_cut[i] + y_add[i] - y_load[i] == 0 for i in self.t_set), name='c_power_balance_eq')

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def solve(self, outputflag:bool=False):

        t_solve = time.time()
        self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status

        if solution['status'] == 2 or  solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            solution['obj'] = m.objVal

            varname = ['y_pos', 'y_neg', 'y_chg', 'y_dis', 'y_S', 'y_PV', 'y_cut', 'y_add', 'y_load',
                       'x_cost_fuel_PWL', 'x_cost_res', 'x_cost_ESS', 'x_cost_cut_PWL',
                       'y_cost_fuel', 'y_cost_ESS', 'y_cost_cut', 'y_cost_add']
            for key in varname:
                solution[key] = []

            sol = m.getVars()
            solution['all_var'] = sol
            for v in sol:
                for key in varname:
                    if v.VarName.split('[')[0] == key:
                        solution[key].append(v.x)
        else:
            print('WARNING planner SP primal status %s -> problem not solved, objective is set to nan' %(solution['status']))
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")
            print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')
            # solutionStatus = 3: Model was proven to be infeasible.
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['obj'] = float('nan')

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

    dirname = '/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/RGD_Mc/export_MILP/'

    PV_trajectory = np.array(pd.read_csv('worst.csv'), dtype=np.float32)[:,0]
    load_trajectory = np.array(pd.read_csv('worst.csv'), dtype=np.float32)[:,1]

    power = read_file(dir=dirname, name='sol_MILP_power')
    reserve_pos = read_file(dir=dirname, name='sol_MILP_reserve_pos')
    reserve_neg = read_file(dir=dirname, name='sol_MILP_reserve_neg')
    charge = read_file(dir=dirname, name='sol_MILP_charge')
    discharge = read_file(dir=dirname, name='sol_MILP_discharge')
    SOC = read_file(dir=dirname, name='sol_MILP_SOC')
    curtailment = read_file(dir=dirname, name='sol_MILP_curtailment')

    SP_primal = SP_primal_LP(PV_trajectory=PV_trajectory, load_trajectory=load_trajectory, power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg, 
                             charge=charge, discharge=discharge, SOC=SOC, curtailment=curtailment)
    
    SP_primal.export_model(dirname + 'SP_primal_LP')
    SP_primal.solve()
    solution = SP_primal.store_solution()

    print('objective SP primal %.2f' %(solution['obj']))
    
    # plt.style.use(['science'])
    # plt.figure()
    # plt.plot(solution['y_chg'], label='y chg')
    # plt.plot(solution['y_dis'], label='y dis')
    # plt.plot(solution['y_S'], label='y S')
    # plt.legend()
    # plt.show()

    # print(solution['all_var'])


    # Get dual values
    # for c in SP_primal.model.getConstrs():
    #     print('The dual value of %s : %g' % (c.constrName, c.pi))