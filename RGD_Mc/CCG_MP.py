import os
import time
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
from Params import PARAMETERS
from Data_read import *
from utils import *

class CCG_MP():
    """
    CCG = Column-and-Constraint Generation.
    MP = Master Problem of the CCG algorithm.
    The MP is a Linear Programming.
    :ivar nb_periods: number of market periods (-)
    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array, load_forecast:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours'] # 1/4 hour
        self.nb_periods = int(24 / self.period_hours) # 96
        self.t_set = range(self.nb_periods)
        self.PV_forecast = PV_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        self.PV_lb = data.PV_lb
        self.PV_ub = data.PV_ub
        self.load_lb = data.load_lb
        self.load_ub = data.load_ub

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
        model = gp.Model("MP")

        # -------------------------------------------------------------------------------------------------------------
        # 2.1 Create First-stage variables
        x = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x') # comtemporary output of DG
        x_pos = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_pos') # reserve capacity of DG
        x_neg = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_neg')
        x_b = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='x_b') # on/off of ESS charge
        x_chg = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_chg') # charge power
        x_dis = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_dis') # discharge power
        x_PV = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_PV') # RE output
        x_cut = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_cut') # pre-dispatch curtailment of RE
        x_load = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_load') # load demand
        x_S = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_SOC') # ESS SOC

        x_cost_fuel = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel")
        x_cost_res = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_res")
        x_cost_ESS = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_ESS")
        x_cost_cut = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_cut")

        theta = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name="theta") # objective

        obj = gp.quicksum(x_cost_fuel[i] + x_cost_res[i] + x_cost_ESS[i] + x_cost_cut[i] for i in self.t_set)
        # -------------------------------------------------------------------------------------------------------------
        # 2.2 Create objective
        model.setObjective(obj + theta, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 2.3 Create constraints
        # 2.3.1 Diesel generators
        model.addConstrs((- x[i] <= - self.u_DG[i] * self.DG_min for i in self.t_set), name='c_x_min')
        model.addConstrs((x[i] <= self.u_DG[i] * self.DG_max for i in self.t_set), name='c_x_max')
        model.addConstrs(((x[i] + x_pos[i]) - (x[i-1] - x_neg[i-1]) <= self.u_DG[i-1] * self.DG_ramp_up + (1 - self.u_DG[i-1]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP1')
        # model.addConstrs((-(x[i] + x_pos[i]) + (x[i-1] - x_neg[i-1]) <= self.u_DG[i] * self.DG_ramp_down + (1 - self.u_DG[i]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP2')
        # model.addConstrs(((x[i] - x_neg[i]) - (x[i-1] + x_pos[i-1]) <= self.u_DG[i-1] * self.DG_ramp_up + (1 - self.u_DG[i-1]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP3')
        model.addConstrs((-(x[i] - x_neg[i]) + (x[i-1] + x_pos[i-1]) <= self.u_DG[i] * self.DG_ramp_down + (1 - self.u_DG[i]) * self.DG_max for i in range(1, self.nb_periods)), name='c_DG_RAMP4')
        model.addConstrs((- x[i] + x_neg[i] <= - self.u_DG[i] * self.DG_min for i in self.t_set), name='c_x_res_min')
        model.addConstrs((x[i] + x_pos[i] <= self.u_DG[i] * self.DG_max for i in self.t_set), name='c_x_res_max')
        model.addConstrs((x_neg[i] <= self.u_DG[i] * self.DG_reserve_down for i in self.t_set), name='c_res_min')
        model.addConstrs((x_pos[i] <= self.u_DG[i] * self.DG_reserve_up for i in self.t_set), name='c_res_max')

        # 2.3.2 Energy storage system
        model.addConstrs((x_chg[i] <= x_b[i] * self.charge_power for i in self.t_set), name='c_chgarge_max') # LP
        model.addConstrs((x_dis[i] <= (1 - x_b[i]) * self.discharge_power for i in self.t_set), name='c_discharge_max') # LP
        model.addConstrs((- x_S[i] <= - self.soc_min for i in self.t_set), name='c_SOC_min')
        model.addConstrs((x_S[i] <= self.soc_max for i in self.t_set), name='c_SOC_max')
        model.addConstr((x_S[0] - (x_chg[0] * self.charge_eff - x_dis[0] / self.discharge_eff) * self.period_hours == self.soc_ini), name='c_SOC_first')
        model.addConstrs((x_S[i] - x_S[i - 1] - (self.charge_eff * x_chg[i] - x_dis[i] / self.discharge_eff) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_SOC_dynamic')
        model.addConstr((x_S[self.nb_periods - 1] == self.soc_end), name='c_SOC_last')
        # 2.3.3 Power balance
        model.addConstrs((x[i] - x_chg[i] + x_dis[i] + x_PV[i] - x_cut[i] - x_load[i] == 0 for i in self.t_set), name='c_power_balance')
        # 2.3.4 Renewable energy and load
        model.addConstrs((x_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_output')
        model.addConstrs((x_load[i] == self.load_forecast[i] for i in self.t_set), name='c_x_load_demand')
        model.addConstrs((x_cut[i] <= self.PV_lb[i] for i in self.t_set), name='c_x_cut')
        # 2.3.5 Cost
        model.addConstrs((x_cost_res[i] == self.cost_DG_pos * x_pos[i] + self.cost_DG_neg * x_neg[i] for i in self.t_set), name='c_cost_fuel_res')
        model.addConstrs((x_cost_ESS[i] == self.cost_ESS_OM_pre * (x_chg[i] + x_dis[i]) for i in self.t_set), name='c_cost_OM_ESS')
        # 2.3.5.1 Cost linearlization
        for i in self.t_set:
            model.addGenConstrPWL(x[i], x_cost_fuel[i], PWL(self.seg_num, self.DG_min, self.DG_max, FC)[0],
                                  PWL(self.seg_num, self.DG_min, self.DG_max, FC)[1])
            model.addGenConstrPWL(x_cut[i], x_cost_cut[i], PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[0],
                                  PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[1])
            
        # -------------------------------------------------------------------------------------------------------------
        # 3. Store variables
        self.allvar = dict()
        self.allvar['x'] = x
        self.allvar['x_pos'] = x_pos
        self.allvar['x_neg'] = x_neg
        self.allvar['x_chg'] = x_chg
        self.allvar['x_dis'] = x_dis
        self.allvar['x_b'] = x_b
        self.allvar['x_S'] = x_S
        self.allvar['x_PV'] = x_PV
        self.allvar['x_cut'] = x_cut
        self.allvar['x_load'] = x_load
        self.allvar['x_cost_fuel'] = x_cost_fuel
        self.allvar['x_cost_res'] = x_cost_res
        self.allvar['x_cost_ESS'] = x_cost_ESS
        self.allvar['x_cost_cut'] = x_cost_cut
        self.allvar['theta'] = theta

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def update_MP(self, PV_trajectory:np.array, load_trajectory:np.array, iteration:int):
        """
        Add the second-stage variables at CCG iteration i.
        :param MP: MP to update in the CCG algorithm.
        :param PV_trajectory: RG trajectory computed by the SP at iteration i.
        :param iteration: update at iteration i.
        :return: the model is directly updated
        """
        # -------------------------------------------------------------------------------------------------------------
        # 4.1 Second-stage variables
        # Incremental output of DG (kW)
        y_pos = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_pos_" + str(iteration))
        y_neg = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_neg_" + str(iteration))
        # Incremental output of ESS (kW)
        y_b = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.BINARY, name="y_b_" + str(iteration))
        y_chg = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_chg_" + str(iteration))
        y_dis = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_dis_" + str(iteration))
        y_S = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="SoC_" + str(iteration))
        # Real-time RG
        y_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV_" + str(iteration))
        y_cut = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cut_" + str(iteration))
        y_add = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_add" + str(iteration))
        y_load = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # Upward/Downward regulation cost of DE1 generator/ES
        y_cost_fuel = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='y_cost_fuel_' + str(iteration))
        y_cost_ESS = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_ESS_" + str(iteration))
        # RG re-dispatch curtailment cost
        y_cost_cut = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cut_" + str(iteration))
        y_cost_add = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_add_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4.2 Add the constraint related to the objective
        # objective
        objective = gp.quicksum(y_cost_fuel[i] + y_cost_ESS[i] + y_cost_cut[i] + y_cost_add[i] for i in self.t_set)
        # theta = MP.model.getVarByname() = only theta variable of the MP model
        self.model.addConstr(self.model.getVarByName('theta') >= objective, name='theta_' + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4.3 Add the constraint related to the feasbility domain of the secondstage variables -> y
        # 4.3.1 cost cst
        self.model.addConstrs((y_cost_fuel[i] == self.cost_DG_pos_re * y_pos[i] + self.cost_DG_neg_re * y_neg[i] for i in self.t_set), name='c_cost_fuel' + str(iteration))
        self.model.addConstrs((y_cost_ESS[i] == self.cost_ESS_OM_re * (y_dis[i] + y_chg[i]) for i in self.t_set), name='c_cost_ESS_res' + str(iteration))
        self.model.addConstrs((y_cost_cut[i] == self.cost_PV_cut_re * y_cut[i] for i in self.t_set), name='c_cost_PV_cut' + str(iteration))
        self.model.addConstrs((y_cost_add[i] == self.cost_PV_add_re * y_add[i] for i in self.t_set), name='c_cost_PV_add' + str(iteration))
        # 4.3.2 DE1 reserve power
        # max/min DG reserve cst: self.model.getVarByName() -> return variables of the model from name, the x_b variable are index 0 to 95

        self.model.addConstrs((y_pos[i] <= self.model.getVars()[i+96] for i in self.t_set), name='c_DG_reserve_max_' + str(iteration))
        self.model.addConstrs((y_neg[i] <= self.model.getVars()[i+192] for i in self.t_set), name='c_DG_reserve_min_' + str(iteration))
        # 4.3.3 ES reserve power
        self.model.addConstrs((y_chg[i] <= y_b[i] * self.charge_power for i in self.t_set), name='c_ESS_chg_re_' + str(iteration))
        self.model.addConstrs((y_dis[i] <= (1 - y_b[i]) * self.discharge_power for i in self.t_set), name='c_ESS_dis_re_' + str(iteration))
        self.model.addConstr((y_S[0] - (y_chg[0] * self.charge_eff - y_dis[0] / self.discharge_eff) * self.period_hours == self.soc_ini), name='c_y_SOC_first_period_' + str(iteration))
        self.model.addConstrs((y_S[i] - y_S[i - 1] - (self.charge_eff * self.model.getVars()[i+384] - self.model.getVars()[i+480] / self.discharge_eff) * self.period_hours
                               - (self.charge_eff * y_chg[i] - y_dis[i] / self.discharge_eff) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_y_S_Incremental_' + str(iteration))
        self.model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_y_SOC_last_period_' + str(iteration))
        self.model.addConstrs((- y_S[i] <= - self.soc_min_re for i in self.t_set), name='c_y_SOC_min_' + str(iteration))
        self.model.addConstrs((y_S[i] <= self.soc_max_re for i in self.t_set), name='c_y_SOC_max_' + str(iteration))
        # 4.3.6 RG generation cst
        self.model.addConstrs((y_PV[i] == PV_trajectory[i] for i in self.t_set), name='c_y_PV_generation_' + str(iteration))
        # 4.3.7 load cst
        self.model.addConstrs((y_load[i] == load_trajectory[i] for i in self.t_set), name='c_y_load_' + str(iteration))
        # 4.3.4 real-time curtailment cst
        self.model.addConstrs((y_cut[i] <= PV_trajectory[i] - self.model.getVars()[i+672] for i in self.t_set), name='c_y_PV_curtailment_' + str(iteration))
        self.model.addConstrs((y_add[i] <= self.model.getVars()[i+672] for i in self.t_set), name='c_y_add' + str(iteration))
        # 4.3.4 power balance equation
        self.model.addConstrs((self.model.getVars()[i] + y_pos[i] - y_neg[i] - self.model.getVars()[i+384] + self.model.getVars()[i+480] - y_chg[i] + y_dis[i] + y_PV[i] - self.model.getVars()[i+672] - y_cut[i] + y_add[i] - y_load[i] == 0 for i in self.t_set), name='c_real-time_power_balance_' + str(iteration))
        
        # -------------------------------------------------------------------------------------------------------------
        # 5. Store the added variables to the MP in a new dict
        self.allvar['var_' + str(iteration)] = dict()
        self.allvar['var_' + str(iteration)]['y_cost_fuel'] = y_cost_fuel
        self.allvar['var_' + str(iteration)]['y_cost_ESS'] = y_cost_ESS
        self.allvar['var_' + str(iteration)]['y_cost_cut'] = y_cost_cut
        self.allvar['var_' + str(iteration)]['y_cost_add'] = y_cost_add
        self.allvar['var_' + str(iteration)]['y_pos'] = y_pos
        self.allvar['var_' + str(iteration)]['y_neg'] = y_neg
        self.allvar['var_' + str(iteration)]['y_b'] = y_b
        self.allvar['var_' + str(iteration)]['y_chg'] = y_chg
        self.allvar['var_' + str(iteration)]['y_dis'] = y_dis
        self.allvar['var_' + str(iteration)]['y_S'] = y_S
        self.allvar['var_' + str(iteration)]['y_PV'] = y_PV
        self.allvar['var_' + str(iteration)]['y_cut'] = y_cut
        self.allvar['var_' + str(iteration)]['y_add'] = y_add
        self.allvar['var_' + str(iteration)]['y_load'] = y_load

        # -------------------------------------------------------------------------------------------------------------
        # 6. Update model to implement the modifications
        self.model.update()

    # True: output a log that is generated during optimization troubleshooting to the console
    def solve(self, LogToConsole:bool=False):
        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):
        
        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (SUbject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            # 0 dimensional variables
            solution['theta'] = self.allvar['theta'].X
            # 1D variable
            solution['x'] = [self.allvar['x'][t].X for t in self.t_set]
            solution['x_cut'] = [self.allvar['x_cut'][t].X for t in self.t_set]
            solution['x_chg'] = [self.allvar['x_chg'][t].X for t in self.t_set]
            solution['x_dis'] = [self.allvar['x_dis'][t].X for t in self.t_set]
            solution['x_S'] = [self.allvar['x_S'][t].X for t in self.t_set]            
            solution['x_pos'] = [self.allvar['x_pos'][t].X for t in self.t_set]
            solution['x_neg'] = [self.allvar['x_neg'][t].X for t in self.t_set]
            solution['x_b'] = [self.allvar['x_b'][t].X for t in self.t_set]
            solution['x_PV'] = [self.allvar['x_PV'][t].X for t in self.t_set]
            solution['x_load'] = [self.allvar['x_load'][t].X for t in self.t_set]
            solution['x_cost_fuel'] = [self.allvar['x_cost_fuel'][t].X for t in self.t_set]
            solution['x_cost_res'] = [self.allvar['x_cost_res'][t].X for t in self.t_set]
            solution['x_cost_ESS'] = [self.allvar['x_cost_ESS'][t].X for t in self.t_set]
            solution['x_cost_cut'] = [self.allvar['x_cost_cut'][t].X for t in self.t_set]
            solution['obj'] = m.objVal
        else:
            print('WARNING MP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            # solutionStatus = 3: Model was proven to be infeasible
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['status'] = float('nan')

        # Timing indicators
        solution['time_building'] = self.time_building_model
        solution['time_solving'] = self.time_solving_model
        solution['time_total'] = self.time_building_model + self.time_solving_model

        return solution
    
    def update_sol(self, MP_sol:dict, i:int):
        """
        Add the solution of the 1 dimensional variables at iteration i.
        :param MP_sol: solution of the MP model at iteration i.
        :param i: index of interation.
        :return: update directly the dict.
        """
        MP_status = MP_sol['status']
        if MP_status == 2 or MP_status == 9:
            MP_sol['var_' + str(i)] = dict()
            # add the solution of the 1 dimensional variables at iteration
            for var in ['y_pos', 'y_neg', 'y_b', 'y_chg', 'y_dis', 'y_S', 'y_PV', 'y_cut', 'y_add', 'y_load',
                        'y_cost_fuel', 'y_cost_ESS', 'y_cost_cut', 'y_cost_add']:
                MP_sol['var_' + str(i)][var] = [self.allvar['var_' + str(i)][var][t].X for t in self.t_set]
        else:
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")
            print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())


    # day = '2018-07-04'
    # dirname = '/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/RGD_Mc/export_MP/'

    # PV_forecast = data.PV_pred
    # load_forecast = data.load_pred

    # MP = CCG_MP(PV_forecast=PV_forecast, load_forecast=load_forecast)
    # MP.export_model(dirname + 'MP')
    # MP.solve()
    # solution = MP.store_solution()

    # FONTSIZE = 20

    # plt.figure(figsize=(16,9))
    # plt.plot(solution['x'], label='DG output')
    # plt.ylabel('kW', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE)
    # plt.yticks(fontsize=FONTSIZE)
    # plt.legend(fontsize=FONTSIZE)
    # plt.title('MILP formulation')
    # plt.savefig(dirname + day + '_DG_units_output_' + '.png')
    # plt.close('all')

    # plt.figure(figsize=(16,9))
    # plt.plot(solution['x_chg'], label='x_chg')
    # plt.plot(solution['x_dis'], label='x_dis')
    # plt.ylabel('kW', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE)
    # plt.yticks(fontsize=FONTSIZE)
    # plt.legend(fontsize=FONTSIZE)
    # plt.title('pre-ESS schedule')
    # plt.savefig(dirname + day + '_pre-ESS_schedule_' + '.png')
    # plt.close('all')    

    # plt.figure(figsize=(16,9))
    # plt.plot(solution['x_pos'], label='x_pos')
    # plt.plot(solution['x_neg'], label='x_neg')
    # plt.ylabel('kW', fontsize=FONTSIZE)
    # plt.xticks(fontsize=FONTSIZE)
    # plt.yticks(fontsize=FONTSIZE)
    # plt.legend(fontsize=FONTSIZE)
    # plt.title('pre-reserve schedule')
    # plt.savefig(dirname + day + '_pre-reserve_schedule_' + '.png')
    # plt.close('all')

    # plt.figure(figsize=(16,9))
    # plt.plot(solution['x_cut'], label='PV curt predict')
    # plt.ylabel('kW', fontsize=FONTSIZE)
    # plt.legend(fontsize=FONTSIZE)
    # plt.savefig(dirname + day + 'x_curt_' + '.png')
    # plt.close('all')
    
    # plt.figure(figsize=(16,9))
    # plt.plot(solution['x'], linewidth=2, label='DG output')
    # plt.plot(solution['x_load'], color='darkorange', linewidth=2, label='load')
    # plt.plot(solution['x_S'], color='green', linewidth=2, label='SOC')
    # plt.plot(solution['x_PV'], color='royalblue', linewidth=2, label='PV generation')
    # plt.plot(solution['x_cut'], color='dimgrey', linestyle='-.', linewidth=2, label='PV curtailment')
    # plt.plot(([hs - eg for hs, eg in zip(solution['x_PV'], solution['x_cut'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    # plt.legend()
    # plt.savefig(dirname + day + 'MILP_day ahead' + '.png')
    # plt.close('all')