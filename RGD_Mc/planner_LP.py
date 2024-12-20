# import os
# import time
# import numpy as np
# import pandas as pd
# import gurobipy as gp
# from gurobipy import GRB

# from root_project import ROOT_DIR

# import matplotlib.pyplot as plt
# from Params import PARAMETERS
# from utils import dump_file
# from utils import read_file
# from Data_read import *


# class Planner_LP():
#     """
#     LP capacity firming formulation: no binary variables to ensure not simultaneous charge and discharge.
#     :ivar nb_periods: number of market periods (-)
#     :ivar period_hours: period duration (hours)
#     :ivar soc_ini: initial state of charge (kWh)
#     :ivar soc_end: final state of charge (kWh)
#     :ivar PV_forecast: PV point forecasts (kW)
#     :ivar load_forecast: load forecast (kW)
#     :ivar x: diesel on/off variable (on = 1, off = 0)
#           shape = (nb_market periods,)

#     :ivar model: a Gurobi model (-)
#     """

#     def __init__(self, RG_forecast:np.array, load_forecast:np.array):
#         """
#         Init the planner.
#         """
#         self.parameters = PARAMETERS # simulation parameters
#         self.period_hours = PARAMETERS['period_hours'] # 1/4 hour
#         self.nb_periods = int(24 / self.period_hours) # 96
#         self.t_set = range(self.nb_periods)

#         # Parameters required for the MP in the CCG algorithm
#         self.RG_forecast = RG_forecast # (kW)
#         self.load_forecast = load_forecast # (kW)

#         # Thermal parameters
#         self.u = PARAMETERS['u'] # on/off
#         self.thermal_min = PARAMETERS['thermal']['thermal_min'] # (kW)
#         self.thermal_max = PARAMETERS['thermal']['thermal_max'] # (kW)
#         self.thermal_ramp_up = PARAMETERS['thermal']['ramp_up'] # (kW)
#         self.thermal_ramp_down = PARAMETERS['thermal']['ramp_down'] # (kW)
#         self.p_rate = PARAMETERS['thermal']['p_rate']

#         # ESS parameters
#         self.EScapacity = PARAMETERS['ES']['capacity']  # (kWh)
#         self.soc_ini = PARAMETERS['ES']['soc_ini']  # (kWh)
#         self.soc_end = PARAMETERS['ES']['soc_end']  # (kWh)
#         self.soc_min = PARAMETERS['ES']['soc_min']  # (kWh)
#         self.soc_max = PARAMETERS['ES']['soc_max']  # (kWh)
#         self.charge_eff = PARAMETERS['ES']['charge_eff']  # (/)
#         self.discharge_eff = PARAMETERS['ES']['discharge_eff']  # (/)
#         self.ES_min = PARAMETERS['ES']['power_min']  # (kW)
#         self.ES_max = PARAMETERS['ES']['power_max']  # (kW)

#         # RG parameters
#         self.RG_min = PARAMETERS['RG']['min_gen']
#         self.RG_max = PARAMETERS['RG']['max_gen']
#         self.RG_ramp_up = PARAMETERS['RG']['ramp_up']
#         self.RG_ramp_down = PARAMETERS['RG']['ramp_down']

#         # load parameters
#         self.load_ramp_up = PARAMETERS['load']['ramp_up']
#         self.load_ramp_down = PARAMETERS['load']['ramp_down']

#         # reserve requirement
#         self.reserve_pos = PARAMETERS['reserve']['reserve_pos']
#         self.reserve_neg = PARAMETERS['reserve']['reserve_neg']

#         # Cost parameters
#         self.cost_a = PARAMETERS['cost']['a_of_Th']
#         self.cost_b = PARAMETERS['cost']['b_of_Th']
#         self.cost_c = PARAMETERS['cost']['c_of_Th']
#         self.cost_m_pos = PARAMETERS['cost']['m_pos_of_Th']
#         self.cost_m_neg = PARAMETERS['cost']['m_neg_of_Th']
#         self.cost_m_pos_re = PARAMETERS['cost']['m_pos_re_of_Th']
#         self.cost_m_neg_re = PARAMETERS['cost']['m_neg_re_of_Th']
#         self.cost_OM_ES = PARAMETERS['cost']['m_O&M_of_ES']
#         self.cost_n_pos = PARAMETERS['cost']['n_pos_of_ES']
#         self.cost_n_neg = PARAMETERS['cost']['n_neg_of_ES']
#         self.cost_n_pos_re = PARAMETERS['cost']['n_pos_re_of_ES']
#         self.cost_n_neg_re = PARAMETERS['cost']['n_neg_re_of_ES']
#         self.cost_m_pre = PARAMETERS['cost']['m_cut_of_preRG']
#         self.cost_m_re = PARAMETERS['cost']['m_cut_of_reRG']

#         # Piecewise linearlization parameters
#         self.seg_num = PARAMETERS['PWL']['num']

#         self.time_building_model = None
#         self.time_solving_model = None

#         # Create model
#         self.model = self.create_model(RG_trajectory=RG_forecast, load_trajectory=load_forecast)

#         # Solve model
#         self.solver_status = None

#     def FC(self, p):
#         if p == 0:
#             return 0
#         elif p != 0:
#             return(self.cost_a * p * p + self.cost_b * p + self.cost_c)

#     def PC(self, g):
#         if g == 0:
#             return 0
#         elif g != 0:
#             return self.cost_m_pre * g * g

#     def RC(self, g):
#         if g == 0:  
#             return 0
#         elif g != 0:
#             return self.cost_m_re * g * g

#     def PWL(self, PWL_num, lb, ub, egg):
#         x = [0]
#         y = [0]
#         for i in range(PWL_num + 1):
#             x.append(lb + (ub - lb) * i / PWL_num)
#             y.append(egg(x[i + 1]))
#         return x, y

#     def create_model(self, RG_trajectory, load_trajectory):
#         """
#         Create the optimization problem.
#         """
#         t_build = time.time()

#         # -------------------------------------------------------------------------------------------------------------
#         # 1. Create model
#         model = gp.Model("planner_MILP_gurobi")

#         # -------------------------------------------------------------------------------------------------------------
#         # 2. create vairables
#         # 2.1 Create First-stage variables -> x
#         # u_ES = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='u_ES') # on/off of ES charge
#         p = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='p') # comtemporary output of thermal
#         r_pos = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_pos') # reserve capacity of thermal
#         r_neg = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_neg') # reserve capacity of thermal
#         r_pos_ES = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_pos_ES') # reserve capacity of ES
#         r_neg_ES = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_neg_ES') # reserve capacity of ES
#         x_chg = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_chg') # charge power
#         x_dis = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_dis') # discharge power
#         x_RG = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_RG') # RG output
#         x_load = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='load') # load demand
#         x_S = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_SOC') # ES SOC
#         x_curt = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='curtailment') # pre-dispatch curtailment of RG

#         # 2.2 Second-stage variables -> y
#         p_pos = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_pos")
#         p_neg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_neg")
#         y_chg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge")
#         y_dis = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge")
#         y_S = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_SOC")
#         y_RG = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_RG")
#         y_curt_pos = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_pos")
#         y_curt_neg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_neg")
#         y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")

#         # -------------------------------------------------------------------------------------------------------------
#         x_cost_fuel = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel")
#         x_cost_OM_ES = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="O&M_cost_OM_ES")
#         x_cost_curt = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_curt") # PWL
#         y_cost_reg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="regulation_cost_thermal")
#         y_cost_reg_ES = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="regulation_cost_ES")
#         y_cost_curt_pos = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="RG_curtail_cost_pos")
#         y_cost_curt_neg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="RG_curtail_cost_neg")

#         # -------------------------------------------------------------------------------------------------------------
#         # 3. Create objective
#         objective = gp.quicksum(x_cost_fuel[i] + x_cost_OM_ES[i] + x_cost_curt[i] + y_cost_reg[i] + y_cost_reg_ES[i] + y_cost_curt_pos[i] + y_cost_curt_neg[i] for i in self.t_set)
#         model.setObjective(objective, GRB.MINIMIZE)

#         # -------------------------------------------------------------------------------------------------------------
#         # 4. Create constraints
#         model.addConstrs((- p[i] + r_neg[i] <= - self.u[i] * self.thermal_min for i in self.t_set), name='c_thermal_min')
#         model.addConstrs((p[i] + r_pos[i] <= self.u[i] * self.thermal_max for i in self.t_set), name='c_thermal_max')
#         model.addConstrs((r_pos[i] <= self.u[i] * self.thermal_ramp_up * self.period_hours for i in self.t_set), name='c_reserve_min')
#         model.addConstrs((r_neg[i] <= self.u[i] * self.thermal_ramp_down * self.period_hours for i in self.t_set), name='c_reserve_max')
#         model.addConstrs(((p[i] + r_pos[i]) - (p[i-1] + r_neg[i-1]) <= self.u[i-1] * self.thermal_ramp_up * self.period_hours + (1 - self.u[i-1]) * self.thermal_max for i in range(1, self.nb_periods)), name='c_thermal_ramping_1')
#         model.addConstrs((- (p[i] + r_pos[i]) + (p[i-1] + r_neg[i-1]) <= self.u[i] * self.thermal_ramp_down * self.period_hours + (1 - self.u[i]) * self.thermal_max for i in range(1, self.nb_periods)), name='c_thermal_ramping_2')
#         model.addConstrs(((p[i] - r_neg[i]) - (p[i-1] + r_pos[i-1]) <= self.u[i-1] * self.thermal_ramp_up * self.period_hours + (1 - self.u[i-1]) * self.thermal_max for i in range(1, self.nb_periods)), name='c_thermal_ramping_3')
#         model.addConstrs((- (p[i] - r_neg[i]) + (p[i-1] + r_pos[i-1]) <= self.u[i] * self.thermal_ramp_down * self.period_hours + (1 - self.u[i]) * self.thermal_max for i in range(1, self.nb_periods)), name='c_thermal_ramping_4')
#         model.addConstrs((p[i] - x_chg[i] + x_dis[i] + x_RG[i] - x_load[i] == 0 for i in self.t_set), name='c_power_balance')
#         model.addConstrs((x_chg[i] <= self.ES_max for i in self.t_set), name='c_chgarge_max') # LP
#         model.addConstrs((x_dis[i] <= self.ES_max for i in self.t_set), name='c_discharge_max') # LP
#         model.addConstrs((- (x_dis[i] - x_chg[i]) + r_neg_ES[i] <= self.ES_min for i in self.t_set), name='c_ES_min')
#         model.addConstrs(((x_dis[i] - x_chg[i]) + r_pos_ES[i] <= self.ES_max for i in self.t_set), name='c_ES_max')
#         model.addConstrs((- x_S[i] <= - self.soc_min for i in self.t_set), name='c_SOC_min')
#         model.addConstrs((x_S[i] <= self.soc_max for i in self.t_set), name='c_SOC_max')
#         model.addConstr((x_S[0] == self.soc_ini), name='c_SOC_first')
#         model.addConstrs((x_S[i] - x_S[i - 1] - ((self.charge_eff * x_chg[i]) - (x_dis[i] / self.discharge_eff)) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_SOC_dynamic')
#         model.addConstr((x_S[self.nb_periods - 1] == self.soc_ini), name='c_SOC_last')
#         model.addConstrs((x_RG[i] == self.RG_forecast[i] for i in self.t_set), name='c_RG_output')
#         model.addConstrs((x_load[i] == self.load_forecast[i] for i in self.t_set), name='c_x_load_demand')
#         model.addConstrs((x_curt[i] <= self.RG_forecast[i] for i in self.t_set), name='c_curtailment')
#         model.addConstrs((r_pos[i] + r_pos_ES[i] >= self.reserve_pos for i in self.t_set), name='c_reserve_pos')
#         model.addConstrs((r_neg[i] + r_neg_ES[i] >= self.reserve_neg for i in self.t_set), name='c_reserve_neg')

#         # 4.2 Second stage constraints
#         model.addConstrs((p_pos[i] <= r_pos[i] for i in self.t_set), name='c_reserve_pos_thermal')
#         model.addConstrs((p_neg[i] <= r_neg[i] for i in self.t_set), name='c_reserve_neg_thermal')
#         model.addConstrs((x_dis[i] + y_dis[i] <= self.ES_max for i in self.t_set), name='c_discharge_re')
#         model.addConstrs((x_chg[i] + y_chg[i] <= self.ES_max for i in self.t_set), name='c_charge_re')
#         model.addConstrs((- y_S[i] <= - self.soc_min for i in self.t_set), name='c_min_S')
#         model.addConstrs((y_S[i] <= self.soc_max for i in self.t_set), name='c_max_S')
#         model.addConstr((y_S[0] == self.soc_ini), name='c_ESS_first_period')
#         model.addConstrs((y_S[i] - y_S[i - 1] - self.period_hours * ((self.charge_eff * (x_chg[i] + y_chg[i]) - (x_dis[i] + y_dis[i]) / self.discharge_eff)) == 0
#                           for i in range(1, self.nb_periods)), name='c_ESS_re-dispatch')
#         model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_ESS_last_period')
#         model.addConstrs((y_RG[i] == RG_trajectory[i] - x_curt[i] for i in self.t_set), name='c_RG_re-dispatch')
#         model.addConstrs((y_load[i] == load_trajectory[i] for i in self.t_set), name='c_load_re-dispatch')
#         model.addConstrs((y_curt_pos[i] <= x_curt[i] for i in self.t_set), name='c_curtailment_pos')
#         model.addConstrs((y_curt_neg[i] <= RG_trajectory[i] - x_curt[i] for i in self.t_set), name='c_curtailment_neg')

#         # -------------------------------------------------------------------------------------------------------------
#         # 5. Store variables
#         self.allvar = dict()
#         self.allvar['p'] = p
#         self.allvar['r_pos'] = r_pos
#         self.allvar['r_neg'] = r_neg
#         self.allvar['r_pos_ES'] = r_pos_ES
#         self.allvar['r_neg_ES'] = r_neg_ES
#         self.allvar['x_chg'] = x_chg
#         self.allvar['x_dis'] = x_dis
#         self.allvar['x_RG'] = x_RG
#         self.allvar['x_load'] = x_load
#         self.allvar['x_S'] = x_S
#         self.allvar['x_curt'] = x_curt
#         self.allvar['p_pos'] = p_pos
#         self.allvar['p_neg'] = p_neg
#         self.allvar['y_chg'] = y_chg
#         self.allvar['y_dis'] = y_dis
#         self.allvar['y_S'] = y_S
#         self.allvar['y_RG'] = y_RG
#         self.allvar['y_curt_pos'] = y_curt_pos
#         self.allvar['y_curt_neg'] = y_curt_neg
#         self.allvar['y_load'] = y_load
#         self.allvar['x_cost_fuel'] = x_cost_fuel
#         self.allvar['x_cost_OM_ES'] = x_cost_OM_ES
#         self.allvar['x_cost_curt'] = x_cost_curt
#         self.allvar['y_cost_reg'] = y_cost_reg
#         self.allvar['y_cost_reg_ES'] = y_cost_reg_ES
#         self.allvar['y_cost_curt_pos'] = y_cost_curt_pos
#         self.allvar['y_cost_curt_neg'] = y_cost_curt_neg

#         self.time_building_model = time.time() - t_build
#         # print("Time spent building the mathematical program: %gs" % self.time_building_model)

#         return model
    
#     def solve(self, outputflag:bool=False):
#         t_solve = time.time()
#         self.model.setParam('OutputFlag', outputflag)
#         self.model.optimize()
#         self.solver_stats = self.model.status
#         self.time_solving_model = time.time() - t_solve

#     def store_solution(self):
#         m = self.model

#         solution = dict()
#         solution['status'] = m.status
#         solution['obj'] = m.objVal

#         # 1 dimensional variables
#         for var in ['p', 'r_pos', 'r_neg', 'r_pos_ES', 'r_neg_ES', 'x_chg', 'x_dis', 'x_RG', 'x_load', 'x_S', 'x_curt', 'p_pos', 'p_neg', 'y_chg', 'y_dis', 'y_S',
#                     'y_RG', 'y_curt_pos', 'y_curt_neg', 'y_load', 'x_cost_fuel', 'x_cost_OM_ES', 'x_cost_curt', 'y_cost_reg', 'y_cost_reg_ES', 'y_cost_curt_pos', 'y_cost_curt_neg']:
#             solution[var] = [self.allvar[var][t].X for t in self.t_set]

#         # Timing indicators
#         solution['time_building'] = self.time_building_model
#         solution['time_solving'] = self.time_solving_model
#         solution['time_total'] = self.time_building_model + self.time_solving_model

#         num_violated_constraints = m.ConstrVio
#         print("Number of violated constraints: ", num_violated_constraints)

#         return solution
    
#     def export_model(self, filename):
#         """
#         Export the pyomo model into a cpxlp format.
#         :param filename: directory and filename of the exported model.
#         """

#         self.model.write("%s.lp" % filename)


# if __name__ == "__main__":
#     # Set the working directory to the root of the project
#     print(os.getcwd())
#     os.chdir(ROOT_DIR)
#     print(os.getcwd())

#     dirname = '/Users/PSL/Desktop/bibleshin/PC_RGD_CCG_Mc_SIU'
#     day = '2018-07-04'

#     RG_solution = np.array(pd.read_csv('PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
#     load_solution = np.array(pd.read_csv('Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
#     RG_forecast = data.PV_pred
#     load_forecast = data.load_egg

#     planner_perfect = Planner_LP(RG_forecast=RG_solution, load_forecast=load_solution)
#     planner_perfect.export_model(dirname + 'LP')
#     planner_perfect.solve()
#     solution_perfect = planner_perfect.store_solution()

#     print('objective oracle %.2f' % (solution_perfect['obj']))

#     dump_file(dir=dirname, name='sol_LP_oracle_p', file=solution_perfect['p'])
#     dump_file(dir=dirname, name='sol_LP_oracle_r_pos', file=solution_perfect['r_pos'])
#     dump_file(dir=dirname, name='sol_LP_oracle_r_neg', file=solution_perfect['r_neg'])
#     dump_file(dir=dirname, name='sol_LP_oracle_r_pos_ES', file=solution_perfect['r_pos_ES'])
#     dump_file(dir=dirname, name='sol_LP_oracle_r_neg_ES', file=solution_perfect['r_neg_ES'])
#     dump_file(dir=dirname, name='sol_LP_oracle_x_chg', file=solution_perfect['x_chg'])
#     dump_file(dir=dirname, name='sol_LP_oracle_x_dis', file=solution_perfect['x_dis'])
#     dump_file(dir=dirname, name='sol_LP_oracle_x_RG', file=solution_perfect['x_RG'])
#     dump_file(dir=dirname, name='sol_LP_oracle_x_load', file=solution_perfect['x_load'])
#     dump_file(dir=dirname, name='sol_LP_oracle_x_S', file=solution_perfect['x_S'])
#     dump_file(dir=dirname, name='sol_LP_oracle_x_curt', file=solution_perfect['x_curt'])

#     # plt.figure()
#     # plt.plot(solution_perfect['y_charge'], label='y_charge')
#     # plt.plot(solution_perfect['y_discharge'], label='y_discharge')
#     # plt.plot(solution_perfect['y_s'], label='y s')
#     # plt.legend()
#     # plt.show()
    
#     planner = Planner_LP(RG_forecast=RG_forecast, load_forecast=load_forecast)
#     planner.solve()
#     solution = planner.store_solution()
#     dump_file(dir=dirname, name='sol_LP_p', file=solution['p'])
#     dump_file(dir=dirname, name='sol_LP_r_pos', file=solution['r_pos'])
#     dump_file(dir=dirname, name='sol_LP_r_neg', file=solution['r_neg'])
#     dump_file(dir=dirname, name='sol_LP_r_pos_ES', file=solution['r_pos_ES'])
#     dump_file(dir=dirname, name='sol_LP_r_neg_ES', file=solution['r_neg_ES'])
#     dump_file(dir=dirname, name='sol_LP_x_chg', file=solution['x_chg'])
#     dump_file(dir=dirname, name='sol_LP_x_dis', file=solution['x_dis'])
#     dump_file(dir=dirname, name='sol_LP_x_RG', file=solution['x_RG'])
#     dump_file(dir=dirname, name='sol_LP_x_load', file=solution['x_load'])
#     dump_file(dir=dirname, name='sol_LP_x_S', file=solution['x_S'])
#     dump_file(dir=dirname, name='sol_LP_x_curt', file=solution['x_curt'])

#     # plt.figure()
#     # plt.plot(solution['x_b'], label='x fist-stage')
#     # plt.plot(solution['y_diesel'], label='diesel')
#     # plt.plot(([hs - eg for hs, eg in zip(solution['y_discharge'], solution['y_charge'])]), label='BESS')
#     # plt.plot(solution['y_load'], label='load')
#     # plt.plot(solution['y_PV'], label='PV generation')
#     # plt.plot(solution['y_curt'], label='PV curtailment')
#     # plt.plot(([hs - eg for hs, eg in zip(solution['y_PV'], solution['y_curt'])]), label='PV output')
#     # plt.title('LP formulation')
#     # plt.legend()
#     # # plt.savefig(dirname+ 'LP_oracle_vs_PVUSA.pdf')
#     # plt.show()


