"""
2023.10.31.
GIST Power System Lab.
Hyun-Su Shin.
Column-and-Constraint Generation (CCG) algorithm to solve a two-stage robust optimization problem in the microgrid scheduling.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CCG_MP import CCG_MP
from CCG_SP_bM_worst import CCG_SP_worst
from SP_primal_LP import SP_primal_LP
from planner_MILP import Planner_MILP
from Data_read import *
from root_project import ROOT_DIR
from Params import PARAMETERS
from utils import *

def ccg_algo(dir:str, tol:float, power:np.array, reserve_pos:np.array, reserve_neg:np.array,
             charge:np.array, discharge:np.array, SOC:np.array, curtailment:np.array,
             GAMMA:int, PI:int, solver_param:dict, day:str, log:bool=False, printconsole:bool=False):
    """
    CCG = Column-and-Constraint Generation
    Column-and-Constraint Generation algorithm.
    Iteration between the MP and SP until convergence criteria is reached.
    :param tol: convergence tolerance.
    :param Pi_RG_t/s, PI: RG/load spatio and temporal budget of uncertainty.
    :param RG/load_max/min: RG/load max/min bound of the uncertainty set (kW).
    :ivar x: pre-dispatch variables.
    :ivar y: re-dispatch variables.
    :param solver_param: Gurobi solver parameters.
    :return: the final preparatory curtailment schedule when the convergence criteria is reached and some data.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # CCG initialization: build the initial MP
    # ------------------------------------------------------------------------------------------------------------------

    # Building the MP
    MP = CCG_MP(PV_forecast=PV_forecast, load_forecast=load_forecast)
    MP.model.update()
    print('MP initialized: %d variables %d constraints' % (len(MP.model.getVars()), len(MP.model.getConstrs())))
    MP.export_model(dir + day + '_CCG_MP_initialized')

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop until convergence criteria is reached
    # ------------------------------------------------------------------------------------------------------------------

    if printconsole:
        print('---------------------------------CCG ITERATION STARTING---------------------------------')

    t_solve = time.time()
    objectives = []
    computation_times = []
    # measure that helps control the trade-off between solution quality and computation time in MILP or MIQP
    mipgap = []
    alpha_pos_list = []
    alpha_neg_list = []
    beta_pos_list = []
    beta_neg_list = []
    gamma_pos_list = []
    gamma_neg_list = []
    SP_dual_status = []
    SP_primal_status = []
    tolerance = 1e5

    # with CCG the convergence is stable.
    tolerance_list = [tolerance]
    iteration = 1
    ESS_count_list = []
    ESS_charge_discharge_list = []
    PV_count_list = []
    PV_cut_add_list = []
    DG_count_list =[]
    DG_pos_neg_list = []
    max_iteration = 50

    while all(i > tol for i in tolerance_list) and iteration < max_iteration:
        logfile = ""
        if log:
            logfile = dir + 'logfile_' + str(iteration) + '.log'
        if printconsole:
            print('i= %s solve SP dual' % (iteration))

        # ------------------------------------------------------------------------------------------------------------------
        # 1. SP part
        # ------------------------------------------------------------------------------------------------------------------

        # 1.1 Solve the SP and get the worst RG and load trajectory to add the new constraints of the MP
        SP_dual = CCG_SP_worst(PV_forecast=PV_forecast, load_forecast=load_forecast, power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg,
                         charge=charge, discharge=discharge, SOC=SOC, curtailment=curtailment, GAMMA=GAMMA, PI=PI, M=M)
        SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
        SP_dual_sol = SP_dual.store_solution()
        SP_dual_status.append(SP_dual_sol['status'])
        mipgap.append(SP_dual.model.MIPGap)
        alpha_pos_list.append(SP_dual_sol['alpha_pos'])
        alpha_neg_list.append(SP_dual_sol['alpha_neg'])
        beta_pos_list.append(SP_dual_sol['beta_pos'])
        beta_neg_list.append(SP_dual_sol['beta_neg'])
        gamma_pos_list.append(SP_dual_sol['gamma_pos'])
        gamma_neg_list.append(SP_dual_sol['gamma_neg'])

        # 1.2 Compute the worst RG, load trajectory from the SP dual solution
        PV_worst_case_from_SP = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
        load_worst_case_from_SP = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'][i] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]
        if printconsole:
            print('     i = %s : SP dual status %s solved in %.1f s MIPGap = %.6f' % (iteration, SP_dual_sol['status'], SP_dual_sol['time_total'], SP_dual.model.MIPGap))

        # 1.3 Solve the primal of the SP to check if the objecitves of the primal and dual are equal to each other
        SP_primal = SP_primal_LP(PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg,
                                 charge=charge, discharge=discharge, SOC=SOC, curtailment=curtailment)
        SP_primal.solve()
        SP_primal_sol = SP_primal.store_solution()
        SP_primal_status.append(SP_primal_sol['status'])

        if printconsole:
            print('     i = %s : SP primal status %s' % (iteration, SP_primal_sol['status']))
            print('     i = %s : SP primal %.1f $ SP dual %.1f $ -> |SP primal - SP dual| = %.2f $' % (iteration, SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))

        # 1.4 SP solved to optimality ? -> Check if there is any simultaneous charge and discharge in the SP primal solution
        if SP_primal_sol['status'] == 2 or SP_primal_sol['status'] == 9: # 2 = optimal, 9 = timelimit has been reached
            ESS_nb_count = check_ESS(SP_primal_sol = SP_primal_sol)
            if ESS_nb_count > 0:
                ESS_charge_discharge_list.append([iteration, SP_primal_sol['y_chg'], SP_primal_sol['y_dis']])
            else:
                ESS_nb_count = float('nan')
            ESS_count_list.append(ESS_nb_count)
            if printconsole:
                print('     i = %s : %s simultaneous charge and discharge' % (iteration, ESS_nb_count))

            PV_nb_count = check_PV(SP_primal_sol=SP_primal_sol)
            if PV_nb_count > 0:
                PV_cut_add_list.append([iteration, SP_primal_sol['y_cut'], SP_primal_sol['y_add']])
            else:
                PV_nb_count = float('nan')
            PV_count_list.append(PV_nb_count)
            if printconsole:
                print('     i = %s : %s simultaneous curtailment and addition' % (iteration, PV_nb_count))

            DG_nb_count = check_DG(SP_primal_sol=SP_primal_sol)
            if DG_nb_count > 0:
                DG_pos_neg_list.append([iteration, SP_primal_sol['y_pos'], SP_primal_sol['y_neg']])
            else:
                DG_nb_count = float('nan')
            DG_count_list.append(DG_nb_count)
            if printconsole:
                print('     i = %s : %s simultaneous reserve pos and neg power' % (iteration, DG_nb_count))

        # ------------------------------------------------------------------------------------------------------------------
        # 2. MP part
        # ------------------------------------------------------------------------------------------------------------------

        # Check Sub Problem status -> bounded or unbounded
        if SP_dual_sol['status'] == 2 or SP_dual_sol['status'] == 9:  # 2 = optimal, 9 = timelimit has been reached
            # Add an optimality cut to MP and solve
            MP.update_MP(PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, iteration=iteration)
            if printconsole:
                print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
            # MP.export_model(dir + 'MP_' + str(iteration))
            if printconsole:
                print('i = %s : solve MP' % (iteration))
            MP.solve()
            MP_sol = MP.store_solution()
            MP.update_sol(MP_sol=MP_sol, i=iteration)
            if MP_sol['status'] == 3 or MP_sol['status'] == 4:
                print('i = %s : WARNING MP status %s -> Create a new MP, increase big-M value and compute a new RG trajectory from SP' % (iteration, MP_sol['status']))

                # MP unbounded of infeasible -> increase big-M's value to get another PV trajectory from the SP
                SP_dual = CCG_SP_worst(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_pos=PV_pos, PV_neg=PV_neg, load_pos=load_pos, load_neg=load_neg,
                                 power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg, curtailment=curtailment, GAMMA=GAMMA, PI=PI, M=M+50)
                SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
                SP_dual_sol = SP_dual.store_solution()

                # Compute a new worst PV trajectory from the SP dual solution
                PV_worst_case_from_SP = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
                load_worst_case_from_SP = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'][i] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]

                # Create a new MP
                MP = CCG_MP()
                MP.model.update()
                MP.update_MP(PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, iteration=iteration)
                if printconsole:
                    print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
                # MP.export_model(dir + 'MP_' + str(iteration))
                if printconsole:
                    print('i = %s : solve new MP' % (iteration))
                MP.solve()
                MP_sol = MP.store_solution()
                MP.update_sol(MP_sol=MP_sol, i=iteration)

            computation_times.append([SP_dual_sol['time_total'], MP_sol['time_total']])

        else: # 4 = Model was proven to be either infeasible or unbounded.
            print('SP is unbounded: a feasibility cut is required to be added to the Master Problem')

        objectives.append([iteration, MP_sol['obj'], SP_dual_sol['obj'], SP_primal_sol['obj']])

        # ------------------------------------------------------------------------------------------------------------------
        # 3. Update: pre-dispatch variables, lower and upper bounds using the updated MP
        # ------------------------------------------------------------------------------------------------------------------

        # Solve the MILP with the worst case trajectory
        planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP)
        planner.solve()
        sol_planner = planner.store_solution()

        # Update x variables
        power = MP_sol['x']
        reserve_pos = MP_sol['x_pos']
        reserve_neg = MP_sol['x_neg']
        charge = MP_sol['x_chg']
        discharge = MP_sol['x_dis']
        SOC = MP_sol['x_S']
        curtailment = MP_sol['x_cut']
        x_cost_fuel = MP_sol['x_cost_fuel']
        x_cost_res = MP_sol['x_cost_res']
        x_cost_ESS = MP_sol['x_cost_ESS']
        x_cost_cut = MP_sol['x_cost_cut']
        y_cost_fuel = MP_sol['var_' + str(iteration)]['y_cost_fuel']
        y_cost_ESS = MP_sol['var_' + str(iteration)]['y_cost_ESS']
        y_cost_cut = MP_sol['var_' + str(iteration)]['y_cost_cut']
        y_cost_add = MP_sol['var_' + str(iteration)]['y_cost_add']
        y_cut = MP_sol['var_' + str(iteration)]['y_cut']
        y_add = MP_sol['var_' + str(iteration)]['y_add']


        # Update the lower and upper bounds
        # MP -> give the lower bound
        # SP -> give the upper bound
        tolerance = abs(MP_sol['obj'] - SP_dual_sol['obj'])
        print('i = %s : |MP - SP dual| = %.2f $' % (iteration, tolerance))
        abs_err = abs(MP_sol['obj'] - sol_planner['obj'])
        tolerance_list.append(tolerance)
        tolerance_list.pop(0)
        if printconsole:
            print('i = %s : MP %.2f $ SP dual %.2f $ -> |MP - SP dual| = %.2f $' % (iteration, MP_sol['obj'], SP_dual_sol['obj'], tolerance))
            print('i = %s : MP %.2f $ MILP %.2f $ -> |MP - MILP| = %.2f $' % (iteration, MP_sol['obj'], sol_planner['obj'], abs_err))
            print(tolerance_list)
            print('                                                                                                       ')

        iteration += 1

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop terminated
    # ------------------------------------------------------------------------------------------------------------------
    if printconsole:
        print('-----------------------------------CCG ITERATION TERMINATED-----------------------------------')
    print('Final iteration  = %s : MP %.2f $ SP dual %.2f $ -> |MP - SP dual| = %.2f $' % (iteration-1, MP_sol['obj'], SP_dual_sol['obj'], tolerance))

    # Export last MP
    MP.export_model(dir + day + '_MP')

    # MP.model.printStats()

    # Dump last engagement plan at iteration
    dump_file(dir=dir, name=day+'_x', file=power)
    dump_file(dir=dir, name=day+'_x_pos', file=reserve_pos)
    dump_file(dir=dir, name=day+'_x_neg', file=reserve_neg)
    dump_file(dir=dir, name=day+'_x_chg', file=charge)
    dump_file(dir=dir, name=day+'_x_dis', file=discharge)
    dump_file(dir=dir, name=day+'_x_S', file=SOC)
    dump_file(dir=dir, name=day+'_x_cut', file=curtailment)
    dump_file(dir=dir, name=day+'_x_cost_fuel', file=x_cost_fuel)
    dump_file(dir=dir, name=day+'_x_cost_res', file=x_cost_res)
    dump_file(dir=dir, name=day+'_x_cost_ESS', file=x_cost_ESS)
    dump_file(dir=dir, name=day+'_x_cost_cut', file=x_cost_cut)

    # print T CPU
    t_total = time.time() - t_solve
    computation_times = np.asarray(computation_times)
    SP_dual_status = np.asarray(SP_dual_status)
    SP_primal_status = np.asarray(SP_primal_status)

    if printconsole:
        print('Total CCG loop t CPU %.1f s' % (t_total))
        print('T CPU (s): Sup Problem max %.1f Master Problem max %.1f' % (computation_times[:, 0].max(), computation_times[:, 1].max()))
        print('nb Sup Problem status 2 %d status 9 %d' % (SP_dual_status[SP_dual_status == 2].shape[0], SP_dual_status[SP_dual_status == 9].shape[0]))

    # Store data
    objectives = np.asarray(objectives)
    df_objectives = pd.DataFrame(index=objectives[:,0], data=objectives[:,1:], columns=['MP', 'SP', 'SP_primal'])

    # Store convergence information
    conv_inf = dict()
    conv_inf['mipgap'] = mipgap
    conv_inf['computation_times'] = computation_times
    conv_inf['SP_status'] = SP_dual_status
    conv_inf['SP_primal_status'] = SP_primal_status
    conv_inf['alpha_pos'] = alpha_pos_list
    conv_inf['alpha_neg'] = alpha_neg_list
    conv_inf['beta_pos'] = beta_pos_list
    conv_inf['beta_neg'] = beta_neg_list
    conv_inf['gamma_pos'] = gamma_pos_list
    conv_inf['gamma_neg'] = gamma_neg_list
    conv_inf['ESS_count'] = ESS_count_list
    conv_inf['ESS_charge_discharge'] = ESS_charge_discharge_list
    conv_inf['PV_count'] = PV_count_list
    conv_inf['PV_cut_add'] = PV_cut_add_list
    conv_inf['DG_count'] = DG_count_list
    conv_inf['DG_pos_neg'] = DG_pos_neg_list

    return power, reserve_pos, reserve_neg, charge, discharge, SOC, curtailment, df_objectives, conv_inf, \
        x_cost_fuel, x_cost_res, x_cost_ESS, x_cost_cut, y_cost_fuel, y_cost_ESS, y_cost_cut, y_cost_add, y_cut, y_add

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 24

# Solver parameters
solver_param = dict()
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 10
solver_param['Threads'] = 1
.8
# Convergence threshold between MP and SP objectives
conv_tol = 5
printconsole = True

day = '2025-01-15'

# --------------------------------------
# Static RO parameters
GAMMA = 96 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
PI = 96
M = 1

# quantile from NE or LSTM
PV_Sandia = True

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    # Create folder
    dirname = '/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/RGD_bM/export_worst/'
    if PV_Sandia:
        dirname += 'PV_Sandia/'
        pdfname = str(PV_Sandia) + '_' + str(GAMMA) + '_' + str(PI)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    
    print('-----------------------------------------------------------------------------------------------------------')
    if PV_Sandia:
        print('CCG: day %s GAMMA %s PI %s' % (day, GAMMA, PI))
    print('-----------------------------------------------------------------------------------------------------------')

    # RG/Load data
    PV_forecast = data.PV_pred
    load_forecast = data.load_pred
    PV_pos = data.PV_pos # (kW) The maximal deviation betwwen the min and forecast PV uncertainty set bounds
    PV_neg = data.PV_neg # (kW) The maximal deviation between the max and forecast PV uncertainty set bounds
    load_pos = data.load_pos # (kw) The maximal deviation between the min and forecast load uncertainty set bounds
    load_neg = data.load_neg # (kW) The maximal deviation between the max and forecast load uncertainty set bounds
    nb_periods = PV_pos.shape[0]

    # plot style
    plt.style.use(['science'])
    plt.rcParams['figure.figsize'] = (7.16, 5.37)
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False
    # plt.rcParams['xtick.labelsize'] = 10
    # plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams['xtick.major.size'] = 5
    # plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['savefig.dpi'] = 1200
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['text.usetex'] = True
    # plt.rc('text', usetex=True)
    x_index = [i for i in range(0, nb_periods)]

    # Store the forecast into a dict
    PV_forecast_dict = dict()
    PV_forecast_dict['forecast'] = PV_forecast

    # Compute the starting point for the first MP = day-ahead planning from the PV using the MILP
    planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_trajectory=PV_forecast, load_trajectory=load_forecast)
    planner.solve()
    sol_planner_ini = planner.store_solution()
    power_ini = sol_planner_ini['x']
    reserve_pos_ini = sol_planner_ini['x_pos']
    reserve_neg_ini = sol_planner_ini['x_neg']
    charge_ini = sol_planner_ini['x_chg']
    discharge_ini = sol_planner_ini['x_dis']
    SOC_ini = sol_planner_ini['x_S']
    curtailment_ini = sol_planner_ini['x_cut']
    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop
    # ------------------------------------------------------------------------------------------------------------------
    final_power, final_reserve_pos, final_reserve_neg, final_charge, final_discharge, final_SOC, \
        final_curtailment, df_objectives, conv_inf , final_x_cost_fuel, \
        final_x_cost_res, final_x_cost_ESS, final_x_cost_cut, \
        final_y_cost_fuel, final_y_cost_ESS, final_y_cost_cut, final_y_cost_add, final_y_cut, final_y_add \
            = ccg_algo(dir=dirname, tol=conv_tol, power=power_ini, reserve_pos=reserve_pos_ini, reserve_neg=reserve_neg_ini,
                       charge=charge_ini, discharge=discharge_ini, SOC=SOC_ini, curtailment=curtailment_ini,
                       GAMMA=GAMMA, PI=PI, solver_param=solver_param, day=day, printconsole=printconsole)
    df_objectives.to_csv(dirname + day + '_obj_MP_SP_' + '.csv')

    print('-----------------------------------------------------------------------------------------------------------')
    print('CCG: day %s GAMMA %s PI %s' % (day, GAMMA, PI))
    print('-----------------------------------------------------------------------------------------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # Get the final worst case RG generation trajectory computed by the Sub Problem
    # ------------------------------------------------------------------------------------------------------------------

    # Get the worst case related to the last engagement plan by using the Sub Problem dual formulation
    SP_dual = CCG_SP_worst(PV_forecast=PV_forecast, load_forecast=load_forecast, power=final_power, reserve_pos=final_reserve_pos, reserve_neg=final_reserve_neg, 
                     charge=final_charge, discharge=final_discharge, SOC=final_SOC, curtailment=final_curtailment, GAMMA=GAMMA, PI=PI, M=M)
    SP_dual.solve(LogToConsole=False, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=10)
    SP_dual_sol = SP_dual.store_solution()

    # Compute the worst RG, load path from the SP dual solution
    PV_worst_case = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
    load_worst_case = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'][i] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]

    phi_PV = [SP_dual_sol['phi_PV'][i] for i in range(nb_periods)]
    phi_data = np.column_stack((np.array(SP_dual_sol['phi_PV']), np.array(SP_dual_sol['phi_cut']), np.array(SP_dual_sol['phi_load']).flatten()))
    np.savetxt('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/result/bM_phi_worst.csv', phi_data, delimiter=',', header='phi_PV,phi_cut,phi_load', comments='', fmt='%.18f')

    dump_file(dir=dirname, name=day + '_PV_worst_case', file=PV_worst_case)
    dump_file(dir=dirname, name=day + '_load_worst_case', file=load_worst_case)
    dump_file(dir='/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/result/', name=day + '_bM_phi_PV_worst', file=SP_dual_sol['phi_PV'])
    dump_file(dir='/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/result/', name=day + '_bM_phi_cut_worst', file=SP_dual_sol['phi_cut'])
    dump_file(dir='/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/result/', name=day + '_bM_phi_load_worst', file=SP_dual_sol['phi_load'])


    # ------------------------------------------------------------------------------------------------------------------
    # Second-stage variables comparison:
    # ------------------------------------------------------------------------------------------------------------------

    # Use the SP primal (SP worst case dispatch max min formulation) to compute the dispatch variables related to the last CCG pre-dispatch computed by the MP
    # Use the worst case dispatch to get the equivalent of the max min formulation
    SP_primal = SP_primal_LP(PV_trajectory=PV_worst_case, load_trajectory=load_worst_case, power=final_power, reserve_pos=final_reserve_pos, reserve_neg=final_reserve_neg,
                             charge=final_charge, discharge=final_discharge, SOC=final_SOC, curtailment=final_curtailment)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # Check if there has been any simultanenaous charge and discharge during all CCG iterations
    # ------------------------------------------------------------------------------------------------------------------

    # 1. Check if there is any simultaneous charge and discharge at the last CCG iteration
    ESS_nb_count = check_ESS(SP_primal_sol=SP_primal_sol)
    PV_nb_count = check_ESS(SP_primal_sol=SP_primal_sol)
    print('CCG last iteration %d simultaneous charge and discharge / %d curtailment and re-generation' % (ESS_nb_count, PV_nb_count))
    # 2. Check if there is any simultaneous charge and discharge over all CCG iteration
    # check if there is nan value (meaning during an iteration the SP primal has not been solved because infeasible, etc)
    ESS_count = conv_inf['ESS_count']
    PV_count = conv_inf['PV_count']
    if sum(np.isnan(ESS_count)) > 0 or sum(np.isnan(PV_count)) > 0:
        print('WARNING %s ESS nan values and %s PV nan values' %(sum(np.isnan(conv_inf['ESS_count'])), sum(np.isnan(conv_inf['PV_count']))))
    # “python list replace nan with 0” Code
    ESS_count = [0 if x != x else x for x in ESS_count]
    PV_count = [0 if x != x else x for x in PV_count]

    print('%d and %d total simultaneous ESS and PV operation over all CCG iterations' % (sum(ESS_count), sum(PV_count)))
    if sum(conv_inf['ESS_count']) > 0:
        plt.figure(figsize=(16,9))
        plt.plot(conv_inf['ESS_count'], 'k', linewidth=2, label='ESS_count')
        plt.ylim(0, max(conv_inf['ESS_count']))
        plt.xlabel('iteration $j$', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        # plt.tight_layout()
        plt.legend()
        plt.savefig(dirname + day + '_ESS_count_' + pdfname + '.pdf')
        # plt.close('all')

        # Plot at each iteration where there has been a simultaneous charge and discharge
        for l in conv_inf['ESS_charge_discharge']:
            plt.figure(figsize = (8,6))
            plt.plot(l[1], linewidth=2, label='charge')
            plt.plot(l[2], linewidth=2, label='discharge')
            plt.ylim(0, PARAMETERS['ESS']['capacity'])
            plt.ylabel('kW', fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE)
            plt.title('simultaneous charge discharge at iteration %s' %(l[0]))
            # plt.tight_layout()
            # plt.close('all')

    if sum(conv_inf['PV_count']) > 0:
        plt.figure(figsize=(16,9))
        plt.plot(conv_inf['PV_count'], 'k', linewidth=2, label='ESS_count')
        plt.ylim(0, max(conv_inf['PV_count']))
        plt.xlabel('iteration $j$', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        # plt.tight_layout()
        plt.legend()
        plt.savefig(dirname + day + '_PV_count_' + pdfname + '.pdf')
        # plt.close('all')

        # Plot at each iteration where there has been a simultaneous charge and discharge
        for l in conv_inf['PV_cut_add']:
            plt.figure(figsize = (8,6))
            plt.plot(l[1], linewidth=2, label='curtailment')
            plt.plot(l[2], linewidth=2, label='addition')
            plt.ylabel('kW', fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE)
            plt.title('simultaneous curtailment and addition at iteration %s' %(l[0]))
            # plt.tight_layout()
            # plt.close('all')

    # ------------------------------------------------------------------------------------------------------------------
    # Check CCG convergence by computing the planning for the PV worst trajectory from CCG last iteration
    # ------------------------------------------------------------------------------------------------------------------
    planner = Planner_MILP(PV_forecast=PV_forecast, load_forecast=load_forecast, PV_trajectory=PV_worst_case, load_trajectory=load_worst_case)
    planner.solve()
    sol_planner = planner.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # First-stage variables comparison: x and objectives
    # ------------------------------------------------------------------------------------------------------------------
    # Convergence plot
    error_MP_SP = np.abs(df_objectives['MP'].values - df_objectives['SP'].values)
    error_SP = np.abs(df_objectives['SP'].values - df_objectives['SP_primal'].values)
    print(error_MP_SP)
    print(error_SP)
    print(conv_inf['mipgap'])

    plt.figure(figsize = (16,9))
    plt.plot(error_MP_SP, marker=10, markersize=10, linewidth=2, label='|MP - SP dual| $')
    plt.plot(error_SP, marker=11, markersize=10, linewidth=2, label='|SP primal - SP dual| $')
    plt.plot(100 * np.asarray(conv_inf['mipgap']), label='SP Dual mipgap %')
    plt.xlabel('Iteration $j$', fontsize=FONTSIZE)
    plt.ylabel('Gap', fontsize=FONTSIZE)
    # plt.ylim(-1, 10)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    # plt.tight_layout()
    plt.savefig(dirname + day + '_error_conv_' + pdfname + '.pdf')
    plt.savefig(dirname + day + '_error_conv.png', dpi=300)
    # plt.close('all')

    print('')
    print('-----------------------CHECK COLUMN AND CONSTRAINT GENERATION CONVERGENCE-----------------------')
    print('Final iteration %s MP %s |MP - SP dual| %.2f $' % (len(df_objectives),
    df_objectives['MP'].values[-1], abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1])))
    print('SP primal %.2f $ SP dual %.2f $ -> |SP primal - SP dual| = %.2f' % (
    SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))
    err_planner_CCG = abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1])
    # print('MILP planner %.2f $ MP CCG %.2f $ -> |MILP planner - MP CCG| = %.2f' % (
    # sol_planner['obj'], df_objectives['MP'].values[-1], err_planner_CCG))

    if err_planner_CCG > conv_tol:
        print('-----------------------WARNING COLUMN AND CONSTRAINT GENERATION IS NOT CONVERGED-----------------------')
        print('abs error %.4f $' % (err_planner_CCG))
    else:
        print('-----------------------COLUMN AND CONSTRAINT GENERATION IS CONVERGED-----------------------')
        print('CCG is converged with |MILP planner - MP CCG| = %.4f $' % (err_planner_CCG))

    plt.figure()
    plt.plot(PV_forecast, color='#1f77b4', linestyle='-', zorder=1, label='$ w^{*}_{t} $')
    plt.plot(load_forecast, color='#ff7f0e', linestyle='--', zorder=2, label='$ l^{*}_{t} $')
    plt.xlabel('Time [h]')
    plt.ylabel('Power [kW]')
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    ax = plt.gca()
    ax.minorticks_off()
    ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_Forecast_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_Forecast_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_Forecast_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(SP_dual_sol['epsilon_pos'], color='#1f77b4', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='#1f77b4', zorder=1, label=r"$\epsilon^+$")
    ax1.plot(SP_dual_sol['epsilon_neg'], color='#d62728', linestyle='--', marker='x', zorder=2, label=r"$\epsilon^-$")
    ax1 = plt.gca()
    ax1.minorticks_off()
    ax1.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('Time [h]')
    ax1.set_xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    ax1.legend(loc='upper right')
    ax1.set_ylabel('PV uncertainty variables')
    ax1.text(0.5, -0.32, '(a)', transform=ax1.transAxes, fontsize=12)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(SP_dual_sol['delta_pos'], color='#2ca02c', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='#2ca02c', zorder=1, label=r"$\delta^+$")
    ax2.plot(SP_dual_sol['delta_neg'], color='#9467bd', linestyle='--', marker='x', zorder=2, label=r"$\delta^-$")
    ax2 = plt.gca()
    ax2.minorticks_off()
    ax2.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Time [h]')
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    ax2.set_ylabel('Load uncertainty variables')
    ax2.text(0.5, -0.32, '(b)', transform=ax2.transAxes, fontsize=12)
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_Uncertainty_variables_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_Uncertainty_variables_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_Uncertainty_variables_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure()
    plt.plot(PV_worst_case, color='#d62728', linestyle='--', marker='o', markerfacecolor='none', markeredgecolor='#d62728', zorder=2, label='$ \hat{w}_t$')
    plt.plot(PV_forecast, color='#1f77b4', linestyle='-', zorder=3, label='$ w^{*}_{t} $')
    plt.plot(PV_forecast + PV_pos, color='#7f7f7f', linestyle=':', zorder=1, alpha=0.8, label='$ w^{*}_{t} + w^{+} $')
    plt.plot(PV_forecast - PV_neg, color='#7f7f7f', linestyle='-.', zorder=1, alpha=0.8, label='$ w^{*}_{t} - w^{-} $')
    plt.fill_between(range(len(PV_forecast)), PV_forecast - PV_neg, PV_forecast + PV_pos, color='#1f77b4', alpha=0.1)
    plt.xlabel('Time [h]')
    plt.ylabel('Power [kW]')
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    ax = plt.gca()
    ax.minorticks_off()
    ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_PV_trajectory_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_PV_trajectory_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_PV_trajectory_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure()
    plt.plot(load_worst_case, color='#d62728', linestyle='--', marker='o', markerfacecolor='none', markeredgecolor='#d62728', zorder=2, label='$ \hat{l}_{t}$')
    plt.plot(load_forecast, color='#ff7f0e', linestyle='-', zorder=3, label='$ l^{*}_{t} $')
    plt.plot(load_forecast + load_pos, color='#7f7f7f', linestyle=':', zorder=1, alpha=0.8, label='$ l^{*}_{t} + l^{+} $')
    plt.plot(load_forecast - load_neg, color='#7f7f7f', linestyle='-.', zorder=1, alpha=0.8, label='$ l^{*}_{t} - l^{-} $')
    plt.fill_between(range(len(load_forecast)), load_forecast - load_neg, load_forecast + load_pos, color='#ff7f0e', alpha=0.1)
    plt.xlabel('Time [h]')
    plt.ylabel('Power [kW]')
    plt.xticks([0, 16, 32, 48, 64, 80, 96], ['0', '4', '8', '12', '16', '20', '24'])
    ax = plt.gca()
    ax.minorticks_off()
    ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    plt.yticks()
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_load_trajectory_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_Load_trajectory_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_Load_trajectory_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure()
    plt.plot(final_SOC, color='#1f77b4', linestyle='-', zorder=1, label='$ s_{t} $')
    plt.plot(SP_primal_sol['y_S'], color='#ff7f0e', linestyle='--', zorder=2, label='$ \hat{s}_t $')
    plt.xlabel('Time [h]')
    plt.ylabel('SOC [kWh]')
    plt.xticks([0, 16, 32, 48, 64, 80, 96], ['0', '4', '8', '12', '16', '20', '24'])
    ax = plt.gca()
    ax.minorticks_off()
    ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_SOC_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_SOC_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_SOC_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure()
    plt.plot(final_power, color='#d62728', linestyle='-', zorder=8, label='$ p_{t} $')
    plt.plot([dg + rp for dg, rp in zip(final_power, final_reserve_pos)], color='#9467bd', linestyle=':', marker='^', markerfacecolor='none', markeredgecolor='#9467bd', zorder=7, label='$ p_{t} + r^{+}_{t} $')
    plt.plot([dg - rn for dg, rn in zip(final_power, final_reserve_neg)], color='#8c564b', linestyle=':', marker='v', markerfacecolor='none', markeredgecolor='#8c564b', zorder=6, label='$ p_{t} + r^{-}_{t} $')
    plt.plot(PV_forecast, color='#1f77b4', linestyle='-.',  marker='o', markerfacecolor='none', markeredgecolor='#1f77b4', zorder=5, label='$ w_{t} $')
    plt.plot([pv - ct for pv, ct in zip(PV_forecast, final_curtailment)], color='#ff7f0e', linestyle='--', marker='o', zorder=3, label='$ w_{t} - w^{\mathrm{cut}}_{t} $')
    plt.plot([ch - dis for ch, dis in zip(final_discharge, final_charge)], color='#bcbd22', linestyle='-.', marker='s', markerfacecolor='none', markeredgecolor='#bcbd22', zorder=2, label='$p^{\mathrm{dis}}_{t} - p^{\mathrm{chg}}_{t}$')
    plt.plot(load_forecast, color='#2ca02c', linestyle='--', zorder=4, label='$ l_{t} $')
    plt.ylim(top=610)
    plt.xlabel('Time [h]')
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    plt.ylabel('Power [kW]')
    ax = plt.gca()
    ax.minorticks_off()
    ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_Pre-dispatch_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_Pre-dispatch_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_Pre-dispatch_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure()
    plt.plot(([dg + dgp - dgn for dg, dgp, dgn in zip(final_power, SP_primal_sol['y_pos'], SP_primal_sol['y_neg'])]), color='#d62728', linestyle='-', zorder=8, label='$ p_{t} + p^{+} - p^{-}_{t} $')
    plt.plot(PV_worst_case, color='#1f77b4', linestyle="-.", marker='o', markerfacecolor='none', markeredgecolor='#1f77b4', zorder=5, label='$ \hat{w}_{t} $')
    plt.plot(load_worst_case, color='#2ca02c', linestyle='--', zorder=4, label='$ \hat{l}_{t}$')
    plt.plot(([pvw - xct - yct + yad for pvw, xct, yct, yad in zip(SP_primal_sol['y_PV'], final_curtailment, SP_primal_sol['y_cut'], SP_primal_sol['y_add'])]), color='#ff7f0e', linestyle='--', marker='o', zorder=3, label='$ \hat{w}_{t} - w^{\mathrm{cut}}_{t} - \hat{w}^{\mathrm{cut}}_{t} + \hat{w}^{\mathrm{add}}_{t} $')
    plt.plot(([xch - xdis + ych - ydis for xch, xdis, ych, ydis in zip(final_discharge, final_charge, SP_primal_sol['y_dis'], SP_primal_sol['y_chg'])]), color='#bcbd22', linestyle='-.', marker='s', markerfacecolor='none', markeredgecolor='#bcbd22', zorder=2, label='$ p^{\mathrm{dis}}_{t} - p^{\mathrm{chg}}_{t} + \hat{p}^{\mathrm{dis}}_{t} - \hat{p}^{\mathrm{chg}}_{t} $')
    plt.xlabel('Time [h]')
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    plt.ylabel('Power [kW]')
    ax = plt.gca()
    ax.minorticks_off()
    ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    plt.legend(ncol=2)
    plt.ylim(top=700)
    plt.legend(ncol=2)
    plt.savefig(dirname + 'svg/' + day + '_Re-dispatch_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_Re-dispatch_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_Re-dispatch_IEEE.pdf', format='pdf')
    plt.close()

    plt.figure(figsize=(7.16, 8.95))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(SP_primal_sol['y_pos'], color='#1f77b4', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='#1f77b4', zorder=1, label='$ \hat{p}^{\mathrm{+}}_{t} $')
    ax1.plot(SP_primal_sol['y_neg'], color='#d62728', linestyle='--', marker='x', zorder=2, label='$ \hat{p}^{\mathrm{-}}_{t} $')
    ax1 = plt.gca()
    ax1.minorticks_off()
    ax1.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('Time [h]')
    ax1.set_xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    ax1.set_ylabel('Power [kW]')
    ax1.text(0.5, -0.28, '(a)', transform=ax1.transAxes, fontsize=12)
    ax1.legend()
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(SP_primal_sol['y_chg'], color='#2ca02c', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='#2ca02c', zorder=1, label='$ \hat{p}^{\mathrm{chg}}_{t} $')
    ax2.plot(SP_primal_sol['y_dis'], color='#9467bd', linestyle='--', marker='x', zorder=2, label='$ \hat{p}^{\mathrm{dis}}_{t} $')
    ax2 = plt.gca()
    ax2.minorticks_off()
    ax2.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Time [h]')
    ax2.set_xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24']) 
    ax2.set_ylabel('Power [kW]')
    ax2.text(0.5, -0.28, '(b)', transform=ax2.transAxes, fontsize=12)
    ax2.legend()
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(SP_primal_sol['y_cut'], color='#ff7f0e', linestyle='-', marker='o', markerfacecolor='none', markeredgecolor='#ff7f0e', zorder=1, label='$ \hat{w}^{\mathrm{cut}}_{t} $')
    ax3.plot(SP_primal_sol['y_add'], color='#17becf', linestyle='--', marker='x', zorder=2, label='$ \hat{w}^{\mathrm{add}}_{t} $')
    ax3 = plt.gca()
    ax3.minorticks_off()
    ax3.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Time [h]')
    ax3.set_xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'])
    ax3.set_ylabel('Power [kW]')
    ax3.text(0.5, -0.28, '(c)', transform=ax3.transAxes, fontsize=12)
    ax3.legend()
    plt.tight_layout()
    plt.savefig(dirname + 'svg/' + day + '_Simultaneous_check_IEEE.svg', format='svg')
    plt.savefig(dirname + 'png/' + day + '_Simultaneous_check_IEEE.png', format='png')
    plt.savefig(dirname + 'pdf/' + day + '_Simultaneous_check_IEEE.pdf', format='pdf')
    plt.close()

    data = np.column_stack((np.array(PV_worst_case), np.array(load_worst_case).flatten()))
    np.savetxt('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/result/worst_case_bM.csv', data, delimiter=',', header='PV_worst,load_worst', comments='', fmt='%.18f')

    data = {"Variable": [], "Value": []}

    # ----------------------------------------------------------------
    for v in SP_dual.model.getVars():
        data["Variable"].append(v.varName)
        data["Value"].append(v.x)

    df = pd.DataFrame(data)
    df.to_excel('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/result/bM_phi_all_worst.xlsx', index=False)

    print('cost fuel:', sum(final_x_cost_fuel) + sum(final_y_cost_fuel))
    print('cost res:', sum(final_x_cost_res))
    print('cost ESS:', sum(final_x_cost_ESS) + sum(final_y_cost_ESS))
    print('x cost cut:', sum(final_x_cost_cut))
    print('cost real:', sum(final_y_cost_fuel) + sum(final_y_cost_ESS) + sum(final_y_cost_cut) + sum(final_y_cost_add))
    print('y cut:', sum(final_y_cut))
    print('y cost cut:', sum(final_y_cost_cut))
    print('y add:', sum(final_y_add))
    print('y cost add:', sum(final_y_cost_add))        