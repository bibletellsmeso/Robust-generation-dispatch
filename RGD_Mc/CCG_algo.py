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
from CCG_SP_Mc import CCG_SP
from SP_primal_LP import SP_primal_LP
from planner_MILP import Planner_MILP
from Data_read import *
from root_project import ROOT_DIR
from Params import PARAMETERS
from utils import read_file, dump_file, check_ESS

def ccg_algo(dir:str, tol:float, power:np.array, reserve_pos:np.array, reserve_neg:np.array, charge:np.array, discharge:np.array, SOC:np.array, curtailment:np.array,
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
    tolerance = 1e20

    # with CCG the convergence is stable.
    tolerance_list = [tolerance] * 2
    iteration = 1
    ESS_count_list = []
    ESS_charge_discharge_list = []
    PV_count_list = []
    PV_cut_add_list = []
    max_iteration = 10

    while all(i < tol for i in tolerance_list) is not True and iteration < max_iteration:
        logfile = ""
        if log:
            logfile = dir + 'logfile_' + str(iteration) + '.log'
        if printconsole:
            print('i= %s solve SP dual' % (iteration))

        # ------------------------------------------------------------------------------------------------------------------
        # 1. SP part
        # ------------------------------------------------------------------------------------------------------------------

        # 1.1 Solve the SP and get the worst RG and load trajectory to add the new constraints of the MP
        SP_dual = CCG_SP(PV_forecast=PV_forecast, load_forecast=load_forecast, power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg,
                         charge=charge, discharge=discharge, SOC=SOC, curtailment=curtailment,
                         M_PV_best=M_PV_best, M_PV_worst=M_PV_worst, M_cut_best=M_cut_best, M_cut_worst=M_cut_worst, M_load_best=M_load_best, M_load_worst=M_load_worst, GAMMA=GAMMA, PI=PI)
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
        SP_primal = SP_primal_LP(PV_trajectory=PV_worst_case_from_SP, load_trajectory=load_worst_case_from_SP,
                                power=power, reserve_pos=reserve_pos, reserve_neg=reserve_neg, charge=charge, discharge=discharge, SOC=SOC, curtailment=curtailment)
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

            PV_nb_count = check_ESS(SP_primal_sol=SP_primal_sol)
            if PV_nb_count > 0:
                PV_cut_add_list.append([iteration, SP_primal_sol['y_cut'], SP_primal_sol['y_add']])
            else:
                PV_nb_count = float('nan')
            PV_count_list.append(PV_nb_count)
            if printconsole:
                print('     i = %s : %s simultaneous curtailment and addition' % (iteration, PV_nb_count))

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

    return power, reserve_pos, reserve_neg, charge, discharge, SOC, curtailment, df_objectives, conv_inf, \
        x_cost_fuel, x_cost_res, x_cost_ESS, x_cost_cut

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 24

# NB periods
nb_periods = 96

# Solver parameters
solver_param = dict()
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 10
solver_param['Threads'] = 1

# Convergence threshold between MP and SP objectives
conv_tol = 5
printconsole = True

day = '2018-07-04'

# --------------------------------------
# Static RO parameters
GAMMA = 96 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
PI = 96

# quantile from NE or LSTM
PV_Sandia = True

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    # Create folder
    dirname = '/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/RGD_Mc/export_CCG/'
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

    M_PV_best = [x for x in read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/', name=day+'_bM_phi_PV_best')]
    M_PV_worst = [x for x in read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/', name=day+'_bM_phi_PV_worst')]
    M_cut_best = [x for x in read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/', name=day+'_bM_phi_cut_best')]
    M_cut_worst = [x for x in read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/', name=day+'_bM_phi_cut_worst')]
    M_load_best = [x for x in read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/', name=day+'_bM_phi_load_best')]
    M_load_worst = [x for x in read_file(dir='/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/', name=day+'_bM_phi_load_worst')]

    nb_periods = PV_pos.shape[0]

    # plot style
    plt.style.use(['science'])
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    x_index = [i for i in range(0, nb_periods)]

    
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
        final_x_cost_res, final_x_cost_ESS, final_x_cost_cut \
            = ccg_algo(dir=dirname, tol=conv_tol, power=power_ini, reserve_pos=reserve_pos_ini, reserve_neg=reserve_neg_ini,
                       charge=charge_ini, discharge=discharge_ini, SOC=SOC_ini, curtailment=curtailment_ini,
                       GAMMA=GAMMA, PI=PI, solver_param=solver_param, day=day, printconsole=printconsole)
    df_objectives.to_csv(dirname + day + 'obj_MP_SP_' + '.csv')

    print('-----------------------------------------------------------------------------------------------------------')
    print('CCG: day %s GAMMA %s PI %s' % (day, GAMMA, PI))
    print('-----------------------------------------------------------------------------------------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # Get the final worst case RG generation trajectory computed by the Sub Problem
    # ------------------------------------------------------------------------------------------------------------------

    # Get the worst case related to the last engagement plan by using the Sub Problem dual formulation
    SP_dual = CCG_SP(PV_forecast=PV_forecast, load_forecast=load_forecast, power=final_power, reserve_pos=final_reserve_pos, reserve_neg=final_reserve_neg, 
                     charge=final_charge, discharge=final_discharge, SOC=final_SOC, curtailment=final_curtailment,
                     M_PV_best=M_PV_best, M_PV_worst=M_PV_worst, M_cut_best=M_cut_best, M_cut_worst=M_cut_worst, M_load_best=M_load_best, M_load_worst=M_load_worst, GAMMA=GAMMA, PI=PI)
    SP_dual.solve(LogToConsole=False, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=10)
    SP_dual_sol = SP_dual.store_solution()

    # Compute the worst RG, load path from the SP dual solution
    PV_worst_case = [PV_forecast[i] + PV_pos[i] * SP_dual_sol['epsilon_pos'][i] - PV_neg[i] * SP_dual_sol['epsilon_neg'][i] for i in range(nb_periods)]
    load_worst_case = [load_forecast[i] + load_pos[i] * SP_dual_sol['delta_pos'][i] - load_neg[i] * SP_dual_sol['delta_neg'][i] for i in range(nb_periods)]
    # dump_file(dir=dirname, name=day + '_PV_worst_case', file=PV_worst_case)
    # dump_file(dir=dirname, name=day + '_load_worst_case', file=load_worst_case)

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
            plt.close('all')

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

    plt.figure(figsize = (16,9))
    plt.plot(error_MP_SP, marker=10, markersize=10, linewidth=2, label='|MP - SP dual| $')
    plt.plot(error_SP, marker=11, markersize=10, linewidth=2, label='|SP primal - SP dual| $')
    plt.plot(100 * np.asarray(conv_inf['mipgap']), label='SP Dual mipgap %')
    plt.xlabel('Iteration $j$', fontsize=FONTSIZE)
    plt.ylabel('Gap', fontsize=FONTSIZE)
    plt.ylim(-1, 50)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    # plt.tight_layout()
    plt.savefig(dirname + day + 'error_conv_' + pdfname + '.pdf')
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

    plt.figure(figsize=(16,9))
    plt.plot(PV_forecast, color='green', marker="D", markersize=6, zorder=3, linewidth=4, label='$ w^{*}_{t} $')
    plt.plot(load_forecast, color='darkgoldenrod', marker='o', markersize=6, zorder=1, linewidth=4,  label='$ l^{*}_{t} $')
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    # plt.tight_layout()
    plt.savefig(dirname + day + '_Forecast_' + pdfname + '.pdf')
    plt.savefig(dirname + day + '_Forecast', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(SP_dual_sol['epsilon_pos'], label=r"$\epsilon^+$")
    plt.plot(SP_dual_sol['epsilon_neg'], label=r"$\epsilon^-$")
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('PV uncertainty variables')

    plt.subplot(2, 1, 2)
    plt.plot(SP_dual_sol['delta_pos'], label=r"$\delta^+$")
    plt.plot(SP_dual_sol['delta_neg'], label=r"$\delta^-$")
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.title('Load uncertainty variables')

    plt.tight_layout()
    plt.savefig('/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/RGD_Mc/export_CCG/PV_Sandia' + day + '_uncertainty_variables.png', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(PV_worst_case, color='crimson', marker="o", markersize=8, linewidth=4, zorder=3, label='$ \hat{p}_t$')
    plt.plot(PV_forecast, 'steelblue', linestyle='solid', marker="s", markersize=8, linewidth=4, label='$ w^{*}_{t} $', zorder=1)
    plt.plot(PV_forecast + PV_pos, 'dimgrey', linestyle=(0, (5, 5)), linewidth=2, label='$ w^{*}_{t} + w^{+} $', zorder=1)
    plt.plot(PV_forecast - PV_neg, 'dimgrey', linestyle=(0, (5, 10)), linewidth=2, label='$ w^{*}_{t} - w^{-} $', zorder=1)
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    # plt.tight_layout()
    plt.savefig(dirname + day + '_PV_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + day + '_PV_trajectory', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(load_worst_case, color='crimson', linestyle='solid', marker="o", markersize=8, linewidth=4, label='$ \hat{l}_t$')
    plt.plot(load_forecast, 'orange', linestyle='solid', marker="d", markersize=8, linewidth=4, label='$ l^{*}_{t} $', zorder=1)
    plt.plot(load_forecast + load_pos, 'dimgrey', linestyle=(0, (5, 5)), linewidth=2, label="$ l^{*}_{t} + l^{+} $", zorder=1)
    plt.plot(load_forecast - load_neg, 'dimgrey', linestyle=(0, (5, 10)), linewidth=2, label="$ l^{*}_{t} - l^{-} $", zorder=1)
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    # plt.tight_layout()
    plt.savefig(dirname + day + '_load_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + day + '_load_trajectory', dpi=300)
    # plt.close('all')

    a = SP_primal_sol['y_chg']
    b = SP_primal_sol['y_dis']
    c = np.zeros(95)
    c[0] = 187.5
    c[94] = 187.5
    for i in range(1,95):
        c[i] = c[i-1] + (a[i] * 0.93 - b[i] / 0.93)/4
    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(c, linewidth=2, label='SOC')
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('SOC (kWh)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    # plt.tight_layout()
    plt.savefig(dirname + day + '_SOC_' + pdfname + '.pdf')
    plt.savefig(dirname + day + '_SOC', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(final_power, color='firebrick', marker='o', markersize=6, zorder=1, linewidth=3, label='DG output')
    plt.plot(([hs + eg for hs, eg in zip(final_power, final_reserve_pos)]), marker='^', markersize=1, zorder=3, linewidth=2, alpha=0.5, label='DG reserve up')
    plt.plot(([hs - eg for hs, eg in zip(final_power, final_reserve_neg)]), marker='v', markersize=1, zorder=3, linewidth=2, alpha=0.5, label='DG reserve down')
    plt.plot(PV_forecast, color='mediumblue', alpha=0.8, linestyle="--", markersize=6, zorder=3, linewidth=3, label='PV prediction')
    plt.plot(([hs - eg for hs, eg in zip(PV_forecast, final_curtailment)]), color='green', marker="D", markersize=6, zorder=3, linewidth=3, label='PV output')
    plt.plot(([hs - eg for hs, eg in zip(final_discharge, final_charge)]), color='gold', markersize=6, zorder=1, linewidth=3,  label='ESS output')
    plt.plot(load_forecast, color='darkgoldenrod', marker='^', markersize=6, zorder=1, linewidth=3, label= 'Load demand')
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=FONTSIZE)
    plt.savefig(dirname + day + '_Pre-dispatch_' + pdfname + '.pdf', dpi=300)
    plt.savefig(dirname + day + '_Pre-dispatch.png', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(([hs + shs - egg for hs, shs, egg in zip(final_power, SP_primal_sol['y_pos'], SP_primal_sol['y_neg'])]), color='firebrick', marker='o', markersize=6, zorder=1, linewidth=3, label='DG output')
    plt.plot(PV_worst_case, color='mediumblue', marker='v', alpha=0.8, linestyle="--", markersize=6, zorder=3, linewidth=3, label='PV worst scenario')
    plt.plot(([hs - shs - eg + egg for hs, shs, eg, egg in zip(SP_primal_sol['y_PV'], final_curtailment, SP_primal_sol['y_cut'], SP_primal_sol['y_add'])]), color='green', marker="D", markersize=6, zorder=2, linewidth=3, label='PV output')
    plt.plot(([shs - hs + egg - eg for shs, hs, egg, eg in zip(final_discharge, final_charge, SP_primal_sol['y_dis'], SP_primal_sol['y_chg'])]), color='gold', markersize=6, zorder=2, linewidth=3, label='ESS output')
    plt.plot(load_worst_case, color='darkgoldenrod', marker='^', markersize=6, zorder=1, linewidth=3, label= 'Load worst scenario')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=2, zorder=1)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=FONTSIZE)
    plt.savefig(dirname + day + '_Result_' + pdfname + '.pdf', dpi=300)
    plt.savefig(dirname + day + '_Result.png', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(SP_primal_sol['y_chg'], linewidth=2, label='real-time charge')
    plt.plot(SP_primal_sol['y_dis'], linewidth=2, label='real-time discharge')
    # plt.ylim(0, PARAMETERS['ES']['capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    # plt.tight_layout()
    plt.savefig(dirname + day + 'realtime_charge_discharge_' + pdfname + '.pdf')
    plt.savefig(dirname + day + 'realtime_charge_discharge_', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(SP_primal_sol['y_cut'], linewidth=2, label='RT curtailment')
    plt.plot(SP_primal_sol['y_add'], linewidth=2, label='RT revise')
    # plt.ylim(0, PARAMETERS['ES']['capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    # plt.tight_layout()
    # plt.savefig(dirname + day + 'realtime_charge_discharge_' + pdfname + '.pdf')
    plt.savefig(dirname + day + 'realtime_curtailment_', dpi=300)
    # plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(final_reserve_pos, linewidth=2, label='reserve pos')
    plt.plot(final_reserve_neg, linewidth=2, label='reserve neg')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    # plt.tight_layout()
    plt.savefig(dirname + day + 'reserve capacity_' + pdfname + '.pdf')
    # plt.close('all')

    # data = np.column_stack(np.array(PV_worst_case, np.array(load_worst_case).flatten()))
    # np.savetxt('worst.csv', data, delimiter=',', header='PV_worst, load_worst', comments='', fmt='%.2f')

    # phi_data = np.column_stack((np.array(SP_dual_sol['phi_PV']), np.array(SP_dual_sol['phi_cut']), np.array(SP_dual_sol['phi_load']).flatten()))
    # np.savetxt('Mc_phi.csv', phi_data, delimiter=',', header='phi_PV,phi_cut,phi_load', comments='', fmt='%.2f')

    # Load data from CSV files
    # phi_worst_from_bM = pd.read_csv('/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/bM_phi_worst.csv', header=0)  # Assuming header is in the first row

    # Define LaTeX-style labels
    # labels = {
    #     'phi_PV': r'$\phi^{\mathrm{PV}}_t$',
    #     'phi_cut': r'$\phi^{\mathrm{cut}}_t$',
    #     'phi_load': r'$\phi^{\mathrm{load}}_t$'
    # }

    # Plot the data
    # fig, axs = plt.subplots(len(labels), 1, figsize=(16, len(labels) * 3), sharex=True)

    # for i, (col, label) in enumerate(labels.items()):
    #     axs[i].plot(phi_bM[col], label=f'{label} (big-M)', linestyle='-', linewidth=4)
    #     axs[i].plot(phi_Mc[col], label=f'{label} (McCormick)', linestyle='--', linewidth=4)
    #     axs[i].set_ylabel(f'{label} values', fontsize=FONTSIZE)
    #     axs[i].legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=FONTSIZE)
    #     axs[i].tick_params(axis='y', labelsize=FONTSIZE)  # Set y-axis tick font size


    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    # plt.tight_layout()
    plt.savefig(dirname + day + '_Phi_' + pdfname + '.pdf', dpi=300)
    plt.savefig(dirname + day + '_Phi.png', dpi=300)
    # plt.close('all')

    data = np.column_stack((np.array(PV_worst_case), np.array(load_worst_case).flatten()))
    np.savetxt('/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/worst_case_Mc.csv', data, delimiter=',', header='PV_worst, load_worst', comments='', fmt='%.18f')

    data = {"Variable": [], "Value": []}

    # ----------------------------------------------------------------
    for v in SP_dual.model.getVars():
        data["Variable"].append(v.varName)
        data["Value"].append(v.x)

    df = pd.DataFrame(data)
    df.to_excel('/Users/PSL/OneDrive/Programing/Python/Robust generation dispatch/result/Mc_phi_all_Mc.xlsx', index=False)