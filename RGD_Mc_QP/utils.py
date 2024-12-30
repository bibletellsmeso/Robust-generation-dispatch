"""
Utils file containing several functions to be used.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import sympy as sp

# from sklearn.preprocessing import StandardScaler
from Params import PARAMETERS

def check_ESS(SP_primal_sol: dict):
    """
    Check if there is any simultanenaous charge and discharge of the ESS.
    :param SP_primal_sol: solution of the SP primal, dict with at least the keys arguments y_chg and y_dis.
    :return: number of simultanenaous charge and discharge.
    """
    df_check = pd.DataFrame(SP_primal_sol['y_chg'], columns=['y_chg'])
    df_check['y_dis'] = SP_primal_sol['y_dis']

    nb_count = 0
    for i in df_check.index:
        if (df_check.loc[i]['y_chg'] > 0) and (df_check.loc[i]['y_dis'] > 0):
            nb_count += 1
    return nb_count

def dump_file(dir: str, name: str, file):
    """
    Dump a file into a picke.
    """
    file_name = open(dir + name + '.pickle', 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def read_file(dir: str, name: str):
    """
    Read a file dumped into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'rb')
    file = pickle.load(file_name)
    file_name.close()

    return file

cost_a = PARAMETERS['cost']['DG_a']
cost_b = PARAMETERS['cost']['DG_b']
cost_c = PARAMETERS['cost']['DG_c']
cost_PV_cut_pre = PARAMETERS['cost']['C_PV_cut_pre']
cost_PV_cut_re = PARAMETERS['cost']['C_PV_cut_re']
cost_PV_add_re = PARAMETERS['cost']['C_PV_add_re']

def FC(p):
    return(cost_a * p * p + cost_b * p + cost_c)

def PC_PV(g):
    return cost_PV_cut_pre * g * g

def RC_PV(g):
    return cost_PV_cut_re * g * g

def RA_PV(g):
    return cost_PV_add_re * g * g

def PWL(PWL_num, lb, ub, quadratic_func):
    x = []
    y = []
    for i in range(PWL_num + 1):
        x.append(lb + (ub - lb) * i / PWL_num)
        y.append(quadratic_func(x[i]))
    return x, y

def PWL_val(PWL_num, lb, ub, quadratic_func, x):
    interval = (ub - lb) / PWL_num
    y = 0
    for i in range(PWL_num):
        if PWL(PWL_num, lb, ub, quadratic_func)[0][i] <= x < PWL(PWL_num, lb, ub, quadratic_func)[0][i + 1]:
            y = (PWL(PWL_num, lb, ub, quadratic_func)[1][i + 1] - PWL(PWL_num, lb, ub, quadratic_func)[1][i]) / interval * (x - PWL(PWL_num, lb, ub, quadratic_func)[0][i]) + PWL(PWL_num, lb, ub, quadratic_func)[1][i]
        else:
            pass
    return y