import os
import pandas as pd
import numpy as np
import random
from PV_model import PV_forecast
import matplotlib.pyplot as plt
np.set_printoptions(precision=6, suppress=True)

# os.listdir: 전체 파일목록 가져오기, endswith(): 특정 확장자 파일인지 확인
file_list = [i for i in os.listdir('./') if i.endswith('.xls')]

for i in file_list:
	SMP = pd.read_excel('{}'.format(i), sheet_name='Sheet1')
	SMP = SMP.drop(index=[0, 1, 26, 27, 28], axis=0)
	SMP = SMP.drop(labels=['육지 SMP 목록'], axis=1)
	SMP = SMP.drop(labels=['Unnamed: {}'.format(i) for i in range(1, 7)], axis=1)
	SMP = np.array(SMP).flatten()
	SMP = np.tile(SMP, reps=1)
	SMP = np.repeat(SMP, repeats=4, axis=0)/4

class data_read:
	def __init__(self):
		self.Sim_time = 96
		self.N_PWL = 10
		self.RTE = 0.93

		self.kpx_PV_data = pd.read_csv('/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/data/KPX_PV.csv', sep=',', names=['Source', 'Location', 'Date', 'Hour', 'Power'], dtype={'Date': str, 'Hour': str, 'Power': str}, encoding='CP949')[1:] # dtype = DataFrame
		self.kpx_PV_data = pd.DataFrame(self.kpx_PV_data, columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

		self.kpx_load = pd.read_csv('/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/data/KPX_Load.csv', sep=',', names=['Date', 'Load_1', 'Load_2', 'Load_3', 'Load_4', 'Load_5', 'Load_6', 'Load_7', 'Load_8', 'Load_9', 'Load_10', 'Load_11',
							      'Load_12', 'Load_13', 'Load_14', 'Load_15', 'Load_16', 'Load_17', 'Load_18', 'Load_19', 'Load_20', 'Load_21', 'Load_22', 'Load_23', 'Load_0'], encoding='CP949')[1:]
		self.kpx_load = self.kpx_load.drop(['Date'], axis=1).to_numpy(dtype=np.float32) / 1000

		self.kpx_PV = []
		for i in range(24):
			self.kpx_PV.append(self.kpx_PV_data[self.kpx_PV_data[:, 0] == i, -1])

		self.PV_var = np.array([])
		self.WT_var = np.array([])
		self.load_var = np.array([])

		for i in range(24):
			self.PV_var = np.append(self.PV_var, np.nanvar(self.kpx_PV[i] / np.max(self.kpx_PV)))
			self.load_var = np.append(self.load_var, np.nanvar(self.kpx_load[:, i - 1] / np.max(self.kpx_load)))

		self.PV_pred = np.array(pd.read_csv('/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/data/PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
		self.PV_pred[self.PV_pred < 0] = 0

		self.load_pred = np.array(pd.read_csv('/Users/Andrew/OneDrive/Programming/Python/Optimization/Robust generation dispatch/data/Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
		self.load_pred = self.load_pred

		self.PV_pos = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)
		self.PV_neg = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)

		self.load_pos = self.load_pred * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)
		self.load_neg = self.load_pred * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)

		self.PV_lb = self.PV_pred - self.PV_neg
		self.PV_ub = self.PV_pred + self.PV_pos
		self.load_lb = self.load_pred - self.load_neg
		self.load_ub = self.load_pred + self.load_pos

data = data_read()