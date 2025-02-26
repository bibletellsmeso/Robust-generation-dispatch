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

		self.kpx_PV_data = pd.read_csv('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/KPX_PV.csv', sep=',', names=['Source', 'Location', 'Date', 'Hour', 'Power'], dtype={'Date': str, 'Hour': str, 'Power': str}, encoding='CP949')[1:] # dtype = DataFrame
		self.kpx_PV_data = pd.DataFrame(self.kpx_PV_data, columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

		self.kpx_load = pd.read_csv('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/KPX_Load.csv', sep=',', names=['Date', 'Load_1', 'Load_2', 'Load_3', 'Load_4', 'Load_5', 'Load_6', 'Load_7', 'Load_8', 'Load_9', 'Load_10', 'Load_11',
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

		self.PV_pred = np.array(PV_forecast.PV_output)
		# self.PV_pred = np.array(pd.read_csv('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
		self.PV_pred[self.PV_pred < 0] = 0

		self.load_pred = np.array(pd.read_csv('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
		self.load_pred = self.load_pred

		# self.PV_pos = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)
		# self.PV_neg = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)

		self.loaded_pv_stats = pd.read_pickle("pv_stats_local.pkl")
		self.loaded_ramping_stats = pd.read_pickle("ramping_stats_local.pkl")
		self.PV_median = np.array(self.loaded_pv_stats["median"])
		self.PV_median[self.PV_median == 0] = 1E-6

		self.PV_pos = self.PV_pred * np.array(self.loaded_pv_stats["upper_95"]) / self.PV_median
		self.PV_pos[self.PV_pos >= 600] = 600
		self.PV_pos = self.PV_pos - self.PV_pred
		self.PV_neg = self.PV_pred * np.array(self.loaded_pv_stats["lower_95"]) / self.PV_median
		self.PV_neg[self.PV_neg <= 0] = 0
		self.PV_neg = self.PV_pred - self.PV_neg

		self.PV_ramp_pos = np.array(self.loaded_ramping_stats["upper_99"])
		self.PV_ramp_pos = self.PV_ramp_pos * self.PV_pred / self.PV_median
		self.PV_ramp_pos = self.PV_ramp_pos[1:]
		self.PV_ramp_neg = np.array(self.loaded_ramping_stats["lower_99"])
		self.PV_ramp_neg = self.PV_ramp_neg * self.PV_pred / self.PV_median
		self.PV_ramp_neg = self.PV_ramp_neg[1:]
		for i in range(95):
			if self.PV_ramp_pos[i] + self.PV_pred[i] > 600:
				self.PV_ramp_pos[i] = 600 - self.PV_pred[i]
			elif self.PV_pred[i] - self.PV_ramp_neg[i] < 0:
				self.PV_ramp_neg[i] = self.PV_pred[i]
		# self.PV_ramp_neg[i]

		self.load_pos = self.load_pred * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)
		self.load_neg = self.load_pred * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)

		self.PV_lb = self.PV_pred - self.PV_neg
		self.PV_lb[self.PV_lb <= 1E-6] = 1E-5
		self.PV_ub = self.PV_pred + self.PV_pos

		self.load_lb = self.load_pred - self.load_neg
		self.load_ub = self.load_pred + self.load_pos



		# self.PV_ramp_pos = self.ramping_sigma_pos
		# self.PV_ramp_neg = self.ramping_sigma_neg
		self.load_ramp_pos = [60] * 96
		self.load_ramp_neg = [60] * 96

		self.a = [np.nan] * 96
		self.b = [np.nan] * 96
		for i in range(1, 96):
			self.a[i] = self.PV_pred[i] + self.PV_ramp_neg[i-1]
			self.b[i] = self.PV_pred[i] + self.PV_ramp_pos[i-1]
		print(self.a)
		print(self.b)
		plt.figure()
		plt.plot(self.PV_pred, marker='D', zorder=3, label='forecast')
		plt.plot(self.PV_pred + self.PV_pos, zorder=1, marker='x', label='95 quantile')
		plt.plot(self.PV_pred - self.PV_neg, zorder=1, marker='o', label='-95 quantile')
		plt.plot(self.a, zorder=4, marker='v', label='ramp down limit')
		plt.plot(self.b, zorder=4, marker='^', label='ramp up limit')
		# plt.fill_between(range(len(self.PV_pred)), self.PV_pred - PV_neg, self.PV_pred + PV_pos, color='#1f77b4', alpha=0.1)
		plt.xlabel('Time [h]')
		plt.ylabel('Power [kW]')
		ax = plt.gca()
		ax.minorticks_off()
		ax.grid(True, which='major', linestyle='-', color='#7f7f7f', linewidth=0.5, alpha=0.5)
		plt.legend()
		plt.tight_layout()
		plt.show()
		for i in range(96):
			print(f"i = {i} | {self.PV_pred[i] - self.PV_neg[i]:.3f} ≤ {self.a[i]:.3f} ≤ {self.PV_pred[i]:.3f} ≤ {self.b[i]:.3f} ≤ {self.PV_pred[i] + self.PV_pos[i]:.3f}")
		print(f"PV_ramp_pos[71] = {self.PV_ramp_pos[71]}")
		print(f"PV_ramp_pos[72] = {self.PV_ramp_pos[72]}")
		print(f"PV_ramp_pos[73] = {self.PV_ramp_pos[73]}")
		print(f"PV_ramp_neg[71] = {self.PV_ramp_neg[71]}")
		print(f"PV_ramp_neg[72] = {self.PV_ramp_neg[72]}")
		print(f"PV_ramp_neg[73] = {self.PV_ramp_neg[73]}")
		print(f"PV_pred[71] = {self.PV_pred[71]}")
		print(f"PV_pred[72] = {self.PV_pred[72]}")
		print(f"PV_pred[73] = {self.PV_pred[73]}")


data = data_read()