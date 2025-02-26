# Developed by HyunSu-Shin on Feb. 2023
# This Code is developed and tested on Python 3.8.13

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import sys

np.set_printoptions(precision=6, suppress=True)

# https://midcdmz.nrel.gov/apps/sitehome.pl?site=NELHA#DOI
# https://midcdmz.nrel.gov/apps/daily.pl?site=NELHA&start=20121101&yr=2023&mo=2&dy=12
# https://PVpmc.sandia.gov/modeling-steps/1-weather-design-inputs/irradiance-and-insolation-2/direct-normal-irradiance/piecewise_decomp-models/

Local = {'local_latitude' : 19.728144, 'local_longitude' : 156.058936, 'standard_longitude' : 150, 'year' : 2018, 'month' : 7, 'day' : 4}
Angle = {'sun_k': 1367, 'panel_azimuth': 180, 'panel_tilted_angle': 25, 'rho': 0.2}
PV = {'a': -3.47, 'b': -0.0594, 'alpha': -0.0035, 'gamma': 0.02, 'EFF_STC': 1, 'Power_Capacity': 450}

# 각도 단위 변경
d2r = math.pi / 180
r2d = 180 / math.pi

class Photovoltaic:
    def __init__(self):
        self.Sim_time = 96 # 15min
        self.Weather_Forecast = np.genfromtxt('/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/Hawaii weather.csv', delimiter=',', skip_header=1, dtype=np.float32)
        self.real_PV = np.loadtxt("/Users/Andrew/OneDrive/Second brain/Programming/Python/Optimization/Robust generation dispatch/data/PV_for_scheduling.txt")
        self.GHI = self.Weather_Forecast[:,2]
        self.Tem = self.Weather_Forecast[:,3]
        self.WS = self.Weather_Forecast[:,4]
        
        self.new_GHI = np.zeros(self.Sim_time)
        self.new_Tem = np.zeros(self.Sim_time)
        self.new_WS = np.zeros(self.Sim_time)

        for i in range(95):
            self.new_GHI[i+1] = self.GHI[15*i]
            self.new_Tem[i+1] = self.Tem[15*i]
            self.new_WS[i+1] = self.WS[15*i]
        self.new_GHI[0] = self.new_GHI[1]
        self.new_Tem[0] = self.new_Tem[1]
        self.new_WS[0] = self.new_WS[1]    
        self.new_GHI[-1] = self.new_GHI[-2]
        self.new_Tem[-1] = self.new_Tem[-2]
        self.new_WS[-1] = self.new_WS[-2]

    def Local_Solar(self, Local):
        # 하와이
        self.local_latitude = Local['local_latitude']               # 위도
        self.local_longitude = Local['local_longitude']             # 경도
        self.standard_longitude = Local['standard_longitude']       # 자오선
        self.year = Local['year']
        self.month = Local['month']
        self.day = Local['day']
        self.x = np.array(range(24))                                # 시변
        self.hour = np.repeat(self.x, 4)           
        self.y = np.array([0, 15, 30, 45])
        self.min = np.tile(self.y, 24)

        # 균시차 (Equaiont of Time), 진태양시와 평균태양시의 차이
        self.day_of_year = np.array(datetime(self.year, self.month, self.day).timetuple().tm_yday)
        
        if (3 < self.month < 10) or (self.month == 3 and self.day >= 21) or (self.month == 9 and self.day <= 22):
             self.ecliptic_obliquity = 23.45
        else:
            self.ecliptic_obliquity = -23.45
        self.solar_declination = self.ecliptic_obliquity * np.pi / 180 * np.sin(2 * np.pi / 365 * (284 + self.day_of_year))

        if 1 <= self.day_of_year <= 106:
            self.EOT = -14.2 * np.sin(np.pi * (self.day_of_year + 7) / 111)
        elif 107 <= self.day_of_year <= 166:
            self.EOT = 4.0 * np.sin(np.pi * (self.day_of_year - 106) / 59)
        elif 167 <= self.day_of_year <= 246:
            self.EOT = -6.5 * np.sin(np.pi * (self.day_of_year - 166) / 80)
        else:
            self.EOT = 16.4 * np.sin(np.pi * (self.day_of_year - 247) / 113)

        # sandia - 이상
        # self.solar_time = self.hour  + self.EOT / 60 + (self.standard_longitude - self.local_longitude) / 15
        # self.hour_angle = r2d * np.pi * (12 - self.solar_time) / 12

        # # 시간각 (Hour Angle), 정남향일 때 0도 기준, 동쪽 [-], 서쪽 [+]
        self.local_hour_decimal = self.hour + self.min / 60
        self.delta_longitude = self.local_longitude - self.standard_longitude
        # 태양 적위 (Solar Declination), 태양의 중심과 지구의 중심을 연결하는 선과 지구 적도면이 이루는 각
        self.solar_time_decimal = (self.local_hour_decimal * 60 + 4 * self.delta_longitude + self.EOT) / 60
        self.solar_time_hour = np.asarray(self.solar_time_decimal, dtype = int)
        self.solar_time_min = (self.solar_time_decimal * 60) % 60
        self.hour_angle = (self.local_hour_decimal * 60 + 4 * self.delta_longitude + self.EOT) / 60 * 15 - 180


        # 태양 고도 (Solar Altitude), 태양과 지표면이 이루는 각
        self.solar_altitude = r2d * np.arcsin(np.cos(d2r * self.local_latitude) * np.cos(d2r * self.solar_declination) * np.cos(d2r * self.hour_angle) + np.sin(d2r * self.local_latitude) * np.sin(d2r * self.solar_declination))

        # 태양 천정각 (Solar Zenith)
        self.solar_zenith = r2d * np.arccos(np.sin(d2r * self.local_latitude) * np.sin(d2r * self.solar_declination) + np.cos(d2r * self.local_latitude) * np.cos(d2r * self.solar_declination) * np.cos(d2r * self.hour_angle))

        # 태양 방위각 - 북향을 기준으로
        self.solar_azimuth = r2d * np.arccos((np.sin(d2r * self.solar_declination) * np.cos(d2r * self.local_latitude) - np.cos(d2r * self.hour_angle) * math.cos(d2r * self.solar_declination) * np.sin(d2r * self.local_latitude)) / np.sin(d2r * (90 - self.solar_altitude)))

        # 태양 방위각 (Solar Azimuth) - 남향을 기준으로 동쪽 [+], 서쪽 [-]
        self.solar_azimuth_2 = r2d * np.arccos(np.sin(d2r * self.solar_altitude) * np.sin(d2r * self.local_latitude) - np.sin(d2r * self.solar_declination)) / (np.cos(d2r * self.solar_altitude) * np.cos(d2r * self.local_latitude))

        # 반사율
        self.cal_rho = 0.012 * self.solar_zenith - 0.04

    def POA(self, Angle):
        self.panel_azimuth = Angle['panel_azimuth']                         # 방위각, 정남형(180) -45~45
        self.panel_tilted_angle = Angle['panel_tilted_angle']               # 패널경사각-에기연, 제주 24

        # 경사면 입사각(The Angle of Incidence) 방법 1
        self.AOI = r2d * np.arccos(np.cos(d2r * self.solar_zenith) * np.cos(d2r * self.panel_tilted_angle) + np.sin(d2r * self.solar_zenith) * np.sin(d2r * self.panel_tilted_angle) * np.cos(d2r * self.solar_azimuth - d2r * self.panel_azimuth))
        
        # 법선면 직달일사량 (Direct Normal Irradiance)
        self.DNI = np.zeros(self.Sim_time)
        self.b = 2 * math.pi * self.day_of_year / 365
        self.Ea = Angle['sun_k'] + (1.00011 + 0.034221 * np.cos(self.b) + 0.00128 * np.sin(self.b) + 0.000719 * np.cos(2 * self.b) + 0.000077 * np.sin(2 * self.b))**2

        self.k_d = np.zeros(self.Sim_time)
        self.k_t = self.new_GHI / (self.Ea * np.cos(d2r * self.solar_zenith))

        for i in range(self.Sim_time):
            if self.k_t[i] <= 0.3:
                self.k_d[i] = 1.02 - 0.254 * self.k_t[i] + 0.0123 * np.sin(d2r * self.solar_altitude[i])
            elif 0.3 < self.k_t[i] < 0.78:
                self.k_d[i] = 1.4 - 1.749 * self.k_t[i] + 0.177 * np.sin(d2r * self.solar_altitude[i])
            elif self.k_t[i] >= 0.78:
                self.k_d[i] = 0.486 * self.k_t[i] - 0.182 * np.sin(d2r * self.solar_altitude[i])

        self.DHI = self.k_d * self.new_GHI
        self.DNI = (self.new_GHI - self.DHI) / np.cos(d2r * self.solar_zenith)
        
        # 경사면 직달일사량 (Plane of Array Beam component)
        self.POA_b = self.DNI * np.cos(d2r * self.AOI)

        # 경사면 확산일사량 (POA Sky-Diffuse component)
        self.POA_d = self.DHI * ((1 + np.cos(d2r * self.panel_tilted_angle))) / 2 + self.new_GHI * (0.12 * d2r * self.solar_zenith - 0.04) * (1 - np.cos(d2r * self.panel_tilted_angle)) / 2

        # 지표면 반사일사량 (Ground-Reflected component)
        self.rho = Angle['rho']
        self.POA_g = self.rho * self.new_GHI * ((1 - np.cos(d2r * self.panel_tilted_angle)) / 2)

        # 경사면 전일사량(Plane of Array Irradiance)
        self.Ir_POA = self.POA_b + self.POA_d + self.POA_g

    def Module_Tem(self, PV):
        # self.alpha = PV['alpha']   # \degc (crystalline Siicon (cSi)), -0.00415 (우리나라 모듈 출력온도계수 평균)
        # self.gamma = PV['gamma']   # free-standing installations = 0.02 / roof integrated systems = 0.056
        # self.T_m = self.new_Tem + self.gamma * self.Ir_POA

        # Sandia Model, Module type: Glass/cell/glass
        self.cons_a = PV['a']       # Mount: Open rack
        self.cons_b = PV['b']
        self.T_m = self.new_Tem + self.Ir_POA * np.exp(self.cons_a + self.cons_b * self.new_WS)

    def Normal_EFF(self):
        for i in range(self.Sim_time):
            if self.new_GHI[i] <= 50:
                self.EFF_norm[i] = 0.91
            elif 50 < self.new_GHI[i] <= 150:
                self.EFF_norm[i] = (0.95-0.91) / (150-50) * (self.new_GHI[i] - 50) + 0.91
            elif 150 < self.new_GHI[i] <= 250:
                self.EFF_norm[i] = (0.98-0.95) / (250-150) * (self.new_GHI[i] - 150) + 0.95
            elif 250 < self.new_GHI[i] <= 350:
                self.EFF_norm[i] = (0.99-0.98) / (350-250) * (self.new_GHI[i] - 250) + 0.98
            elif 350 < self.new_GHI[i] <= 450:
                self.EFF_norm[i] = (1.00-0.99) / (450-350) * (self.new_GHI[i] - 350) + 0.99 
            elif 450 < self.new_GHI[i] <= 550:
                self.EFF_norm[i] = (1.02-1.00) / (550-450) * (self.new_GHI[i] - 450) + 1.00
            elif 550 < self.new_GHI[i] <= 650:
                self.EFF_norm[i] = (1.03-1.02) / (650-550) * (self.new_GHI[i] - 550) + 1.02
            elif 650 < self.new_GHI[i] <= 750:
                self.EFF_norm[i] = (1.04-1.03) / (750-650) * (self.new_GHI[i] - 650) + 1.03
            elif 750 < self.new_GHI[i] <= 850:
                self.EFF_norm[i] = (1.03-1.04) / (850-750) * (self.new_GHI[i] - 750) + 1.04
            elif 850 < self.new_GHI[i] <= 950:
                self.EFF_norm[i] = (1.01-1.03) / (950-850) * (self.new_GHI[i] - 850) + 1.03
            elif 950 < self.new_GHI[i] <= 1050:
                self.EFF_norm[i] = (0.99-1.01) / (1050-950) * (self.new_GHI[i] - 950) + 1.01
            elif 1050 < self.new_GHI[i] <= 1150:
                self.EFF_norm[i] = (0.98-0.99) / (1150-1050) * (self.new_GHI[i] - 1050) + 0.99
            elif 1150 < self.new_GHI[i] <= 1250:
                self.EFF_norm[i] = (0.97-0.98) / (1250-1150) * (self.new_GHI[i] - 1150) + 0.98
            else:
                self.EFF_norm[i] = 0.97
        return self.EFF_norm

    def PV_Gen(self, PV):
        self.alpha = PV['alpha']
        self.P_nom =  PV['Power_Capacity']

        self.EFF_norm = np.zeros(self.Sim_time)
        self.EFF_norm = self.Normal_EFF()
        self.PV_output = self.EFF_norm * (1 + self.alpha * (self.T_m - 25)) * (self.Ir_POA / 1000) * self.P_nom
        # for i in range(self.Sim_time):
        #     if self.PV_output[i] > 0:
        #         self.PV_output[i] = self.PV_output[i]
        #     else:
        #         self.PV_output[i] = 0
        self.PV_output_g = self.EFF_norm * (1 + self.alpha * (self.new_Tem - 25)) * (self.new_GHI / 1000) * self.P_nom

        self.error = (self.real_PV - self.PV_output)
        self.rmse = 1 / np.sqrt(self.Sim_time) * np.sqrt(np.sum(self.error**2))
        self.relative_error = (self.PV_output - self.real_PV) / self.real_PV * 100
        self.MAE = 1 / self.Sim_time * np.sum(np.abs(self.error))

        self.PV_stack = np.zeros(self.Sim_time)
        for i in range(24-1):
                self.PV_stack[4*(i+1)] = self.PV_stack[4*i] + self.PV_output[4*i] + self.PV_output[4*i+1] + self.PV_output[4*i+2] + self.PV_output[4*i+3]
        self.time_h = np.zeros(self.Sim_time)
        for i in range(self.Sim_time-1):
            self.time_h[0] = 0
            self.time_h[i+1] = self.time_h[i] + 1
'''
        plt.style.use(['science'])
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False

        # 고도 및 방위각
        fig = plt.figure(1, figsize=(12,9))
        plt.xlabel('Time step(15m)', fontsize=18)
        plt.ylabel(r'Degree($^{\circ}$)', fontsize=18)
        # plt.suptitle("Solar Altitude and Azimuth", fontsize = 14)
        plt.plot(self.solar_altitude, '-', lw=2, color='mediumblue', label = 'Altitude')
        plt.plot(self.solar_azimuth, '-', lw=2, color='crimson', label = 'Azimuth')
        plt.tight_layout()
        fig.legend(fontsize=18)

        # 일사량 비교
        fig = plt.figure(2, figsize=(12,9))
        plt.xlabel('Time step(15m)', fontsize=18)
        plt.ylabel(r'Irradiance(W/${m^2}$)', fontsize=18)
        # plt.suptitle("Horizontal and Plane of array Irradiance", fontsize = 14)
        plt.plot(self.new_GHI, '--', lw=2, marker=10, color='mediumblue', label = 'Ir_global', zorder=2)
        plt.plot(self.Ir_POA, '-', lw=2, marker=11, color='crimson', label = 'Ir_local', zorder=1)
        plt.tight_layout()
        fig.legend(fontsize=18)

        # 온도 비교
        fig = plt.figure(3, figsize=(12,9))
        plt.xlabel('Time step(15m)', fontsize=18)
        plt.ylabel(r'Temperature($^{\circ}$C)', fontsize=18)
        # plt.suptitle("Temperature and Module Temperature", fontsize = 14)
        plt.plot(self.new_Tem, '--', lw=2, color='mediumblue', label = 'T_global', zorder=2)
        plt.plot(self.T_m, '-', lw=2, color='crimson', label = 'T_local', zorder=1)
        plt.tight_layout()
        fig.legend(fontsize=18)

        fig = plt.figure(4, figsize=(12,9))
        plt.xlabel('Time step(15m)', fontsize=18)
        plt.ylabel('PV output(kW)', fontsize=18)
        plt.plot(self.real_PV, '--', lw=2, color='mediumblue', label = 'Real')
        plt.plot(self.PV_output, '-', lw=2, color='crimson', label = 'Predict')
        plt.tight_layout()
        fig.legend(fontsize=18)

        # 일사량 및 온도와 발전량 관계
        fig, ax1 = plt.subplots(1, figsize=(12,9))    # 직접 figure 객체를 생성
        ax1.plot(self.PV_output , 'o-', lw=2, color='crimson', label='PV output', zorder=2)
        ax1.plot(self.new_GHI , '-', lw=2, color='mediumblue', label='GHI', zorder=3)
        ax1.set_xlabel('Time step(15m)', fontsize=18)
        ax1.set_ylabel(r'GHI(W/${m^2}$) & PV output(kWh)', fontsize=18)
        # plt.suptitle("PV Output Correlation", fontsize = 14)
        # ax1.minorticks_on()
        ax2 = ax1.twinx()
        ax2.plot(self.new_Tem , '--', lw=2, color='#FA7268', label='Tem', zorder=1)
        ax2.set_ylabel(r'Temperature($^{\circ}$C)', fontsize=18)       
        ax1.patch.set_visible(False)
        ax1.set_zorder(1)
        lines = []
        labels = []
        for ax in fig.axes:
            axLine , axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)
        
        ax1.legend(lines, labels, loc="best", fontsize=18)
        
        # 발전량 계산
        fig, ax1 = plt.subplots(1, figsize=(12,9))
        ax1.plot(self.PV_output, '-', marker='o', lw=2, color='crimson', label='PV output_local', zorder=2)
        # ax1.plot(self.Predict_PV, '--', lw=2, color='forestgreen', alpha=0.7, label='Predcition', zorder=1)
        # ax1.plot(self.PV_output_g, '-', lw=2, marker=10, color='mediumblue', label='PV output_global', zorder=0)
        ax1.plot(self.T_m, '-.', lw=2, color='#FA7268', label='Module temperate', zorder=1)             
        ax1.hlines(y=self.P_nom, xmin=0, xmax=96, color='blueviolet', linestyles='solid', label='Capacity', lw=2, zorder=1)
        ax1.set_xlabel('Time step(15m)', fontsize=18)
        ax1.set_ylabel(r'PV power(kW) & Temperature($^{\circ}$C)', fontsize=18)
        ax1.plot(self.Ir_POA, '--', lw=2, color='mediumblue', label=' POA irradiance', zorder=3) 
        # plt.suptitle("PV output according to POA Irradiance and Module Temperature ", fontsize = 14)
        ax1.tick_params(which='major', axis='both', direction='in')
        ax1.tick_params(which='minor', axis='both', direction='in')      
        ax1.patch.set_visible(False)
        ax1.set_zorder(1)

        ax2 = ax1.twinx()   
        ax2.bar(self.time_h, self.PV_stack, color='lightsteelblue', label='Total output', width=2)    
        # ax2.plot(self.new_GHI, '-.', lw=2, label='new_GHI', zorder=3)
        ax2.set_ylabel(r'Irradiance(W/${m^2}$) & Total output(kWh)', fontsize=18)
        ax2.set_zorder(-1)

        lines = []
        labels = []
        for ax in fig.axes:
            axLine , axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)
        
        ax1.legend(lines, labels, loc="center left", fontsize=18)
        
        plt.show()
'''
PV_forecast = Photovoltaic()
PV_forecast.Local_Solar(Local)
PV_forecast.POA(Angle)
PV_forecast.Module_Tem(PV)
PV_forecast.PV_Gen(PV)

result = np.stack((PV_forecast.solar_altitude, PV_forecast.solar_azimuth, PV_forecast.solar_zenith, PV_forecast.AOI, PV_forecast.new_GHI, PV_forecast.k_t, PV_forecast.k_d, PV_forecast.DHI, PV_forecast.DNI, PV_forecast.POA_b, PV_forecast.POA_d, PV_forecast.POA_g, PV_forecast.Ir_POA, PV_forecast.new_Tem, PV_forecast.T_m, PV_forecast.EFF_norm, PV_forecast.PV_output, PV_forecast.real_PV, PV_forecast.PV_stack), axis=1)
result = pd.DataFrame(result, columns = ['Altitude', 'Azimuth', 'Zenith', 'AOI', 'GHI', 'k_t', 'k_d', 'DHI', 'DNI', 'POA_b', 'POA_d', 'POA_g','Ir_POA', 'Tem', 'T_m', 'EFF_norm', 'PV_Gen', 'Real', 'Total'])

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# print(round(result, 4))
# print(PV_forecast.rmse)
# print(PV_forecast.MAE)

# error = np.zeros(60)
# for i in range(20, 80):
#     error[i-20] = PV_forecast.relative_error[i]
# plt.figure()
# plt.plot(error, label='error')
# plt.legend()
# plt.tight_layout()
# plt.show()
