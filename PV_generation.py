import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 데이터 경로 설정 (경로 수정)
data_folder = r"C:\Users\Andrew\OneDrive\Second brain\Programming\Python\Optimization\Robust generation dispatch\data"
path_pattern = os.path.join(data_folder, "25109_21.29_-157.86_*.csv")  # 모든 CSV 파일 찾기

# 📌 태양광 발전량 계산 공식
def calculate_pv_generation(E, T, E0=1000, P0=600, T0=25, gamma_P=-0.0035):
    pv_power = (E / E0) * P0 * (1 + gamma_P * (T - T0))
    pv_power = np.clip(pv_power, 0, P0)  # 0W ~ 600W 범위로 제한
    return pv_power

# 📌 데이터 불러오기 및 병합
def load_and_merge_data(path_pattern):
    files = glob.glob(path_pattern)  # 파일 리스트 가져오기
    all_data = []

    print(f"🔍 찾은 파일 개수: {len(files)}")
    if len(files) == 0:
        print("❌ 파일을 찾지 못했습니다. 경로를 확인하세요.")
        return None

    for file in files:
        try:
            file_size = os.path.getsize(file)  # 파일 크기 확인
            if file_size == 0:
                print(f"⚠️ 경고: {file} 파일이 비어 있음. 건너뜀.")
                continue  # 빈 파일은 건너뛴다.

            df = pd.read_csv(file, skiprows=1)  # 첫 번째 행을 버리고 두 번째 행을 컬럼명으로 사용
            df.columns = df.iloc[0]  #  이후 첫 번째 행을 컬럼명으로 설정
            df = df[1:].reset_index(drop=True)  # 컬럼명으로 설정한 뒤 첫 번째 행 제거
            df = df.apply(pd.to_numeric, errors='ignore')  # 데이터 변환
            df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])  # datetime 컬럼 생성 (Local Time 기준)
            df = df.dropna(axis=1, how='all')  # 불필요한 NaN 컬럼 제거
            df['pv_power'] = calculate_pv_generation(df['GHI'], df['Temperature'])  # 발전량 계산 (GHI를 E로 사용)
            df = df[['datetime', 'GHI', 'Temperature', 'pv_power']]  # 필요한 컬럼만 선택

            all_data.append(df)
            print(f"✅ 파일 로드 성공: {file}, 데이터 개수: {len(df)}")

        except Exception as e:
            print(f"❌ 파일 로드 실패: {file}, 오류: {e}")

    if len(all_data) == 0:
        print("❌ 모든 파일을 로드하는 데 실패했습니다. 경로와 파일 형식을 확인하세요.")
        return None

    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 총 데이터 개수: {len(merged_df)}")
    print(f"📌 데이터 최소/최대 시간: {merged_df['datetime'].min()} ~ {merged_df['datetime'].max()}")
    return merged_df

# 📌 15분 단위로 보간 + 누락된 시간 추가
def resample_to_15min(data):
    full_time_range = pd.date_range(start=data['datetime'].min().floor('D'),
                                    end=data['datetime'].max().ceil('D'),
                                    freq='15T')
    print(f"📌 보간된 시간 범위: {full_time_range.min()} ~ {full_time_range.max()}")
    
    data = data.set_index('datetime').reindex(full_time_range).interpolate().reset_index()
    data.rename(columns={'index': 'datetime'}, inplace=True)
    
    data.fillna(0, inplace=True) # NaN 값이 있다면 0으로 채우기 (태양광 발전량이 없던 시간도 포함)
    
    return data

# 📌 ±1.96σ, ±2.58σ 통계 계산 함수
def calculate_sigma_stats(data):
    unique_times = data['datetime'].dt.time.unique()
    print(f"📌 그룹화된 시간 개수: {len(unique_times)}개")

    epsilon = 1e-6

    pv_stats = data.groupby(data['datetime'].dt.time)['pv_power'].agg(['mean', 'std'])
    pv_stats['-1.96sigma'] = (pv_stats['mean'] - 1.96 * pv_stats['std'])
    pv_stats['+1.96sigma'] = pv_stats['mean'] + 1.96 * pv_stats['std']

    pv_stats['-1.96sigma_scaled'] = pv_stats['-1.96sigma'] / np.maximum(pv_stats['mean'], epsilon)
    pv_stats['+1.96sigma_scaled'] = pv_stats['+1.96sigma'] / np.maximum(pv_stats['mean'], epsilon)

    data['ramping'] = data['pv_power'].diff()
    ramping_stats = data.groupby(data['datetime'].dt.time)['ramping'].agg(['mean', 'std'])
    ramping_stats['-1.96sigma'] = ramping_stats['mean'] - 1.96 * ramping_stats['std']
    ramping_stats['+1.96sigma'] = ramping_stats['mean'] + 1.96 * ramping_stats['std']
    ramping_stats['-2.58sigma'] = ramping_stats['mean'] - 3 * ramping_stats['std']
    ramping_stats['+2.58sigma'] = ramping_stats['mean'] + 3 * ramping_stats['std']

    ramping_stats['-1.96sigma_scaled'] = ramping_stats['-1.96sigma'] / np.maximum(ramping_stats['mean'], epsilon)
    ramping_stats['+1.96sigma_scaled'] = ramping_stats['+1.96sigma'] / np.maximum(ramping_stats['mean'], epsilon)
    ramping_stats['-2.58sigma_scaled'] = ramping_stats['-2.58sigma'] / np.maximum(ramping_stats['mean'], epsilon)
    ramping_stats['+2.58sigma_scaled'] = ramping_stats['+2.58sigma'] / np.maximum(ramping_stats['mean'], epsilon)

    return pv_stats, ramping_stats

# 📌 96개 타임 스텝별 확률 분포 및 램핑 분석 함수
def plot_combined_distributions(data):
    data['ramping'] = data['pv_power'].diff()
    time_steps = data['datetime'].dt.time.unique()
    
    plt.figure(figsize=(12, 6))
    for time in time_steps:
        subset = data[data['datetime'].dt.time == time]
        sns.kdeplot(subset['pv_power'], label=str(time), alpha=0.5)
    plt.xlabel("PV Power Generation (W)")
    plt.ylabel("Density")
    plt.title("Probability Distributions of PV Power for 96 Time Steps")
    plt.legend(loc='upper right', fontsize=6, ncol=4)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for time in time_steps:
        subset = data[data['datetime'].dt.time == time]
        sns.kdeplot(subset['ramping'].dropna(), label=str(time), alpha=0.5)
    plt.xlabel("Ramping (ΔPV Power, W)")
    plt.ylabel("Density")
    plt.title("Ramping Distributions for 96 Time Steps")
    plt.legend(loc='upper right', fontsize=6, ncol=4)
    plt.grid(True)
    plt.show()

# 📌 실행 코드
merged_data = load_and_merge_data(path_pattern)

if merged_data is not None:
    merged_data = resample_to_15min(merged_data)
    pv_stats_local, ramping_stats_local = calculate_sigma_stats(merged_data)
    print(pv_stats_local.to_string())
    print(ramping_stats_local.to_string())
    plot_combined_distributions(merged_data)
    pv_stats_local.to_pickle("pv_stats_local.pkl")
    ramping_stats_local.to_pickle("ramping_stats_local.pkl")