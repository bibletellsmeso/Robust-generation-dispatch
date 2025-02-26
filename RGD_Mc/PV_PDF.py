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
            df.columns = df.iloc[0]  # 이후 첫 번째 행을 컬럼명으로 설정
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
    
    data.fillna(0, inplace=True)  # NaN 값이 있다면 0으로 채우기 (태양광 발전량이 없던 시간도 포함)
    
    return data

# 📌 불대칭성을 반영한 ±시그마(신뢰구간) 계산 함수
def calculate_sigma_stats(data):
    # helper 함수: 주어진 시리즈에서 평균, 양측(평균보다 큰/작은) 표준편차를 구한 후,
    # multiplier를 곱하여 하한과 상한을 산출
    def compute_bounds(series, multiplier):
        m = series.mean()
        # 평균보다 큰 값들의 표준편차 (양측 시그마)
        pos_data = series[series > m]
        pos_std = pos_data.std() if len(pos_data) > 1 else 0
        # 평균보다 작은 값들의 표준편차 (음측 시그마)
        neg_data = series[series < m]
        neg_std = neg_data.std() if len(neg_data) > 1 else 0
        lower_bound = m - multiplier * neg_std
        upper_bound = m + multiplier * pos_std
        return m, lower_bound, upper_bound, pos_std, neg_std

    # 그룹별(각 시간대)로 PV 발전량에 대해 ±2시그마 범위 산출
    pv_stats_list = []
    for t, group in data.groupby(data['datetime'].dt.time):
        m, lower_bound, upper_bound, pos_std, neg_std = compute_bounds(group['pv_power'], multiplier=2)
        pv_stats_list.append({
            'time': t,
            'mean': m,
            'lower_2sigma': lower_bound,
            'upper_2sigma': upper_bound,
            'pos_std': pos_std,
            'neg_std': neg_std,
            'std': group['pv_power'].std()  # 전체 표준편차
        })
    pv_stats = pd.DataFrame(pv_stats_list).set_index('time')

    # 램핑: 발전량의 시계열 차분으로 계산
    # 주의: diff()로 인해 하루 내 15분 간격 발전량은 96개인데, 램핑은 자연스럽게 95개가 됨.
    data['ramping'] = data['pv_power'].diff()
    
    ramping_stats_list = []
    for t, group in data.groupby(data['datetime'].dt.time):
        # diff()로 인해 NaN이 포함될 수 있으므로 drop
        valid_ramping = group['ramping'].dropna()
        if len(valid_ramping) == 0:
            m = np.nan
            lower_bound = np.nan
            upper_bound = np.nan
            pos_std = np.nan
            neg_std = np.nan
        else:
            m, lower_bound, upper_bound, pos_std, neg_std = compute_bounds(valid_ramping, multiplier=3)
        ramping_stats_list.append({
            'time': t,
            'mean': m,
            'lower_3sigma': lower_bound,
            'upper_3sigma': upper_bound,
            'pos_std': pos_std,
            'neg_std': neg_std,
            'std': valid_ramping.std()
        })
    ramping_stats = pd.DataFrame(ramping_stats_list).set_index('time')

    return pv_stats, ramping_stats

# 📌 96개 타임스텝별 발전량과 95개(자연산출)의 램핑 분포 시각화 함수
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
    plt.title("Ramping Distributions for 96 Time Steps (실제 데이터는 하루 95개)")
    plt.legend(loc='upper right', fontsize=6, ncol=4)
    plt.grid(True)
    plt.show()

# 📌 실행 코드
merged_data = load_and_merge_data(path_pattern)

if merged_data is not None:
    merged_data = resample_to_15min(merged_data)
    pv_stats_local, ramping_stats_local = calculate_sigma_stats(merged_data)
    
    print("----- PV Power Statistics (각 시간대 ±2시그마) -----")
    print(pv_stats_local.to_string())
    
    print("----- Ramping Statistics (각 시간대 ±3시그마) -----")
    print(ramping_stats_local.to_string())
    
    plot_combined_distributions(merged_data)
    
    pv_stats_local.to_pickle("pv_stats_local.pkl")
    ramping_stats_local.to_pickle("ramping_stats_local.pkl")
