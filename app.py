import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 포지션별 모델 파일 경로
model_paths = {
    "P": './models/P_golden_glove_model_1984_2023.keras',
    "C": './models/C_golden_glove_model_1984_2023.keras',
    "1B": './models/1B_golden_glove_model_1984_2023.keras',
    "2B": './models/2B_golden_glove_model_1984_2023.keras',
    "3B": './models/3B_golden_glove_model_1984_2023.keras',
    "SS": './models/SS_golden_glove_model_1984_2023.keras',
    "Outfielders": './models/Outfielders_golden_glove_model_1984_2023.keras',
    "DH": './models/DH_golden_glove_model_1984_2023.keras'
}

# 모델 로드 함수
def load_position_model(position):
    if position in model_paths:
        return tf.keras.models.load_model(model_paths[position])
    return None

# 타자와 투수의 특징 정의
hitter_features = [
    'WAR', 'oWAR', 'dWAR', 'G', 'PA', 'ePA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'TB',
    'RBI', 'SB', 'CS', 'BB', 'HP', 'IB', 'SO', 'GDP', 'SH', 'SF', 'AVG', 'OBP', 'SLG',
    'OPS', 'R/ePA', 'wRC+'
]

pitcher_features = [
    'WAR', 'G', 'GS', 'GR', 'GF', 'CG', 'SHO', 'W', 'L', 'S', 'HD', 'IP', 'ER', 'R', 'rRA', 
    'TBF', 'H', '2B', '3B', 'HR', 'BB', 'HP', 'IB', 'SO', 'ROE', 'BK', 'WP', 'ERA', 'RA9', 
    'rRA9', 'rRA9pf', 'FIP', 'WHIP'
]

# 페이지 레이아웃 설정
st.set_page_config(page_title="KBO 골든글러브 수상자 예측모델", page_icon="⚾", layout="wide")
st.title("KBO 골든글러브 수상자 예측모델")

# 사이드바에 파일 업로드 설정
uploaded_file = st.sidebar.file_uploader("선수 성적 CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    # 업로드된 CSV 데이터 불러오기
    data = pd.read_csv(uploaded_file)
    st.sidebar.write("업로드된 데이터:")
    st.sidebar.write(data)

    # 포지션 선택
    position = st.sidebar.selectbox("포지션을 선택하세요", list(model_paths.keys()))
    st.write(f"### 선택된 포지션: {position}")

    # CSV 파일에 선택한 포지션이 포함되어 있는지 확인
    if position not in data['Position'].unique():
        st.warning(f"선택한 포지션 '{position}'이(가) 업로드된 CSV 파일에 존재하지 않습니다. 다른 포지션을 선택해주세요.")
    else:
        # 모델 로드 및 데이터 전처리
        model = load_position_model(position)
        if model:
            # 포지션에 따라 다른 특징 리스트 사용
            if position == 'P':
                features = pitcher_features  # 투수 특징 사용
            else:
                features = hitter_features   # 타자 특징 사용

            # 예측에 사용할 특징만 선택
            input_data = data[data['Position'] == position][features].fillna(0)  # 해당 포지션의 데이터만 사용
            st.write(f"### 사용할 특징: {features}")

            # 데이터가 올바른 형식인지 확인
            st.write("### 예측에 사용될 데이터")
            st.write(input_data.head())

            # 스케일링 수행
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)

            # 예측 수행
            predictions = model.predict(input_data_scaled)
            data['수상 확률'] = 0  # 기본 값 설정
            data.loc[data['Position'] == position, '수상 확률'] = predictions
            data['순위'] = data['수상 확률'].rank(ascending=False)
            data = data.sort_values(by='수상 확률', ascending=False)

            # 예측 결과 테이블 표시
            st.write(f"### {position} 포지션의 예측 결과")
            st.write(data[['Name', 'Team', 'Position', '수상 확률', '순위']])

            # 1위 선수의 능력치 시각화
            top_player = data[data['Position'] == position].iloc[0]
            st.write(f"### {position} 포지션의 1위 선수: {top_player['Name']} ({top_player['Team']})")
            top_stats = top_player[features]
            fig, ax = plt.subplots()
            sns.barplot(x=top_stats.index, y=top_stats.values, ax=ax)
            ax.set_title(f"{top_player['Name']}의 주요 성적 지표")
            st.pyplot(fig)
        else:
            st.write("선택한 포지션의 모델이 로드되지 않았습니다.")
else:
    st.write("선수 성적 CSV 파일을 업로드해주세요.")
