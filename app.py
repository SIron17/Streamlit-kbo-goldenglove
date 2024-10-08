import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from math import pi

# 폰트 설정 제거 (기본 폰트 사용)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Streamlit 페이지 설정
st.set_page_config(page_title="KBO 골든글러브 예측모델", page_icon="⚾", layout="wide")
st.title("KBO 골든글러브 수상자 예측모델")

# 각 포지션별 저장된 모델 경로 설정
model_paths = {
    'P': './models/P_golden_glove_model_1984_2023.keras',
    'C': './models/C_golden_glove_model_1984_2023.keras',
    '1B': './models/1B_golden_glove_model_1984_2023.keras',
    '2B': './models/2B_golden_glove_model_1984_2023.keras',
    '3B': './models/3B_golden_glove_model_1984_2023.keras',
    'SS': './models/SS_golden_glove_model_1984_2023.keras',
    'Outfielders': './models/Outfielders_golden_glove_model_1984_2023.keras',
    'DH': './models/DH_golden_glove_model_1984_2023.keras'
}

# 각 포지션별 주요 피처 설정
position_features = {
    'P': ['WAR', 'W', 'L', 'S', 'HD', 'IP', 'ER', 'R', 'rRA', 'H', 'HR', 'BB', 'HP', 'SO', 'ERA', 'RA9', 'rRA9', 'rRA9pf', 'FIP', 'WHIP'],
    'C': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+'],
    '1B': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+'],
    '2B': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+'],
    '3B': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+'],
    'SS': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+'],
    'Outfielders': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+'],
    'DH': ['WAR', 'oWAR', 'dWAR', 'R', 'H', '2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+']
}

# CSV 파일 업로드 받기
uploaded_hitter_file = st.sidebar.file_uploader("타자 성적 CSV 파일 업로드", type=["csv"])
uploaded_pitcher_file = st.sidebar.file_uploader("투수 성적 CSV 파일 업로드", type=["csv"])

# 데이터가 업로드되었는지 확인
if uploaded_hitter_file and uploaded_pitcher_file:
    # CSV 데이터 읽기
    hitter_data = pd.read_csv(uploaded_hitter_file)
    pitcher_data = pd.read_csv(uploaded_pitcher_file)

    # 외야수 포지션 통합
    hitter_data['Position'] = hitter_data['Position'].replace({'LF': 'Outfielders', 'CF': 'Outfielders', 'RF': 'Outfielders'})

    # 각 포지션별 예측 수행 및 결과 저장
    final_candidates = pd.DataFrame()

    for pos, model_path in model_paths.items():
        if not os.path.exists(model_path):
            st.warning(f"{pos} 모델을 찾을 수 없습니다. 경로: {model_path}")
            continue

        model = tf.keras.models.load_model(model_path)

        if pos == 'P':
            pos_data = pitcher_data
        else:
            pos_data = hitter_data[hitter_data['Position'] == pos]

        if pos_data.empty:
            st.warning(f"CSV 파일에 '{pos}' 포지션 데이터가 없습니다. 해당 포지션을 건너뜁니다.")
            continue

        features = position_features[pos]
        pos_data_filled = pos_data[features].fillna(0)
        scaler = StandardScaler()
        pos_data_scaled = scaler.fit_transform(pos_data_filled)

        pos_data['GoldenGlove_Prob'] = model.predict(pos_data_scaled).flatten()

        top_candidates = pos_data.nlargest(3, 'GoldenGlove_Prob')

        final_candidates = pd.concat([final_candidates, top_candidates[['Name', 'Position', 'GoldenGlove_Prob']]], ignore_index=True)

        top_player = top_candidates.iloc[0]
        st.write(f"### {pos} 포지션의 1위 선수: {top_player['Name']} ({top_player['Position']})")

        labels = list(features)
        num_vars = len(labels)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        player_stats = top_player[labels].values.flatten().tolist()
        player_stats += player_stats[:1]

        # 그래프 크기와 폰트 조정
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))  # 크기 축소
        ax.fill(angles, player_stats, color='b', alpha=0.25)
        ax.plot(angles, player_stats, color='b', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)  # 폰트 크기 축소
        ax.set_title(f"{top_player['Name']}의 주요 성적 지표", size=12, color='blue', y=1.1)
        st.pyplot(fig)

    st.write("### 골든글러브 수상자 예측 결과")
    st.dataframe(final_candidates)

    st.download_button("결과 다운로드", final_candidates.to_csv(index=False).encode('utf-8-sig'), "golden_glove_top_candidates_2024.csv", "text/csv")

else:
    st.write("타자와 투수의 성적 CSV 파일을 모두 업로드해주세요.")
