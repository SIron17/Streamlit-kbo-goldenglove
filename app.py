# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from math import pi

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

def plot_performance_metrics(player_stats, labels, title):
    """선수의 성적 지표를 방사형 그래프로 시각화하는 함수"""
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]
    
    player_stats = player_stats.flatten().tolist()
    player_stats += player_stats[:1]

    fig, ax = plt.subplots(figsize=(1.5, 1.5), subplot_kw=dict(polar=True))
    ax.fill(angles, player_stats, color='b', alpha=0.25)
    ax.plot(angles, player_stats, color='b', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(title, size=8, y=1.6)

    # 각 지표의 수치 표시
    for i, (angle, value) in enumerate(zip(angles, player_stats)):
        ax.text(angle, value + 0.1, f"{value:.2f}", horizontalalignment='center', size=5, color='blue', weight='bold')
        offset = 0.1 if value < 0 else -0.1
        ax.text(angle, value + offset, f"{value:.2f}", horizontalalignment='center', size=5, color='black')

    return fig

# CSV 파일 업로드 받기
uploaded_hitter_file = st.sidebar.file_uploader("타자 성적 CSV 파일 업로드", type=["csv"])
uploaded_pitcher_file = st.sidebar.file_uploader("투수 성적 CSV 파일 업로드", type=["csv"])
position_selection = st.sidebar.selectbox("세부 기록을 볼 포지션 선택", list(model_paths.keys()))

# 데이터가 업로드되었는지 확인 후 모델 실행
if uploaded_hitter_file and uploaded_pitcher_file:
    hitter_data = pd.read_csv(uploaded_hitter_file)
    pitcher_data = pd.read_csv(uploaded_pitcher_file)
    hitter_data['Position'] = hitter_data['Position'].replace({'LF': 'Outfielders', 'CF': 'Outfielders', 'RF': 'Outfielders'})

    # 각 포지션별 예측 수행 및 결과 저장
    final_candidates = pd.DataFrame()

    for pos, model_path in model_paths.items():
        if not os.path.exists(model_path):
            st.warning(f"{pos} 모델을 찾을 수 없습니다. 경로: {model_path}")
            continue

        model = tf.keras.models.load_model(model_path)

        pos_data = pitcher_data if pos == 'P' else hitter_data[hitter_data['Position'] == pos]
        if pos_data.empty:
            st.warning(f"CSV 파일에 '{pos}' 포지션 데이터가 없습니다. 해당 포지션을 건너뜁니다.")
            continue

        features = position_features[pos]
        pos_data_filled = pos_data[features].fillna(0)
        scaler = StandardScaler()
        pos_data_scaled = scaler.fit_transform(pos_data_filled)
        pos_data['GoldenGlove_Prob'] = model.predict(pos_data_scaled).flatten()

        top_candidates = pos_data.nlargest(3 if pos != 'Outfielders' else 5, 'GoldenGlove_Prob')
        final_candidates = pd.concat([final_candidates, top_candidates[['Name', 'Position', 'GoldenGlove_Prob']]], ignore_index=True)

        # 선택한 포지션의 선수 기록과 그래프만 표시
        if pos == position_selection:
            st.write(f"### {pos} 포지션의 주요 선수 기록")
            st.write(top_candidates)

            # Outfielders의 경우 1위, 2위, 3위 그래프 표시
            if pos == 'Outfielders':
                titles = ["1st Performance Metrics", "2nd Performance Metrics", "3rd Performance Metrics"]
                for idx, player in top_candidates.head(3).iterrows():
                    labels = list(features)
                    player_stats = player[labels].values
                    fig = plot_performance_metrics(player_stats, labels, titles[idx])
                    st.pyplot(fig)

            else:
                # 나머지 포지션은 1위 그래프만 출력
                labels = list(features)
                player_stats = top_candidates.iloc[0][labels].values
                fig = plot_performance_metrics(player_stats, labels, "1st Performance Metrics")
                st.pyplot(fig)

    # 전체 예측 결과 표시 및 중앙 정렬
    st.write("### 골든글러브 수상자 예측 결과")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.dataframe(final_candidates, width=800)
    st.markdown("</div>", unsafe_allow_html=True)

    # 결과 다운로드 버튼
    st.download_button("결과 다운로드", final_candidates.to_csv(index=False).encode('utf-8-sig'), "golden_glove_top_candidates_2024.csv", "text/csv")

else:
    st.write("타자와 투수의 성적 CSV 파일을 모두 업로드해주세요.")
