import streamlit as st  
import xgboost as xgb
import numpy as np
import pandas as pd
from io import BytesIO

# 加载模型
model_beta = xgb.XGBRegressor()
model_beta.load_model("xgboost_knn_5_beta.model")
model_c = xgb.XGBRegressor()
model_c.load_model("xgboost_knn_5_c.model")

# 最小和最大值，用于归一化
X_min = np.array([0, 1.06, 21.69, 2.19, 0.17, 2.2, 0.95, 1.23])
X_max = np.array([1, 2.245, 880, 94, 4.85, 140.36, 1.76, 2.44])
beta_min, beta_max = 18.22, 67.7
c_min, c_max = 0, 9.6

# 页面配置
st.set_page_config(page_title="Lunar Soil Prediction App", layout="wide", initial_sidebar_state="expanded")

# 页面标题
st.title("🌕 Lunar Soil Prediction App")

# 使用侧边栏来输入数据
st.sidebar.header("🔧 Input Parameters")

# 创建输入表单
相对密实度 = st.sidebar.number_input("Relative density", min_value=0.0, max_value=1.0, value=0.5, format="%.2f")
试样密度 = st.sidebar.number_input("Sample density (g/cm³)", min_value=0.1, max_value=3.0, value=1.5, format="%.2f")
D60 = st.sidebar.number_input("D60 (μm)", min_value=1.0, max_value=1000.0, value=50.0, format="%.2f")
D10 = st.sidebar.number_input("D10 (μm)", min_value=1.0, max_value=1000.0, value=10.0, format="%.2f")
Cc = st.sidebar.number_input("Cc", min_value=0.0, max_value=10.0, value=1.0, format="%.2f")
Cu = st.sidebar.number_input("Cu", min_value=0.0, max_value=150.0, value=10.0, format="%.2f")
最小密度 = st.sidebar.number_input("Minimum density (g/cm³)", min_value=0.1, max_value=3.0, value=1.0, format="%.2f")
最大密度 = st.sidebar.number_input("Maximum density (g/cm³)", min_value=0.2, max_value=4.0, value=1.5, format="%.2f")

# 初始化历史记录
if 'history' not in st.session_state:
    st.session_state.history = []

# 创建按钮在同一行
col1, col2, col3 = st.sidebar.columns([1, 1, 1])  # 调整列宽
if col1.button("Submit 🚀"):
    X_input = np.array([相对密实度, 试样密度, D60, D10, Cc, Cu, 最小密度, 最大密度])

    # 归一化
    X_normalized = (X_input - X_min) / (X_max - X_min)

    # 预测
    pred_beta_norm = model_beta.predict(np.array([X_normalized]))
    pred_c_norm = model_c.predict(np.array([X_normalized]))

    # 反归一化
    pred_beta = pred_beta_norm * (beta_max - beta_min) + beta_min
    pred_c = pred_c_norm * (c_max - c_min) + c_min

    # 保存历史记录
    st.session_state.history.append((pred_beta[0], pred_c[0], 相对密实度, 试样密度, D60, D10, Cc, Cu, 最小密度, 最大密度))

    # 显示结果
    st.success(f"✅ Prediction Results: \n\n - φ = {pred_beta[0]:.2f}\n - Cohesion (*c*) = {pred_c[0]:.2f}")

# 显示历史预测结果
if st.session_state.history:
    st.subheader("📜 Prediction History")
    
    # 创建包含输入参数的历史记录DataFrame
    history_df = pd.DataFrame(st.session_state.history, columns=["φ", "Cohesion (*c*)", "Relative density", "Sample density (g/cm³)", "D60 (μm)", "D10 (μm)", "Cc", "Cu", "Minimum density (g/cm³)", "Maximum density (g/cm³)"])
    
    # 添加索引列
    history_df.index += 1  # 从1开始
    history_df.index.name = 'Index'
    
    st.dataframe(history_df, use_container_width=True)

    # 导出历史记录按钮
    if col2.button("Export History 📥"):
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            history_df.to_excel(writer, sheet_name='History', index=True)
        
        output.seek(0)

        # 生成下载链接
        st.download_button(
            label="Download Excel file",
            data=output,
            file_name="prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 清除历史记录按钮
    if col3.button("Clear History ❌"):
        st.session_state.history.clear()
        st.success("🗑️ History cleared!")

# 页脚
st.markdown("""---\nDeveloped by WP.\n""")

# 自定义样式
st.markdown(
    """
    <style>
        .stButton > button {
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            padding: 5px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            cursor: pointer;
            border-radius: 12px;
            width: auto; /* 自动调整按钮宽度 */
            white-space: nowrap; /* 防止换行 */
        }
        .stNumberInput input {
            font-size: 16px;
            padding: 10px;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        .stDataFrame thead th {
            background-color: #4CAF50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

