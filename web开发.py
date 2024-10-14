import streamlit as st
import xgboost as xgb
import numpy as np

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

# 创建表单
st.title("Lunar Soil Prediction App")

相对密实度 = st.number_input("Relative density", min_value=0.0, max_value=1.0, value=0.5)
试样密度 = st.number_input("Sample density (g/cm³)", min_value=1.06, max_value=2.245, value=1.5)
D60 = st.number_input("D60 ({}m)".format('\u03bc'), min_value=21.69, max_value=880.0, value=50.0)
D10 = st.number_input("D10 ({}m)".format('\u03bc'), min_value=2.19, max_value=94.0, value=10.0)
Cc = st.number_input("Cc", min_value=0.17, max_value=4.85, value=1.0)
Cu = st.number_input("Cu", min_value=2.2, max_value=140.36, value=10.0)
最小密度 = st.number_input("Minimum density (g/cm³)", min_value=0.95, max_value=1.76, value=1.0)
最大密度 = st.number_input("Maximum density (g/cm³)", min_value=1.23, max_value=2.44, value=1.5)

if st.button("预测"):
    X_input = np.array([相对密实度, 试样密度, D60, D10, Cc, Cu, 最小密度, 最大密度])

    # 归一化
    X_normalized = (X_input - X_min) / (X_max - X_min)

    # 预测
    pred_beta_norm = model_beta.predict(np.array([X_normalized]))
    pred_c_norm = model_c.predict(np.array([X_normalized]))

    # 反归一化
    pred_beta = pred_beta_norm * (beta_max - beta_min) + beta_min
    pred_c = pred_c_norm * (c_max - c_min) + c_min

    st.write(f"预测结果: beta = {pred_beta[0]}, c = {pred_c[0]}")



