#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 22:14
# @Author  : Ken
# @Software: PyCharm
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st


def get_standard_data(x):
    train_data = pd.read_excel('train_data.xlsx')
    X = train_data.iloc[:, :12]
    # 标准化数据
    scaler = StandardScaler()
    X.iloc[:, :] = scaler.fit_transform(X)
    x.iloc[:, :] = scaler.transform(x)
    return x


# 标题,居中
st.markdown("<h1 style='text-align: center; color: black;'>RWHs Financial Viability Predictor</h1",
            unsafe_allow_html=True)

st.markdown("""
    <style>
    .stNumberInput input {
        background-color: #C0DBF8;
    }
    </style>
    """, unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    A = st.number_input('Water tariff (USD/m³)', max_value=24.0, min_value=0.1, step=0.01)
    B = st.number_input('Rainfall (mm/year)', max_value=4839.0, min_value=105.0, step=0.1)
    C = st.number_input('Coefficient of Variation', max_value=2.0, min_value=0.1, step=0.1)
    D = st.number_input('Catchment area (m²)', max_value=159971.0, min_value=37.0, step=0.1)
    E = st.number_input('Total built area  (m²)', max_value=900000.0, min_value=40.0, step=0.1)
    F = st.number_input('Number of occupants', max_value=20000, min_value=1)
with right:
    G = st.number_input('Demand (m³/d)', max_value=1912.0, min_value=0.1, step=0.1)
    H = st.number_input('Building Height (m)', max_value=530.0, min_value=3.5, step=0.1)
    I = st.number_input('Tank sizes (m³)', max_value=1600.0, min_value=0.2, step=0.1)
    J = st.number_input('Discount rate (%)', max_value=15.0, min_value=1.0, step=0.1)
    K = st.number_input('Energy cost (USD/kWh)', max_value=0.332, min_value=0.057, step=0.001)
    L = st.number_input('Project lifetime (Year)', max_value=60, min_value=10)

input_df = pd.DataFrame([A, B, C, D, E, F, G, H, I, J, K, L]).T
input_df.columns = ['Water tariff (USD/m³)', 'Rainfall (mm/year)', 'Coefficient of Variation', 'Catchment area (m²)',
                    'Total built area  (m²)', 'Number of occupants', 'Demand (m³/d)', 'Building Height (m)',
                    'Tank sizes (m³)', 'Discount rate (%)', 'Energy cost (USD/kWh)', 'Project lifetime (Year)']
x_df = get_standard_data(input_df)
if st.button('Predicting'):
    result_list = []
    for variable_name in ['BCR', 'NPV', 'PBP']:
        model = joblib.load(variable_name + '_best_model.pkl')
        predictions = model.predict(x_df)
        result = ''
        if variable_name == 'PBP' and predictions[0] > 60:
            result = 'Invalid value'
        else:
            result = str(round(predictions[0], 2))
        result_list.append({'variable_name': variable_name, 'predict_value': result})
    df = pd.DataFrame(result_list).T
    df.columns = df.iloc[0]
    df.drop('variable_name', inplace=True)
    st.balloons()
    # st.table(df)

    html_table = df.to_html(classes='table table-striped')
    # 为表头添加背景色
    html_table = html_table.replace('<thead>', '<thead style="background-color: #FB0046;">')  # 设置表头背景色为红色
    # 为第一行添加背景色
    html_table = html_table.replace('<tr>', '<tr style="background-color: #1A72DE;">', 1)  # 设置第一行背景色为蓝色
    # 在 Streamlit 中显示 HTML 表格
    st.markdown(html_table, unsafe_allow_html=True)
