# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Diabetes Data Dashboard",
    layout="centered",
    page_icon="🩺",
)

# Sidebar
st.sidebar.title("Diabetes Analysis Dashboard")

page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction", "Visualization"]
)

# Load Dataset
df = pd.read_csv("diabetes.csv", sep="\t")  
df = df.loc[:, ~df.columns.duplicated()]   

st.title("Diabetes Dataset Analysis")

# PAGE 1 — INTRODUCTION
if page == "Introduction":
    st.subheader("Data Preview")
    rows = st.slider("Select number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("No missing values found")
    else:
        st.warning("Dataset contains missing values")

    st.subheader("Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

# PAGE 2 — VISUALIZATION
elif page == "Visualization":
    st.subheader("Data Visualization")

    col_x = st.selectbox("Select X-axis variable", df.columns, index=0)
    col_y = st.selectbox("Select Y-axis variable", df.columns, index=1)

    tab1, tab2, tab3 = st.tabs(
        ["Bar Chart", "Line Chart", "Correlation Heatmap"]
    )

    # Ensure selected columns exist and are numeric for plotting
    cols_to_plot = [col for col in [col_x, col_y] if col in df.columns]

    with tab1:
        st.subheader("Bar Chart")
        st.bar_chart(
            df[cols_to_plot].sort_values(by=col_x),
            use_container_width=True
        )

    with tab2:
        st.subheader("Line Chart")
        st.line_chart(
            df[cols_to_plot].sort_values(by=col_x),
            use_container_width=True
        )

    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_numeric.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm"
        )
        st.pyplot(fig)
