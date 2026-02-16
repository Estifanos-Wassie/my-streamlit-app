import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Diabetes Health Dashboard 🩺",
    layout="centered",
    page_icon="🩺",
)

st.sidebar.title("Diabetes Health Dashboard 🩺")
page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction 📘", "Data Exploration 📊", "Visualization 📈"]
)

st.image("diabetes_image.png")
st.write(" ")
st.write(" ")
st.write(" ")

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df = load_data()

st.title("Diabetes Dataset Analysis 🩺")

if page == "Introduction 📘":
    st.subheader("Data Preview")
    rows = st.slider("Select number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("No missing values found")
    else:
        st.warning(f"Dataset contains {missing.sum()} missing values")

    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", int(missing.sum()))

    if st.button("Show Summary Statistics"):
        st.dataframe(df.describe())

elif page == "Data Exploration 📊":
    st.subheader("Dataset Overview")
    st.dataframe(df)

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

elif page == "Visualization 📈":
    st.subheader("Interactive Visualizations")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(
        ["Bar Chart 📊", "Line Chart 📈", "Correlation Heatmap 🔥"]
    )

    with tab1:
        col_x = st.selectbox("Select X-axis variable", numeric_cols, key="bar_x")
        col_y = st.selectbox("Select Y-axis variable", numeric_cols, key="bar_y")

        if col_x and col_y:
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x=col_x, y=col_y, ax=ax_bar)
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

    with tab2:
        col_x = st.selectbox("Select X variable", numeric_cols, key="line_x")
        col_y = st.selectbox("Select Y variable", numeric_cols, key="line_y")

        if col_x != col_y:
            df_sorted = df.sort_values(by=col_x)
            st.line_chart(
                df_sorted.set_index(col_x)[col_y],
                use_container_width=True
            )
        else:
            st.warning("Please select two different variables")

    with tab3:
        df_numeric = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_numeric.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("""
This dashboard explores the Diabetes dataset with:
- Data preview and statistics
- Interactive charts
- Correlation analysis
""")
