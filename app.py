import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Manufacturing Efficiency AI Dashboard", layout="wide")

st.title("🏭 AI-Based Manufacturing Efficiency Dashboard")

uploaded_file = st.file_uploader("Upload Manufacturing Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(df.describe())

    # KPI SECTION
    st.subheader("Key Performance Indicators")

    col1, col2, col3 = st.columns(3)

    if "Production_Count" in df.columns:
        col1.metric("Total Production", int(df["Production_Count"].sum()))

    if "Error_Rate" in df.columns:
        col2.metric("Average Error Rate", round(df["Error_Rate"].mean(),2))

    if "Efficiency_Class" in df.columns:
        high_eff = (df["Efficiency_Class"] == "High").sum()
        col3.metric("High Efficiency Machines", high_eff)

    # GRAPH SECTION
    st.subheader("Efficiency Class Distribution")

    if "Efficiency_Class" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x="Efficiency_Class", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Production Trend")

    if "Production_Count" in df.columns:
        fig, ax = plt.subplots()
        df["Production_Count"].plot(ax=ax)
        st.pyplot(fig)

    st.subheader("Sensor Data Distribution")

    if "Sensor_Value" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df["Sensor_Value"], kde=True, ax=ax)
        st.pyplot(fig)

    # HEATMAP
    st.subheader("Feature Correlation")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # AI MODEL
    st.subheader("AI Efficiency Prediction Model")

    if "Efficiency_Class" in df.columns:

        df = df.dropna()

        X = df.select_dtypes(include=['int64','float64'])

        y = df["Efficiency_Class"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        model = RandomForestClassifier()

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test,pred)

        st.success(f"Model Accuracy: {round(acc*100,2)} %")

else:
    st.info("Upload your dataset to generate AI dashboard")
