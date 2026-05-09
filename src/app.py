import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Air Quality App", layout="wide")

st.title("🌍 Air Quality Prediction App")

# ---------------- SAFE MODEL LOADING ----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# ---------------- MENU ----------------
menu = st.sidebar.selectbox("Menu", ["Dataset","Visualization","Prediction"])

# ---------------- DATASET ----------------
if menu == "Dataset":
    st.subheader("📊 Dataset Info")

    try:
        data = pd.read_csv("data.csv")
        st.write(data.head())
        st.write("Shape:", data.shape)
        st.write("Missing Values:")
        st.write(data.isnull().sum())
    except Exception as e:
        st.warning(f"Dataset not found or error: {e}")

# ---------------- VISUALIZATION ----------------
elif menu == "Visualization":

    st.subheader("Data Visualization")

    try:
        data = pd.read_csv("data.csv")

        # Fill missing for visualization
        data.fillna(method='ffill', inplace=True)

        viz_option = st.selectbox(
            "Choose Visualization",
            ["PM2.5 Distribution", "PM2.5 vs Temperature", "Correlation Heatmap"]
        )

        # --------- 1. Distribution ---------
        if viz_option == "PM2.5 Distribution":
            fig, ax = plt.subplots()
            sns.histplot(data['PM2.5'], kde=True, ax=ax)
            ax.set_title("PM2.5 Distribution")
            st.pyplot(fig)

        # --------- 2. Scatter ---------
        elif viz_option == "PM2.5 vs Temperature":
            fig, ax = plt.subplots()
            sns.scatterplot(x='TEMP', y='PM2.5', data=data, ax=ax)
            ax.set_title("PM2.5 vs Temperature")
            st.pyplot(fig)

        # --------- 3. Heatmap ---------
        elif viz_option == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(12,12))
            corr = data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Visualization Error: {e}")

# ---------------- PREDICTION ----------------
elif menu == "Prediction":

    st.subheader("🔮 Predict PM2.5")

    try:
        PM10 = st.number_input("PM10", 0.0)
        SO2 = st.number_input("SO2", 0.0)
        NO2 = st.number_input("NO2", 0.0)
        CO = st.number_input("CO", 0.0)
        O3 = st.number_input("O3", 0.0)
        TEMP = st.number_input("Temperature", 0.0)
        PRES = st.number_input("Pressure", 0.0)
        DEWP = st.number_input("Dew Point", 0.0)
        RAIN = st.number_input("Rainfall", 0.0)
        WSPM = st.number_input("Wind Speed", 0.0)

        if st.button("Predict"):

            # Validate input
            if any(v is None for v in [PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM]):
                st.warning("Please fill all inputs")
            else:
                input_data = np.array([[PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM]])

                try:
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)

                    st.success(f"Predicted PM2.5: {prediction[0]:.2f}")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    except Exception as e:
        st.error(f"Input Error: {e}")