import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn import metrics # type: ignore
import time

import warnings
warnings.filterwarnings('ignore')

st.markdown("## 🌟 Personal Fitness Tracker 🏋️‍♂️")
st.markdown("### 🔥 Track your fitness and predict your calorie burn! 🔥")

st.sidebar.header("🎯 User Input Parameters:")
def user_input_features():
    age = st.sidebar.slider("📅 Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("⚖️ BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("⏳ Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("💓 Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡️ Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("👤 Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.markdown("---")
st.markdown("### 📊 Your Parameters:")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

calories = pd.read_csv(r"C:\Users\AVANTIKA\OneDrive\Desktop\fitness_tracker\calories.csv")
exercise = pd.read_csv(r"C:\Users\AVANTIKA\OneDrive\Desktop\fitness_tracker\exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)

prediction = random_reg.predict(df)

st.markdown("---")
st.markdown("### 🎯 Prediction: ")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.markdown(f"💪 **{round(prediction[0], 2)} kilocalories** 🔥")

st.markdown("---")
st.markdown("### 🔄 Similar Results:")
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.markdown("---")
st.markdown("### ℹ️ General Information:")

boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.markdown(f"📅 You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of other users.")
st.markdown(f"⏳ Your exercise duration is higher than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of other users.")
st.markdown(f"💓 Your heart rate is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of other users during exercise.")
st.markdown(f"🌡️ Your body temperature is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of other users during exercise.")

# Visualization Section
st.markdown("---")
st.markdown("### 📊 Data Visualizations:")

# Heatmap for correlation
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 6))
corr = exercise_train_data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, square=True, linewidth=0.5, vmin=0, vmax=1, cmap='Blues', ax=ax)
st.pyplot(fig)

# Bar chart for gender distribution
st.subheader("📊 Gender Distribution")
fig, ax = plt.subplots()
sns.countplot(data=exercise_df, x="Gender", ax=ax)
st.pyplot(fig)
