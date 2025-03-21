import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import time
st.set_page_config(layout="wide")  # Set wide layout for better spacing
  # Main content (3x) and right sidebar (1x)
st.markdown(
    """
    <style>
    /* Change slider track color */
    div[data-baseweb="slider"] > div {
        background: linear-gradient(to right,rgb(173, 1000, 230), #ADD8E6);
    }

    /* Change slider handle color */
    div[data-baseweb="slider"] > div > div {
        background-color: #ff5733 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write(
    """
    <style>
    .stApp {
        background-color:rgb(139, 0, 0);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import warnings
warnings.filterwarnings('ignore')

st.write("Tracker Your Fitness")
#st.image("", use_column_width=True)

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    with st.sidebar.expander("Select Your Age"):
        st.write("Age")
        age = st.slider("Age: ", 10, 100, 30)
    
    with st.sidebar.expander("Select Your BMI"):
        st.write("BMI")
        bmi = st.slider("BMI: ", 15, 40, 20)
    
    with st.sidebar.expander("Select Your Workout Time"):
        st.write("duration")
        duration = st.slider("Duration (min): ", 0, 35, 15)
    
    with st.sidebar.expander("Select Your Heart Rate"):
        st.write("Heart Rate")
        heart_rate = st.slider("Heart Rate: ", 60, 130, 80)
    
    with st.sidebar.expander("Select Your Body Temparature"):
        st.write("Body Temparature")
        body_temp = st.slider("Body Temperature (C): ", 36, 42, 38)
    
    with st.sidebar.expander("Select Your Sex"):
        st.write("Sex")
        gender_button = st.selectbox("Choose option", ["Male", "Female"])
    
    gender = 1 if gender_button == "Male" else 0
    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features
col1, col2 = st.columns([3.8, 4.5])
df = user_input_features()
with col1:
    st.write("---")
    st.header("Your Parameters: ")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)
with col2:
    st.write("---")
    st.header("Prediction: ")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

    st.write(f"{round(prediction[0], 2)} **kilocalories**")
col3, col4 = st.columns([4, 3])
with col3:
    st.write("---")
    st.header("Similar Results: ")
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)

# Find similar results based on predicted calories
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.write(similar_data.sample(5))
with col4:
    st.write("---")

    st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

    st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
    st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
    st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
    st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")


