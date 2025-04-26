import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    with open('ash_xg_boost.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'ash_xg_boost.pkl' is in the same directory.")
    exit()

# Function to load data (replace with your actual data loading - from CSV or DB)
def load_data():
    # Placeholder: Load your data here
    # For example, if you have a CSV:
    # df = pd.read_csv("titanic.csv")  
    #
    # Or if you load from your MySQL database (adjust connection details):
    import pymysql
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='Dob@1980',  # Replace with your password
        port=3306,
        database='tumor' # Replace with your database name
    )
    query = "SELECT * FROM titanic"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df = load_data()

# Preprocessing function (as in your notebook)
def preprocess_data(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Embarked'] = le.fit_transform(df['Embarked'])
    # Add any other preprocessing steps here
    return df

df = preprocess_data(df.copy()) # Process a copy to avoid modifying original

# Streamlit App
def main():
    st.title("Titanic Survival Prediction")

    # Feature selection based on your model
    pclass = st.selectbox("Pclass", options=df['Pclass'].unique())
    sex = st.selectbox("Sex", options=["male", "female"]) # Original labels
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.slider("SibSp", 0, 8, 0)
    parch = st.slider("Parch", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 500.0, 20.0)
    embarked = st.selectbox("Embarked", options=["S", "C", "Q"]) # Original labels

    # Convert sex and embarked back to numerical
    sex_encoded = 1 if sex == "male" else 0
    embarked_encoded = df[df['Embarked'] == embarked]['Embarked'].values[0]

    # Create input dataframe for prediction
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The passenger is predicted to have survived.")
        else:
            st.error("The passenger is predicted not to have survived.")

if __name__ == '__main__':
    main()