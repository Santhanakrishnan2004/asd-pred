import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, label_binarize, OneHotEncoder
import pandas as pd
import pickle

# model
model = tf.keras.models.load_model("model.h5")

# load
with open("label_encoder_class.pkl", "rb") as file:
    label_encoder_class = pickle.load(file)
with open("label_encoder_ethnicity.pkl", "rb") as file:
    label_encoder_ethnicity = pickle.load(file)
with open("label_encoder_fammem.pkl", "rb") as file:
    label_encoder_fammem = pickle.load(file)
with open("label_encoder_jaundice.pkl", "rb") as file:
    label_encoder_jaundice = pickle.load(file)
with open("label_encoder_sex.pkl", "rb") as file:
    label_encoder_sex = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# streamlit app
st.title("ASD  prediction (0 - no) & ( 1- yes)")

A1 = st.selectbox("Does your child look at you when you call his/her name?", [0, 1])
A2 = st.selectbox("How easy is it for you to get eye contact with your child? ", [0, 1])
A3 = st.selectbox(
    "Does your child point to indicate that s/he wants something? e.g. a toy that is out of reach ",
    [0, 1],
)
A4 = st.selectbox("Does your child point to share interest with you? :", [0, 1])
A5 = st.selectbox("Does your child pretend? ", [0, 1])
A6 = st.selectbox("Does your child follow where you’re looking? :", [0, 1])
A7 = st.selectbox(
    "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? :",
    [0, 1],
)
A8 = st.selectbox("Would you describe your child’s first words as:", [0, 1])
A9 = st.selectbox("Does your child use simple gestures? (e.g. wave goodbye) ", [0, 1])
A10 = st.selectbox(
    "Does your child stare at nothing with no apparent purpose? ", [0, 1]
)
Age_Mons = st.slider("Age of your child in months ", 0, 100)
QchatScore = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10
Ethnicity = st.selectbox("Ethnicity", label_encoder_ethnicity.categories_[0])
Sex = st.selectbox("Sex if male -1 , female is  0", [0, 1])
Jaundice = st.selectbox("has Jaundice", label_encoder_jaundice.classes_)
Family_mem_with_ASD = st.selectbox("is family member has asd no -0 , yes 1", [0, 1])


# Example input data
input_data = pd.DataFrame(
    {
        "A1": [A1],
        "A2": [A2],
        "A3": [A3],
        "A4": [A4],
        "A5": [A5],
        "A6": [A6],
        "A7": [A7],
        "A8": [A8],
        "A9": [A9],
        "A10": [A10],
        "Age_Mons": [Age_Mons],
        "Qchat-10-Score": [QchatScore],
        "Sex": [Sex],
        "Jaundice": [label_encoder_jaundice.transform([Jaundice])[0]],
        "Family_mem_with_ASD": [Family_mem_with_ASD],
    }
)


geo_encoded = label_encoder_ethnicity.transform([[Ethnicity]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=label_encoder_ethnicity.get_feature_names_out(["Ethnicity"])
)

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"asd Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The child has likely to have asd.")
else:
    st.write("The child has not likely to have asd..")



data = {
    "s.no": [1, 2, 3],
    "ANN": ["0.875", "0.7126436781609196", "0.775" ],
    "SVC": ["0.895", "0.7529411764705882", "0.8"],
    "Random Forest": ["0.815", "0.6542056074766355", "0.875"],
    "Decision Tree": ["0.875", "0.6575342465753424", "0.6"],
    "Logistic Regression": ["0.88", "0.7", "0.7"],
}

# Convert the data into a DataFrame
df = pd.DataFrame(data, index=["Accuracy score", "F1-Score", "Recall Score", ""])

# Streamlit app
st.title("Comparison Table of Machine Learning Models")
st.table(df)
