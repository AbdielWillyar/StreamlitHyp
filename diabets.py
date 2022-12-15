# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:37:19 2022

@author: Abdiel Wilyar
"""


import streamlit as st 
import numpy as np 
from keras.models import load_model
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


# heading
st.markdown("<h1 style='text-align: center; color: blue;'>DiabetDet</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>...a Hypertentcy detection system</h4><br>", unsafe_allow_html=True)

X_train = pd.read_csv("C:/Python/VS/UKSW/AI/X_train.csv")
name = st.text_input('What is your name?').capitalize()

#Get the feature input from the user
def get_user_input():

    Glucose = st.number_input('Tingkat glukosa')
    BloodPressure = st.number_input('Nilai Tekanan Darah:')
    SkinThickness = st.number_input('Nilai ketebalan kulit:')
    Insulin = st.number_input('Tingkat insulin:')
    BMI = st.number_input('Nilai BMI:')
    DiabetesPedigreeFunction = st.number_input('Silisilah Diabetes (nilai)')
    Age = st.number_input('Usia')
    
    user_data = (Glucose, BloodPressure, SkinThickness, 
                 Insulin, BMI, DiabetesPedigreeFunction, Age)
    
    input_data_as_numpy_array= np.asarray(user_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    scaler = MinMaxScaler().fit(X_train)
    x = scaler.transform(input_data_reshaped)
    return x

user_input = get_user_input()

bt = st.button('Get Result')

model  = load_model("C:/Python/VS/UKSW/AI/diabets.h5")

if bt:
    
    prediction = model.predict(user_input)
    prediction = np.argmax(prediction, axis=1)

    if (prediction[0]== 0):
        st.write("Horeee!", name, "Tidak Berisiko Diabetes.")
        
    else:
        st.write(name,", Berisiko Diabetes. Konsultasikan dengan dokter.")