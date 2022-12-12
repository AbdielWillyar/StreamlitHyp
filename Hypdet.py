# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:37:19 2022

@author: Abdiel Wilyar
"""


import streamlit as st 
import numpy as np 
from keras.models import load_model
#import pandas as pd 
#import streamlit.components.v1 as components
from sklearn import preprocessing


# heading
st.markdown("<h1 style='text-align: center; color: blue;'>HypeDetect</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>...a Hypertentcy detection system</h4><br>", unsafe_allow_html=True)

st.write("Hipertensi adalah pengertian medis dari penyakit tekanan darah tinggi. Kondisi ini dapat menyebabkan berbagai macam komplikasi kesehatan yang membahayakan nyawa jika dibiarkan. Bahkan, gangguan ini dapat menyebabkan peningkatan risiko terjadinya penyakit jantung, stroke, hingga kematian. " )

name = st.text_input('What is your name?').capitalize()

#Get the feature input from the user
def get_user_input():

    age = st.number_input('Usia dalam kategori. 1 = 18-24, 9 = 60-64, 13 = 80 atau lebih')
    sex = st.number_input('Jenis Kelamin (0 = perempuan, 1 = laki-laki)')
    HighChol = st.number_input('HighColestrol? (0: tidak, 1: iya)')
    CholCheck = st.number_input('CholCheck dalam 5 Tahun (: tidak, 1: iya)')
    BMI = st.number_input('BMI')
    Smoker = st.number_input('Pernah merokok 100 batang sepanjang hidup (0: tidak, 1: iya)')
    HeartDiseaseorAttack = st.number_input('Penyakit Jantung (0: tidak, 1: iya)')
    PhysActivity = st.number_input('Aktivitas Fisik dalam 30 hari terkahir - tidak termasuk pekerjaan (0: tidak, 1: iya)')
    Fruits = st.number_input('Mengkonsumsi Buah satu kali atau lebih per hari (0 = tidak, 1 = ya)')
    Veggies = st.number_input('Mengkonsumsi Sayuran 1 kali atau lebih per hari (0 = tidak, 1 = ya)')
    HvyAlcoholConsump = st.number_input('Laki-laki dewasa: lebih dari 14 minuman per minggu. Wanita dewasa: lebih dari 7 minuman per minggu (0 = tidak, 1 = ya)')
    GenHlth = st.number_input('Apakah Anda akan mengatakan bahwa secara umum kesehatan Anda adalah: (skala 1-5) 1 = sangat baik, 2 = sangat baik, 3 = baik, 4 = cukup baik, 5 = buruk')
    DiffWalk = st.number_input('Apakah Anda mengalami kesulitan serius saat berjalan atau menaiki tangga? 0 = tidak, 1 = ya')
    
 
    user_data = (age, sex, HighChol, CholCheck, BMI, Smoker,
                 HeartDiseaseorAttack, PhysActivity,
                 Fruits,Veggies,HvyAlcoholConsump,GenHlth,DiffWalk)
    input_data_as_numpy_array= np.asarray(user_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    return input_data_reshaped

user_input = get_user_input()

bt = st.button('Get Result')

mm_scaler = preprocessing.MinMaxScaler()
data = mm_scaler.fit_transform(user_input)
model  = load_model("hypmod.h5")

if bt:
    
    prediction = model.predict(data)
    prediction = np.argmax(prediction, axis=1)

    if prediction == 1:
        st.write(name,", Berisiko Hypertency. Konsultasikan dengan dokter.")
        
    else:
        st.write('Horeee!', name, 'Tidak terdeteksi Hypertency.')
