import pandas as pd
import streamlit as st
import pickle
import requests
import streamlit as st
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import numpy as np
import os


df = pd.read_csv("G:\ML\car_price_prediction\main_folder\Cleanned_Car.csv")

with open('G:\ML\car_price_prediction\main_folder\linear_regg.pkl', 'rb') as file:
    pipe = pickle.load(file)

def get_car_names():
    return df['name']



def get_companies():

    return df['company'].unique()


def get_fuel_types():


    return df['fuel_type'].unique()

def do_prediction(item , model, year, km_travelled ,fuel_type):
    # Open the pickle file in binary read mode
    y = pipe.predict(pd.DataFrame([[item, model , year, km_travelled, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

    return y


st.set_page_config(layout="wide")


# Add some spacing
st.write("")
st.write("")

# Add a beautifully formatted title and subtitle
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to My Beautiful Car Price Prediction </h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #FF6347;'>Empowering my ML journey</h2>", unsafe_allow_html=True)


# Create a selectbox input field for single selection
selected_item = st.selectbox(
    "Select your favorite CAR from the list:",
    options= get_car_names()
)
# Create a selectbox input field for single selection
selected_model = st.selectbox(
    "Select your favorite MODEL :",
    options= get_companies()
)


# Create a selectbox input field for single selection

# Create a slider to select a year greater than 2013
year = st.slider("Select a year:", min_value=2014, max_value=2024, value=2016)


# Create a selectbox input field for single selection
selected_fuel_type = st.selectbox(
    "Select FUEL TYPE",
    options= get_fuel_types()
)

# Create a number input widget
km_travelled = st.number_input('Enter a number:', min_value=0, max_value=1000000, value=0, step=1)



# Add a call to action
if st.button("Predict Price"):

    y = do_prediction(selected_item,selected_model,year , km_travelled , selected_fuel_type  )
    st.write(f"Predicted Car price is :   {round(y[0], 2)}")

# Path to your car image
image_path = os.path.join('images', 'car_img.png')
# Display the car image using Streamlit
st.image(image_path, width=100, use_column_width=False)

