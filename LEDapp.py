import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Define a function to make predictions
def predict_rating(input_data, model):
    prediction = model.predict(input_data)
    return prediction[0]

# Define the Streamlit app
def main():
    st.title('Smartphone Price Prediction')
    st.write('This app predicts the rating of a smartphone based on its features.')

    # Choose model
    model_name = st.radio('Select Model', ['Random Forest (Random)', 'Random Forest (Grid)', 'Neural Networks (Random)', 'Neural Networks (Grid)', 'Gradient Boosting (Random)', 'Gradient Boosting (Grid)'])
    file = open(f'{model_name}.pkl', 'rb')
    
    model = pickle.load(file)
    print("====",model)
    model.fit(X_train, y_train)

    # Create input fields for user input
    ratings = st.number_input('Ratings', value=3, min_value=0, max_value=5)
    rom = st.number_input('ROM', value=64, min_value=0, max_value=256, step=1)
    battery_power = st.number_input('Battery Power', value=4000, min_value=500, max_value=6000, step=1)
    ram = st.number_input('RAM', value=4, min_value=0, max_value=8, step=1)
    selfi_cam = st.number_input('Selfi Cam', value=16, min_value=0, max_value=61, step=1)
    primary_cam = st.number_input('Primary Cam', value=20, min_value=0, max_value=64, step=1)
    # Define the list of brands
    brands = [
    "Alcatel", "Apple", "Blacear", "Blacerry", "Black Shark", "BlackZone", "Callbar", "Detel", "Dublin", "Easyfone", "Ecotel", "F-Fook", 
    "Forme", "GAMMA", "Gee", "Gfive", "Good One", "Google", "Grabo", "GreenBerry", 
    "Heemax", "Hicell", "Honor", "Huawei", "I Kall", "Infinix", "InFocus", "Inovu", 
    "Intex", "iQOO", "Itel", "JIVI", "Jmax", "Karbonn", "Kechaoda", "Lava", "Lenovo", 
    "LG", "Mafe", "Megus", "Meizu", "Mi", "Moto", "MTR", 
    "Muphone", "Mymax", "Nexus", "Nokia", "OnePlus", "OPPO", "Peace", "POCO", 
    "Q-Tel", "Realme", "Redmi", "Unknown"
    ]


    # Create a dictionary to store brand values
    brand_values = {f'Brand_{brand}': 0 for brand in brands}

    # Make the selected brand 1
    selected_brand = st.selectbox('Brand', brands)
    brand_values[f'Brand_{selected_brand}'] = 1

    # Make a prediction and display the result
    if st.button('Predict'):


        # input data without model
        feature_names = ['Ratings', 'ROM', 'Battery_Power', 'RAM', 'Selfi_Cam', 'Primary_Cam'] + list(brand_values.keys())
        input_data = np.array([ratings, rom, battery_power, ram, selfi_cam, primary_cam] + list(brand_values.values())).reshape(1, -1)
        input_data = pd.DataFrame(input_data, columns=feature_names)

        # Fit the scaler on the training data with the same feature names used during the fit step
        scaler = MinMaxScaler().fit(X_train[feature_names])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data[feature_names])
        prediction = predict_rating(input_data_scaled, model)
        st.write(f'The predicted price is {prediction}')
        st.write(f'The model used is {model}')

if __name__ == '__main__':
    main()