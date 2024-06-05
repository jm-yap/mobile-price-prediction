import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Define a function to make predictions
def predict_rating(input_data, model):
    prediction = model.predict(input_data)
    return prediction[0]

# Define the Streamlit app
def main():
    feature_names = ['Ratings','Brand_Apple', 'ROM', 'Brand_Samsung', 'Battery_Power', 'RAM', 'Selfi_Cam', 'Primary_Cam']
    st.title('Smartphone Price Prediction')
    st.write('This app predicts the rating of a smartphone based on its features.')

    # Choose model
    model_name = st.radio('Select Model', ['Random Forest (Random)', 'Random Forest (Grid)', 'Neural Networks (Random)', 'Neural Networks (Grid)', 'Gradient Boosting (Random)', 'Gradient Boosting (Grid)'])

    if model_name == 'Random Forest (Random)':
        model = RandomForestRegressor(bootstrap=False, max_depth=7, max_features='sqrt',
                      min_samples_leaf=2, min_samples_split=5, n_estimators=126,
                      random_state=1)
    elif model_name == 'Random Forest (Grid)':
        model = RandomForestRegressor(max_depth=7, max_features='sqrt', n_estimators=50,
                      random_state=1)
    elif model_name == 'Neural Networks (Random)':
        model = MLPRegressor(alpha=0.016218168996279973, hidden_layer_sizes=478,
             learning_rate_init=0.009203793237806832, max_iter=1000,
             random_state=69)
    elif model_name == 'Neural Networks (Grid)':
        model = MLPRegressor(alpha=0.01, hidden_layer_sizes=(50, 25), learning_rate_init=0.1,
             max_iter=500, random_state=427)
    elif model_name == 'Gradient Boosting (Random)':
        model = GradientBoostingRegressor(learning_rate=0.006400495458402238, max_depth=6,
                          n_estimators=276, random_state=42)
    else:
        model = GradientBoostingRegressor(learning_rate=0.05, random_state=42)
    
    model.fit(X_train, y_train)

    # Create input fields for user input
    ratings = st.number_input('Ratings', value=3, min_value=0, max_value=5)
    brand = st.radio('Brand', ['Apple', 'Samsung', 'Other'])
    rom = st.number_input('ROM', value=64, min_value=0, max_value=256, step=1)
    battery_power = st.number_input('Battery Power', value=4000, min_value=500, max_value=5000, step=1)
    ram = st.number_input('RAM', value=4, min_value=0, max_value=8, step=1)
    selfi_cam = st.number_input('Selfi Cam', value=16, min_value=0, max_value=20, step=1)
    primary_cam = st.number_input('Primary Cam', value=20, min_value=0, max_value=25, step=1)

    if brand == 'Apple':
      brand_apple = 1
      brand_samsung = 0
    elif brand == 'Samsung':
      brand_apple = 0
      brand_samsung = 1
    else:
      brand_apple = 0
      brand_samsung = 0

    # Make a prediction and display the result
    if st.button('Predict'):
        input_data = [ratings, brand_apple, rom, brand_samsung, battery_power, ram, selfi_cam, primary_cam]
        # Convert input data to DataFrame with feature names
        input_data = pd.DataFrame([input_data], columns=feature_names)
        # Fit the scaler on the training data
        scaler = MinMaxScaler().fit(X_train)
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        prediction = predict_rating(input_data_scaled, model)
        st.write(f'The predicted price is {prediction}')

if __name__ == '__main__':
    main()