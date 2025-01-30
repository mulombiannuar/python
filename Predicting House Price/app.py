import streamlit as st
import joblib
import pandas as pd

prediction_model = joblib.load('house_price_prediction_model.pkl')  
st.title('Linear Regression Prediction App')
st.write('Enter feature value to get prediction')

feature_value = st.number_input('Enter feature value', min_value=500, max_value=50000, value=1200)
if st.button('Predict House Price'):
    prediction = prediction_model.predict(pd.DataFrame([[feature_value]], columns=['HouseSize']))
    #st.write(f'Prediction value is : {prediction[0]:.2f}')
    st.write('Prediction value is :', round(prediction[0], 2)) 
    
    