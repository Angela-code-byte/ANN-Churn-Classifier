import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


#load trained model
model = tf.keras.models.load_model('model.keras')

#Load the encoders and scalar

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


#Streamlit app
st.title('Customer Churn Prediction')

#User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', 0, 10)
num_of_products = st.number_input('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#prepare the input data
input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]], 
    'Geography': [geography],     
    'Age': [age],            
    'Balance': [balance],           
    'CreditScore': [credit_score],           
    'EstimatedSalary': [estimated_salary],           
    'Tenure': [tenure],           
    'NumOfProducts': [num_of_products],           
    'HasCrCard': [has_cr_card],           
    'IsActiveMember': [is_active_member]      
      })


# One-hot encode Geography (Fixing the error here)
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Drop the original 'Geography' column and merge encoded data
input_data = pd.concat([input_data.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure feature order matches what was used during training
expected_features = scaler.feature_names_in_  # Get expected feature names from the scaler
input_data = input_data.reindex(columns=expected_features, fill_value=0)  # Reorder and fill missing columns


#scale the input data
input_data_scaled=scaler.transform(input_data)


#prediction_churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


#display the prediction
if prediction_proba > 0.5:  
    st.write(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")
else:   
    st.write(f"The customer is likely to stay with the bank with a probability of {1-prediction_proba:.2f}")