import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

st.cache(allow_output_mutation=True)
def load_pipeline_and_model():
    """load the pipeline object for preprocessing and the ml model"""

    with open('models/xgb_preprocessing.pkl', 'rb') as f:
        pipeline_xgb = pickle.load(f)

    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    return pipeline_xgb, xgb_model

def main():

     # load preprocessing pipeline and model
    pipeline_xgb, xgb_model = load_pipeline_and_model()

    # side bar and title
    st.sidebar.header('Water Pump Features')
    st.header('Water Pump Functionality Prediction App')


    # get feature values
    amount_tsh = st.sidebar.slider(" Total static head (amount water available to waterpoint)", 0, 400000, 1)
    funder = st.sidebar.selectbox('Who funded the well', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    gps_height = st.sidebar.slider("Altitude of the well", 0, 3000, 1)s
    installer = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    basin = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    region = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    population = st.sidebar.slider("Artist's Number of Followers", 0, 40000, 1)
    public_meeting = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    scheme_management = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    permit = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    extraction_type_class = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    management_group = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    payment_type = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    quality_group = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    quantity_group = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    source_class = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    waterpoint_type_group = st.sidebar.selectbox('Genre', ('Pop', 'Rock', 'Rap', 'Country', 'EDM', 'Other'))
    age = st.sidebar.slider("Age of the Water Pump", 0, 100, 1)    

    # create input matrix with user response
    input_features = pd.DataFrame(columns=['amount_tsh', 'funder', 'gps_height', 'installer', 'basin', 'region',
       'population', 'public_meeting', 'scheme_management', 'permit',
       'extraction_type_class', 'management_group', 'payment_type',
       'quality_group', 'quantity_group', 'source_class',
       'waterpoint_type_group', 'age'])
    

    input_features.loc[0] = [amount_tsh, funder, gps_height, installer, basin, region,
        population, public_meeting, scheme_management, permit,
        extraction_type_class, management_group, payment_type, quality_group,
        quantity_group, source_class, waterpoint_type_group, age]


    # create button that generates prediction
    if st.button('Predict Water Pump Condition'):
        input_features_processed = pipeline_xgb.transform(input_features)
        prediction = xgb_model.predict(input_features_processed)[0]
        st.success(f'Water Pump Condition: {prediction}')

    pass

['amount_tsh', 'funder', 'gps_height', 'installer', 'basin', 'region',
       'population', 'public_meeting', 'scheme_management', 'permit',
       'extraction_type_class', 'management_group', 'payment_type',
       'quality_group', 'quantity_group', 'source_class',
       'waterpoint_type_group', 'status_group', 'age']