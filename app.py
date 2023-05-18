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
    funder = st.sidebar.selectbox('Who funded the well', ('Government Of Tanzania', 'Danida', 'Hesawa', 'Rwssp', 'World Bank', 'Kkkt','World Vision','Unicef', 'Other'))
    gps_height = st.sidebar.slider("Altitude of the well", 0, 3000, 1)
    installer = st.sidebar.selectbox('Installer', ('Government', 'DWE', 'RWE', 'Commu', 'DANIDA', 'Other'))
    basin = st.sidebar.selectbox('Basin', ('Lake Victoria', 'Pangani', 'Rufiji', 'Internal', 'Lake Tanganyika', 'Wami / Ruvu', 'Lake Nyasa','Ruvuma / Southern Coast', 'Lake Rukwa'))
    region = st.sidebar.selectbox('Region', ('Iringa', 'Shinyanga', 'Mbeya', 'Kilimanjaro', 'Morogoro', 'Arusha', 'Kagera', 'Mwanza', 'Kigoma', 'Ruvuma','Pwani', 'Tanga', 'Dodoma', 'Singida', 'Mara', 'Tabora', 'Rukwa', 'Mtwara', 'Manyara', 'Lindi', 'Dar es Salaam'))
    population = st.sidebar.slider("Population of Community", 0, 40000, 1)
    public_meeting = st.sidebar.selectbox('Public Meeting', ('TRUE', 'FALSE'))
    scheme_management = st.sidebar.selectbox('Scheme Management', ('VWC', 'WUG', 'Water authority', 'WUA', 'Water Board', 'Parastatal', 'Private operator', 'Company', 'SWC', 'Trust', 'Other'))
    permit = st.sidebar.selectbox('Permit', ('TRUE', 'FALSE'))
    extraction_type_class = st.sidebar.selectbox('Extraction Type', ('gravity', 'handpump', 'submersible', 'motorpump', 'rope pump', 'wind-powered', 'other'))
    management_group = st.sidebar.selectbox('Management Group', ('user-group', 'commercial', 'parastatal', 'unknown', 'other'))
    payment_type = st.sidebar.selectbox('Payment Type', ('never pay', 'per bucket', 'monthly', 'on failure', 'annually', 'unknown','other'))
    quality_group = st.sidebar.selectbox('Quality Group', ('good', 'salty', 'milky', 'colored', 'fluoride', 'unknown'))
    quantity_group = st.sidebar.selectbox('Quantity Group', ('enough', 'insufficient', 'dry', 'seasonal', 'unknown'))
    source_class = st.sidebar.selectbox('Source Class', ('groundwater', 'surface', 'unknown'))
    waterpoint_type_group = st.sidebar.selectbox('Waterpoint Type Group', ('communal standpipe', 'hand pump', 'spring', 'cattle trough', 'dam', 'other'))
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

        prediction_map = {0:'Functional',
                          1: 'Functional, but needs repair',
                          2: 'Not Functional'}

        st.success(f'Water Pump Condition: {prediction_map[prediction]}')

    pass

if __name__ == '__main__':
    main()