import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from PIL import Image

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

    # load image
    image = Image.open('waterpump_img.png')

    # sidebar and title
    st.sidebar.header('Water Pump Features')

    # Display header and image
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(image, use_column_width=True)
    with col2:
        st.title("Predicting Functionality of Water Pump App")
    #st.header('Predicting Functionality of Water Pump App')
   

   # add text
    st.write('Predict the functionality of water pumps in Tanzania with the XGBoost model!')
    st.write('To access the codebase for this application, please visit the following GitHub repository: https://github.com/anair123/Detecting-Faulty-Water-Pumps-With-Machine-Learning')


    # get feature values
    amount_tsh = st.sidebar.slider("The amount of water available to the waterpoint", 0, 400000, 1)
    funder = st.sidebar.selectbox('Who funded the well', ('Government Of Tanzania', 'Danida', 'Hesawa', 'Rwssp', 'World Bank', 'Kkkt','World Vision','Unicef', 'Other'))
    gps_height = st.sidebar.slider("The altitude of the well", 0, 3000, 1)
    installer = st.sidebar.selectbox('Installer', ('Government', 'DWE', 'RWE', 'Commu', 'DANIDA', 'Other'))
    basin = st.sidebar.selectbox('Geographic water basin', ('Lake Victoria', 'Pangani', 'Rufiji', 'Internal', 'Lake Tanganyika', 'Wami / Ruvu', 'Lake Nyasa','Ruvuma / Southern Coast', 'Lake Rukwa'))
    region = st.sidebar.selectbox('Region', ('Iringa', 'Shinyanga', 'Mbeya', 'Kilimanjaro', 'Morogoro', 'Arusha', 'Kagera', 'Mwanza', 'Kigoma', 'Ruvuma','Pwani', 'Tanga', 'Dodoma', 'Singida', 'Mara', 'Tabora', 'Rukwa', 'Mtwara', 'Manyara', 'Lindi', 'Dar es Salaam'))
    population = st.sidebar.slider("Population of Community", 0, 40000, 1)
    public_meeting = st.sidebar.selectbox('Public Meeting', ('TRUE', 'FALSE'))
    scheme_management = st.sidebar.selectbox('Who operates the waterpoint', ('VWC', 'WUG', 'Water authority', 'WUA', 'Water Board', 'Parastatal', 'Private operator', 'Company', 'SWC', 'Trust', 'Other'))
    permit = st.sidebar.selectbox('Is the waterpoint permitted?', ('TRUE', 'FALSE'))
    extraction_type_class = st.sidebar.selectbox('The kind of extraction the waterpoint uses', ('gravity', 'handpump', 'submersible', 'motorpump', 'rope pump', 'wind-powered', 'other'))
    management_group = st.sidebar.selectbox('How the waterpoint is managed', ('user-group', 'commercial', 'parastatal', 'unknown', 'other'))
    payment_type = st.sidebar.selectbox('Payment Type', ('never pay', 'per bucket', 'monthly', 'on failure', 'annually', 'unknown','other'))
    quality_group = st.sidebar.selectbox('The quality of the water', ('good', 'salty', 'milky', 'colored', 'fluoride', 'unknown'))
    quantity_group = st.sidebar.selectbox('The quantity of the water', ('enough', 'insufficient', 'dry', 'seasonal', 'unknown'))
    source_class = st.sidebar.selectbox('The source of the water', ('groundwater', 'surface', 'unknown'))
    waterpoint_type_group = st.sidebar.selectbox('The type of waterpoint', ('communal standpipe', 'hand pump', 'spring', 'cattle trough', 'dam', 'other'))
    age = st.sidebar.slider("Age of the Water Pump", 0, 100, 1)    

    # create input matrix with user response
    input_features = pd.DataFrame(columns=['amount_tsh', 'funder', 'gps_height', 'installer', 'basin', 'region',
       'population', 'public_meeting', 'scheme_management', 'permit',
       'extraction_type_class', 'management_group', 'payment_type',
       'quality_group', 'quantity_group', 'source_class',
       'waterpoint_type_group', 'age'])
    
    # store input features in a pandas data frame
    input_features.loc[0] = [amount_tsh, funder, gps_height, installer, basin, region,
        population, public_meeting, scheme_management, permit,
        extraction_type_class, management_group, payment_type, quality_group,
        quantity_group, source_class, waterpoint_type_group, age]

    st.text("")
    st.text("")
    st.write('Click the button below after selecting the features o the water pump of interest:')
    # create button that generates prediction
    if st.button('Predict Water Pump Condition'):

        # make a prediction with the xgboost model
        input_features_processed = pipeline_xgb.transform(input_features)
        prediction = xgb_model.predict(input_features_processed)[0]

        # convert the yielded number to a prediction
        prediction_map = {0:'Functional',
                          1: 'Functional, but needs repair',
                          2: 'Not Functional'}

        # show prediction with color depending on the outcome
        if prediction == 0:
            st.success(f'Water Pump Condition: {prediction_map[prediction]}')
        elif prediction == 1:
            st.warning(f'Water Pump Condition: {prediction_map[prediction]}')
        elif prediction == 2:
            st.error(f'Water Pump Condition: {prediction_map[prediction]}') 


if __name__ == '__main__':
    main()