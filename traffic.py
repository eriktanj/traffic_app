# Import libraries
import streamlit as st
import pandas as pd
import pickle
from mapie.regression import MapieRegressor
import warnings
import numpy as np
warnings.filterwarnings('ignore')

st.title('Traffic Volume Predictor') 

st.write("Utilize our advanced Machine Learning application to predict traffic volume.")

# Display the image
st.image('traffic_image.gif', width = 1000)

# Load the pre-trained model from the pickle file
xg_pickle = open('xg_traffic.pickle', 'rb') 
xg = pickle.load(xg_pickle) 
xg_pickle.close()

# Create a sidebar for input collection
st.sidebar.image('traffic_sidebar.jpg', width = 500, caption = 'Traffic Volume Predictor')
st.sidebar.header('**Input Features**')
st.sidebar.write("You can either upload your data file or manually enter traffic features.")

###

# Option 1: Asking users to input their data as a file
with st.sidebar.expander('Option 1: Upload CSV File'):
    st.write('Upload a CSV file containing traffic details.')
    traffic_file = st.file_uploader('Choose a CSV file')
    st.subheader('Sample Data for Upload')
    st.write(pd.read_csv('traffic_data_user.csv').head(5))
    st.warning('Ensure your uploaded file has the same column names and data types as shown above.')

# Option 2: Asking users to input their data using a form in the sidebar
with st.sidebar.expander('Option 2: Fill Out Form'):
    st.write('Enter the traffic details manually using the form below.')
    default_df = pd.read_csv('Traffic_Volume.csv')
    default_df['date_time'] = pd.to_datetime(default_df['date_time'])
    default_df['month'] = default_df['date_time'].dt.month_name()
    default_df['weekday'] = default_df['date_time'].dt.day_name()
    default_df['hour'] = default_df['date_time'].dt.hour
    default_df = default_df.drop('date_time', axis=1)
    
    holiday = st.selectbox(
    "Choose whether today is a designated holiday or not",
    options=default_df["holiday"].fillna('None').unique()
    )
    temp = st.number_input(
        "Average temperature in Kelvin",
        min_value=float(default_df["temp"].min()),
        max_value=float(default_df["temp"].max()),
        step=0.1
    )
    rain_1h = st.number_input(
        "Amount in mm of rain that occurred in the hour",
        min_value=float(default_df["rain_1h"].min()),
        max_value=float(default_df["rain_1h"].max()),
        step=0.1
    )
    snow_1h = st.number_input(
        "Amount in mm of snow that occurred in the hour",
        min_value=float(default_df["snow_1h"].min()),
        max_value=float(default_df["snow_1h"].max()),
        step=0.1
    )
    clouds_all = st.number_input(
        "Percentage of cloud cover",
        min_value=float(default_df["clouds_all"].min()),
        max_value=float(default_df["clouds_all"].max()),
        step=1.0
    )
    weather_main = st.selectbox(
        "Choose the current weather",
        options=default_df["weather_main"].unique()
    )
    month = st.selectbox(
        "Choose month",
        options=default_df["month"].unique(),
    )
    weekday = st.selectbox(
        "Choose day of the week",
        options=default_df["weekday"].unique()
    )
    hour = st.selectbox(
        "Choose hour",
        options=default_df["hour"].unique()
    )
         
    button = st.button('Submit Form Data')


# If no file is provided, then allow user to provide inputs using the form
if traffic_file is None and not button:
    st.badge("Please choose a data input method to proceed.", color="blue")
if button:
    st.success("Form data submitted successfully.")
if traffic_file is None:
    # Encode the inputs for model prediction
    a = st.slider("Select alpha value for estimating prediction intervals", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns = ['traffic_volume'])

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
    encode_df['holiday'] = encode_df['holiday'].apply(lambda x: 1 if x != 'None' else 0)

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    
    # Using predict() with new data provided by the user
    new_prediction = xg.predict(user_encoded_df, alpha=a)

    # Extract mean prediction and bounds
    y_mean = int(new_prediction[0])
    y_lower = int(new_prediction[1][0, 0])
    y_upper = int(new_prediction[1][0, 1])

    # Show the predicted species on the app
    st.header(":green[**Predicting Traffic Volume...**]",)
    with st.container(border=True):
        st.write("Predicted Traffic Volume")
        st.header("${:,.2f}".format(y_mean))
    st.write("**Prediction Interval {}%**: [{:,.2f}, {:,.2f}]".format(int((1-a)*100), y_lower, y_upper))

else:
   # Loading data
   user_df = pd.read_csv(traffic_file) # User provided data
   original_df = pd.read_csv('Traffic_Volume.csv') # Original data to create ML model
   st.success("CSV file uploaded successfully.")
   a = st.slider("Select alpha value for estimating prediction intervals", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

   original_df['holiday'] = original_df['holiday'].apply(lambda x: 1 if x != 'None' else 0)

   original_df['date_time'] = pd.to_datetime(original_df['date_time'])
   original_df['month'] = original_df['date_time'].dt.month_name()
   original_df['weekday'] = original_df['date_time'].dt.day_name()
   original_df['hour'] = original_df['date_time'].dt.hour
   original_df = original_df.drop('date_time', axis=1)

   # Remove output (traffic_volume) column from original data
   original_df = original_df.drop(columns = ['traffic_volume'])

   # Ensure the order of columns in user data is in the same order as that of original data
   user_df = user_df[original_df.columns]

   # Concatenate two dataframes together along rows (axis = 0)
   combined_df = pd.concat([original_df, user_df], axis = 0)

   # Number of rows in original dataframe
   original_rows = original_df.shape[0]

   # Create dummies for the combined dataframe
   combined_df_encoded = pd.get_dummies(combined_df)

   # Split data into original and user dataframes using row index
   original_df_encoded = combined_df_encoded[:original_rows]
   user_df_encoded = combined_df_encoded[original_rows:]

   # Using predict() with new data provided by the user
   new_prediction = xg.predict(user_df_encoded, alpha=a)
   
   y_pred, y_pis = new_prediction
   user_df['Predicted Volume'] = np.round(y_pred.flatten()).astype(int)
   user_df['Lower Limit'] = np.round(y_pis[:, 0]).astype(int)
   user_df['Upper Limit'] = np.round(y_pis[:, 1]).astype(int)

   # Show the predicted prices on the app
   st.header("**Prediction Results with {}% Confidence Level**".format(int((1-a)*100)))
   st.dataframe(user_df)


st.header("**Model Performance and Inference**")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs Actual", "Coverage Plot"])

# Tab 1: Feature Importance
with tab1:
    st.write("### Feature Importance")
    st.image('feature_importance.svg')
    st.caption("Relative importance of features in prediction.")

# Tab 2: Histogram of Residuals
with tab2:
    st.write("### Histogram of Residuals")
    st.image('distribution_of_residuals.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")

# Tab 3: Predicted vs Actual
with tab3:
    st.write("### Predicted vs Actual")
    st.image('predicted_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")

# Tab 4: Coverage Plot
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_plot.svg')
    st.caption("Range of predictions with confidence intervals.")
