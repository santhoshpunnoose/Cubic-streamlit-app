# Import libraries

import streamlit as st
import pandas as pd
import joblib

# Load our model pipeline object

model =joblib.load("cubicmodel.joblib")

# Add Title and Instruction

st.title("Cubic Regression")
st.subheader("Enter value between 0 and 10 to get the prediction")

# Age iput form
X = st.number_input(
    label = "01. Enter the X value",
    min_value = 0.000,
    max_value = 10.000,
    value=5.000)

st.sidebar.title("About This App")
st.sidebar.markdown("""
This app uses a quadratic regression model to predict outcomes based on a single input variable **X**.
Cubic models are useful when the relationship between variables follows a more complex curve, allowing for two bends in the data. This makes them ideal for capturing patterns that change direction more than once, such as growth followed by decline or vice versa.
Try entering different values to see how predictions change!
""")

# Submit inputs to model

if st.button("Submit For Prediction"):
    
    # Store our data in dataframe for prediction
    new_data = pd.DataFrame({"X": [X]})
    
    
    # Apply model pipeline to the input data and extract probability prediction
    
    #pred_proba = model.predict_proba(new_data)[0][1]
    pred_value = model.predict(new_data)[0]
    
    # Output prediction
    
    st.subheader(f"Based on the X value provided, the predicted value is {pred_value:.2f}")
    
st.markdown("---")
st.markdown("Created by Santhosh | Powered by Streamlit & scikit-learn")