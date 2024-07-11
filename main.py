import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved Keras model
model = load_model('prediction_model.keras')

# Function to predict house prices
def predict_price(features):
    # Process features as needed (scaling, encoding, etc.)
    # Example: scaling features assuming 'preprocessor' is a fitted scaler
    # features = preprocessor.transform(features)

    # Make predictions using the loaded model
    predictions = model.predict(features)

    return predictions

# Function to display example inputs and explanations
def show_example_inputs():
    st.subheader('Example Inputs and Explanations')
    st.markdown("""
    Below are example inputs for features related to predicting house prices, along with brief explanations:

    - **Longitude**: Represents the geographic location, which can affect property values based on proximity to amenities and urban centers.
    - **Latitude**: Similar to longitude, geographic location impacts property values based on local demand and environmental factors.
    - **Housing Median Age**: Indicates the age of housing units in the area, influencing maintenance costs and property attractiveness.
    - **Total Rooms**: Total number of rooms can reflect the size and perceived value of the property.
    - **Total Bedrooms**: Number of bedrooms is a key factor influencing property size and family suitability.
    - **Population**: Higher population density may indicate demand pressure on housing and influence property prices.
    - **Households**: Number of households can reflect local community dynamics and housing demand.
    - **Median Income**: Median income of residents correlates strongly with housing affordability and price levels.

    Adjust these inputs to explore how they impact the predicted house price.
    """)

# Streamlit UI
st.title('House Price Prediction')


# Example input fields (replace with actual UI input elements)
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.number_input('Housing Median Age')
total_rooms = st.number_input('Total Rooms')
total_bedrooms = st.number_input('Total Bedrooms')
population = st.number_input('Population')
households = st.number_input('Households')
median_income = st.number_input('Median Income')
ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# Button to show example inputs and explanations
if st.button('Show Example Inputs and Explanations'):
    show_example_inputs()

# Example feature array (replace with actual feature handling)
features = np.array([[longitude, latitude, housing_median_age, total_rooms,
                      total_bedrooms, population, households, median_income]])

# Perform any necessary preprocessing on 'features' to match the expected input shape
# Example: One-hot encode 'ocean_proximity' if needed

# Button to predict
if st.button('Predict'):
    # Ensure 'features' has the correct shape (1, 13) if needed
    # Example: Add one-hot encoded 'ocean_proximity' feature
    if ocean_proximity == 'NEAR BAY':
        features = np.append(features, [[1, 0, 0, 0, 0]], axis=1)
    elif ocean_proximity == '<1H OCEAN':
        features = np.append(features, [[0, 1, 0, 0, 0]], axis=1)
    elif ocean_proximity == 'INLAND':
        features = np.append(features, [[0, 0, 1, 0, 0]], axis=1)
    elif ocean_proximity == 'NEAR OCEAN':
        features = np.append(features, [[0, 0, 0, 1, 0]], axis=1)
    elif ocean_proximity == 'ISLAND':
        features = np.append(features, [[0, 0, 0, 0, 1]], axis=1)

    # Make sure 'features' has shape (1, 13)
    assert features.shape == (1, 13), f'Expected shape (1, 13), got {features.shape}'

    # Perform prediction with the model
    prediction = predict_price(features)
    st.write(f'Predicted Price: ${prediction[0][0]:,.2f}')
