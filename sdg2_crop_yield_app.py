import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Weather data fetch function
def get_weather_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"]
        }
    else:
        return None

# Title and description
st.title("Crop Yield Predictor")
st.markdown("This AI-powered app supports **United Nation Sustainable Development Goal Two: Zero Hunger** by predicting crop yields using environmental data and machine learning.")

# Sidebar input for city and rainfall
st.sidebar.header("Weather Data Input")
city = st.sidebar.text_input("Enter City Name", "Nairobi")
api_key = "7df502202fb4c527011480a96a4e6838"

# Get weather data
weather = get_weather_data(city, api_key)

if weather:
    st.sidebar.metric("Temperature (°C)", weather["temperature"])
    st.sidebar.metric(" Humidity (%)", weather["humidity"])
    st.sidebar.metric(" Wind Speed (m/s)", weather["wind_speed"])
    st.sidebar.metric(" Pressure (hPa)", weather["pressure"])
else:
    st.sidebar.warning("⚠️ Unable to fetch weather data. Please check your city name or API key.")

# Load and show dataset
st.subheader("Dataset Preview")
df = pd.read_csv("crop_yield_data.csv")
st.dataframe(df.head())

# Machine Learning
X = df[['Temperature', 'Rainfall', 'Humidity', 'WindSpeed']]
y = df['CropYield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []
predictions = {}

rainfall_input = st.slider("Rainfall (mm)", 50, 200, 100)

# Train, evaluate, and predict
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    yield_prediction = model.predict([[weather["temperature"], rainfall_input, weather["humidity"], weather["wind_speed"]]])[0]
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R² Score": r2_score(y_test, y_pred)
    })
    predictions[name] = yield_prediction

# Results table
st.subheader("Model Performance")
st.dataframe(pd.DataFrame(results))

# Prediction outputs
st.subheader("Predicted Crop Yield (Tons/Ha)")
for model_name, pred in predictions.items():
    st.success(f"{model_name}: {pred:.2f} tons/ha")
