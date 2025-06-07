# SDG2: Crop Yield Prediction using Machine Learning

**Project Title:** AI for Sustainable Agriculture  
**SDG Goal Addressed:** SDG 2 – Zero Hunger  
**Author:** Jefther Simeon Afuyo  
**Course:** AI for Sustainable Development – Week 2 Assignment (PLP Academy)  


## Project Overview

This project leverages **machine learning** to predict crop yield using environmental and agricultural features such as rainfall, temperature, humidity, wind speed, pressure, and fertilizer usage. The focus is global but with a particular emphasis on Kenya's agricultural context.

By providing accurate yield forecasts, it empowers farmers, policymakers, and stakeholders to make informed decisions to improve food security and promote sustainable farming practices.

The app uses multiple models for robust prediction and performance comparison:  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**  
- **Linear Regression**

Real-time weather data is fetched via OpenWeatherMap API to enrich prediction inputs.


## Problem Statement

Sub-Saharan Africa faces chronic food insecurity due to unpredictable crop yields influenced by climate variability, insufficient data tools, and farming practice challenges. This project builds a data-driven model to guide efficient food production and agricultural planning, helping to reduce hunger and support SDG 2.


## Machine Learning Approach

- **Models Used:**  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
  - Linear Regression  

- **ML Type:** Supervised Learning (Regression)  
- **Target Variable:** Crop Yield (kg/ha)  
- **Features:**  
  - Rainfall  
  - Temperature  
  - Humidity  
  - Wind Speed  
  - Pressure  
  - Fertilizer Usage (where available)  

- **Libraries Used:**  
  - `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`, `requests`


## Dataset

- **Source:**  
  - Historical crop yield and weather data compiled from [World Bank Open Data](https://data.worldbank.org/), [UN SDG Database](https://unstats.un.org/sdgs/indicators/database/), and Kaggle agricultural datasets.  
  - Real-time weather data accessed via OpenWeatherMap API.

- **Coverage:** Global with Kenya focus; covers major staple crops.

- **Preprocessing:** Data cleaned, normalized, and combined for model training.


**How to Run**
**1. Run Jupyter Notebook (for training and evaluation)**
pip install -r requirements.txt
jupyter notebook SDG2_Crop_Yield_Prediction.ipynb

Execute cells to load data, train models, compare performance, and visualize results.

**2. Run Streamlit Web App (for real-time prediction)**
pip install -r requirements.txt
streamlit run sdg2_crop_yield_app.py

Access at: http://localhost:8501

Enter your city (supports global locations, with Kenya focus).

Choose ML model and view crop yield predictions alongside weather data.

**Model Evaluation Summary**
Model	MAE	MSE	R² Score
Random Forest	2.13	7.54	0.91
Gradient Boosting	2.05	7.20	0.92
Linear Regression	4.32	19.87	0.68

Gradient Boosting Regressor slightly outperforms Random Forest; both outperform Linear Regression.

**Ethical Reflection**
**Data Bias:**
Regional data gaps and inconsistent data collection may affect model generalizability. Continuous retraining with diverse datasets is recommended.

**Fairness:**
Comparing multiple models ensures balanced and transparent predictions.

**Sustainability Impact:**
Empowers farmers and policymakers with data-driven insights to enhance food security and promote sustainable agriculture aligned with SDG 2.

**Future Work**
Incorporate satellite and remote sensing imagery (CNNs) for enhanced yield predictions.

Expand datasets to include socio-economic factors impacting agriculture.

Deploy the app on cloud platforms for global access and scalability.

Add crop-specific models and forecasts.

**Why This Matters**
“AI can be the bridge between innovation and sustainability.” — UN Tech Envoy

This project exemplifies AI's power to solve real-world global challenges like hunger by enabling smarter, more sustainable farming systems worldwide.

**Contact**
Author: Jefther Simeon Afuyo
afuyojefther@gmail.com
+254 796 090 806
GitHub: https://github.com/Jefther37
