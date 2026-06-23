# Air Quality Index (AQI) Prediction

A data science project to predict AQI levels based on pollutant data.

## Dataset
- Source: [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

## What this project does
- Cleans and preprocesses AQI data
- Visualizes AQI distribution
- Trains Linear Regression and Random Forest models
- Saves the best model for future use

## Results
- Linear Regression MAE: 31.23542376378991 | R2: 0.8062193112323
- Random Forest MAE: 21.59214949678314 | R2: 0.9071607302558777

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python aqi.py`