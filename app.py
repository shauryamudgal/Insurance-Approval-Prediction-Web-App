from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import plotly
import json
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the model and preprocessing
model = joblib.load('system.pkl')

# Define your numerical and categorical columns (same as in your notebook)
numerical_cols = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 
                 'max_torque', 'max_power', 'displacement']
categorical_cols = ['area_cluster', 'make', 'segment', 'model', 'fuel_type', 
                   'engine_type', 'rear_brakes_type', 'transmission_type', 'steering_type']

# Recreate your preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('scaler', StandardScaler()), ('power', PowerTransformer())]), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Preprocess the data
        processed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        proba = model.predict_proba(processed_data)[0]
        
        # Create visualization
        fig = px.bar(x=['No Claim', 'Claim'], y=proba, 
                    title='Prediction Probability', 
                    labels={'y': 'Probability', 'x': 'Outcome'})
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('result.html', 
                             prediction=prediction,
                             probabilities=proba,
                             graphJSON=graphJSON)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)