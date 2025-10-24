from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

main = Blueprint('main', __name__)

# Load the trained model and preprocessor
try:
    model = joblib.load('models/telco_churn_model.pkl')
    preprocessor = joblib.load('models/feature_preprocessor.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print(" Model and preprocessor loaded successfully!")
    print(f"Model: {model_info['model_name']}, AUC: {model_info['auc_score']:.4f}")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None
    preprocessor = None

def preprocess_single_input(input_data, preprocessor):
    """Preprocess a single customer record for prediction"""
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Feature engineering (same as during training)
    df['value_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    
    # Service count
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    df['service_count'] = df[service_columns].sum(axis=1)
    
    # Handle categorical variables (same encoding as training)
    categorical_mappings = {
        'gender': {'Female': 1, 'Male': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-hot encoding for other categorical variables
    categorical_for_dummies = ['Contract', 'InternetService', 'PaymentMethod']
    
    for col in categorical_for_dummies:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Ensure all training features are present
    for feature in preprocessor['feature_names']:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match training
    df = df[preprocessor['feature_names']]
    
    # Scale numerical features
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 'service_count']
    existing_num_cols = [col for col in numerical_columns if col in df.columns]
    
    df[existing_num_cols] = preprocessor['scaler'].transform(df[existing_num_cols])
    
    return df

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get form data
        form_data = {
            'tenure': float(request.form['tenure']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
            'gender': request.form['gender'],
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod']
        }
        
        # Preprocess input
        processed_data = preprocess_single_input(form_data, preprocessor)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0, 1]
        churn_prediction = model.predict(processed_data)[0]
        
        # Interpret results
        risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.4 else "Low"
        
        # Generate recommendations
        recommendations = generate_business_recommendations(form_data, churn_probability)
        
        return jsonify({
            'success': True,
            'churn_probability': round(churn_probability * 100, 2),
            'churn_prediction': bool(churn_prediction),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'model_info': {
                'name': model_info.get('model_name', 'Unknown'),
                'auc_score': round(model_info.get('auc_score', 0), 4)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def generate_business_recommendations(customer_data, churn_prob):
    """Generate actionable business recommendations"""
    recommendations = []
    
    # High risk immediate actions
    if churn_prob > 0.7:
        recommendations.append(" IMMEDIATE ACTION: Customer at very high risk of churn!")
        recommendations.append(" Assign to retention specialist immediately")
        recommendations.append(" Proactive outreach from customer success team")
    
    # Contract-based recommendations
    if customer_data['Contract'] == 'Month-to-month':
        recommendations.append(" Offer contract upgrade discount (15-20% off for 1-year commitment)")
    elif customer_data['Contract'] == 'One year':
        recommendations.append(" Consider early renewal incentive for 2-year contract")
    
    # Service-based recommendations
    if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No':
        recommendations.append(" Offer free trial of Online Security services")
    
    if customer_data['TechSupport'] == 'No':
        recommendations.append(" Proactively offer tech support package")
    
    # Payment and billing recommendations
    if customer_data['PaymentMethod'] == 'Electronic check':
        recommendations