from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

main = Blueprint('main', __name__)

# Load models
try:
    model = joblib.load('models/best_churn_model.pkl')
    print("✅ Loaded best_churn_model.pkl")
except:
    try:
        model = joblib.load('models/quick_churn_model.pkl')
        print("✅ Loaded quick_churn_model.pkl")
    except:
        print("❌ No model found")
        model = None

try:
    model_comparison = joblib.load('models/model_comparison_results.pkl')
    print("✅ Loaded model_comparison_results.pkl")
except:
    model_comparison = {'best_model_name': 'Random Forest', 'best_score': 0.85}
    print("ℹ️ Using default model info")

@main.route('/')
def home():
    """Home page with prediction form"""
    model_info = {
        'name': model_comparison.get('best_model_name', 'Random Forest'),
        'auc': round(model_comparison.get('best_score', 0.85), 4),
        'balance_method': 'smote'
    }
    return render_template('index.html', model_info=model_info)

@main.route('/predict', methods=['POST'])
def predict_churn():
    """API endpoint for churn prediction"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Prediction model not available. Please train the model first.'
        }), 503
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Preprocess the input
        processed_data = preprocess_customer_data(form_data)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0, 1]
        churn_prediction = model.predict(processed_data)[0]
        
        # Interpret results
        if churn_probability > 0.7:
            risk_level = "Very High"
            risk_color = "danger"
        elif churn_probability > 0.5:
            risk_level = "High"
            risk_color = "warning"
        elif churn_probability > 0.3:
            risk_level = "Medium"
            risk_color = "info"
        else:
            risk_level = "Low"
            risk_color = "success"
        
        # Generate recommendations
        recommendations = generate_business_recommendations(form_data, churn_probability)
        
        return jsonify({
            'success': True,
            'prediction': {
                'churn_probability': round(churn_probability * 100, 2),
                'churn_prediction': bool(churn_prediction),
                'risk_level': risk_level,
                'risk_color': risk_color
            },
            'recommendations': recommendations,
            'model_info': {
                'name': model_comparison.get('best_model_name', 'Random Forest'),
                'auc_score': model_comparison.get('best_score', 0.85)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 400

@main.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

def preprocess_customer_data(input_data):
    """Preprocess single customer data for prediction"""
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Convert to appropriate data types
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Basic feature engineering
    df['value_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['is_new_customer'] = (df['tenure'] <= 3).astype(int)
    df['is_loyal_customer'] = (df['tenure'] > 24).astype(int)
    
    # Service count
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    df['total_services'] = df[service_columns].sum(axis=1)
    
    # Binary encoding
    binary_mappings = {
        'gender': {'Female': 1, 'Male': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-hot encoding for critical features
    if 'Contract' in df.columns:
        contract_dummies = pd.get_dummies(df['Contract'], prefix='Contract')
        df = pd.concat([df, contract_dummies], axis=1)
    
    if 'InternetService' in df.columns:
        internet_dummies = pd.get_dummies(df['InternetService'], prefix='InternetService')
        df = pd.concat([df, internet_dummies], axis=1)
    
    if 'PaymentMethod' in df.columns:
        payment_dummies = pd.get_dummies(df['PaymentMethod'], prefix='PaymentMethod')
        df = pd.concat([df, payment_dummies], axis=1)
    
    # Ensure we have all expected features
    expected_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 
        'is_new_customer', 'is_loyal_customer', 'total_services',
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'PaymentMethod_Bank transfer', 'PaymentMethod_Credit card', 
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the features we need
    df = df[expected_features]
    
    return df

def generate_business_recommendations(customer_data, churn_prob):
    """Generate actionable business recommendations"""
    recommendations = []
    
    # Risk-based recommendations
    if churn_prob > 0.7:
        recommendations.append("🚨 CRITICAL: Immediate retention action required!")
        recommendations.append("📞 Assign to premium retention specialist")
        recommendations.append("💰 Offer personalized retention package")
    elif churn_prob > 0.5:
        recommendations.append("⚠️ HIGH RISK: Proactive retention needed")
        recommendations.append("🎯 Schedule personal check-in call")
        recommendations.append("💡 Review service usage patterns")
    
    # Contract-based recommendations
    if customer_data.get('Contract') == 'Month-to-month':
        recommendations.append("📝 Offer 15% discount for 1-year contract commitment")
    elif customer_data.get('Contract') == 'One year':
        recommendations.append("🔄 Consider early renewal with upgrade incentive")
    
    # Service-based recommendations
    if customer_data.get('OnlineSecurity') == 'No' and customer_data.get('InternetService') != 'No':
        recommendations.append("🔒 Offer free 3-month trial of Online Security")
    
    if customer_data.get('TechSupport') == 'No':
        recommendations.append("🛠️ Proactively offer tech support assessment")
    
    # Tenure-based recommendations
    tenure = int(customer_data.get('tenure', 0))
    if tenure <= 6:
        recommendations.append("👋 Welcome package: Offer complimentary service for 1 month")
    elif tenure > 24:
        recommendations.append("🏆 Loyalty reward: Special discount for long-term customers")
    
    if len(recommendations) < 3:
        recommendations.extend([
            "📊 Monitor customer engagement metrics",
            "🎁 Consider small loyalty incentives", 
            "✅ Maintain regular quality check-ins"
        ])
    
    return recommendations[:6]