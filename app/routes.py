from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

main = Blueprint('main', __name__)

# Load trained models and preprocessors
try:
    model = joblib.load('models/best_churn_model.pkl')
    preprocessor = joblib.load('models/feature_preprocessor.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    model_comparison = joblib.load('models/model_comparison_results.pkl')
    
    print("✅ All artifacts loaded successfully!")
    print(f"Best model: {model_comparison['best_model_name']}")
    print(f"Best AUC: {model_comparison['best_score']:.4f}")
    
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    model = None
    preprocessor = None

def preprocess_customer_data(input_data):
    """Preprocess single customer data for prediction"""
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Convert to appropriate data types
    df['tenure'] = int(df['tenure'])
    df['MonthlyCharges'] = float(df['MonthlyCharges'])
    df['TotalCharges'] = float(df['TotalCharges'])
    
    # Feature engineering (same as training)
    df['value_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['avg_monthly_value'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] <= 3).astype(int)
    df['is_loyal_customer'] = (df['tenure'] > 24).astype(int)
    df['tenure_squared'] = df['tenure'] ** 2
    
    # Service count
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    df['total_services'] = df[service_columns].sum(axis=1)
    df['has_premium_services'] = (df['total_services'] >= 3).astype(int)
    
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
    
    # One-hot encoding
    multi_category_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    
    for col in multi_category_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Ensure all training features are present
    for feature in preprocessor['feature_names']:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns
    df = df[preprocessor['feature_names']]
    
    # Scale numerical features
    numerical_columns = preprocessor['numerical_columns']
    existing_num_cols = [col for col in numerical_columns if col in df.columns]
    
    df[existing_num_cols] = preprocessor['scaler'].transform(df[existing_num_cols])
    
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
    if customer_data['Contract'] == 'Month-to-month':
        recommendations.append("📝 Offer 15% discount for 1-year contract commitment")
    elif customer_data['Contract'] == 'One year':
        recommendations.append("🔄 Consider early renewal with upgrade incentive")
    
    # Service-based recommendations
    if customer_data['OnlineSecurity'] == 'No' and customer_data['InternetService'] != 'No':
        recommendations.append("🔒 Offer free 3-month trial of Online Security")
    
    if customer_data['TechSupport'] == 'No':
        recommendations.append("🛠️ Proactively offer tech support assessment")
    
    # Tenure-based recommendations
    tenure = int(customer_data['tenure'])
    if tenure <= 6:
        recommendations.append("👋 Welcome package: Offer complimentary service for 1 month")
    elif tenure > 24:
        recommendations.append("🏆 Loyalty reward: Special discount for long-term customers")
    
    # Payment recommendations
    if 'Electronic check' in customer_data['PaymentMethod']:
        recommendations.append("💳 Suggest automatic payment for convenience discount")
    
    if len(recommendations) < 3:
        recommendations.extend([
            "📊 Monitor customer engagement metrics",
            "🎁 Consider small loyalty incentives",
            "✅ Maintain regular quality check-ins"
        ])
    
    return recommendations[:6]  # Return top 6 recommendations

@main.route('/')
def home():
    """Home page with prediction form"""
    model_info = {
        'name': model_comparison.get('best_model_name', 'Unknown'),
        'auc': round(model_comparison.get('best_score', 0), 4),
        'balance_method': preprocessor.get('balance_method', 'smote') if preprocessor else 'Unknown'
    }
    
    return render_template('index.html', model_info=model_info)

@main.route('/predict', methods=['POST'])
def predict_churn():
    """API endpoint for churn prediction"""
    if model is None or preprocessor is None:
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
                'name': model_comparison['best_model_name'],
                'auc_score': model_comparison['best_score']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 400

@main.route('/model-info')
def get_model_info():
    """API endpoint to get model information"""
    if model_comparison is None:
        return jsonify({'error': 'Model information not available'}), 404
    
    return jsonify({
        'best_model': model_comparison['best_model_name'],
        'best_auc': model_comparison['best_score'],
        'all_models': model_comparison['all_models']
    })

@main.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'feature_info_loaded': feature_info is not None
    }
    
    return jsonify(status)