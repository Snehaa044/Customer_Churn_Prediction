from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback

main = Blueprint('main', __name__)

# Load models
try:
    model = joblib.load('models/best_churn_model.pkl')
    print("✅ Loaded best_churn_model.pkl")
except Exception as e:
    print(f"❌ Error loading best_churn_model.pkl: {e}")
    try:
        model = joblib.load('models/quick_churn_model.pkl')
        print("✅ Loaded quick_churn_model.pkl")
    except Exception as e:
        print(f"❌ Error loading quick_churn_model.pkl: {e}")
        model = None

try:
    model_comparison = joblib.load('models/model_comparison_results.pkl')
    print("✅ Loaded model_comparison_results.pkl")
except Exception as e:
    print(f"❌ Error loading model_comparison_results.pkl: {e}")
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
        print("📥 Received prediction request...")
        
        # Get form data
        form_data = request.form.to_dict()
        print(f"Form data: {form_data}")
        
        # Check required fields
        required_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
        missing_fields = [field for field in required_fields if field not in form_data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Preprocess the input
        processed_data = preprocess_customer_data(form_data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0, 1]
        churn_prediction = model.predict(processed_data)[0]
        
        print(f"Prediction - Probability: {churn_probability:.4f}, Class: {churn_prediction}")
        
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
        print(f"❌ Prediction error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 400

@main.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': model_comparison.get('best_model_name', 'Unknown')
    })

@main.route('/debug')
def debug_info():
    """Debug endpoint to check model and feature information"""
    debug_info = {
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else 'None',
        'model_features': len(model.feature_importances_) if model and hasattr(model, 'feature_importances_') else 0,
        'expected_features_sample': []
    }
    
    if model and hasattr(model, 'feature_importances_'):
        debug_info['expected_features_sample'] = [f'feature_{i}' for i in range(min(10, len(model.feature_importances_)))]
    
    return jsonify(debug_info)

@main.route('/test')
def test_prediction():
    """Test endpoint with sample data"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Sample test data
    test_data = {
        'tenure': '12',
        'MonthlyCharges': '70.35', 
        'TotalCharges': '844.20',
        'gender': 'Male',
        'Partner': 'Yes',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check'
    }
    
    try:
        processed_data = preprocess_customer_data(test_data)
        churn_probability = model.predict_proba(processed_data)[0, 1]
        
        return jsonify({
            'success': True,
            'test_prediction': f'Churn probability: {churn_probability:.4f}',
            'processed_features': processed_data.shape[1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def preprocess_customer_data(input_data):
    """Preprocess single customer data for prediction"""
    try:
        print("🔄 Starting data preprocessing...")
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input columns: {list(df.columns)}")
        
        # Convert to appropriate data types with proper error handling
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill NaN values before converting to int
        df['tenure'] = df['tenure'].fillna(0).astype(int)
        df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        print(f"After numeric conversion - Tenure: {df['tenure'].iloc[0]}, Monthly: {df['MonthlyCharges'].iloc[0]}, Total: {df['TotalCharges'].iloc[0]}")
        
        # Basic feature engineering
        df['value_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'].replace(0, 1) + 1)  # Avoid division by zero
        df['is_new_customer'] = (df['tenure'] <= 3).astype(int)
        df['is_loyal_customer'] = (df['tenure'] > 24).astype(int)
        
        # Service count - handle missing columns gracefully
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
            else:
                df[col] = 0  # Default value if column missing
        
        df['total_services'] = df[service_columns].sum(axis=1)
        
        # Binary encoding with default values
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
            else:
                df[col] = 0  # Default value
        
        # One-hot encoding for categorical features
        categorical_columns = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
        
        for col in categorical_columns:
            if col in df.columns:
                # Get unique values and create dummies
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
        
        # Drop original categorical columns if they exist
        df = df.drop(categorical_columns, axis=1, errors='ignore')
        
        print(f"After encoding: {df.shape[1]} features")
        
        # Get the feature names that the model expects
        # First, try to get from the preprocessor if available
        try:
            preprocessor = joblib.load('models/feature_preprocessor.pkl')
            expected_features = preprocessor.get('feature_names', [])
            print(f"Loaded expected features from preprocessor: {len(expected_features)}")
        except:
            # Fallback: get from model or create default list
            if hasattr(model, 'feature_importances_'):
                expected_features = [f'feature_{i}' for i in range(len(model.feature_importances_))]
            else:
                # Comprehensive fallback feature list
                expected_features = [
                    'tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 
                    'is_new_customer', 'is_loyal_customer', 'total_services',
                    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
                    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
                    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
                    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
                ]
            print(f"Using fallback features: {len(expected_features)}")
        
        # Ensure we have all expected features
        missing_features = []
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            print(f"Added {len(missing_features)} missing features")
        
        # Select only the expected features (in correct order)
        df = df[expected_features]
        
        print(f"✅ Final processed data shape: {df.shape}")
        print(f"First few features: {list(df.columns[:10])}")
        
        return df
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise e

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