import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

from utils import save_artifact, plot_class_balance

class TelcoFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_names = []
        self.scaler = StandardScaler()
        self.preprocessor = {}
        
    def create_features(self):
        """Create new features based on business insight"""
        print("🛠️ CREATING NEW FEATURES...")
        
        # 1. Customer value features
        self.df['value_ratio'] = self.df['MonthlyCharges'] / (self.df['TotalCharges'] + 1)
        self.df['avg_monthly_value'] = self.df['TotalCharges'] / (self.df['tenure'] + 1)
        
        # 2. Tenure-based features
        self.df['is_new_customer'] = (self.df['tenure'] <= 3).astype(int)
        self.df['is_loyal_customer'] = (self.df['tenure'] > 24).astype(int)
        self.df['tenure_squared'] = self.df['tenure'] ** 2
        
        # 3. Service usage features
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Convert service columns to binary
        for col in service_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
        
        self.df['total_services'] = self.df[service_columns].sum(axis=1)
        self.df['has_premium_services'] = (self.df['total_services'] >= 3).astype(int)
        
        # 4. Payment and contract features
        self.df['is_monthly_contract'] = (self.df['Contract'] == 'Month-to-month').astype(int)
        self.df['is_electronic_payment'] = self.df['PaymentMethod'].str.contains('electronic', case=False).astype(int)
        
        print(f"Created {8} new features")
        return self.df
    
    def encode_categorical_variables(self):
        """Encode all categorical variables"""
        print("🔤 ENCODING CATEGORICAL VARIABLES...")
        
        # Binary categorical variables
        binary_mappings = {
            'gender': {'Female': 1, 'Male': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
        
        # Multi-category variables (one-hot encoding)
        multi_category_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
        
        for col in multi_category_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
        
        print(f"After encoding: {self.df.shape[1]} features")
        return self.df
    
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using various techniques"""
        print(f"\n⚖️ HANDLING CLASS IMBALANCE USING {method.upper()}...")
        
        print("Before balancing:")
        print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        if method == 'smote':
            # SMOTE: Synthetic Minority Over-sampling Technique
            balancer = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)
            
        elif method == 'undersample':
            # Random undersampling of majority class
            balancer = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)
            
        elif method == 'combine':
            # Combine SMOTE and undersampling
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            pipeline = ImbPipeline(steps=[('over', over), ('under', under)])
            X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
        
        else:  # No balancing
            X_balanced, y_balanced = X_train, y_train
        
        print("After balancing:")
        balanced_counts = pd.Series(y_balanced).value_counts()
        print(f"Class distribution: {balanced_counts.to_dict()}")
        print(f"Balancing ratio: {balanced_counts[0]/balanced_counts[1]:.2f}:1")
        
        return X_balanced, y_balanced
    
    def prepare_model_data(self, test_size=0.2, balance_method='smote'):
        """Prepare final dataset for modeling with imbalance handling"""
        print("\n📊 PREPARING MODEL DATA...")
        
        # Separate features and target
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Original features: {len(self.feature_names)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train, balance_method)
        
        # Scale numerical features
        numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio', 
                           'avg_monthly_value', 'tenure_squared']
        
        # Only scale columns that exist
        existing_num_cols = [col for col in numerical_columns if col in X_train_balanced.columns]
        
        X_train_balanced[existing_num_cols] = self.scaler.fit_transform(X_train_balanced[existing_num_cols])
        X_test[existing_num_cols] = self.scaler.transform(X_test[existing_num_cols])
        
        # Store preprocessor info
        self.preprocessor = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_columns': existing_num_cols,
            'balance_method': balance_method
        }
        
        print(f"\n📈 FINAL DATA SHAPES:")
        print(f"Training set (balanced): {X_train_balanced.shape}")
        print(f"Testing set: {X_test.shape}")
        print(f"Features used: {len(self.feature_names)}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def save_artifacts(self):
        """Save preprocessor and feature names"""
        save_artifact(self.preprocessor, 'models/feature_preprocessor.pkl')
        
        # Save feature list
        feature_info = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names)
        }
        save_artifact(feature_info, 'models/feature_info.pkl')

# Complete feature engineering pipeline
def run_feature_engineering(data_path, balance_method='smote'):
    """Run complete feature engineering pipeline"""
    print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load cleaned data
    df = pd.read_csv(data_path)
    
    # Initialize feature engineer
    engineer = TelcoFeatureEngineer(df)
    
    # Execute pipeline
    df_with_features = engineer.create_features()
    df_encoded = engineer.encode_categorical_variables()
    
    # Prepare model data with imbalance handling
    X_train, X_test, y_train, y_test = engineer.prepare_model_data(balance_method=balance_method)
    
    # Save artifacts
    engineer.save_artifacts()
    
    # Save processed data
    df_encoded.to_csv('data/processed/final_processed_data.csv', index=False)
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETED!")
    print(f"• Final features: {len(engineer.feature_names)}")
    print(f"• Training shape: {X_train.shape}")
    print(f"• Test shape: {X_test.shape}")
    print(f"• Balance method: {balance_method}")
    
    return X_train, X_test, y_train, y_test, engineer.feature_names

if __name__ == "__main__":
    # Run with SMOTE balancing (you can change to 'undersample' or 'combine')
    X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
        'data/processed/cleaned_churn_data.csv',
        balance_method='smote'
    )