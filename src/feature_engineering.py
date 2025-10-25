import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import save_artifact, plot_class_balance

class TelcoFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_names = []
        self.scaler = StandardScaler()
        self.preprocessor = {}

    # ====================================================
    # 1️⃣ Feature creation
    # ====================================================
    def create_features(self):
        print("🛠️ CREATING NEW FEATURES...")

        # Customer value features
        self.df['value_ratio'] = self.df['MonthlyCharges'] / (self.df['TotalCharges'] + 1)
        self.df['avg_monthly_value'] = self.df['TotalCharges'] / (self.df['tenure'] + 1)

        # Tenure features
        self.df['is_new_customer'] = (self.df['tenure'] <= 3).astype(int)
        self.df['is_loyal_customer'] = (self.df['tenure'] > 24).astype(int)
        self.df['tenure_squared'] = self.df['tenure'] ** 2

        # Service usage
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']

        for col in service_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})

        self.df['total_services'] = self.df[service_columns].sum(axis=1)
        self.df['has_premium_services'] = (self.df['total_services'] >= 3).astype(int)

        # Contract & payment
        self.df['is_monthly_contract'] = (self.df['Contract'] == 'Month-to-month').astype(int)
        self.df['is_electronic_payment'] = self.df['PaymentMethod'].str.contains('electronic', case=False).astype(int)

        print(f"✅ Created 8 new features.")
        return self.df

    # ====================================================
    # 2️⃣ Encoding
    # ====================================================
    def encode_categorical_variables(self):
        print("🔤 ENCODING CATEGORICAL VARIABLES...")

        # Binary mappings
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

        # One-hot encoding for multi-category variables
        multi_category_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
        for col in multi_category_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                dummies.columns = dummies.columns.str.replace(' ', '_').str.replace('-', '_')
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)

        print(f"✅ After encoding: {self.df.shape[1]} total features.")
        return self.df

    # ====================================================
    # 3️⃣ Handle imbalance
    # ====================================================
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        print(f"\n⚖️ HANDLING CLASS IMBALANCE USING {method.upper()}...")
        print("Before balancing:", pd.Series(y_train).value_counts().to_dict())

        X_train_np, y_train_np = X_train.values, y_train.values
        if method == 'smote':
            try:
                balancer = SMOTE(random_state=42, k_neighbors=5)
                X_balanced, y_balanced = balancer.fit_resample(X_train_np, y_train_np)
            except Exception as e:
                print(f"SMOTE failed: {e}. Using original data.")
                X_balanced, y_balanced = X_train_np, y_train_np
        elif method == 'undersample':
            try:
                balancer = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = balancer.fit_resample(X_train_np, y_train_np)
            except Exception as e:
                print(f"Undersampling failed: {e}. Using original data.")
                X_balanced, y_balanced = X_train_np, y_train_np
        else:
            X_balanced, y_balanced = X_train_np, y_train_np

        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        print("After balancing:", pd.Series(y_balanced).value_counts().to_dict())
        return X_balanced, y_balanced

    # ====================================================
    # 4️⃣ Prepare model data
    # ====================================================
    def prepare_model_data(self, test_size=0.2, balance_method='smote'):
        print("\n📊 PREPARING MODEL DATA...")

        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']

        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.to_numeric(y, errors='coerce').fillna(0)

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_bal, y_train_bal = self.handle_class_imbalance(X_train, y_train, balance_method)

        # Scale numerical features
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio',
                    'avg_monthly_value', 'tenure_squared']
        num_cols = [col for col in num_cols if col in X_train_bal.columns]

        X_train_bal[num_cols] = self.scaler.fit_transform(X_train_bal[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])

        # Save preprocessing info
        self.preprocessor = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_columns': num_cols,
            'balance_method': balance_method
        }

        print(f"✅ Final training shape: {X_train_bal.shape}")
        return X_train_bal, X_test, y_train_bal, y_test

    # ====================================================
    # 5️⃣ Save artifacts
    # ====================================================
    def save_artifacts(self):
        print("💾 Saving preprocessor and feature info...")
        save_artifact(self.preprocessor, 'models/feature_preprocessor.pkl')
        save_artifact({
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names)
        }, 'models/feature_info.pkl')

    # ====================================================
    # ✅ Utility to transform new data for prediction
    # ====================================================
    @staticmethod
    def transform_for_prediction(raw_input: dict):
        """Apply same transformations for a single prediction."""
        df_input = pd.DataFrame([raw_input])

        # Reapply feature creation & encoding
        engineer = TelcoFeatureEngineer(df_input)
        df_feat = engineer.create_features()
        df_encoded = engineer.encode_categorical_variables()
        df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Load preprocessor
        preproc = joblib.load('models/feature_preprocessor.pkl')
        feature_info = joblib.load('models/feature_info.pkl')

        scaler = preproc['scaler']
        num_cols = preproc['numerical_columns']
        for col in num_cols:
            if col in df_encoded.columns:
                df_encoded[col] = scaler.transform(df_encoded[[col]])

        # Align with training columns
        expected_cols = feature_info['feature_names']
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_cols]

        return df_encoded

# ====================================================
# Full feature engineering pipeline
# ====================================================
def run_feature_engineering(data_path, balance_method='smote'):
    print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(data_path)
    engineer = TelcoFeatureEngineer(df)

    df = engineer.create_features()
    df = engineer.encode_categorical_variables()

    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        print(f"⚠️ Converting string columns to numeric: {list(string_cols)}")
        df[string_cols] = df[string_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = engineer.prepare_model_data(balance_method=balance_method)
    engineer.save_artifacts()

    df.to_csv('data/processed/final_processed_data.csv', index=False)

    print("=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETED!")
    print(f"• Features: {len(engineer.feature_names)}")
    print(f"• Train shape: {X_train.shape}")
    print(f"• Test shape: {X_test.shape}")
    print(f"• Balance method: {balance_method}")

    return X_train, X_test, y_train, y_test, engineer.feature_names


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
            'data/processed/cleaned_churn_data.csv',
            balance_method='smote'
        )
    except Exception as e:
        print(f"❌ SMOTE failed: {e}, retrying without balancing...")
        X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
            'data/processed/cleaned_churn_data.csv',
            balance_method='none'
        )
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import save_artifact, plot_class_balance

class TelcoFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_names = []
        self.scaler = StandardScaler()
        self.preprocessor = {}

    # ====================================================
    # 1️⃣ Feature creation
    # ====================================================
    def create_features(self):
        print("🛠️ CREATING NEW FEATURES...")

        # Customer value features
        self.df['value_ratio'] = self.df['MonthlyCharges'] / (self.df['TotalCharges'] + 1)
        self.df['avg_monthly_value'] = self.df['TotalCharges'] / (self.df['tenure'] + 1)

        # Tenure features
        self.df['is_new_customer'] = (self.df['tenure'] <= 3).astype(int)
        self.df['is_loyal_customer'] = (self.df['tenure'] > 24).astype(int)
        self.df['tenure_squared'] = self.df['tenure'] ** 2

        # Service usage
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies']

        for col in service_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})

        self.df['total_services'] = self.df[service_columns].sum(axis=1)
        self.df['has_premium_services'] = (self.df['total_services'] >= 3).astype(int)

        # Contract & payment
        self.df['is_monthly_contract'] = (self.df['Contract'] == 'Month-to-month').astype(int)
        self.df['is_electronic_payment'] = self.df['PaymentMethod'].str.contains('electronic', case=False).astype(int)

        print(f"✅ Created 8 new features.")
        return self.df

    # ====================================================
    # 2️⃣ Encoding
    # ====================================================
    def encode_categorical_variables(self):
        print("🔤 ENCODING CATEGORICAL VARIABLES...")

        # Binary mappings
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

        # One-hot encoding for multi-category variables
        multi_category_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
        for col in multi_category_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                dummies.columns = dummies.columns.str.replace(' ', '_').str.replace('-', '_')
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)

        print(f"✅ After encoding: {self.df.shape[1]} total features.")
        return self.df

    # ====================================================
    # 3️⃣ Handle imbalance
    # ====================================================
    def handle_class_imbalance(self, X_train, y_train, method='smote'):
        print(f"\n⚖️ HANDLING CLASS IMBALANCE USING {method.upper()}...")
        print("Before balancing:", pd.Series(y_train).value_counts().to_dict())

        X_train_np, y_train_np = X_train.values, y_train.values
        if method == 'smote':
            try:
                balancer = SMOTE(random_state=42, k_neighbors=5)
                X_balanced, y_balanced = balancer.fit_resample(X_train_np, y_train_np)
            except Exception as e:
                print(f"SMOTE failed: {e}. Using original data.")
                X_balanced, y_balanced = X_train_np, y_train_np
        elif method == 'undersample':
            try:
                balancer = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = balancer.fit_resample(X_train_np, y_train_np)
            except Exception as e:
                print(f"Undersampling failed: {e}. Using original data.")
                X_balanced, y_balanced = X_train_np, y_train_np
        else:
            X_balanced, y_balanced = X_train_np, y_train_np

        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        print("After balancing:", pd.Series(y_balanced).value_counts().to_dict())
        return X_balanced, y_balanced

    # ====================================================
    # 4️⃣ Prepare model data
    # ====================================================
    def prepare_model_data(self, test_size=0.2, balance_method='smote'):
        print("\n📊 PREPARING MODEL DATA...")

        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']

        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.to_numeric(y, errors='coerce').fillna(0)

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_bal, y_train_bal = self.handle_class_imbalance(X_train, y_train, balance_method)

        # Scale numerical features
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'value_ratio',
                    'avg_monthly_value', 'tenure_squared']
        num_cols = [col for col in num_cols if col in X_train_bal.columns]

        X_train_bal[num_cols] = self.scaler.fit_transform(X_train_bal[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])

        # Save preprocessing info
        self.preprocessor = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_columns': num_cols,
            'balance_method': balance_method
        }

        print(f"✅ Final training shape: {X_train_bal.shape}")
        return X_train_bal, X_test, y_train_bal, y_test

    # ====================================================
    # 5️⃣ Save artifacts
    # ====================================================
    def save_artifacts(self):
        print("💾 Saving preprocessor and feature info...")
        save_artifact(self.preprocessor, 'models/feature_preprocessor.pkl')
        save_artifact({
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names)
        }, 'models/feature_info.pkl')

    # ====================================================
    # ✅ Utility to transform new data for prediction
    # ====================================================
    @staticmethod
    def transform_for_prediction(raw_input: dict):
        """Apply same transformations for a single prediction."""
        df_input = pd.DataFrame([raw_input])

        # Reapply feature creation & encoding
        engineer = TelcoFeatureEngineer(df_input)
        df_feat = engineer.create_features()
        df_encoded = engineer.encode_categorical_variables()
        df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Load preprocessor
        preproc = joblib.load('models/feature_preprocessor.pkl')
        feature_info = joblib.load('models/feature_info.pkl')

        scaler = preproc['scaler']
        num_cols = preproc['numerical_columns']
        for col in num_cols:
            if col in df_encoded.columns:
                df_encoded[col] = scaler.transform(df_encoded[[col]])

        # Align with training columns
        expected_cols = feature_info['feature_names']
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_cols]

        return df_encoded

# ====================================================
# Full feature engineering pipeline
# ====================================================
def run_feature_engineering(data_path, balance_method='smote'):
    print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(data_path)
    engineer = TelcoFeatureEngineer(df)

    df = engineer.create_features()
    df = engineer.encode_categorical_variables()

    string_cols = df.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        print(f"⚠️ Converting string columns to numeric: {list(string_cols)}")
        df[string_cols] = df[string_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = engineer.prepare_model_data(balance_method=balance_method)
    engineer.save_artifacts()

    df.to_csv('data/processed/final_processed_data.csv', index=False)

    print("=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETED!")
    print(f"• Features: {len(engineer.feature_names)}")
    print(f"• Train shape: {X_train.shape}")
    print(f"• Test shape: {X_test.shape}")
    print(f"• Balance method: {balance_method}")

    return X_train, X_test, y_train, y_test, engineer.feature_names


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
            'data/processed/cleaned_churn_data.csv',
            balance_method='smote'
        )
    except Exception as e:
        print(f"❌ SMOTE failed: {e}, retrying without balancing...")
        X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
            'data/processed/cleaned_churn_data.csv',
            balance_method='none'
        )
