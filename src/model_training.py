import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class TelcoModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_importance = None
        
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """Train and compare multiple machine learning models"""
        print("🤖 TRAINING MULTIPLE MACHINE LEARNING MODELS...")
        
        # Define models to try
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = model.score(X_test, y_test)
            
            # Store results
            results[name] = {
                'model': model,
                'auc': auc_score,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
            
            # Update best model
            if auc_score > self.best_score:
                self.best_score = auc_score
                self.best_model = model
                self.best_model_name = name
        
        self.models = results
        print(f"\n🏆 BEST MODEL: {self.best_model_name} with AUC: {self.best_score:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n🎯 PERFORMING HYPERPARAMETER TUNING FOR {self.best_model_name.upper()}...")
        
        if self.best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            
        elif self.best_model_name == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
        
        return self.best_model
    
    def evaluate_best_model(self, X_test, y_test, feature_names):
        """Comprehensive evaluation of the best model"""
        print(f"\n📊 COMPREHENSIVE EVALUATION OF {self.best_model_name.upper()}...")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(2, 2, 2)
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 15 features
            top_features = self.feature_importance.head(15)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importance')
            plt.tight_layout()
        
        # Precision-Recall curve
        plt.subplot(2, 2, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # ROC Curve
        plt.subplot(2, 2, 4)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.text(0.6, 0.2, f'AUC = {self.best_score:.4f}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Business metrics
        print("\n💼 BUSINESS METRICS:")
        tn, fp, fn, tp = cm.ravel()
        
        print(f"True Positives (Correctly predicted churn): {tp}")
        print(f"False Positives (False alarms): {fp}")
        print(f"True Negatives (Correctly predicted no churn): {tn}")
        print(f"False Negatives (Missed churn): {fn}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {2 * (precision * recall) / (precision + recall):.4f}")
        
        return {
            'confusion_matrix': cm,
            'feature_importance': self.feature_importance,
            'auc_score': self.best_score
        }
    
    def save_model(self, model_path, preprocessor_path):
        """Save the trained model and preprocessor"""
        # Save model
        joblib.dump(self.best_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'auc_score': self.best_score,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_info, 'models/model_info.pkl')
        
        print(" MODEL TRAINING COMPLETED!")

# Complete training pipeline
def run_model_training():
    """Complete model training pipeline"""
    # Load processed data
    from feature_engineering import run_feature_engineering
    
    X_train, X_test, y_train, y_test, feature_names, df_processed = run_feature_engineering(
        'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )
    
    # Initialize trainer
    trainer = TelcoModelTrainer()
    
    # Train multiple models
    results = trainer.train_multiple_models(X_train, X_test, y_train, y_test)
    
    # Hyperparameter tuning
    best_model = trainer.hyperparameter_tuning(X_train, y_train)
    
    # Evaluate best model
    evaluation_results = trainer.evaluate_best_model(X_test, y_test, feature_names)
    
    # Save model
    trainer.save_model('models/telco_churn_model.pkl', 'models/feature_preprocessor.pkl')
    
    return trainer, evaluation_results

if __name__ == "__main__":
    trainer, results = run_model_training()