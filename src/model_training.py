import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our utility functions
from utils import evaluate_model_with_balance, save_artifact, load_artifact

class AdvancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.cv_results = {}
        
    def train_with_cross_validation(self, X_train, X_test, y_train, y_test):
        """Train multiple models with cross-validation"""
        print("🤖 TRAINING MODELS WITH CROSS-VALIDATION...")
        
        # Define models with initial parameters
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model_info in models.items():
            print(f"\n🎯 TRAINING {name.upper()}...")
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_estimator = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_mean_score = grid_search.best_score_
            
            # Store results
            self.models[name] = {
                'model': best_estimator,
                'best_params': grid_search.best_params_,
                'auc_score': auc_score,
                'cv_score': cv_mean_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'grid_search': grid_search
            }
            
            print(f"✅ {name} - Best params: {grid_search.best_params_}")
            print(f"   Cross-val AUC: {cv_mean_score:.4f}")
            print(f"   Test AUC: {auc_score:.4f}")
            
            # Update best model
            if auc_score > self.best_score:
                self.best_score = auc_score
                self.best_model = best_estimator
                self.best_model_name = name
        
        print(f"\n🏆 BEST OVERALL MODEL: {self.best_model_name}")
        print(f"   Best Test AUC: {self.best_score:.4f}")
        
        return self.models
    
    def calibrate_probabilities(self, X_train, y_train):
        """Calibrate probabilities for better reliability"""
        print(f"\n🎯 CALIBRATING PROBABILITIES FOR {self.best_model_name.upper()}...")
        
        # Use Platt scaling for calibration
        calibrated_model = CalibratedClassifierCV(self.best_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        self.calibrated_model = calibrated_model
        print("✅ Probability calibration completed")
        
        return calibrated_model
    
    def evaluate_all_models(self, X_test, y_test, feature_names):
        """Comprehensive evaluation of all trained models"""
        print(f"\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        evaluation_results = {}
        
        for name, model_info in self.models.items():
            print(f"\n🔍 EVALUATING {name.upper()}...")
            
            # Evaluate model
            results = evaluate_model_with_balance(
                y_test, 
                model_info['predictions'], 
                model_info['probabilities'],
                model_name=name
            )
            
            evaluation_results[name] = results
        
        # Compare all models
        self._plot_model_comparison(evaluation_results)
        
        # Feature importance for tree-based models
        self._analyze_feature_importance(feature_names)
        
        return evaluation_results
    
    def _plot_model_comparison(self, evaluation_results):
        """Plot comparison of all models"""
        metrics = ['auc_score', 'f1_score', 'precision', 'recall']
        model_names = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[i].set_title(f'Model Comparison - {metric.upper()}')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_feature_importance(self, feature_names):
        """Analyze and plot feature importance for tree-based models"""
        print("\n📈 ANALYZING FEATURE IMPORTANCE...")
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                # Get feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Plot top 15 features
                plt.figure(figsize=(12, 8))
                top_features = importance_df.head(15)
                
                sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
                plt.title(f'Top 15 Feature Importance - {name.upper()}')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(f'feature_importance_{name}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\n🎯 {name.upper()} - Top 5 Important Features:")
                for _, row in importance_df.head().iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")
                
                self.models[name]['feature_importance'] = importance_df
    
    def save_trained_models(self):
        """Save all trained models and results"""
        print("\n💾 SAVING MODELS AND RESULTS...")
        
        # Save best model
        save_artifact(self.best_model, 'models/best_churn_model.pkl')
        
        # Save calibrated model if exists
        if hasattr(self, 'calibrated_model'):
            save_artifact(self.calibrated_model, 'models/calibrated_churn_model.pkl')
        
        # Save model comparison results
        model_comparison = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'all_models': {}
        }
        
        for name, model_info in self.models.items():
            model_comparison['all_models'][name] = {
                'auc_score': model_info['auc_score'],
                'cv_score': model_info['cv_score'],
                'best_params': model_info['best_params']
            }
            
            if 'feature_importance' in model_info:
                # Save top features
                top_features = model_info['feature_importance'].head(10)[['feature', 'importance']]
                model_comparison['all_models'][name]['top_features'] = top_features.to_dict('records')
        
        save_artifact(model_comparison, 'models/model_comparison_results.pkl')
        
        print("✅ ALL MODELS AND RESULTS SAVED SUCCESSFULLY!")

# Complete training pipeline
def run_complete_training(balance_method='smote'):
    """Run the complete model training pipeline"""
    print("🚀 STARTING COMPLETE MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load processed data
    from feature_engineering import run_feature_engineering
    
    X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
        'data/processed/cleaned_churn_data.csv',
        balance_method=balance_method
    )
    
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Train models with cross-validation
    models = trainer.train_with_cross_validation(X_train, X_test, y_train, y_test)
    
    # Calibrate probabilities
    calibrated_model = trainer.calibrate_probabilities(X_train, y_train)
    
    # Evaluate all models
    evaluation_results = trainer.evaluate_all_models(X_test, y_test, feature_names)
    
    # Save models
    trainer.save_trained_models()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"• Best model: {trainer.best_model_name}")
    print(f"• Best AUC score: {trainer.best_score:.4f}")
    print(f"• Number of models trained: {len(models)}")
    print(f"• Balance method used: {balance_method}")
    
    return trainer, evaluation_results

if __name__ == "__main__":
    # Run complete training pipeline
    trainer, results = run_complete_training(balance_method='smote') 