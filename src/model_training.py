import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our utility functions
from utils import evaluate_model_with_balance, save_artifact, load_artifact

class SimpleModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def train_models_simple(self, X_train, X_test, y_train, y_test):
        """Train models without parallel processing to avoid Windows issues"""
        print("🤖 TRAINING MODELS (SIMPLE MODE)...")
        
        # Define models with basic parameters (no grid search)
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                class_weight='balanced_subsample'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
        }
        
        for name, model in models.items():
            print(f"\n🎯 TRAINING {name.upper()}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_pred_proba)
                accuracy = model.score(X_test, y_test)
                
                # Store results
                self.models[name] = {
                    'model': model,
                    'auc_score': auc_score,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✅ {name} - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
                
                # Update best model
                if auc_score > self.best_score:
                    self.best_score = auc_score
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        if self.best_model:
            print(f"\n🏆 BEST MODEL: {self.best_model_name} with AUC: {self.best_score:.4f}")
        else:
            print("\n❌ No models were successfully trained")
            
        return self.models
    
    def evaluate_models(self, X_test, y_test, feature_names):
        """Comprehensive evaluation of trained models"""
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
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            self._analyze_feature_importance(feature_names)
        
        return evaluation_results
    
    def _plot_model_comparison(self, evaluation_results):
        """Plot comparison of all models"""
        if not evaluation_results:
            print("No evaluation results to plot")
            return
            
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
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 15 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'Top 15 Feature Importance - {self.best_model_name.upper()}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{self.best_model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n🎯 {self.best_model_name.upper()} - Top 5 Important Features:")
            for _, row in importance_df.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            self.models[self.best_model_name]['feature_importance'] = importance_df
    
    def save_trained_models(self):
        """Save all trained models and results"""
        print("\n💾 SAVING MODELS AND RESULTS...")
        
        if self.best_model:
            # Save best model
            save_artifact(self.best_model, 'models/best_churn_model.pkl')
            
            # Save model comparison results
            model_comparison = {
                'best_model_name': self.best_model_name,
                'best_score': self.best_score,
                'all_models': {}
            }
            
            for name, model_info in self.models.items():
                model_comparison['all_models'][name] = {
                    'auc_score': model_info['auc_score'],
                    'accuracy': model_info['accuracy']
                }
                
                if 'feature_importance' in model_info:
                    # Save top features
                    top_features = model_info['feature_importance'].head(10)[['feature', 'importance']]
                    model_comparison['all_models'][name]['top_features'] = top_features.to_dict('records')
            
            save_artifact(model_comparison, 'models/model_comparison_results.pkl')
            
            print("✅ MODELS AND RESULTS SAVED SUCCESSFULLY!")
        else:
            print("❌ No models to save")

# Complete training pipeline
def run_simple_training():
    """Run simplified model training pipeline"""
    print("🚀 STARTING SIMPLIFIED MODEL TRAINING PIPELINE")
    print("="*60)
    
    try:
        # Load processed data from feature engineering
        feature_preprocessor = load_artifact('models/feature_preprocessor.pkl')
        feature_info = load_artifact('models/feature_info.pkl')
        
        if feature_preprocessor is None or feature_info is None:
            print("❌ Preprocessed data not found. Running feature engineering first...")
            from feature_engineering import run_feature_engineering
            X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
                'data/processed/cleaned_churn_data.csv',
                balance_method='class_weight'  # Use class_weight to avoid SMOTE issues
            )
        else:
            # Load the processed data
            processed_data = pd.read_csv('data/processed/final_processed_data.csv')
            X = processed_data.drop('Churn', axis=1)
            y = processed_data['Churn']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            feature_names = feature_info['feature_names']
            
            print(f"Loaded preprocessed data: {X_train.shape}, {X_test.shape}")
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Running feature engineering from scratch...")
        from feature_engineering import run_feature_engineering
        X_train, X_test, y_train, y_test, feature_names = run_feature_engineering(
            'data/processed/cleaned_churn_data.csv',
            balance_method='class_weight'
        )
    
    # Initialize trainer
    trainer = SimpleModelTrainer()
    
    # Train models
    models = trainer.train_models_simple(X_train, X_test, y_train, y_test)
    
    if models:
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X_test, y_test, feature_names)
        
        # Save models
        trainer.save_trained_models()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"• Best model: {trainer.best_model_name}")
        print(f"• Best AUC score: {trainer.best_score:.4f}")
        print(f"• Number of models trained: {len(models)}")
        
        return trainer, evaluation_results
    else:
        print("\n❌ MODEL TRAINING FAILED!")
        return None, None

if __name__ == "__main__":
    # Run simplified training pipeline
    trainer, results = run_simple_training()