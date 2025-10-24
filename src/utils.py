import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

def plot_class_balance(y, title="Class Distribution"):
    """Plot class distribution to visualize imbalance"""
    plt.figure(figsize=(10, 6))
    
    class_counts = y.value_counts()
    colors = ['lightblue', 'lightcoral']
    
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar', color=colors)
    plt.title(f'{title} - Count')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = len(y)
    for i, count in enumerate(class_counts):
        plt.text(i, count + 10, f'{count/total:.1%}', ha='center')
    
    plt.subplot(1, 2, 2)
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title(f'{title} - Percentage')
    
    plt.tight_layout()
    plt.savefig('class_balance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return class_counts

def evaluate_model_with_balance(y_true, y_pred, y_pred_proba, model_name=""):
    """Comprehensive evaluation considering class imbalance"""
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'='*50}")
    
    # Classification report
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n🔍 CONFUSION MATRIX ANALYSIS:")
    print(f"True Positives (TP): {tp} - Correctly predicted churn")
    print(f"False Positives (FP): {fp} - False alarms")
    print(f"True Negatives (TN): {tn} - Correctly predicted no churn")
    print(f"False Negatives (FN): {fn} - Missed churn cases")
    
    # Calculate business metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 KEY METRICS:")
    print(f"Precision: {precision:.4f} - How many predicted churns were actual churns")
    print(f"Recall (Sensitivity): {recall:.4f} - How many actual churns were caught")
    print(f"Specificity: {specificity:.4f} - How many actual non-churns were correctly identified")
    print(f"F1-Score: {f1:.4f} - Balance between precision and recall")
    print(f"AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}")
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc_score': roc_auc_score(y_true, y_pred_proba)
    }

def save_artifact(obj, file_path):
    """Save model or preprocessor with error handling"""
    try:
        joblib.dump(obj, file_path)
        print(f"✅ Successfully saved: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving {file_path}: {e}")
        return False

def load_artifact(file_path):
    """Load model or preprocessor with error handling"""
    try:
        obj = joblib.load(file_path)
        print(f"✅ Successfully loaded: {file_path}")
        return obj
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None