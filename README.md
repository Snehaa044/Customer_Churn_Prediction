# 📊 Telco Customer Churn Prediction

This is an **end-to-end machine learning project** that predicts whether a telecom customer is likely to churn.  
Built with **Python, Scikit-learn, Flask, and Imbalanced-learn**.

## 🚀 Project Pipeline

1. **Data Preprocessing** → Handle missing values, encode categories  
2. **Feature Engineering** → Create meaningful business features  
3. **Model Training** → Train Logistic Regression, Random Forest, Gradient Boosting  
4. **Evaluation** → Generate metrics, confusion matrices, AUC  
5. **Deployment** → Flask web app for live churn prediction

## 🧠 Best Model
| Model | AUC | Accuracy | Recall |
|--------|-----|-----------|--------|
| Logistic Regression | 0.848 | 0.75 | 0.80 |

## 🛠️ Tech Stack
- Python, Pandas, Scikit-learn, Imbalanced-learn  
- Matplotlib, Seaborn  
- Flask (for deployment)

## 🧩 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run feature engineering and training
python src/feature_engineering.py
python src/model_training.py

# 4. Start Flask app
python main.py
