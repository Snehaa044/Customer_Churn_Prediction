from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists('models/telco_churn_model.pkl'):
        print("⚠️  WARNING: Model not found! Please run the training pipeline first.")
        print("Run: python src/full_pipeline.py")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)