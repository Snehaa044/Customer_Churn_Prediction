from app import create_app
import os

def check_dependencies():
    """Check if we have the basic requirements"""
    # Check for dataset
    if not os.path.exists('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        print("❌ Dataset not found. Please download from Kaggle and place in data/raw/")
        return False
    
    # Check for any model
    model_files = [
        'models/best_churn_model.pkl',
        'models/quick_churn_model.pkl'
    ]
    
    has_model = any(os.path.exists(model_file) for model_file in model_files)
    if not has_model:
        print("❌ No trained model found.")
        print("💡 Please run: python src/quick_train.py")
        return False
    
    return True

if __name__ == '__main__':
    print("🚀 STARTING TELCO CHURN PREDICTION APPLICATION")
    print("="*50)
    
    # Check dependencies
    if check_dependencies():
        print("✅ All dependencies found!")
        print("🌐 Starting web server...")
        print("📍 Application available at: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Create and run the application
        app = create_app()
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Application cannot start due to missing files")