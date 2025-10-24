from app import create_app
import os

# Create Flask application
app = create_app()

def check_dependencies():
    """Check if all required files and models exist"""
    required_files = [
        'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'models/best_churn_model.pkl',
        'models/feature_preprocessor.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ MISSING REQUIRED FILES:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please run the training pipeline first:")
        print("   python src/data_analysis.py")
        print("   python src/feature_engineering.py") 
        print("   python src/model_training.py")
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
        
        # Run the application
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Application cannot start due to missing files")