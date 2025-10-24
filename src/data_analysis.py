import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our utility functions
from utils import plot_class_balance

class TelcoDataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        
    def load_and_clean_data(self):
        """Load and perform initial data cleaning"""
        print("📊 LOADING TELCO CUSTOMER CHURN DATASET...")
        
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        
        print(f"Original dataset shape: {self.df.shape}")
        
        # Initial data inspection
        print("\n🔍 INITIAL DATA INSPECTION:")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        
        # Handle missing values
        print("\n🧹 HANDLING MISSING VALUES...")
        
        # TotalCharges has empty strings that should be converted to NaN
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Check missing values
        missing_values = self.df.isnull().sum()
        print("Missing values per column:")
        print(missing_values[missing_values > 0])
        
        # Fill missing TotalCharges with 0 (likely new customers)
        self.df['TotalCharges'].fillna(0, inplace=True)
        
        # Remove customerID (not useful for modeling)
        if 'customerID' in self.df.columns:
            self.df.drop('customerID', axis=1, inplace=True)
            
        print(f"Dataset shape after cleaning: {self.df.shape}")
        return self.df
    
    def analyze_target_variable(self):
        """Comprehensive analysis of the target variable (Churn)"""
        print("\n🎯 ANALYZING TARGET VARIABLE (CHURN)...")
        
        # Convert Churn to binary
        self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Plot class distribution
        class_counts = plot_class_balance(self.df['Churn'], 'Churn Distribution')
        
        # Calculate churn statistics
        churn_rate = self.df['Churn'].mean() * 100
        no_churn_rate = (1 - self.df['Churn'].mean()) * 100
        
        print(f"\n📈 CHURN STATISTICS:")
        print(f"Churn Rate: {churn_rate:.2f}%")
        print(f"No Churn Rate: {no_churn_rate:.2f}%")
        print(f"Imbalance Ratio: {no_churn_rate/churn_rate:.2f}:1")
        
        # This shows we have class imbalance (73:27 ratio)
        print("💡 INSIGHT: Dataset has class imbalance - we'll need to handle this in modeling")
        
        return self.df['Churn']
    
    def analyze_feature_distributions(self):
        """Analyze distributions of all features"""
        print("\n📊 ANALYZING FEATURE DISTRIBUTIONS...")
        
        # Separate numerical and categorical features
        self.numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        
        print(f"Numerical features: {self.numerical_cols}")
        print(f"Categorical features: {self.categorical_cols}")
        
        # Analyze numerical features
        self._analyze_numerical_features()
        
        # Analyze categorical features
        self._analyze_categorical_features()
        
        return self.numerical_cols, self.categorical_cols
    
    def _analyze_numerical_features(self):
        """Analyze numerical features distribution and relationship with churn"""
        print("\n🔢 ANALYZING NUMERICAL FEATURES...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(self.numerical_cols):
            # Distribution by churn
            self.df[self.df['Churn'] == 0][col].hist(alpha=0.7, label='No Churn', ax=axes[i], bins=20, color='skyblue')
            self.df[self.df['Churn'] == 1][col].hist(alpha=0.7, label='Churn', ax=axes[i], bins=20, color='salmon')
            axes[i].set_title(f'Distribution of {col} by Churn')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        # Correlation heatmap
        numerical_df = self.df[self.numerical_cols + ['Churn']]
        correlation_matrix = numerical_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=axes[3])
        axes[3].set_title('Correlation Matrix of Numerical Features')
        
        plt.tight_layout()
        plt.savefig('numerical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical insights
        print("\n📈 NUMERICAL FEATURES INSIGHTS:")
        for col in self.numerical_cols:
            churn_mean = self.df[self.df['Churn'] == 1][col].mean()
            no_churn_mean = self.df[self.df['Churn'] == 0][col].mean()
            print(f"{col}: Churn mean = {churn_mean:.2f}, No-churn mean = {no_churn_mean:.2f}")
    
    def _analyze_categorical_features(self):
        """Analyze categorical features and their relationship with churn"""
        print("\n📈 ANALYZING CATEGORICAL FEATURES...")
        
        # Plot top 6 categorical features with highest churn rate variation
        top_features = []
        for col in self.categorical_cols:
            churn_rates = self.df.groupby(col)['Churn'].mean()
            variation = churn_rates.max() - churn_rates.min()
            top_features.append((col, variation))
        
        # Sort by variation and take top 6
        top_features.sort(key=lambda x: x[1], reverse=True)
        top_6_features = [feat[0] for feat in top_features[:6]]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(top_6_features):
            churn_rate = self.df.groupby(col)['Churn'].mean().sort_values(ascending=False)
            
            bars = axes[i].bar(range(len(churn_rate)), churn_rate.values, color='steelblue')
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Churn Rate')
            axes[i].set_xticks(range(len(churn_rate)))
            axes[i].set_xticklabels(churn_rate.index, rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(churn_rate.values):
                axes[i].text(j, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key insights
        print("\n💡 KEY CATEGORICAL INSIGHTS:")
        for col in top_6_features[:3]:  # Top 3 most influential
            churn_rates = self.df.groupby(col)['Churn'].mean().sort_values(ascending=False)
            highest = churn_rates.index[0]
            lowest = churn_rates.index[-1]
            print(f"{col}: Highest churn in '{highest}' ({churn_rates.iloc[0]:.1%}), "
                  f"Lowest in '{lowest}' ({churn_rates.iloc[-1]:.1%})")
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        print("\n💼 GENERATING BUSINESS INSIGHTS...")
        
        insights = []
        
        # Insight 1: Overall metrics
        total_customers = len(self.df)
        churning_customers = self.df['Churn'].sum()
        insights.append(f"Total customers: {total_customers:,}")
        insights.append(f"Churning customers: {churning_customers:,} ({self.df['Churn'].mean():.1%})")
        
        # Insight 2: Contract type impact
        contract_impact = self.df.groupby('Contract')['Churn'].mean()
        for contract, rate in contract_impact.items():
            insights.append(f"Contract '{contract}': {rate:.1%} churn rate")
        
        # Insight 3: Internet service impact
        internet_impact = self.df.groupby('InternetService')['Churn'].mean()
        for service, rate in internet_impact.items():
            insights.append(f"Internet service '{service}': {rate:.1%} churn rate")
        
        # Insight 4: Tenure segments
        self.df['tenure_segment'] = pd.cut(self.df['tenure'], 
                                         bins=[0, 12, 24, 36, 60, 72],
                                         labels=['0-1yr', '1-2yr', '2-3yr', '3-5yr', '5-6yr'])
        tenure_impact = self.df.groupby('tenure_segment')['Churn'].mean()
        for segment, rate in tenure_impact.items():
            insights.append(f"Tenure {segment}: {rate:.1%} churn rate")
        
        # Print insights
        print("📊 BUSINESS INSIGHTS SUMMARY:")
        for insight in insights:
            print(f"• {insight}")
        
        return insights

# Main execution function
def run_complete_analysis(data_path):
    """Run the complete data analysis pipeline"""
    print("🚀 STARTING COMPLETE DATA ANALYSIS PIPELINE")
    print("="*60)
    
    analyzer = TelcoDataAnalyzer(data_path)
    
    # Execute analysis steps
    df_clean = analyzer.load_and_clean_data()
    churn_series = analyzer.analyze_target_variable()
    numerical_cols, categorical_cols = analyzer.analyze_feature_distributions()
    insights = analyzer.generate_business_insights()
    
    print("\n" + "="*60)
    print("✅ DATA ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"• Cleaned dataset shape: {df_clean.shape}")
    print(f"• Numerical features: {len(numerical_cols)}")
    print(f"• Categorical features: {len(categorical_cols)}")
    print(f"• Churn rate: {churn_series.mean():.2%}")
    
    return analyzer, df_clean

if __name__ == "__main__":
    # Run the analysis
    analyzer, df_clean = run_complete_analysis('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Save cleaned data
    df_clean.to_csv('data/processed/cleaned_churn_data.csv', index=False)
    print("💾 Cleaned data saved to: data/processed/cleaned_churn_data.csv")