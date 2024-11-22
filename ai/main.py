import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

class InstagramInfluencePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def preprocess_data(self, data):
        """Convert string numbers with k, m, b suffixes to numerical values"""
        def convert_to_number(value):
            if isinstance(value, str):
                value = value.lower().strip()
                multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
                for suffix, multiplier in multipliers.items():
                    if suffix in value:
                        try:
                            return float(value.replace(suffix, '')) * multiplier
                        except ValueError:
                            return None
                if '%' in value:
                    try:
                        return float(value.replace('%', '')) / 100
                    except ValueError:
                        return None
            return value

        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Columns that need conversion
        numeric_columns = ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 
                         'new_post_avg_like', 'total_likes']
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(convert_to_number)
        
        # Handle missing values
        df['country'] = df['country'].fillna('Unknown')
        
        # Encode categorical variables
        df['country_encoded'] = self.label_encoder.fit_transform(df['country'])
        
        # Drop unnecessary columns
        columns_to_drop = ['channel_info', 'country']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'influence_score']
        
        return df
    
    def train(self, data):
        """Train the model on preprocessed data"""
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        
        # Split features and target
        X = processed_data[self.feature_names]
        y = processed_data['influence_score']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to DataFrame to maintain feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        # Initialize and train the model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return metrics, X_test_scaled, y_test, y_pred
    
    def plot_results(self, X, y_true, y_pred):
        """Plot feature importance and prediction results"""
        # Feature importance plot
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Actual vs Predicted plot
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Influence Score')
        plt.ylabel('Predicted Influence Score')
        plt.title('Actual vs Predicted Influence Scores')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename='influence_predictor.pkl'):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, filename)
            print(f"Model saved as {filename}")
        else:
            print("No trained model to save")
    
    def load_model(self, filename='influence_predictor.pkl'):
        """Load a trained model"""
        saved_objects = joblib.load(filename)
        self.model = saved_objects['model']
        self.scaler = saved_objects['scaler']
        self.label_encoder = saved_objects['label_encoder']
        self.feature_names = saved_objects['feature_names']
        print("Model loaded successfully")
    
    def predict(self, data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        processed_data = self.preprocess_data(data)
        scaled_data = self.scaler.transform(processed_data[self.feature_names])
        scaled_data = pd.DataFrame(scaled_data, columns=self.feature_names)
        predictions = self.model.predict(scaled_data)
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Load your dataset
    file_path = 'ai/top_insta_influencers_data(1).csv'
    data = pd.read_csv(file_path)
    
    # Print column names to verify
    print("Original column names:", data.columns.tolist())
    
    # Initialize and train the model
    predictor = InstagramInfluencePredictor()
    metrics, X_test, y_test, y_pred = predictor.train(data)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    predictor.plot_results(X_test, y_test, y_pred)
    
    # Save the model
    predictor.save_model()
