import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Train the influencer score prediction model'

    def handle(self, *args, **kwargs):
        # Load the dataset (Assuming CSV format)
        df = pd.read_csv('ai/top_insta_influencers_data(1).csv')  # Make sure to put the correct path to your CSV file

        # Preprocess the data
        # Handle missing country values, but we won't use 'country' in the model anymore
        df['country'].fillna('Unknown', inplace=True)  # Handle missing country values
        
        # Define feature columns and target variable
        features = ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 
                    'new_post_avg_like', 'total_likes']  # Removed 'country_encoded'
        target = 'influence_score'  # Assuming 'influence_score' is your target column
        
        X = df[features]
        y = df[target]

        # Convert string numbers with k, m, b suffixes and percentages to numeric values
        def convert_to_number(value):
            if isinstance(value, str):
                value = value.lower().strip()
                # Handle 'k', 'm', 'b' suffixes for numbers
                multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
                for suffix, multiplier in multipliers.items():
                    if suffix in value:
                        try:
                            return float(value.replace(suffix, '')) * multiplier
                        except ValueError:
                            return None
                
                # Handle percentage values (e.g., '0.85%' -> 0.0085)
                if '%' in value:
                    try:
                        return float(value.replace('%', '')) / 100  # Convert percentage to a decimal
                    except ValueError:
                        return None
            
            return value  # If no conversion needed, return the value as is
        
        # Apply conversion to numeric columns
        for col in ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 
                    'new_post_avg_like', 'total_likes']:
            X[col] = X[col].apply(convert_to_number)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.stdout.write(self.style.SUCCESS(f'Model trained successfully! MSE: {mse}'))

        # Save the trained model to a file
        joblib.dump(model, 'ai/influence_predictor.pkl')  # Save the model
        self.stdout.write(self.style.SUCCESS(f'Model saved as ai/influence_predictor.pkl'))
