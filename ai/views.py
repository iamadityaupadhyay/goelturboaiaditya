from django.shortcuts import render
from django.http import JsonResponse
import joblib
import pandas as pd
from .forms import InfluenceForm

# Load the trained model
model = joblib.load('ai/influence_predictor.pkl')

def preprocess_data(data):
    """Preprocess input data for prediction"""
    def convert_to_number(value):
        """Convert string numbers with k, m, b suffixes to numerical values"""
        if isinstance(value, str):
            value = value.lower().strip()
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
        return value  # if no conversion needed, return the value as is

    # Convert the form data into a DataFrame
    df = pd.DataFrame([data])
    
    # Apply conversion to numeric columns
    for col in ['posts', 'followers', 'avg_likes', 'sixty_day_eng_rate', 
                'new_post_avg_like', 'total_likes']:
        df[col] = df[col].apply(convert_to_number)
    
    # Encode country (example encoding, modify as needed)
    df['country_encoded'] = df['country'].map({'Unknown': 0, 'USA': 1, 'India': 2})
    df.drop(columns=['country'], inplace=True)  # Drop original country column

    return df

def predict_influence(request):
    if request.method == 'POST':
        form = InfluenceForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Preprocess the input data (convert strings like "3.7k" to numeric)
            processed_data = preprocess_data(data)

            # Make prediction using the model
            prediction = model.predict(processed_data)

            # Return the prediction result as JSON
            return JsonResponse({'predicted_influence_score': prediction[0]})

    else:
        form = InfluenceForm()

    return render(request, 'index.html', {'form': form})
