# flask, scikit-learn, pandas, pickle-mixin, flask_cors
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from flask_cors import CORS

# Load data and model
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

app = Flask(__name__, static_folder='static', template_folder="templates")
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/')
def home():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('total_sqft')

        # Validate input
        if not all([location, bhk, bath, sqft]):
            return "Error: All fields are required", 400
        
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)

        # Check if location exists in data
        if location not in data['location'].values:
            return "Error: Invalid location", 400

        # Prepare input and predict
        input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_data)[0] * 1e5

        return str(np.round(prediction, 2))
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
