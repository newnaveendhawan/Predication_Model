# flask, scikit-learn, pandas, pickle-mixin, flask_cors
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

app = Flask(__name__, static_folder='static', template_folder="templates")
@app.route('/')
def home():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['post'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))

    sqft = float(request.form.get('total_sqft'))
    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0]*1e5

    print("Prediction:", prediction)
    # Return the prediction as a string
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
