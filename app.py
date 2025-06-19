from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['cn_ratio']),
            float(request.form['moisture']),
            int(request.form['aeration']),
            {'Vegetable': 2, 'Leaves': 1, 'Mixed': 0}[request.form['waste_type']],
            float(request.form['bin_size'])
        ]
        pred = model.predict([np.array(features)])
        return render_template('index.html', prediction_text=f"Estimated Decomposition Time: {int(pred[0])} days")
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
