import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('pickle_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    double_features = [float(x) for x in request.form.values()]
    final_features = [np.array(double_features)]
    prediction = model.predict(final_features)
    
    if prediction[0] == 0.0:
        output = 'Benign'
    else :  
        output = 'Malignant'
    return render_template('index.html', prediction_text='{}'.format(output))

if __name__=="__main__":
    app.run(debug = True)