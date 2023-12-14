from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Corrected file paths
pipe = pickle.load(open('Model/pipe.pkl', 'rb'))
df = pickle.load(open('Model/df.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


# Corrected file paths
pipe = pickle.load(open('Model/pipe.pkl', 'rb'))
df = pickle.load(open('Model/df.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        company = request.form['company']
        laptop_type = request.form['laptop_type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = request.form['touchscreen']
        ips = request.form['ips']
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Perform prediction using the model
        features = np.array([[company, laptop_type, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os]])
        price_prediction = pipe.predict(features)[0]

        return render_template('result.html', prediction=price_prediction)

if __name__ == '__main__':
    app.run(debug=True)
