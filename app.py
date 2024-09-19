# app.py
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models and scaler
best_model = joblib.load('models/diabetes_model.pkl')
scaler = joblib.load('models/scaler.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        action = request.form.get('action')

        if action == 'Predict':

            # Get input data from form
            form_data = {
                'Pregnancies': request.form.get('Pregnancies'),
                'Glucose': request.form.get('Glucose'),
                'BloodPressure': request.form.get('BloodPressure'),
                'SkinThickness': request.form.get('SkinThickness'),
                'Insulin': request.form.get('Insulin'),
                'BMI': request.form.get('BMI'),
                'DiabetesPedigreeFunction': request.form.get('DiabetesPedigreeFunction'),
                'Age': request.form.get('Age')
            }

            input_data = [
                int(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                int(request.form['Age'])
            ]

            # Convert the input data to a numpy array
            input_data_np = np.array(input_data).reshape(1, -1)

            # Standardize the input data using the saved scaler
            input_data_scaled = scaler.transform(input_data_np)

            # Predict diabetes using the Logistic Regression model
            diabetes_prediction = best_model.predict(input_data_scaled)

            # Predict the cluster using the KMeans model
            cluster = kmeans_model.predict(input_data_scaled)

            # Prepare the result message
            if diabetes_prediction[0] == 1:
                diabetes_result = 'The person is diabetic.'
            else:
                diabetes_result = 'The person is not diabetic.'
            
            if cluster[0] == 0:
                cluster_result = "Hypertension, Gestational Diabetes, Metabolic Syndrome."
            elif cluster[0] == 1:
                cluster_result = "Cardiovascular Diseases, Dyslipidemia, Hypotension, Non-alcoholic Fatty Liver Disease (NAFLD)."
            elif cluster[0] == 2:
                cluster_result = "Polycystic Ovary Syndrome (PCOS), Insulin Resistance Syndrome, Gallbladder Disease."

            # Return the result
            return render_template('index.html', diabetes_result=diabetes_result, cluster_result=cluster_result, form_data=form_data)

        else:
            return redirect(url_for('home'))

    return render_template('index.html', form_data={})

if __name__ == '__main__':
    app.run(debug=True, port="8080")
