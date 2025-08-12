from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Get form input
        data = request.form
        credit_score = int(data['CreditScore'])
        geography = data['Geography']
        gender = data['Gender']
        age = int(data['Age'])
        tenure = int(data['Tenure'])
        balance = float(data['Balance'])
        num_products = int(data['NumOfProducts'])
        has_cr_card = int(data['HasCrCard'])
        is_active = int(data['IsActiveMember'])
        salary = float(data['EstimatedSalary'])

        # Feature Engineering
        credit_score_group = 'Low' if credit_score <= 669 else ('Medium' if credit_score <= 739 else 'High')
        credit_utilization = balance / credit_score
        interaction_score = num_products + has_cr_card + is_active
        balance_to_salary_ratio = balance / salary
        credit_score_age_interaction = credit_score * age

        # Encoding
        geo_enc = encoders['Geography'].transform([geography])[0]
        gender_enc = encoders['Gender'].transform([gender])[0]
        score_group_enc = encoders['CreditScoreGroup'].transform([credit_score_group])[0]

        features = np.array([[
            credit_score,
            geo_enc,
            gender_enc,
            age,
            tenure,
            balance,
            num_products,
            has_cr_card,
            is_active,
            salary,
            score_group_enc,
            credit_utilization,
            interaction_score,
            balance_to_salary_ratio,
            credit_score_age_interaction
        ]])

        # Scaling
        scale_idx = [3, 0, 5, 9, 11, 13, 14]
        features[:, scale_idx] = scaler.transform(features[:, scale_idx])

        # Predict
        pred = model.predict(features)[0]
        result = "Customer will Churn ❌" if pred == 1 else "Customer will Stay ✅"

        return f"<h2>Prediction: {result}</h2><br><a href='/'>Back</a>"

    except Exception as e:
        return f"<h3>Error: {e}</h3><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
