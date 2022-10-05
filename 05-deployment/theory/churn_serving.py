import pickle
import numpy as np
from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

with open('churn-model.bin', 'rb') as f_in:
    dv, model = pickle.load((f_in))

# customer = {
#             'customerid': '8879-zkjof',
#             'gender': 'female',
#             'seniorcitizen': 0,
#             'partner': 'no',
#             'dependents': 'no',
#             'tenure': 41,
#             'phoneservice': 'yes',
#             'multiplelines': 'no',
#             'internetservice': 'dsl',
#             'onlinesecurity': 'yes',
#             'onlinebackup': 'no',
#             'deviceprotection': 'yes',
#             'techsupport': 'yes',
#             'streamingtv': 'yes',
#             'streamingmovies': 'yes',
#             'contract': 'one_year',
#             'paperlessbilling': 'yes',
#             'paymentmethod': 'bank_transfer_(automatic)',
#             'monthlycharges': 79.85,
#             'totalcharges': 3320.75,
#             }

# prediction = predict_single(customer, dv, model)

# print('Prediction: %.3f' % prediction)

# if prediction >= 0.5:
#     print('Verdict: Churn')
# else:
#     print('Verdict: Not churn')

app = Flask('churn')

@app.route('/predict', methods=['POST']) #A
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
    'churn_probability': float(prediction),
    'churn': bool(churn),
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)