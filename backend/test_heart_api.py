import requests

# API endpoint
url = "http://127.0.0.1:5000/predict_heart"

# Example heart disease input (replace with your own test values)
data = {
    "age": 63,
    "sex": 1,         # 1 = male, 0 = female
    "cp": 3,          # chest pain type
    "trestbps": 145,  # resting blood pressure
    "chol": 233,      # serum cholesterol
    "fbs": 1,         # fasting blood sugar > 120 mg/dl
    "restecg": 0,     # resting electrocardiographic results
    "thalach": 150,   # maximum heart rate achieved
    "exang": 0,       # exercise induced angina
    "oldpeak": 2.3,   # ST depression induced by exercise
    "slope": 0,       # slope of the peak exercise ST segment
    "ca": 0,          # number of major vessels (0–3)
    "thal": 1         # thalassemia
}

# Send POST request
response = requests.post(url, json=data)

# Print prediction result
print("Status Code:", response.status_code)
print("Response:", response.json())
