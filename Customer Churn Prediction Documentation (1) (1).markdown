# Customer Churn Prediction System Documentation

## Overview
This document outlines the process for building, deploying, and monitoring a customer churn prediction system using a Random Forest model. The system uses scikit-learn and pandas for training, Flask for API serving, Docker for containerization.

## Step 1: Training the Random Forest Model

### Objective
Train a Random Forest model to predict customer churn based on historical data.

### Tools
- **Scikit-learn**: For model training and evaluation.
- **Pandas**: For data preprocessing.
- **Joblib**: For model serialization.

### Process
1. **Data Preparation**:
   - Input: CSV file (`customer_data.csv`) with features (e.g., `tenure`, `usage`, `age`, `monthly_charges`, `contract_type`) and target (`churn`: 1 for churned, 0 for not).
   - For demonstration, synthetic data is generated with 1000 samples.
   - Categorical features (`contract_type`) are one-hot encoded using `pd.get_dummies`.

2. **Model Training**:
   - Split data into 80% training and 20% testing sets.
   - Train a Random Forest Classifier (`n_estimators=100`, `random_state=42`).
   - Evaluate using accuracy and classification report.

3. **Model Serialization**:
   - Save the trained model to `churn_model.pkl` using `joblib`.

### Code
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Generate synthetic data (replace with pd.read_csv('customer_data.csv') for real data)
np.random.seed(42)
n_samples = 1000
data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'usage': np.random.uniform(0, 100, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'monthly_charges': np.random.uniform(20, 150, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Preprocess: Encode categorical features
df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)

# Split features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'churn_model.pkl')
```

## Step 2: Building the Flask API

### Objective
Serve the trained model via a REST API for real-time predictions.

### Tools
- **Flask**: Lightweight web framework for API.
- **Joblib**: To load the model.
- **Pandas**: For input data preprocessing.

### Process
1. Create `app.py` with a `/predict` endpoint accepting JSON input (e.g., `{"tenure": 12, "usage": 50, "age": 30, "monthly_charges": 70, "contract_type": "Month-to-month"}`).
2. Preprocess input: Encode categorical variables and align with training data columns.
3. Return prediction (`churn`: 0 or 1) and churn probability.

### Code
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)
    # Align columns to match training data
    missing_cols = set(['contract_type_One year', 'contract_type_Two year']) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    prediction = model.predict(df)
    prob = model.predict_proba(df)[0][1]
    return jsonify({'churn': int(prediction[0]), 'probability': prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Testing
Run `python app.py` and test with:
```
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"tenure": 12, "usage": 50, "age": 30, "monthly_charges": 70, "contract_type": "Month-to-month"}'
```

## Step 3: Containerizing with Docker

### Objective
Package the Flask app for scalable deployment.

### Tools
- **Docker**: For containerization.

### Process
1. Create a `Dockerfile` to build a lightweight Python image.
2. Include dependencies in `requirements.txt`.
3. Build and test the container locally.

### Files
**`requirements.txt`**:
```
flask
scikit-learn
pandas
joblib
numpy
prometheus-client
```

**`Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Commands
```
docker build -t churn-predictor .
docker run -p 5000:5000 churn-predictor
```
