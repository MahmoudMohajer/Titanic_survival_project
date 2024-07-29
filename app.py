from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
model = joblib.load('titanic_model.joblib')

def preprocess_data(df):
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[features]
    
    # Handle missing values (not needed for single input, but included for completeness)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables to numerical
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Age*Class'] = df['Age'] * df['Pclass']
    
    return df

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None

    if request.method == 'POST':
        # Get form data
        input_data = pd.DataFrame({
            'Pclass': [int(request.form['pclass'])],
            'Sex': [request.form['sex']],
            'Age': [float(request.form['age'])],
            'SibSp': [int(request.form['sibsp'])],
            'Parch': [int(request.form['parch'])],
            'Fare': [float(request.form['fare'])],
            'Embarked': [request.form['embarked']]
        })

        # Preprocess the input data
        processed_data = preprocess_data(input_data)

        # Ensure all expected columns are present
        expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'FamilySize', 'IsAlone', 'Age*Class']
        for col in expected_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0

        # Ensure columns are in the correct order
        processed_data = processed_data[expected_columns]

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)