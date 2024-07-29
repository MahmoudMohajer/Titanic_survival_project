# Titanic Survival Prediction App

This Flask web application predicts the probability of survival for passengers on the Titanic using machine learning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MahmoudMohajer/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask app:
   ```
   python app.py
   ```

2. Open a web browser and go to `http://localhost:5000`.

3. Fill in the passenger details in the form:
   - Pclass: Passenger class (1, 2, or 3)
   - Sex: Male or Female
   - Age: Age in years
   - SibSp: Number of siblings/spouses aboard
   - Parch: Number of parents/children aboard
   - Fare: Passenger fare
   - Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

4. Click the "Predict" button to see the survival probability.

## Features

- Web interface for easy input of passenger details
- Real-time prediction using a trained machine learning model
- Displays both the binary survival prediction and the probability of survival

## Model

The prediction model used in this app is a machine learning classifier trained on the Titanic dataset. The model considers various features such as passenger class, sex, age, and family size to make predictions.

### Model Training

The model was trained using the following process:

1. Data preprocessing:
   - Handling missing values
   - Converting categorical variables to numerical
   - Feature engineering (e.g., creating 'FamilySize', 'IsAlone', and 'Age*Class' features)

2. Feature selection:
   - Selected features: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

3. Model selection:
   - Compared Random Forest and Gradient Boosting Classifiers
   - Used GridSearchCV for hyperparameter tuning

4. Evaluation:
   - Used cross-validation and a separate validation set to assess model performance
   - Evaluated using accuracy, classification report, and confusion matrix

The best performing model was saved and is used in this application for making predictions.

## Contributing

Contributions to improve the app are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---
