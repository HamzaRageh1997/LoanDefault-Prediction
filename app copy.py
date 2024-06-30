from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, mean_squared_error, classification_report, accuracy_score, log_loss

app = Flask(__name__)

# Load the data and preprocess it
default = pd.read_csv('C:/Users/hamza/OneDrive/Desktop/Master Books/flask/template/templates/loan_history.csv')
dataMapping_Education = {
    "PhD": 1,
    "Master's": 2,
    "Bachelor's": 3,
    "High School": 4
}
dataMapping_LoanPurpose = {
    "Auto": 1,
    "Other": 2,
    "Business": 3,
    "Education": 4,
    "Home": 5
}
default['EducationNum'] = default['Education'].map(dataMapping_Education)
default['LoanPurposeNum'] = default['LoanPurpose'].map(dataMapping_LoanPurpose)
default = default[['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'LoanTerm',
                  'EducationNum', 'LoanPurposeNum', 'Default']]
x_default = default[['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'LoanTerm', 'EducationNum', 'LoanPurposeNum']]
y_default = default[['Default']]
x_train, x_test, y_train, y_test = train_test_split(x_default, y_default, test_size=0.30, random_state=15)

# Train the model
dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=2)
dt_model.fit(x_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = float(request.form['age'])
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = float(request.form['credit_score'])
        months_employed = float(request.form['months_employed'])
        num_credit_lines = float(request.form['num_credit_lines'])
        loan_term = float(request.form['loan_term'])
        education_num = float(request.form['education_num'])
        loan_purpose_num = float(request.form['loan_purpose_num'])

        # Make a prediction
        prediction = dt_model.predict([[age, income, loan_amount, credit_score, months_employed, num_credit_lines, loan_term, education_num, loan_purpose_num]])
        if prediction[0] == 0:
            result = 'Loan will not default'
        else:
            result = 'Loan will default'

        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')