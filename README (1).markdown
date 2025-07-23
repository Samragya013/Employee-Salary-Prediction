# Employee Salary Prediction Project

## Overview
The Employee Salary Prediction App is a Flask-based web application that predicts whether an individual’s annual income exceeds $50,000 based on demographic and employment data. It uses a pre-trained machine learning model (`best_model.pkl`) to process inputs from a web form, uploaded CSV files, or PDFs. The app supports:
- Manual input via a form, redirecting to a results page with predictions, a probability pie chart, and context-aware messages.
- PDF uploads, parsing 12 specific features to display results.

The app is optimized for deployment on Render, a Platform as a Service (PaaS), with secure environment variables, logging, and input validation.

## Features
- **Input Methods**:
  - Web form for manual input of 12 features: `age`, `workclass`, `educational-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.
  - PDF upload with regex-based parsing of feature values.
- **Output**:
  - Results page (`/results`) showing prediction (`>50K` or `≤50K`), probability pie chart, and context-aware advice (e.g., career or education recommendations).
- **Validation**: Robust checks for numerical ranges (e.g., `age: 17–90`) and categorical values, with default fallbacks and user feedback via flash messages.
- **Logging**: Debug and error logs for troubleshooting, visible in Render’s dashboard.
- **Deployment**: Optimized for Render with Gunicorn, ephemeral filesystem compatibility, and secure environment variables.

## Project Structure
```
salary_prediction_app/
├── app.py                  # Main Flask application
├── best_model.pkl          # Pre-trained scikit-learn model
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html         # Form and file upload page
│   ├── results.html       # Prediction results page
├── README.md              # Project documentation
```

## Requirements
- **Python**: 3.8+ 
- **Dependencies**: Listed in `requirements.txt`:
  ```
  flask==3.0.3
  pandas==2.2.3
  scikit-learn==1.7.0
  joblib==1.5.1
  numpy==1.24.3
  PyPDF2==3.0.1
  gunicorn==23.0.0
  ```
- **Git and GitHub**: For version control.
- **Model**: `best_model.pkl` trained with `scikit-learn==1.7.0`, expecting 12 features in order: `age`, `workclass`, `educational-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.

## Local Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<Samragya013>/employee-salary-prediction.git
   cd salary_prediction_app
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App Locally**:
   ```bash
   python app.py
   ```
   Access at `http://localhost:5000`.

5. **Test Locally**:
   - **Form Submission**: Enter values (e.g., `age: 35`, `workclass: Private`) and submit to view results.
     
   - **PDF Upload**: Upload a PDF with text like:
     ```
     Age: 35
     Workclass: Private
     Educational Number: 13
     Marital Status: Married-civ-spouse
     Occupation: Exec-managerial
     Relationship: Husband
     Race: White
     Gender: Male
     Capital Gain: 5000
     Capital Loss: 0
     Hours per Week: 50
     Native Country: United-States
     ```

## Troubleshooting
- **Feature Order Error**:
  - If you see `The feature names should match those that were passed during fit`, ensure `best_model.pkl` was trained with `scikit-learn==1.7.0` and matches the feature order in `app.py`:
    ```python
    ['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 
     'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 
     'hours-per-week', 'native-country']
    ```
  - Inspect the model’s expected features by adding:
    ```python
    logger.info(f"Model expected features: {model.feature_names_in_}")
    ```
    after `model = joblib.load("best_model.pkl")`. Update `expected_feature_order` if needed.
    
## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.
