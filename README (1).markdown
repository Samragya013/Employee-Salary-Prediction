# Salary Prediction App

## Overview
The Salary Prediction App is a Flask-based web application that predicts whether an individual’s annual income exceeds $50,000 based on demographic and employment data. It uses a pre-trained machine learning model (`best_model.pkl`) to process inputs from a web form, uploaded CSV files, or PDFs. The app supports:
- Manual input via a form, redirecting to a results page with predictions, a probability pie chart, and context-aware messages.
- Single-row CSV uploads, redirecting to the results page.
- Multi-row CSV uploads, generating a downloadable CSV with predictions and probabilities.
- PDF uploads, parsing 12 specific features to display results.

The app is optimized for deployment on Render, a Platform as a Service (PaaS), with secure environment variables, logging, and input validation.

## Features
- **Input Methods**:
  - Web form for manual input of 12 features: `age`, `workclass`, `educational-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.
  - CSV upload for single or multiple records.
  - PDF upload with regex-based parsing of feature values.
- **Output**:
  - Results page (`/results`) showing prediction (`>50K` or `≤50K`), probability pie chart, and context-aware advice (e.g., career or education recommendations).
  - Multi-row CSV uploads return a downloadable CSV with predictions and probabilities.
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
├── uploads/               # Temporary storage for uploaded files
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
```

## Requirements
- **Python**: 3.8+ (Render uses 3.8.13 by default).
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
- **Git and GitHub**: For version control and Render deployment.
- **Render Account**: Sign up at [render.com](https://render.com).
- **Model**: `best_model.pkl` trained with `scikit-learn==1.7.0`, expecting 12 features in order: `age`, `workclass`, `educational-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.

## Local Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/salary-prediction-app.git
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
   - **Single-Row CSV**: Upload a CSV with one row, e.g.:
     ```
     age,workclass,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
     35,Private,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,50,United-States
     ```
     Verify redirection to `/results` with prediction, pie chart, and messages.
   - **Multi-Row CSV**: Upload a CSV with multiple rows, e.g.:
     ```
     age,workclass,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
     35,Private,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,5000,0,50,United-States
     25,Self-emp-not-inc,9,Never-married,Other-service,Not-in-family,Black,Female,0,0,30,United-States
     45,Federal-gov,10,Divorced,Adm-clerical,Unmarried,White,Female,0,1500,40,Canada
     ```
     Verify download of `predicted_salary_classes.csv`.
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

## Deployment on Render
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create Render Web Service**:
   - Log in to [dashboard.render.com](https://dashboard.render.com).
   - Click **New > Web Service**, select your `salary-prediction-app` repository.
   - Configure:
     - **Name**: `salary-prediction-app`
     - **Environment**: Python 3
     - **Region**: Closest (e.g., Oregon for US West)
     - **Root Directory**: Leave blank
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Instance Type**: Free (spins down after inactivity; upgrade for persistent uptime)
     - **Environment Variables**:
       ```
       PYTHON_VERSION=3.8.13
       SECRET_KEY=<random_string>  # Generate: python -c "import secrets; print(secrets.token_hex(16))"
       ```
   - Click **Create Web Service**. Build takes 5–10 minutes (free tier).

3. **Access the App**:
   - Use the provided URL (e.g., `https://salary-prediction-app.onrender.com`).
   - Test all features as described in local testing.

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
- **Build Failure**:
  - Check Render’s **Events** tab for logs.
  - Ensure `best_model.pkl` and `requirements.txt` are in the repository.
  - Verify dependency versions match `requirements.txt`.
- **File Upload Issues**:
  - Ensure `uploads/` is created and files are deleted after processing.
  - Render’s ephemeral filesystem is compatible since files are temporary.
- **Logs**: Check Render’s **Events** tab for debug info (e.g., `Preprocessed DataFrame columns` or error messages).

## Notes
- **Free Tier**: Spins down after ~15 minutes of inactivity, causing a 30-second delay on the next request. Upgrade to a paid plan ($7/month+) for persistent uptime.
- **Security**: Set a secure `SECRET_KEY` in Render’s environment variables.
- **Model Compatibility**: Ensure `best_model.pkl` was trained with `scikit-learn<=1.7.0`. Re-train if necessary.
- **Updates**: Push changes to GitHub to trigger automatic re-deployment on Render.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License.