from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io
import os
import PyPDF2
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Required for flash messages
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = joblib.load("best_model.pkl")

# Initialize LabelEncoders for categorical features
encoders = {}
categorical_cols = ['workclass', 'marital-status', 'occupation', 
                    'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols:
    encoders[col] = LabelEncoder()

# Define feature options based on adult.csv dataset
feature_options = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                  'Local-gov', 'State-gov'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 
                       'Separated', 'Widowed'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                   'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                   'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                   'Transport-moving', 'Protective-serv'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                     'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'gender': ['Male', 'Female'],
    'native-country': ['United-States', 'England', 'Puerto-Rico', 
                      'Canada', 'Germany', 'India', 
                      'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 
                      'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 
                      'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
                      'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Guatemala', 
                      'Nicaragua']
}

# Fit encoders with all possible categories
for col in categorical_cols:
    encoders[col].fit(feature_options[col])

# Preprocess input data
def preprocess_input(df):
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = encoders[col].transform(df_encoded[col])
    return df_encoded

# Parse PDF content to extract input data
def parse_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
        
        # Define default input data
        input_data = {
            'age': 30,
            'workclass': 'Private',
            'educational-num': 10,
            'marital-status': 'Never-married',
            'occupation': 'Prof-specialty',
            'relationship': 'Not-in-family',
            'race': 'White',
            'gender': 'Male',
            'capital-gain': 0,
            'capital-loss': 0,
            'hours-per-week': 40,
            'native-country': 'United-States'
        }
        
        # Extract key-value pairs using regex
        patterns = {
            'age': r'Age:\s*(\d+)',
            'workclass': r'Workclass:\s*([^\n]+)',
            'educational-num': r'Educational Number:\s*(\d+)',
            'marital-status': r'Marital Status:\s*([^\n]+)',
            'occupation': r'Occupation:\s*([^\n]+)',
            'relationship': r'Relationship:\s*([^\n]+)',
            'race': r'Race:\s*([^\n]+)',
            'gender': r'Gender:\s*([^\n]+)',
            'capital-gain': r'Capital Gain:\s*(\d+)',
            'capital-loss': r'Capital Loss:\s*(\d+)',
            'hours-per-week': r'Hours per Week:\s*(\d+)',
            'native-country': r'Native Country:\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                    input_data[key] = int(value)
                else:
                    # Validate categorical values
                    if value in feature_options[key]:
                        input_data[key] = value
                    else:
                        flash(f'Invalid {key} in PDF: {value}. Using default: {input_data[key]}', 'warning')
        
        return input_data
    except Exception as e:
        flash(f'Error parsing PDF: {str(e)}', 'danger')
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                if file.filename.endswith('.pdf'):
                    # Handle PDF upload
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)
                    input_data = parse_pdf(file_path)
                    os.remove(file_path)
                    if input_data:
                        return redirect(url_for('results', **input_data))
                    else:
                        return redirect(url_for('index'))
                elif file.filename.endswith('.csv'):
                    # Handle CSV upload
                    return redirect(url_for('batch_predict'))
                else:
                    flash('Please upload a valid PDF or CSV file', 'danger')
                    return redirect(url_for('index'))
            
            # Handle form submission
            input_data = {
                'age': int(request.form.get('age', 30)),
                'workclass': request.form.get('workclass', 'Private'),
                'educational-num': int(request.form.get('educational-num', 10)),
                'marital-status': request.form.get('marital-status', 'Never-married'),
                'occupation': request.form.get('occupation', 'Prof-specialty'),
                'relationship': request.form.get('relationship', 'Not-in-family'),
                'race': request.form.get('race', 'White'),
                'gender': request.form.get('gender', 'Male'),
                'capital-gain': int(request.form.get('capital-gain', 0)),
                'capital-loss': int(request.form.get('capital-loss', 0)),
                'hours-per-week': int(request.form.get('hours-per-week', 40)),
                'native-country': request.form.get('native-country', 'United-States')
            }
            return redirect(url_for('results', **input_data))
        
        except Exception as e:
            flash(f'Error processing input: {str(e)}', 'danger')
    
    return render_template('index.html', feature_options=feature_options)

@app.route('/results')
def results():
    try:
        # Collect input data from query parameters
        input_data = {
            'age': int(request.args.get('age', 30)),
            'workclass': request.args.get('workclass', 'Private'),
            'educational-num': int(request.args.get('educational-num', 10)),
            'marital-status': request.args.get('marital-status', 'Never-married'),
            'occupation': request.args.get('occupation', 'Prof-specialty'),
            'relationship': request.args.get('relationship', 'Not-in-family'),
            'race': request.args.get('race', 'White'),
            'gender': request.args.get('gender', 'Male'),
            'capital-gain': int(request.args.get('capital-gain', 0)),
            'capital-loss': int(request.args.get('capital-loss', 0)),
            'hours-per-week': int(request.args.get('hours-per-week', 40)),
            'native-country': request.args.get('native-country', 'United-States')
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess and predict
        input_processed = preprocess_input(input_df)
        prediction = model.predict(input_processed)[0]
        prediction = '>50K' if prediction == 1 else '≤50K'
        probabilities = model.predict_proba(input_processed)[0] * 100
        probabilities = {'<=50K': round(probabilities[0], 2), '>50K': round(probabilities[1], 2)}
        
        # Generate context-aware messages
        messages = []
        message_color = '#28A745' if probabilities['>50K'] > probabilities['<=50K'] else '#DC3545'
        higher_class = '>50K' if probabilities['>50K'] > probabilities['<=50K'] else '≤50K'
        higher_prob = max(probabilities['>50K'], probabilities['<=50K'])
        
        # Primary message based on higher probability
        if higher_prob >= 80:
            messages.append({
                'text': f"Strong confidence: {higher_prob}% likelihood of earning {higher_class} annually.",
                'color': message_color
            })
        elif higher_prob >= 50:
            messages.append({
                'text': f"Moderate confidence: {higher_prob}% chance of earning {higher_class} annually.",
                'color': message_color
            })
        else:
            messages.append({
                'text': f"Uncertain prediction: Only {higher_prob}% confidence for earning {higher_class}.",
                'color': message_color
            })
        
        # Additional contextual messages
        if higher_class == '>50K':
            messages.append({
                'text': "This suggests a strong career trajectory. Consider leveraging skills in high-demand fields like tech or management.",
                'color': '#28A745'
            })
            if input_data['educational-num'] < 10:
                messages.append({
                    'text': "Higher education could further boost your earning potential.",
                    'color': '#6c757d'
                })
            if input_data['hours-per-week'] < 40:
                messages.append({
                    'text': "Increasing weekly work hours may enhance income prospects.",
                    'color': '#6c757d'
                })
        else:
            messages.append({
                'text': "Opportunities for growth exist. Upskilling or exploring new career paths could increase earnings.",
                'color': '#DC3545'
            })
            if input_data['occupation'] in ['Other-service', 'Handlers-cleaners', 'Farming-fishing']:
                messages.append({
                    'text': f"Your occupation ({input_data['occupation']}) may have limited earning potential. Consider specialized training.",
                    'color': '#6c757d'
                })
            if input_data['capital-gain'] == 0:
                messages.append({
                    'text': "Investing or exploring side ventures could improve financial outcomes.",
                    'color': '#6c757d'
                })
        
        return render_template('results.html', 
                             prediction=prediction, 
                             probabilities=probabilities, 
                             messages=messages)
    
    except Exception as e:
        flash(f'Error processing prediction: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save and read uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            batch_data = pd.read_csv(file_path)
            
            # Validate columns
            expected_cols = ['age', 'workclass', 'educational-num', 
                            'marital-status', 'occupation', 'relationship', 'race', 
                            'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 
                            'native-country']
            if not all(col in batch_data.columns for col in expected_cols):
                flash('CSV file must contain all required columns', 'danger')
                os.remove(file_path)
                return redirect(url_for('index'))
            
            # Check if CSV has exactly one row
            if len(batch_data) == 1:
                # Validate and extract data for single-row CSV
                input_data = {}
                for col in expected_cols:
                    value = batch_data.iloc[0][col]
                    if col in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                        try:
                            input_data[col] = int(value)
                        except (ValueError, TypeError):
                            flash(f'Invalid {col} in CSV: {value}. Using default.', 'warning')
                            input_data[col] = {'age': 30, 'educational-num': 10, 'capital-gain': 0, 
                                              'capital-loss': 0, 'hours-per-week': 40}.get(col)
                    elif col in categorical_cols:
                        if pd.isna(value) or str(value).strip() not in feature_options[col]:
                            flash(f'Invalid {col} in CSV: {value}. Using default.', 'warning')
                            input_data[col] = {'workclass': 'Private', 'marital-status': 'Never-married', 
                                              'occupation': 'Prof-specialty', 'relationship': 'Not-in-family', 
                                              'race': 'White', 'gender': 'Male', 
                                              'native-country': 'United-States'}.get(col)
                        else:
                            input_data[col] = str(value).strip()
                
                os.remove(file_path)
                return redirect(url_for('results', **input_data))
            
            # Process batch CSV (multiple rows)
            batch_processed = preprocess_input(batch_data)
            batch_preds = model.predict(batch_processed)
            batch_probs = model.predict_proba(batch_processed) * 100
            batch_data['PredictedClass'] = np.where(batch_preds == 1, '>50K', '≤50K')
            batch_data['Probability_<=50K (%)'] = batch_probs[:, 0].round(2)
            batch_data['Probability_>50K (%)'] = batch_probs[:, 1].round(2)
            
            # Save predictions to CSV
            output = io.StringIO()
            batch_data.to_csv(output, index=False)
            output.seek(0)
            
            # Clean up
            os.remove(file_path)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name='predicted_salary_classes.csv'
            )
        
        except Exception as e:
            flash(f'Error processing batch prediction: {str(e)}', 'danger')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('index'))
    
    flash('Please upload a valid CSV file', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)