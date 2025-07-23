import os
import logging
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io
import PyPDF2
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-secret-key')
app.config['UPLOAD_FOLDER'] = 'Uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    model = joblib.load("best_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Initialize LabelEncoders for categorical features
encoders = {}
categorical_cols = ['workclass', 'marital-status', 'occupation', 
                    'relationship', 'race', 'gender', 'native-country']
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

# Define expected feature order for the model
expected_feature_order = ['age', 'workclass', 'educational-num', 'marital-status', 
                         'occupation', 'relationship', 'race', 'gender', 
                         'capital-gain', 'capital-loss', 'hours-per-week', 
                         'native-country']

# Fit encoders with all possible categories
for col in categorical_cols:
    encoders[col] = LabelEncoder().fit(feature_options[col])

# Preprocess input data with correct feature order
def preprocess_input(df):
    try:
        # Validate input columns
        missing_cols = [col for col in expected_feature_order if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in input DataFrame: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Select and reorder columns to match model expectations
        df_encoded = df[expected_feature_order].copy()
        
        # Transform categorical columns
        for col in categorical_cols:
            df_encoded[col] = encoders[col].transform(df_encoded[col].astype(str))
        
        logger.debug(f"Preprocessed DataFrame columns: {list(df_encoded.columns)}")
        return df_encoded
    except Exception as e:
        logger.error(f"Error in preprocess_input: {str(e)}")
        raise

# Parse PDF content to extract input data
def parse_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
        
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
                    try:
                        num_value = int(value)
                        if key == 'age' and not (17 <= num_value <= 90):
                            raise ValueError("Age out of range (17-90)")
                        if key == 'educational-num' and not (1 <= num_value <= 16):
                            raise ValueError("Educational-num out of range (1-16)")
                        if key == 'hours-per-week' and not (1 <= num_value <= 100):
                            raise ValueError("Hours-per-week out of range (1-100)")
                        input_data[key] = num_value
                    except ValueError as e:
                        flash(f"Invalid {key} in PDF: {value}. {str(e)}. Using default.", 'warning')
                else:
                    if value in feature_options[key]:
                        input_data[key] = value
                    else:
                        flash(f"Invalid {key} in PDF: {value}. Using default: {input_data[key]}", 'warning')
        
        return input_data
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        flash(f"Error parsing PDF: {str(e)}", 'danger')
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                if file.filename.endswith('.pdf'):
                    input_data = parse_pdf(file_path)
                    os.remove(file_path)
                    if input_data:
                        return redirect(url_for('results', **input_data))
                    return redirect(url_for('index'))
                elif file.filename.endswith('.csv'):
                    return redirect(url_for('batch_predict', filename=file.filename))
                else:
                    os.remove(file_path)
                    flash('Please upload a valid PDF or CSV file', 'danger')
                    return redirect(url_for('index'))
            
            # Handle form submission
            input_data = {}
            for key in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                try:
                    value = request.form.get(key, {'age': 30, 'educational-num': 10, 'capital-gain': 0, 
                                                  'capital-loss': 0, 'hours-per-week': 40}.get(key))
                    num_value = int(value)
                    if key == 'age' and not (17 <= num_value <= 90):
                        raise ValueError("Age must be between 17 and 90")
                    if key == 'educational-num' and not (1 <= num_value <= 16):
                        raise ValueError("Educational-num must be between 1 and 16")
                    if key == 'hours-per-week' and not (1 <= num_value <= 100):
                        raise ValueError("Hours-per-week must be between 1 and 100")
                    input_data[key] = num_value
                except ValueError as e:
                    flash(f"Invalid {key}: {value}. {str(e)}. Using default.", 'warning')
                    input_data[key] = {'age': 30, 'educational-num': 10, 'capital-gain': 0, 
                                       'capital-loss': 0, 'hours-per-week': 40}.get(key)
            
            for key in ['workclass', 'marital-status', 'occupation', 'relationship', 
                        'race', 'gender', 'native-country']:
                value = request.form.get(key, {'workclass': 'Private', 'marital-status': 'Never-married', 
                                              'occupation': 'Prof-specialty', 'relationship': 'Not-in-family', 
                                              'race': 'White', 'gender': 'Male', 
                                              'native-country': 'United-States'}.get(key))
                if value in feature_options[key]:
                    input_data[key] = value
                else:
                    flash(f"Invalid {key}: {value}. Using default: {input_data[key]}", 'warning')
            
            logger.info(f"Form submission processed: {input_data}")
            return redirect(url_for('results', **input_data))
        
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            flash(f"Error processing input: {str(e)}", 'danger')
            return redirect(url_for('index'))
    
    return render_template('index.html', feature_options=feature_options)

@app.route('/results')
def results():
    try:
        input_data = {}
        for key in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            try:
                value = request.args.get(key, {'age': 30, 'educational-num': 10, 'capital-gain': 0, 
                                              'capital-loss': 0, 'hours-per-week': 40}.get(key))
                num_value = int(value)
                if key == 'age' and not (17 <= num_value <= 90):
                    raise ValueError("Age must be between 17 and 90")
                if key == 'educational-num' and not (1 <= num_value <= 16):
                    raise ValueError("Educational-num must be between 1 and 16")
                if key == 'hours-per-week' and not (1 <= num_value <= 100):
                    raise ValueError("Hours-per-week must be between 1 and 100")
                input_data[key] = num_value
            except ValueError as e:
                flash(f"Invalid {key}: {value}. {str(e)}. Using default.", 'warning')
                input_data[key] = {'age': 30, 'educational-num': 10, 'capital-gain': 0, 
                                   'capital-loss': 0, 'hours-per-week': 40}.get(key)
        
        for key in ['workclass', 'marital-status', 'occupation', 'relationship', 
                    'race', 'gender', 'native-country']:
            value = request.args.get(key, {'workclass': 'Private', 'marital-status': 'Never-married', 
                                          'occupation': 'Prof-specialty', 'relationship': 'Not-in-family', 
                                          'race': 'White', 'gender': 'Male', 
                                          'native-country': 'United-States'}.get(key))
            if value in feature_options[key]:
                input_data[key] = value
            else:
                flash(f"Invalid {key}: {value}. Using default: {input_data[key]}", 'warning')
        
        input_df = pd.DataFrame([input_data])
        input_processed = preprocess_input(input_df)
        prediction = model.predict(input_processed)[0]
        prediction = '>50K' if prediction == 1 else '≤50K'
        probabilities = model.predict_proba(input_processed)[0] * 100
        probabilities = {'<=50K': round(probabilities[0], 2), '>50K': round(probabilities[1], 2)}
        
        messages = []
        message_color = '#28A745' if probabilities['>50K'] > probabilities['<=50K'] else '#DC3545'
        higher_class = '>50K' if probabilities['>50K'] > probabilities['<=50K'] else '≤50K'
        higher_prob = max(probabilities['>50K'], probabilities['<=50K'])
        
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
        
        logger.info(f"Prediction for input: {input_data}, Result: {prediction}")
        return render_template('results.html', 
                             prediction=prediction, 
                             probabilities=probabilities, 
                             messages=messages)
    
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        flash(f"Error processing prediction: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files or not request.files['file'].filename:
        flash('No file uploaded', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        batch_data = pd.read_csv(file_path)
        if not all(col in batch_data.columns for col in expected_feature_order):
            missing_cols = [col for col in expected_feature_order if col not in batch_data.columns]
            flash(f"CSV file must contain all required columns: {', '.join(expected_feature_order)}. Missing: {', '.join(missing_cols)}", 'danger')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        if len(batch_data) == 1:
            input_data = {}
            for col in expected_feature_order:
                value = batch_data.iloc[0][col]
                if col in ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
                    try:
                        num_value = int(value)
                        if col == 'age' and not (17 <= num_value <= 90):
                            raise ValueError("Age must be between 17 and 90")
                        if col == 'educational-num' and not (1 <= num_value <= 16):
                            raise ValueError("Educational-num must be between 1 and 16")
                        if col == 'hours-per-week' and not (1 <= num_value <= 100):
                            raise ValueError("Hours-per-week must be between 1 and 100")
                        input_data[col] = num_value
                    except (ValueError, TypeError) as e:
                        flash(f"Invalid {col} in CSV: {value}. {str(e)}. Using default.", 'warning')
                        input_data[col] = {'age': 30, 'educational-num': 10, 'capital-gain': 0, 
                                           'capital-loss': 0, 'hours-per-week': 40}.get(col)
                elif col in categorical_cols:
                    value_str = str(value).strip()
                    if pd.isna(value) or value_str not in feature_options[col]:
                        flash(f"Invalid {col} in CSV: {value}. Using default.", 'warning')
                        input_data[col] = {'workclass': 'Private', 'marital-status': 'Never-married', 
                                           'occupation': 'Prof-specialty', 'relationship': 'Not-in-family', 
                                           'race': 'White', 'gender': 'Male', 
                                           'native-country': 'United-States'}.get(col)
                    else:
                        input_data[col] = value_str
            
            os.remove(file_path)
            logger.info(f"Single-row CSV processed: {input_data}")
            return redirect(url_for('results', **input_data))
        
        batch_processed = preprocess_input(batch_data)
        batch_preds = model.predict(batch_processed)
        batch_probs = model.predict_proba(batch_processed) * 100
        batch_data['PredictedClass'] = np.where(batch_preds == 1, '>50K', '≤50K')
        batch_data['Probability_<=50K (%)'] = batch_probs[:, 0].round(2)
        batch_data['Probability_>50K (%)'] = batch_probs[:, 1].round(2)
        
        output = io.StringIO()
        batch_data.to_csv(output, index=False)
        output.seek(0)
        
        os.remove(file_path)
        logger.info(f"Batch CSV processed: {len(batch_data)} rows")
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predicted_salary_classes.csv'
        )
    
    except Exception as e:
        logger.error(f"Error processing batch prediction: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        flash(f"Error processing batch prediction: {str(e)}", 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
