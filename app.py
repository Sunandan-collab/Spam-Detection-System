from flask import Flask, request, jsonify, render_template
import pickle
import os
import traceback
from model import SpamDetector

app = Flask(__name__)

# Initialize the spam detector
detector = SpamDetector()

# Check if model files exist, otherwise train the model
try:
    print("Attempting to load existing model...")
    if not detector.load_model():
        print("Training new model...")
        if os.path.exists('emails.csv'):
            detector.train('emails.csv')
            detector.save_model()
        else:
            print("Error: Email_test.csv not found in the current directory.")
            print(f"Current working directory: {os.getcwd()}")
            print("Please place the dataset file in this directory.")
except Exception as e:
    print(f"Error during model initialization: {e}")
    print(traceback.format_exc())
    print("Training new model...")
    try:
        if os.path.exists('emails.csv'):
            detector.train('emails.csv')
            detector.save_model()
        else:
            print("Error: Email_test.csv not found in the current directory.")
            print(f"Current working directory: {os.getcwd()}")
            print("Please place the dataset file in this directory.")
    except Exception as e:
        print(f"Failed to train model: {e}")
        print(traceback.format_exc())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        result = detector.predict(text)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        if os.path.exists('emails.csv'):
            accuracy = detector.train('emails.csv')
            detector.save_model()
            return jsonify({'success': True, 'accuracy': accuracy})
        else:
            return jsonify({'success': False, 'error': 'Dataset file not found'}), 404
    except Exception as e:
        print(f"Error during retraining: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)