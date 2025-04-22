# Spam Detection System

A modern web application that uses a Naive Bayes classifier to detect spam emails. The system features a Flask backend and a responsive, user-friendly interface.

## Features

- Machine learning-based spam detection using Multinomial Naive Bayes
- Text preprocessing including tokenization, stemming, and stop word removal
- Real-time analysis with probability scores
- Modern, responsive user interface
- Interactive results visualization

## Project Structure

```
Spam-Detection-System/
├── app.py                  # Flask application
├── model.py                # Spam detection model
├── emails.csv          # Training dataset
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   └── js/
│       └── script.js       # Frontend functionality
├── templates/
│   └── index.html          # Web interface
├── vectorizer.pkl          # Saved vectorizer (generated after training)
├── spam_model.pkl          # Saved model (generated after training)
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sunandan-collab/Spam-Detection-System.git
   cd Spam-Detection-System
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the training dataset (`emails.csv`) in the project root.

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`

3. Enter the email text in the textarea and click "Analyze" to see the results.

## Model Details

The spam detection model uses a Multinomial Naive Bayes classifier, which is particularly effective for text classification tasks. The process includes:

1. **Text Preprocessing**:
   - Converting to lowercase
   - Removing special characters and numbers
   - Tokenization
   - Removing stopwords
   - Stemming

2. **Feature Extraction**:
   - Using CountVectorizer to convert text to numerical features

3. **Classification**:
   - Training the Naive Bayes model on the processed features
   - Outputting spam probability scores

## API Endpoints

- `GET /` - Serves the web interface
- `POST /predict` - Analyzes text for spam. Expects JSON with a `text` field.
- `POST /retrain` - Retrains the model using the dataset

