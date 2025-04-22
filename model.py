import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Warning: NLTK download failed. If stopwords are missing, manually download them.")

class SpamDetector:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        
    def preprocess_text(self, text):
        # Handle None or non-string inputs
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        # Join back into a string
        return ' '.join(tokens)
    
    def train(self, data_path):
        # Check if data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file {data_path} not found.")
            
        # Load data
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        print("\n" + "="*50)
        print("SPAM DETECTOR TRAINING PROCESS")
        print("="*50)
        print(f"Dataset shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        
        # Check class distribution
        spam_count = df['spam'].sum()
        ham_count = len(df) - spam_count
        print("\nClass Distribution:")
        print(f"Spam messages: {spam_count} ({spam_count/len(df)*100:.2f}%)")
        print(f"Ham messages: {ham_count} ({ham_count/len(df)*100:.2f}%)")
        
        # Preprocess the text
        print("\nPreprocessing text...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split into training and testing sets
        print("Splitting dataset into training (80%) and testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['spam'], test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Vectorize the text
        print("\nVectorizing text...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Feature matrix shape: {X_train_vectorized.shape}")
        
        # Train the model
        print("\nTraining Multinomial Naive Bayes model...")
        self.model.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        y_pred_proba = self.model.predict_proba(X_test_vectorized)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print detailed metrics
        print(f"\nAccuracy: {accuracy:.4f} (Percentage of correctly classified messages)")
        print(f"Precision: {precision:.4f} (Percentage of true spam among detected spam)")
        print(f"Recall: {recall:.4f} (Percentage of detected spam among all actual spam)")
        print(f"F1 Score: {f1:.4f} (Harmonic mean of precision and recall)")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Confusion Matrix with explanation
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nConfusion Matrix Explanation:")
        print(f"True Negatives (Ham correctly identified as Ham): {cm[0][0]}")
        print(f"False Positives (Ham incorrectly identified as Spam): {cm[0][1]}")
        print(f"False Negatives (Spam incorrectly identified as Ham): {cm[1][0]}")
        print(f"True Positives (Spam correctly identified as Spam): {cm[1][1]}")
        
        # Calculate and display additional metrics
        misclassification_rate = 1 - accuracy
        false_positive_rate = cm[0][1] / (cm[0][0] + cm[0][1])
        false_negative_rate = cm[1][0] / (cm[1][0] + cm[1][1])
        
        print("\nAdditional Metrics:")
        print(f"Misclassification Rate: {misclassification_rate:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f} (Percentage of ham classified as spam)")
        print(f"False Negative Rate: {false_negative_rate:.4f} (Percentage of spam classified as ham)")
        
        # Model information
        print("\nModel Parameters:")
        print(f"Class priors: {self.model.class_log_prior_.tolist()}")
        print(f"Number of features: {self.model.feature_count_.shape[1]}")
        
        print("\n" + "="*50 + "\n")
        
        return accuracy
    
    def predict(self, text):
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        # Get probability
        probability = self.model.predict_proba(text_vectorized)[0][1]
        
        return {
            'is_spam': bool(prediction),
            'spam_probability': float(probability),
            'ham_probability': float(1 - probability)
        }
    
    def save_model(self, vectorizer_path='vectorizer.pkl', model_path='spam_model.pkl'):
        print(f"Saving model to {vectorizer_path} and {model_path}...")
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
                
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print("Model saved successfully.")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, vectorizer_path='vectorizer.pkl', model_path='spam_model.pkl'):
        # Check if files exist and have content
        if not os.path.exists(vectorizer_path) or os.path.getsize(vectorizer_path) == 0:
            print(f"Warning: {vectorizer_path} is empty or does not exist. Need to train model first.")
            return False
        
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            print(f"Warning: {model_path} is empty or does not exist. Need to train model first.")
            return False
        
        try:
            # Load the files if they exist and have content
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    detector = SpamDetector()
    try:
        print("\n" + "*"*70)
        print("*" + " "*26 + "SPAM DETECTOR" + " "*27 + "*")
        print("*" + " "*22 + "Training and Evaluation" + " "*23 + "*")
        print("*"*70)
        
        accuracy = detector.train('emails.csv')
        
        # Save the trained model
        detector.save_model()
        
        # Test with sample texts
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS")
        print("="*50)
        
        test_messages = [
            "Subject: FREE OFFER - Increase your sales now with our marketing services",
            "Subject: Meeting Tomorrow - Please bring your presentation materials",
            "Subject: URGENT - You've WON $5,000,000 in our lottery!",
            "Subject: Project status update and next steps"
        ]
        
        for i, message in enumerate(test_messages):
            result = detector.predict(message)
            print(f"\nExample {i+1}: \"{message}\"")
            print(f"Classification: {'SPAM' if result['is_spam'] else 'HAM'}")
            print(f"Spam probability: {result['spam_probability']:.4f}")
            print(f"Ham probability: {result['ham_probability']:.4f}")
        
        print("\n" + "*"*70)
        print("Spam detection model training and evaluation complete.")
        print("*"*70 + "\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")