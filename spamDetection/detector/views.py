import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import emoji
from joblib import dump, load
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from django.shortcuts import render
from .forms import MessageForm
import logging

# Set up logging for tracking and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize necessary resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english')) - {'not', 'no', 'without'}  # Keep negation words
lemmatizer = WordNetLemmatizer()
cached_lemmas = {}  # Cache lemmatized words to speed up processing

# Preprocess text by cleaning and lemmatizing

def preprocess_text(text):
    """
    Clean and preprocess text by removing URLs, emails, and unnecessary characters.
    Convert emojis to text and apply lemmatization while filtering out stopwords.

    Args:
        text (str): Raw input text.

    Returns:
        str: Processed text ready for model input.
    """
    try:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
        text = re.sub(r'[^a-zA-Z0-9\s@#]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize spaces and lowercase
        text = emoji.demojize(text)  # Convert emojis to descriptive text

        processed = [
            cached_lemmas.setdefault(word, lemmatizer.lemmatize(word))
            for word in text.split() if word not in stop_words
        ]
        return ' '.join(processed)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return text

# Load and preprocess the dataset from a CSV file

def load_dataset(filepath):
    """
    Load dataset from file, remove duplicates, handle missing values, and preprocess the text.

    Args:
        filepath (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Cleaned and processed dataset.
    """
    try:
        dataset = pd.read_csv(filepath).drop_duplicates(subset=['text']).dropna(subset=['text', 'spam'])
        dataset['text'] = dataset['text'].map(preprocess_text)
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=['text', 'spam'])

# Build a machine learning pipeline for spam classification

def build_pipeline():
    """
    Create and return a pipeline with TF-IDF for text vectorization, SMOTE for class balancing, 
    and a VotingClassifier combining Naive Bayes, Logistic Regression, and Random Forest.

    Returns:
        Pipeline: Configured machine learning pipeline.
    """
    return ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ('smote', SMOTE(random_state=42)),
        ('classifier', VotingClassifier(
            estimators=[
                ('nb', MultinomialNB(alpha=0.1)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42))
            ],
            voting='soft'
        ))
    ])

# Evaluate the model using test data and log the results

def evaluate_model(model, X_test, y_test):
    """
    Assess model performance using accuracy, classification report, confusion matrix, and ROC-AUC.

    Args:
        model (Pipeline): Trained model.
        X_test (pd.Series): Test features.
        y_test (pd.Series): True labels.
    """
    try:
        y_pred = model.predict(X_test)
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        if hasattr(model, 'predict_proba'):
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        logger.error(f"Evaluation error: {e}")

# Load dataset, train model, and evaluate performance
filepath = 'data/emails.csv'
dataset = load_dataset(filepath)

if not dataset.empty:
    X = dataset['text']
    y = dataset['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = build_pipeline()
    try:
        pipeline.fit(X_train, y_train)
        logger.info("Model training complete.")
        dump(pipeline, 'spam_pipeline.pkl')  # Save model to file
        evaluate_model(pipeline, X_test, y_test)
    except Exception as e:
        logger.error(f"Error during training: {e}")

    model_loaded = load('spam_pipeline.pkl')

# Predict if a message is spam or ham

def predict_message(message):
    """
    Classify a message as spam or ham.

    Args:
        message (str): Message to classify.

    Returns:
        str: "Spam" or "Ham" based on prediction.
    """
    try:
        if not message.strip():
            return "Invalid input"
        processed_message = preprocess_text(message)
        prediction = model_loaded.predict([processed_message])
        return "Spam" if prediction[0] == 1 else "Ham"
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error in prediction"

# Handle web requests and display predictions

def home(request):
    """
    Render home page and handle form submissions for spam classification.

    Args:
        request (HttpRequest): Incoming HTTP request.

    Returns:
        HttpResponse: Home page with form and prediction result.
    """
    result = None
    form = MessageForm(request.POST or None)
    if form.is_valid():
        message = form.cleaned_data['text']
        result = predict_message(message)
    return render(request, 'home.html', {'form': form, 'result': result})
