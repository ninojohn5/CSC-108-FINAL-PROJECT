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
from collections import Counter

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton for NLTK resources
class NLTKResources:
    """Singleton for initializing and caching NLTK resources."""
    _stop_words = None
    _lemmatizer = None

    @classmethod
    def initialize(cls):
        if not cls._stop_words or not cls._lemmatizer:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            cls._stop_words = set(stopwords.words('english')) - {'not', 'no', 'without'}
            cls._lemmatizer = WordNetLemmatizer()
        return cls._stop_words, cls._lemmatizer

# Text preprocessing
cached_lemmas = {}

def preprocess_text(text):
    """Preprocess a single text input."""
    try:
        stop_words, lemmatizer = NLTKResources.initialize()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-zA-Z0-9\s@#]', ' ', text)  # Remove unwanted characters
        text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize spaces and lowercase
        text = emoji.demojize(text)  # Convert emojis to text
        processed = [
            cached_lemmas.setdefault(word, lemmatizer.lemmatize(word))
            for word in text.split()
            if word not in stop_words
        ]
        return ' '.join(processed)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return text

# Load dataset with chunking
try:
    dataset_path = 'data/emails.csv'
    dataset = pd.read_csv(dataset_path, usecols=['text', 'spam']).drop_duplicates(subset=['text']).dropna()
    dataset['text'] = dataset['text'].map(preprocess_text)
    if dataset.empty:
        raise ValueError("Dataset is empty after preprocessing.")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    dataset = pd.DataFrame(columns=['text', 'spam'])

# Check if dataset is valid
if not dataset.empty:
    X = dataset['text']
    y = dataset['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Check class distribution
    class_distribution = Counter(y_train)
    logger.info(f"Class distribution: {class_distribution}")
    use_smote = class_distribution[1] / class_distribution[0] < 0.5  # Adjust threshold

    # Build pipeline
    pipeline_steps = [('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)))]
    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=42)))
    pipeline_steps.append(('classifier', VotingClassifier(
        estimators=[
            ('nb', MultinomialNB(alpha=0.1)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42))
        ],
        voting='soft'
    )))
    pipeline = ImbPipeline(pipeline_steps)

    # Train model
    try:
        pipeline.fit(X_train, y_train)
        logger.info("Model training complete.")
        dump(pipeline, 'spam_pipeline.pkl')
    except Exception as e:
        logger.error(f"Error during training: {e}")

    # Evaluation function
    def evaluate_model(model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            if hasattr(model, 'predict_proba'):
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                logger.info(f"ROC-AUC: {roc_auc}")
        except Exception as e:
            logger.error(f"Evaluation error: {e}")

    evaluate_model(pipeline, X_test, y_test)

    # Load trained model
    model_loaded = load('spam_pipeline.pkl')

    def predict_message(message):
        """Predict whether a message is spam or ham."""
        try:
            if not model_loaded:
                return "Model not loaded. Contact admin."
            if not message.strip():
                return "Invalid input. Please enter a non-empty message."
            message = preprocess_text(message)
            prediction = model_loaded.predict([message])
            return "Spam" if prediction[0] == 1 else "Ham"
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error in prediction"

    # Django view
    def home(request):
        """Handle home page requests."""
        result = None
        if not dataset.empty:
            if request.method == "POST":
                form = MessageForm(request.POST)
                if form.is_valid():
                    message = form.cleaned_data['text']
                    result = predict_message(message)
                else:
                    result = "Invalid input. Please enter a valid message."
            else:
                form = MessageForm()
        else:
            form = None
            result = "Dataset unavailable. Contact admin."
        return render(request, 'home.html', {'form': form, 'result': result})
else:
    logger.error("No dataset available. Application will not proceed.")
