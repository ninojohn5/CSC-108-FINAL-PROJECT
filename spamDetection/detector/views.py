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

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preprocessing components
stop_words = None
lemmatizer = None
cached_lemmas = {}

def init_nltk_resources():
    """Initialize NLTK resources and set stopwords/lemmatizer."""
    global stop_words, lemmatizer
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'without'}
    lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess a single text input efficiently."""
    if not stop_words or not lemmatizer:
        init_nltk_resources()

    try:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-zA-Z0-9\s@#]', ' ', text)  # Remove unwanted characters
        text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize spaces and lowercase
        text = emoji.demojize(text)  # Convert emojis
        processed = []
        for word in text.split():
            if word in stop_words:
                continue
            if word not in cached_lemmas:
                cached_lemmas[word] = lemmatizer.lemmatize(word)
            processed.append(cached_lemmas[word])
        return ' '.join(processed)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return text

# Load and validate dataset
try:
    dataset = pd.read_csv('data/emails.csv').drop_duplicates(subset=['text']).dropna(subset=['text', 'spam'])
    if dataset.empty:
        raise ValueError("Dataset is empty after preprocessing.")
    dataset['text'] = dataset['text'].map(preprocess_text)
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    dataset = pd.DataFrame(columns=['text', 'spam'])

if not dataset.empty:
    X = dataset['text']
    y = dataset['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Build and train pipeline
    pipeline = ImbPipeline([
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

    try:
        pipeline.fit(X_train, y_train)
        logger.info("Model training complete.")
        dump(pipeline, 'spam_pipeline.pkl')
    except Exception as e:
        logger.error(f"Error during training: {e}")

    # Evaluate model
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

    # Load model
    model_loaded = load('spam_pipeline.pkl')

    def predict_message(message):
        try:
            if not message.strip():
                return "Invalid input"
            message = preprocess_text(message)
            prediction = model_loaded.predict([message])
            return "Spam" if prediction[0] == 1 else "Ham"
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error in prediction"

    # Django view
    def home(request):
        result = None
        if request.method == "POST":
            form = MessageForm(request.POST)
            if form.is_valid():
                message = form.cleaned_data['text']
                result = predict_message(message)
        else:
            form = MessageForm()
        return render(request, 'home.html', {'form': form, 'result': result})
else:
    logger.error("No dataset available. Application will not proceed.")