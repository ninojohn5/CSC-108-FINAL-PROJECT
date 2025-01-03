import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import emoji
from joblib import dump, load
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import logging
from django.shortcuts import render
from .forms import MessageForm

# Configure logging to capture errors and important events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary resources for text preprocessing
nltk.download('stopwords')  # Download stopwords for English
nltk.download('wordnet')   # Download WordNet lemmatizer data

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Retain negative stopwords to preserve meaning in certain contexts
important_stopwords = {'not', 'no', 'without'}
filtered_stopwords = stop_words - important_stopwords

# Function to preprocess text data
def preprocess_text(text):
    """Cleans and preprocesses input text for machine learning models."""
    try:
        # Remove URLs from text
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove non-alphanumeric characters, except allowed symbols
        text = re.sub(r'[^a-zA-Z0-9\s@#]', ' ', text)
        # Replace multiple spaces with a single space and convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()
        # Convert emojis to text representation
        text = emoji.demojize(text)
        # Lemmatize words to their base form
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
        # Remove stopwords, keeping important negative ones
        text = ' '.join(word for word in text.split() if word not in filtered_stopwords)
        return text
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return text  # Return original text on error

# Load datasets with error handling
try:
    dataset1 = pd.read_csv('data/emails.csv')  # Load first dataset
    dataset2 = pd.read_csv('data/spam.csv', encoding='ISO-8859-1')  # Load second dataset
except FileNotFoundError as e:
    logger.error(f"Dataset not found: {e.filename}. Check the data directory.")
    dataset1 = pd.DataFrame(columns=['text', 'spam'])  # Fallback to empty dataframe
    dataset2 = pd.DataFrame(columns=['text', 'spam'])

# Combine datasets and clean duplicates or missing values
dataset = pd.concat([dataset1, dataset2], ignore_index=True)
dataset.drop_duplicates(subset=['text'], inplace=True)
dataset.dropna(subset=['text', 'spam'], inplace=True)

# Apply preprocessing to the text column
dataset['text'] = dataset['text'].apply(preprocess_text)

# Separate features (X) and target variable (y)
X = dataset['text']
y = dataset['spam']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set up pipeline for preprocessing and modeling
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),  # Convert text to TF-IDF features
    ('smote', SMOTE(random_state=42)),  # Handle imbalanced classes using SMOTE
    ('classifier', VotingClassifier(  # Use an ensemble of classifiers for better performance
        estimators=[
            ('nb', MultinomialNB()),  # Naive Bayes classifier
            ('lr', LogisticRegression(max_iter=1000)),  # Logistic Regression
            ('rf', RandomForestClassifier())  # Random Forest
        ], 
        voting='soft'  # Use soft voting to average probabilities
    ))
])

# Define hyperparameter grid for model tuning
parameters = [
    {'classifier__nb__alpha': [0.01, 0.1, 0.5, 1.0]},  # Alpha values for Naive Bayes
    {'classifier__lr__C': [0.1, 1, 10], 'classifier__lr__solver': ['lbfgs', 'liblinear'], 'classifier__lr__penalty': ['l2']},  # Logistic Regression params
    {'classifier__rf__n_estimators': [100, 200], 'classifier__rf__max_depth': [10, 20]},  # Random Forest params
]

# Use cross-validation and randomized search for hyperparameter optimization
cv = StratifiedKFold(n_splits=5)
grid_search = RandomizedSearchCV(
    pipeline, parameters, cv=cv, n_iter=10, n_jobs=-1, scoring='accuracy'
)

# Train the model with grid search
try:
    grid_search.fit(X_train, y_train)
    logger.info("Model training complete.")
except Exception as e:
    logger.error(f"Error during grid search: {e}")

# Save the trained model to disk (only once)
def save_model(model, filename='spam_pipeline.pkl'):
    """Save the trained model to disk."""
    try:
        dump(model, filename)  # Save model using joblib
        logger.info(f"Model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

# Evaluate the trained model
try:
    y_pred = grid_search.predict(X_test)  # Predict test set labels
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # Log accuracy
    logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")  # Detailed metrics
    logger.info(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")  # Confusion matrix
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])}")  # ROC-AUC score

    # Additional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")
except Exception as e:
    logger.error(f"Error during evaluation: {e}")

# Save model after training
save_model(grid_search)

# Function to predict spam or ham for new messages
def predict_message(message):
    """Predicts whether a message is spam or ham."""
    try:
        if not message.strip():
            return "Invalid input"  # Handle empty input
        # Load the saved model and preprocess the message (only once)
        model = load('spam_pipeline.pkl')
        message = preprocess_text(message)
        prediction = model.predict([message])
        return "Spam" if prediction[0] == 1 else "Ham"
    except Exception as e:
        logger.error(f"Error predicting message: {e}")
        return "Error in prediction"

# Django view to handle form submission and prediction
def home(request):
    """Renders the home page and processes form submissions."""
    result = None
    if request.method == "POST":
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']  # Extract message from form
            result = predict_message(message)  # Get prediction result
    else:
        form = MessageForm()  # Initialize an empty form
    return render(request, 'home.html', {'form': form, 'result': result})  # Render the template
