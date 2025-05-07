import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text

def train_model():
    """
    Trains a logistic regression model on the feedback data.

    Returns:
        model: Trained LogisticRegression model.
        vectorizer: Fitted TfidfVectorizer for text transformation.
    """
    try:
        # Load the dataset
        data = pd.read_csv("/Users/zach/Documents/sentiment_analysis_project/sample data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("The file path '/Users/zach/Documents/sentiment_analysis_project/sample data.csv' does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The CSV file is malformed.")

    # Validate required columns
    required_columns = {"feedback", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Drop rows with missing values in required columns
    data = data.dropna(subset=["feedback", "label"])

    # Ensure labels are numeric (if not, map them to integers)
    if not pd.api.types.is_numeric_dtype(data["label"]):
        label_mapping = {"Negative": 0, "Positive": 1, "Neutral": 2}
        # Check if all labels are in the mapping
        unknown_labels = set(data["label"]) - set(label_mapping.keys())
        if unknown_labels:
            raise ValueError(f"Found unknown labels: {unknown_labels}")
        data["label"] = data["label"].map(label_mapping)

    # Print class distribution
    class_counts = data["label"].value_counts()
    print("Class distribution:")
    print(class_counts)

    # Check if there are at least two unique classes with sufficient samples
    min_samples_per_class = 2
    if (class_counts < min_samples_per_class).any():
        stratify = None
    else:
        stratify = data["label"]

    # Preprocess the feedback column
    data["feedback"] = data["feedback"].apply(clean_text)

    # Split the data into training and testing sets
    X_train, _, y_train, _ = train_test_split(
        data["feedback"], 
        data["label"], 
        test_size=0.2, 
        random_state=42,
        stratify=stratify
    )

    # Transform the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train the logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vec, y_train)

    return model, vectorizer

def predict_sentiment(model, vectorizer, text):
    """
    Predicts the sentiment of a given text using the trained model.

    Args:
        model: Trained LogisticRegression model
        vectorizer: Fitted TfidfVectorizer
        text: Text to analyze

    Returns:
        int: Predicted sentiment (0: Negative, 1: Positive, 2: Neutral)
    """
    # Transform the input text
    text_vec = vectorizer.transform([text])
    
    # Make prediction
    return model.predict(text_vec)[0]