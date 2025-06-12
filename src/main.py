"""
Lab1: Spam Email Classification

This script implements a spam email classifier using three different models:
1. Support Vector Machine (SVM)
2. Logistic Regression
3. Naive Bayes


Dataset:
1. Enron Email Dataset: https://www2.aueb.gr/users/ion/data/enron-spam/ (enron1)
    - enron1.tar.gz
2. Kaggle https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset
3. SpamAssassin Public Corpus: https://spamassassin.apache.org/old/publiccorpus/ (2003)
    - 20030228_easy_ham.tar.bz2
    - 20030228_easy_ham_2.tar.bz2
    - 20030228_spam.tar.bz2
    - 20030228_spam_2.tar.bz2
    - 20050311_spam_2.tar.bz2
"""

import pandas as pd #Handle cvs file
import re #Handle regex 
from bs4 import BeautifulSoup #Handle HTML tags
import ssl #Handle SSL errors
import nltk  #Handle text: tokenization, stemming, stopwords
from nltk.corpus import stopwords
from collections import Counter  # For word frequency counting
from sklearn.feature_extraction.text import TfidfVectorizer #Vectorization
from sklearn.model_selection import train_test_split #Split dataset
import os #Handle file paths
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Models related
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

#Visualization related
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Bypass SSL verification for stopwords downloading
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def convert_enron_to_csv(dataset_path, output_path):
    """
    Convert Enron email dataset to CSV format.
    
    The Enron dataset is organized in the following structure:
    enron1/
    ├── ham/     # Contains legitimate emails
    └── spam/    # Contains spam emails
    
    Each email is stored as a separate text file in its respective folder.
    
    Args:
        dataset_path (str): Path to the enron1 directory containing ham and spam folders
        output_path (str): Path where the output CSV file will be saved
        
    Returns:
        None: Creates a CSV file with columns 'text' and 'spam' (0 for ham, 1 for spam)
    """
    rows = []

    for label, label_value in [('ham', 0), ('spam', 1)]:
        folder = os.path.join(dataset_path, label)
        print(f"In-processing: {folder}")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='latin-1') as file:
                content = file.read().strip()
                rows.append({'text': content, 'spam': label_value})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Enron dataset: {output_path} ({len(df)} email)")

def convert_spamassassin_to_csv(base_dir: str, output_csv: str):
    """
    Convert SpamAssassin email dataset to CSV format.
    
    The SpamAssassin dataset is organized in the following structure:
    spamassassin/
    ├── spam/spam     # Contains spam emails
    └── ham/ham      # Contains legitimate emails
    
    Each email is stored as a separate text file in its respective folder.
    The dataset includes multiple archives:
    - 20030228_easy_ham.tar.bz2
    - 20030228_easy_ham_2.tar.bz2
    - 20030228_spam.tar.bz2
    - 20030228_spam_2.tar.bz2
    - 20050311_spam_2.tar.bz2
    
    Args:
        base_dir (str): Path to the directory containing spam and ham folders
        output_csv (str): Path where the output CSV file will be saved
        
    Returns:
        None: Creates a CSV file with columns 'text' and 'spam' (0 for ham, 1 for spam)
    """
    folders = [
        ("spam/spam", 1),
        ("ham/ham", 0)
    ]

    all_data = []

    for folder_name, label in folders:
        folder_path = os.path.join(base_dir, folder_name)
        print(f"In-processing: {folder_name}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read().strip()
                    all_data.append({'text': content, 'spam': label})
            except Exception as e:
                print(f"Reading error {file_path}: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)  
    print(f"Spamassassin dataset: {output_csv} ({len(df)} email)")


def read_data(stop_words_en):
    """
    Read and concatenate all datasets (Kaggle, Enron, Spamassassin).
    Returns:
        pd.DataFrame: Combined DataFrame of all emails.
    """

    ''' Processing datasets
    enron_dataset_folder = "resources/enron1" 
    enron_output_csv = "datasets/enron1.csv"
    convert_enron_to_csv(enron_dataset_folder, enron_output_csv)

    spamassassin_dataset_folder = "resources/spamassassin" 
    spamassassin_output_csv = "datasets/spamassassin.csv"
    convert_spamassassin_to_csv(spamassassin_dataset_folder, spamassassin_output_csv)
    '''

    df1 = pd.read_csv("datasets/kaggle.csv") # Kaggle
    df2 = pd.read_csv('datasets/enron1.csv') # Enron
    df3 = pd.read_csv('datasets/spamassassin.csv') # Spamassassin
    df_all = pd.concat([df1, df2, df3], ignore_index=True)
    # Remove empty or whitespace-only emails
    df_all = df_all[df_all['text'].str.strip().astype(bool)]
    # Remove duplicate emails based on content
    df_all = df_all.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
    # Clean text
    df_all["clean_text"] = df_all["text"].apply(clean_text, args=(stop_words_en,))
    # Remove emails that become empty after cleaning
    df_all = df_all[df_all['clean_text'].str.strip().astype(bool)]
    return df_all

def clean_text(text, stop_words_en):
    # 1 Check for NaN or float values
    if pd.isna(text) or isinstance(text, float):
        return ""
    # 2 to lower case
    text = text.lower()
    # 3. remove special characters, keep only letters, spaces and special_chars_to_track
    # to follow "Character frequency: The frequency of specific characters (e.g., exclamation marks, dollar signs). "
    pattern = rf'[^a-z{escaped_specials}\s]'
    text = re.sub(pattern, '', text)
    # 4. remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words_en]
    
    return " ".join(words)



def summarize_data(df: pd.DataFrame, text_col='text', label_col='spam', top_n=10):
    """
    Summarize basic statistics for the spam/ham dataset.

    Args:
        df (pd.DataFrame): DataFrame containing email data
        text_col (str): Name of the column containing email content
        label_col (str): Name of the column containing spam label (0 = ham, 1 = spam)
        top_n (int): Number of most common words to display
    """

    print("OVERVIEW SPAM/HAM DATASET")
    print("-" * 50)

    # Total number of emails
    total = len(df)
    print(f"Total emails: {total}")

    # Number and ratio by label
    label_counts = df[label_col].value_counts()
    label_ratio = df[label_col].value_counts(normalize=True)

    spam_count = label_counts.get(1, 0)
    ham_count = label_counts.get(0, 0)

    print(f"Number of SPAM emails: {spam_count} ({label_ratio.get(1, 0)*100:.2f}%)")
    print(f"Number of HAM emails : {ham_count} ({label_ratio.get(0, 0)*100:.2f}%)")

    # Empty and duplicate emails
    num_empty = (df[text_col].fillna('').str.strip() == '').sum()
    num_duplicates = df.duplicated(subset=text_col).sum()

    print(f"Empty emails: {num_empty}")
    print(f"Duplicate emails: {num_duplicates}")

    # Text length
    df['char_count'] = df[text_col].fillna('').astype(str).apply(len)
    df['word_count'] = df[text_col].fillna('').astype(str).apply(lambda x: len(x.split()))

    print(f"Average length (characters): Spam={df[df[label_col]==1]['char_count'].mean():.0f}, Ham={df[df[label_col]==0]['char_count'].mean():.0f}")
    print(f"Average length (words)   : Spam={df[df[label_col]==1]['word_count'].mean():.0f}, Ham={df[df[label_col]==0]['word_count'].mean():.0f}")

    # Most common words
    def get_top_words(texts):
        all_words = []
        for text in texts:
            if pd.isna(text):
                continue
            words = re.findall(r'\b\w+\b', str(text).lower())
            all_words.extend(words)
        return Counter(all_words).most_common(top_n)

    print(f"\nTop {top_n} most common words in SPAM:")
    for word, freq in get_top_words(df[df[label_col]==1][text_col]):
        print(f"   - {word}: {freq}")

    print(f"\nTop {top_n} most common words in HAM:")
    for word, freq in get_top_words(df[df[label_col]==0][text_col]):
        print(f"   - {word}: {freq}")

    print("-" * 50)

#Feature engineering
def prepare_manual_features(df: pd.DataFrame, text_col='clean_text'):
    """
    Add simple features to help the model classify spam emails more effectively.

    Features include:
    - is_html_email: 1 if there is an HTML tag
    - num_links: number of links (http/https)
    - num_capital_words: number of fully capitalized words
    - num_words: total number of words in the email
    - num_chars: total number of characters
    - char_*: number of occurrences of each special character in the special_chars_to_track array

    Args:
        df (pd.DataFrame): DataFrame containing emails
        text_col (str): Name of the column containing email text

    Returns:
        pd.DataFrame: DataFrame with new feature columns
    """
    def detect_html(text):
        if pd.isna(text):
            return 0
        return int(bool(re.search(r'<\s*html|<\s*body|<\s*div|<\s*table|<\s*span|<\s*a\s+href', str(text), re.IGNORECASE)))
    
    def count_links(text):
        if pd.isna(text):
            return 0
        return len(re.findall(r'https?://|www\.', str(text), re.IGNORECASE))
    
    def count_caps(text):
        if pd.isna(text):
            return 0
        return len([word for word in str(text).split() if word.isupper() and len(word) > 1])
    
    def count_words(text):
        if pd.isna(text):
            return 0
        return len(str(text).split())
    
    def count_chars(text):
        if pd.isna(text):
            return 0
        return len(str(text))

    # Add basic features
    df['is_html_email'] = df[text_col].apply(detect_html)
    df['num_links'] = df[text_col].apply(count_links)
    df['num_capital_words'] = df[text_col].apply(count_caps)
    df['num_words'] = df[text_col].apply(count_words)
    df['num_chars'] = df[text_col].apply(count_chars)

    # Define manual features list
    manual_features = [
        'is_html_email',
        'num_links',
        'num_capital_words',
        'num_words',
        'num_chars'
    ]

    # Add special character features
    for char in special_chars_to_track:
        safe_char = char if char.isalnum() else f'char_{repr(char)[1:-1].replace("\\", "\\")}'
        col_name = f'char_{safe_char}'
        df[col_name] = df[text_col].fillna('').apply(lambda x: x.count(char))
        manual_features.append(col_name)

    return df, manual_features

def combine_tfidf_and_manual_features(df, text_col, manual_features, max_tfidf_features=10000):
    """
    Combine TF-IDF vector with manual features.
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        text_col (str): Column containing email text
        manual_features (List[str]): List of manual feature columns
        max_tfidf_features (int): Maximum number of features for TF-IDF

    Returns:
        X_final: Combined feature matrix (sparse matrix)
        y: Label vector (spam/ham)
        vectorizer: Trained TF-IDF vectorizer (can be reused)
    """
    # 1. TF-IDF
    # Vectorization
    # What does TfidfVectorizer do?
    # 1. Tokenizes df_all["clean_text"].
    # 2. Creates a dictionary of up to 5,000 words, suitable for a simple lab
    # 3. Converts each text into a vector with the length of the dictionary
    vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
    X_tfidf = vectorizer.fit_transform(df[text_col].fillna(''))

    # 2. Manual features → convert to sparse matrix
    scaler = MinMaxScaler()
    X_manual_scaled = scaler.fit_transform(df[manual_features])
    X_manual_sparse = csr_matrix(X_manual_scaled)  #

    # 3. Combine
    X_final = hstack([X_tfidf, X_manual_sparse])

    # 4. Labels
    y = df['spam'].values

    return X_final, y, vectorizer
    
#special characters to vectorization feature
special_chars_to_track = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
    '{', '|', '}', '~'
]
escaped_specials = ''.join([re.escape(char) for char in special_chars_to_track])

def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.grid(False)
    plt.show()

def classify_with_svm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Support Vector Machine (SVM) classifier.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        tuple: (y_pred, accuracy, classification_report)
    """
    # Initialize and train SVM model
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    # Predict
    y_pred = svm_model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred)

    print("SVM Accuracy:", accuracy)
    print("SVM Details:")
    print(report)
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, "SVM")
    
    return y_pred, accuracy, report

def classify_with_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Logistic Regression classifier.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        tuple: (y_pred, accuracy, classification_report)
    """
    # Initialize and train Logistic Regression model
    logistic_model = LogisticRegression(max_iter=2000)
    logistic_model.fit(X_train, y_train)
    # Predict
    y_pred = logistic_model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred)

    print("Logistic Regression Accuracy:", accuracy)
    print("Logistic Regression Details:")
    print(report)
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Logistic Regression")
    
    return y_pred, accuracy, report

def classify_with_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Naive Bayes classifier.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        tuple: (y_pred, accuracy, classification_report)
    """
    # Initialize and train Naive Bayes model
    naive_model = MultinomialNB()
    naive_model.fit(X_train, y_train)

    # Predict
    y_pred = naive_model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    print("Naive Bayes Accuracy:", accuracy)
    print("Naive Bayes Details:")
    print(report)
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Naive Bayes")
    
    return y_pred, accuracy, report

def main():
    nltk.download('stopwords')
    stop_words_en = set(stopwords.words('english'))
    
    # Read datasets
    df_all = read_data(stop_words_en)
    # Print overview
    summarize_data(df_all)
    # Add manual features
    df_all, manual_features = prepare_manual_features(df_all, "clean_text")
    # Combine tfidf and manual features
    X_final, y, vectorizer = combine_tfidf_and_manual_features(df_all, "clean_text", manual_features)
    # Split dataset into training and testing: 20% test, 80% train
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Train and evaluate models
    y_pred_svm, svm_accuracy, svm_report = classify_with_svm(X_train, X_test, y_train, y_test)
    y_pred_lr, lr_accuracy, lr_report = classify_with_logistic_regression(X_train, X_test, y_train, y_test)
    y_pred_nb, nb_accuracy, nb_report = classify_with_naive_bayes(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
