import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from prefect import task, flow
import warnings
warnings.filterwarnings('ignore')

# Initialize WordNet lemmatizer and Porter stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Define the function to categorize sentiment
def categorize_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating <= 2:
        return 'Negative'
    else:
        return 'Neutral'

# Define the function to preprocess text
def preprocess(text, flag):
    """
    Preprocess the text data by removing special characters, converting to lowercase,
    tokenizing, removing stop words, and lemmatizing/stemming.
    """
    sentence = re.sub(r'[^a-zA-Z]', ' ', text)
    sentence = sentence.lower()
    tokens = sentence.split()
    clean_tokens = [token for token in tokens if token not in stopwords.words("english")]
    if flag == 'stem':
        clean_tokens = [stemmer.stem(token) for token in clean_tokens]
    else:
        clean_tokens = [lemmatizer.lemmatize(token) for token in clean_tokens]
    return ' '.join(clean_tokens)

# Define the data loading function
@task
def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    return pd.read_csv(file_path)

# Define the text preprocessing function
@task
def preprocess_data(df, flag):
    """
    Preprocess the data by categorizing sentiment, selecting relevant columns, and
    transforming the text.
    """
    df1 = df[['Review text', 'Ratings']]
    df1.isnull().sum()
    df1.dropna(inplace=True)
    df1['Sentiment'] = df1['Ratings'].apply(categorize_sentiment)
    X = df1['Review text']
    y = df1['Sentiment']
    X_transformed = X.apply(lambda x: preprocess(x, flag))
    return X_transformed, y

# Define the data splitting function
@task
def split_data(X_transformed, y):
    """
    Split the preprocessed data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, random_state=1)
    return X_train, X_test, y_train, y_test

# Define the ML pipeline initialization function
@task
def define_pipelines():
    """
    Initialize ML pipelines with predefined configurations.
    """
    pipelines = {
        'naive_bayes': Pipeline([
           ('vectorization', CountVectorizer()),
            ('classifier', MultinomialNB())
        ]),
        'decision_tree': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', DecisionTreeClassifier())
        ]),
        'logistic_regression': Pipeline([
            ('vectorization', CountVectorizer()),
            ('classifier', LogisticRegression())
        ])
    }
    return pipelines

# Define the parameter grid initialization function
@task
def define_parameter_grids():
    """
    Initialize the parameter grids for the GridSearchCV.
    """
    param_grids = {
        'naive_bayes': [
            {
                'vectorization': [CountVectorizer(), TfidfVectorizer()],
                'vectorization__max_features' : [1000, 1500, 2000, 5000], 
                'classifier__alpha' : [1, 10]
            }
        ],
        'decision_tree': [
            {
                'vectorization': [CountVectorizer(), TfidfVectorizer()],
                'vectorization__max_features' : [1000, 1500, 2000, 5000],
                'classifier__max_depth': [None, 5, 10]
            }
        ],
        'logistic_regression': [
            {
                'vectorization': [CountVectorizer(), TfidfVectorizer()],
                'vectorization__max_features' : [1000, 1500, 2000, 5000], 
                'classifier__C': [0.1, 1, 10], 
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear'], # Changed solver to liblinear
                'classifier__class_weight': ['balanced']
            }
        ]
    }
    return param_grids

# Define the GridSearchCV function
@task
def perform_grid_search(pipeline, X_train, y_train, param_grid):
    """
    Perform GridSearchCV to find the best hyperparameters for a given pipeline.
    """
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='f1', return_train_score=True)
    grid_search.fit(X_train, y_train)
    return grid_search

@flow(name="Sentiment Review Analysis Model Development Workflow")
def workflow():
    file_path = "C:/Users/VISHRUTH NIMALAN/Downloads/data.csv"

    # Load data
    data = load_data(file_path)

    # Preprocess data
    X_transformed, y = preprocess_data(data, flag='lemma') # Changed flag to 'lemma'

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_transformed, y)

    # Define pipelines
    pipelines = define_pipelines()

    # Define parameter grids
    param_grids = define_parameter_grids()

    # Perform GridSearchCV
    grid_search_results = {}
    for algo, pipeline in pipelines.items():
        grid_search = perform_grid_search(pipeline, X_train, y_train, param_grids[algo])
        grid_search_results[algo] = grid_search


if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="* * * * *"
    )