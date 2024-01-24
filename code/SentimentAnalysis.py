import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to plot pie chart
def plot_pie_chart(data, labels, colors, title):
    data.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=colors)
    plt.title(title)
    plt.show()

# Function for text preprocessing
def preprocess_text(features):
    processed_features = [re.sub(r'\W', ' ', str(feature)) for feature in features]
    return processed_features

# Function to train Random Forest model
def train_random_forest(X_train, y_train):
    classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_rf.fit(X_train, y_train)
    return classifier_rf

# Function to evaluate model accuracy
def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Read CSV file into a DataFrame for model training
df_train = pd.read_csv('Twitter_Data.csv')

# Clean DataFrame of null rows
df_train.dropna(axis=0, how='any', inplace=True)

# Text preprocessing for model training
features_train = df_train.iloc[:, 0].values
labels_train = df_train.iloc[:, 1].values
processed_features_train = preprocess_text(features_train)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text vectorization using TF-IDF for Random Forest
vectorizer_rf = TfidfVectorizer(max_features=3038, min_df=7, max_df=0.8, stop_words=stop_words)
processed_features_rf_train = vectorizer_rf.fit_transform(processed_features_train).toarray()

# Split data into training and testing sets for Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(processed_features_rf_train, labels_train, test_size=0.2)

# Train Random Forest classifier
classifier_rf = train_random_forest(X_train_rf, y_train_rf)

# Evaluate model accuracy on the test set
accuracy_rf = evaluate_model(classifier_rf, X_test_rf, y_test_rf)
print(f'Model Accuracy: {accuracy_rf * 100:.2f}%')

# Read CSV file into a DataFrame for prediction
df_predict = pd.read_csv('Agriculture_Tweets.csv')

# Clean DataFrame of null rows
df_predict.dropna(axis=0, how='any', inplace=True)

# Text preprocessing for prediction
features_predict = df_predict.iloc[:, 0].values
processed_features_predict = preprocess_text(features_predict)

# Text vectorization using TF-IDF for Random Forest prediction
processed_features_rf_predict = vectorizer_rf.transform(processed_features_predict).toarray()

# Predict sentiments for Agriculture_Tweets.csv
y_pred_rf_predict = classifier_rf.predict(processed_features_rf_predict)

# Plot a pie chart of predicted sentiments
sentiment_counts_predict = pd.Series(y_pred_rf_predict).value_counts()
colors_predict = ['red', 'yellow', 'green']
labels_predict = ['Negative', 'Neutral', 'Positive']

plot_pie_chart(sentiment_counts_predict, labels_predict, colors_predict, 'Predicted Sentiments Pie Chart')
