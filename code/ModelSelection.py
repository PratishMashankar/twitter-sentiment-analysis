import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read CSV file into a DataFrame
df1 = pd.read_csv('Twitter_Data.csv')

# Clean DataFrame of null rows
df1.dropna(axis=0, how='any', inplace=True)

# Set plot dimensions
plt.rcParams["figure.figsize"] = (8, 6)

# Pie chart of sentiment
df1['category'].value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["green", "yellow", "red"])
plt.show()

# Text preprocessing
features = df1.iloc[:, 0].values
labels = df1.iloc[:, 1].values

processed_features = [re.sub(r'\W', ' ', str(feature)) for feature in features]

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Varying max_features for TF-IDF vectorization
max_features_values = [250, 500, 750]
results = {'KNN': {}, 'RandomForest': {}}

for max_features in max_features_values:
    # Text vectorization using TF-IDF for KNN
    vectorizer_knn = TfidfVectorizer(max_features=max_features, min_df=7, max_df=0.8, stop_words=stop_words)
    processed_features_knn = vectorizer_knn.fit_transform(processed_features).toarray()

    # Split data into training and testing sets for KNN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(processed_features_knn, labels, test_size=0.2)

    # Standardize features using StandardScaler for KNN
    scaler_knn = StandardScaler()
    X_train_knn = scaler_knn.fit_transform(X_train_knn)
    X_test_knn = scaler_knn.transform(X_test_knn)

    # Train K-Nearest Neighbors classifier
    classifier_knn = KNeighborsClassifier(n_neighbors=300)
    classifier_knn.fit(X_train_knn, y_train_knn)

    # Predictions and evaluation for KNN
    y_pred_knn = classifier_knn.predict(X_test_knn)
    accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
    results['KNN'][max_features] = accuracy_knn

    # Text vectorization using TF-IDF for Random Forest
    vectorizer_rf = TfidfVectorizer(max_features=max_features, min_df=7, max_df=0.8, stop_words=stop_words)
    processed_features_rf = vectorizer_rf.fit_transform(processed_features).toarray()

    # Split data into training and testing sets for Random Forest
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(processed_features_rf, labels, test_size=0.2)

    # Train Random Forest classifier
    classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_rf.fit(X_train_rf, y_train_rf)

    # Predictions and evaluation for Random Forest
    y_pred_rf = classifier_rf.predict(X_test_rf)
    accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
    results['RandomForest'][max_features] = accuracy_rf

# Plotting the accuracy comparison
plt.plot(max_features_values, list(results['KNN'].values()), label='KNN')
plt.plot(max_features_values, list(results['RandomForest'].values()), label='Random Forest')
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: KNN vs Random Forest')
plt.legend()
plt.show()
