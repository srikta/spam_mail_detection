import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # New import for SMOTE
import joblib

# Load and clean the data
file_path = 'email_spam.csv'
df = pd.read_csv(file_path)
df_cleaned = df.drop_duplicates()

# Map 'Category' to numerical values
df_cleaned['Category'] = df_cleaned['Category'].map({'spam': 0, 'ham': 1})

# Drop NaN values in the 'Category' column
df_cleaned = df_cleaned.dropna(subset=['Category'])

# Define feature and target variables
X = df_cleaned['Message']
y = df_cleaned['Category']

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)  # SMOTE instance
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample training data

# Train the model using Logistic Regression
model = LogisticRegression()
model.fit(X_train_smote, y_train_smote)  # Train on SMOTE-resampled data

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))


joblib.dump(model, 'spam_classifier_model_SMOTE.pkl')
