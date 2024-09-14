import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib



file_path = 'email_spam.csv'
df = pd.read_csv(file_path)
#print(df.head())
#print(df.columns)
#print(df.info())
#print(df.isnull().sum())
#print(df.duplicated().sum())
df_cleaned = df.drop_duplicates()
#print(df_cleaned.duplicated().sum())

df_cleaned['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

#print(df_cleaned.head())
#print(df_cleaned.info())
#print(df_cleaned.dtypes)

# Check for NaN values in 'category' column
nan_count = df_cleaned['Category'].isna().sum()

#print(f"Number of NaN values in 'category' column: {nan_count}")

df_cleaned = df_cleaned.dropna(subset=['Category'])

#print(df.isna().sum())

x = df_cleaned['Message']
y = df_cleaned['Category']

vectorizer = TfidfVectorizer(stop_words='english')
x_vectorized = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2,random_state=42, stratify=y)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Confusion matrix:\n',confusion_matrix(y_test,y_pred))
print('Classification Report:\n',classification_report(y_test, y_pred))

joblib.dump(model, 'spam_classifier_model.pkl')