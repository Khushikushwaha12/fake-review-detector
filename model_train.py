import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Data load karein
df = pd.read_csv('reviews_data.csv')

# Column names se extra spaces aur quotes hatana
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

# Agar ab bhi error aaye toh pehle column ko hi review maan lo
if 'review' not in df.columns:
    df.rename(columns={df.columns[0]: 'review', df.columns[1]: 'label'}, inplace=True)

# Model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'].astype(str))
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# Files save karein
joblib.dump(model, 'review_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Success! Model training poori hui.")