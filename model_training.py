import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("news_dataset.csv")  # Must have 'headline' and 'category' columns

# Basic preprocessing
df['headline'] = df['headline'].str.lower()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['headline'], df['category'], test_size=0.2, stratify=df['category'], random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "category_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
