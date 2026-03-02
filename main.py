# ============================================
# SENTIMENT ANALYSIS - MOVIE REVIEWS PROJECT
# ============================================

# STEP 1: IMPORT TOOLS
print("Loading libraries...")
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# STEP 2: LOAD DATASET
print("\nLoading dataset...")
df = pd.read_csv('data/imdb.csv')
print(f"Total reviews loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))

# STEP 3: CLEAN THE TEXT
print("\nCleaning text...")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                                      # make lowercase
    text = re.sub(r'<.*?>', '', text)                       # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)                # remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]       # remove common words
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)
print("Text cleaning done!")

# STEP 4: PREPARE DATA
print("\nPreparing data...")
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# STEP 5: CONVERT TEXT TO NUMBERS (TF-IDF)
print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Conversion done!")

# STEP 6: TRAIN 3 MODELS AND COMPARE
print("\nTraining 3 models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(max_iter=2000)
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"  {name} Accuracy: {accuracy * 100:.2f}%")

# STEP 7: SHOW BEST MODEL
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\nBest Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy!")

# STEP 8: DETAILED REPORT FOR BEST MODEL
print("\n--- Detailed Report for Best Model ---")
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_vec)
print(classification_report(y_test, y_pred_best))

# STEP 9: VISUALIZATIONS
print("\nCreating visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Model Accuracy Comparison
model_names = list(results.keys())
accuracies = [v * 100 for v in results.values()]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = axes[0].bar(model_names, accuracies, color=colors)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([80, 100])
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.2f}%', ha='center', fontsize=11, fontweight='bold')

# Chart 2: Sentiment Distribution
sentiment_counts = df['sentiment'].value_counts()
axes[1].pie(sentiment_counts, labels=sentiment_counts.index,
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Dataset Sentiment Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results.png')
print("Visualization saved as results.png")
plt.show()

# STEP 10: LIVE PREDICTOR
print("\n" + "="*50)
print("LIVE SENTIMENT PREDICTOR")
print("="*50)
print("Type any movie review and I'll tell you if it's Positive or Negative!")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter your review: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    print(f"Sentiment: {prediction.upper()} ✓\n")

print("\nPROJECT COMPLETE!")


